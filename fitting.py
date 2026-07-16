"""BP3333 fitting routines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize

from .database import database_seed
from .geometry import ReferenceAirfoil, extract_bp3333_seed, read_airfoil
from .model import BP3333Parameters, generate_airfoil


@dataclass(frozen=True)
class FitResult:
    """Result returned by :func:`fit_airfoil`."""

    airfoil: str
    params: BP3333Parameters
    mae: float
    max_abs_error: float
    rms: float
    n_evaluations: int
    success: bool
    message: str
    optimizer: str
    rt_root_strategy: str
    pairing_method: str
    chord_normalised: bool
    chord_le: tuple[float, float]
    chord_te: tuple[float, float]
    geometry: dict[str, np.ndarray]


def fit_airfoil(
    path: str | Path,
    n_eval: int = 501,
    n_per_segment: int = 260,
    max_nfev: int = 900,
    use_database_seed: bool = False,
    optimizer: str = "slsqp",
    rt_root_strategy: str = "smallest",
    pairing_method: str = "auto",
    normalise_chord: bool | None = None,
    leading_edge: tuple[float, float] | None = None,
    trailing_edge: tuple[float, float] | None = None,
    ga_maxiter: int = 500,
    ga_population: int = 150,
    random_seed: int = 42,
) -> FitResult:
    """Fit pure BP3333 parameters to one reference airfoil file.

    The default path runs deterministic multi-start SLSQP. If every SLSQP run
    fails, a Differential Evolution genetic search is used automatically.
    """
    if optimizer not in {"slsqp", "least_squares", "ga"}:
        raise ValueError("optimizer must be 'slsqp', 'least_squares', or 'ga'.")
    if rt_root_strategy not in {"smallest", "midpoint"}:
        raise ValueError("rt_root_strategy must be 'smallest' or 'midpoint'.")
    if ga_maxiter < 1 or ga_population < 5:
        raise ValueError("GA requires ga_maxiter >= 1 and ga_population >= 5.")

    ref = read_airfoil(
        path,
        n_eval=n_eval,
        pairing_method=pairing_method,
        normalise_chord=normalise_chord,
        leading_edge=leading_edge,
        trailing_edge=trailing_edge,
    )
    seed = extract_bp3333_seed(ref)
    seed = _regularise_negative_camber_seed(seed)
    lower, upper = _parameter_bounds(ref, seed)
    x0 = np.clip(BP3333Parameters(**seed).to_vector(), lower, upper)

    best_x: np.ndarray | None = None
    best_cost = np.inf
    best_message = "No optimization result."
    best_success = False
    best_optimizer = "initial"
    n_evaluations = 0
    slsqp_succeeded = False
    successful_x: np.ndarray | None = None
    successful_cost = np.inf
    successful_message = ""
    successful_optimizer = ""

    extra_starts = []
    if use_database_seed:
        db_seed = database_seed(ref.name, Path.cwd())
        if db_seed is not None:
            db_vector = np.clip(BP3333Parameters(**db_seed).to_vector(), lower, upper)
            extra_starts.append(db_vector)

    starts = _multistart_vectors(x0, lower, upper, ref, extra_starts=extra_starts)
    for start in starts:
        try:
            generate_airfoil(
                BP3333Parameters.from_vector(start),
                n_per_segment=n_per_segment,
                rt_root_strategy=rt_root_strategy,
            )
        except Exception:
            continue
        start_cost = float(np.linalg.norm(_surface_residual(
            start,
            ref,
            n_per_segment,
            rt_root_strategy=rt_root_strategy,
            reference_seed=seed,
        )))
        if start_cost < best_cost:
            best_cost = start_cost
            best_x = start.copy()
            best_message = "Best deterministic start."
            best_success = False
            best_optimizer = "initial"

        if optimizer == "ga":
            continue
        if optimizer == "slsqp":
            result_x, result_cost, result_nfev, result_success, result_message = _slsqp_fit(
                start,
                lower,
                upper,
                ref,
                n_per_segment=n_per_segment,
                max_nfev=max_nfev,
                rt_root_strategy=rt_root_strategy,
                reference_seed=seed,
            )
            slsqp_succeeded = slsqp_succeeded or (result_x is not None and result_success)
        elif optimizer == "least_squares":
            result_x, result_cost, result_nfev, result_success, result_message = _least_squares_fit(
                start,
                lower,
                upper,
                ref,
                n_per_segment=n_per_segment,
                max_nfev=max_nfev,
                rt_root_strategy=rt_root_strategy,
                reference_seed=seed,
            )

        n_evaluations += result_nfev
        if result_x is None:
            continue
        if result_success and result_cost < successful_cost:
            successful_x = result_x.astype(float)
            successful_cost = result_cost
            successful_message = result_message
            successful_optimizer = optimizer
        if result_cost < best_cost:
            best_cost = result_cost
            best_x = result_x.astype(float)
            best_message = result_message
            best_success = result_success
            best_optimizer = optimizer

    if successful_x is not None:
        best_x = successful_x
        best_cost = successful_cost
        best_message = successful_message
        best_success = True
        best_optimizer = successful_optimizer

    if optimizer == "ga" or (optimizer == "slsqp" and not slsqp_succeeded):
        ga_x, ga_cost, ga_nfev, ga_success, ga_message = _ga_fit(
            best_x if best_x is not None else x0,
            lower,
            upper,
            ref,
            n_per_segment=n_per_segment,
            maxiter=ga_maxiter,
            population_size=ga_population,
            random_seed=random_seed,
            rt_root_strategy=rt_root_strategy,
            reference_seed=seed,
        )
        n_evaluations += ga_nfev
        if ga_x is not None:
            best_x = ga_x
            best_cost = ga_cost
            best_success = ga_success
            best_optimizer = "ga"
            prefix = "GA fallback after SLSQP failure" if optimizer == "slsqp" else "GA"
            best_message = f"{prefix}: {ga_message}"

    if best_x is None:
        raise ValueError(f"No feasible BP3333 parameter set found for {ref.name}.")

    params = BP3333Parameters.from_vector(best_x)
    geometry = generate_airfoil(
        params,
        n_per_segment=n_per_segment,
        rt_root_strategy=rt_root_strategy,
    )
    upper_error, lower_error = _surface_errors(geometry, ref)
    all_error = np.concatenate([upper_error, lower_error])

    return FitResult(
        airfoil=ref.name,
        params=params,
        mae=float(np.mean(np.abs(all_error))),
        max_abs_error=float(np.max(np.abs(all_error))),
        rms=float(np.sqrt(np.mean(all_error**2))),
        n_evaluations=n_evaluations,
        success=best_success,
        message=best_message,
        optimizer=best_optimizer,
        rt_root_strategy=rt_root_strategy,
        pairing_method=ref.pairing_method,
        chord_normalised=ref.chord_normalised,
        chord_le=ref.chord_le,
        chord_te=ref.chord_te,
        geometry=geometry,
    )


def _surface_residual(
    values: np.ndarray,
    ref: ReferenceAirfoil,
    n_per_segment: int,
    rt_root_strategy: str,
    reference_seed: dict[str, float] | None = None,
) -> np.ndarray:
    """Residual vector against sampled upper and lower reference surfaces."""
    try:
        params = BP3333Parameters.from_vector(values)
        geometry = generate_airfoil(
            params,
            n_per_segment=n_per_segment,
            rt_root_strategy=rt_root_strategy,
        )
        upper_error, lower_error = _surface_errors(geometry, ref)
        residual = np.concatenate([upper_error, lower_error])

        penalty = _shape_penalties(values, geometry, ref, reference_seed=reference_seed)
        if penalty.size:
            residual = np.concatenate([residual, penalty])
        return residual
    except Exception:
        return np.full(ref.x_eval.size * 2 + 4, 1e2)


def _objective(
    values: np.ndarray,
    ref: ReferenceAirfoil,
    n_per_segment: int,
    rt_root_strategy: str,
    reference_seed: dict[str, float] | None = None,
) -> float:
    """Scalar objective, mirroring the original BP3434 SLSQP style."""
    residual = _surface_residual(
        values,
        ref,
        n_per_segment,
        rt_root_strategy=rt_root_strategy,
        reference_seed=reference_seed,
    )
    return float(np.linalg.norm(residual))


def _least_squares_fit(
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ref: ReferenceAirfoil,
    n_per_segment: int,
    max_nfev: int,
    rt_root_strategy: str,
    reference_seed: dict[str, float] | None = None,
) -> tuple[np.ndarray | None, float, int, bool, str]:
    """Optional least-squares backend retained for comparison studies."""
    result = optimize.least_squares(
        lambda values: _surface_residual(
            values,
            ref,
            n_per_segment,
            rt_root_strategy=rt_root_strategy,
            reference_seed=reference_seed,
        ),
        start,
        bounds=(lower, upper),
        x_scale=np.maximum(np.abs(start), 1e-3),
        loss="soft_l1",
        f_scale=2e-4,
        max_nfev=max_nfev,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    try:
        generate_airfoil(
            BP3333Parameters.from_vector(result.x),
            n_per_segment=n_per_segment,
            rt_root_strategy=rt_root_strategy,
        )
    except Exception:
        return None, np.inf, int(result.nfev), bool(result.success), str(result.message)
    cost = _objective(
        result.x,
        ref,
        n_per_segment,
        rt_root_strategy=rt_root_strategy,
        reference_seed=reference_seed,
    )
    return result.x.astype(float), cost, int(result.nfev), bool(result.success), str(result.message)


def _slsqp_fit(
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ref: ReferenceAirfoil,
    n_per_segment: int,
    max_nfev: int,
    rt_root_strategy: str,
    reference_seed: dict[str, float] | None = None,
) -> tuple[np.ndarray | None, float, int, bool, str]:
    """SLSQP backend following the original BP3434 optimisation pattern."""
    result = optimize.minimize(
        lambda values: _objective(
            values,
            ref,
            n_per_segment,
            rt_root_strategy=rt_root_strategy,
            reference_seed=reference_seed,
        ),
        start,
        method="SLSQP",
        bounds=optimize.Bounds(lower, upper),
        constraints=_nonlinear_constraints(rt_root_strategy),
        options={"maxiter": max_nfev, "disp": False, "ftol": 1e-11},
    )
    try:
        generate_airfoil(
            BP3333Parameters.from_vector(result.x),
            n_per_segment=n_per_segment,
            rt_root_strategy=rt_root_strategy,
        )
    except Exception:
        return None, np.inf, int(result.nfev), bool(result.success), str(result.message)
    cost = _objective(
        result.x,
        ref,
        n_per_segment,
        rt_root_strategy=rt_root_strategy,
        reference_seed=reference_seed,
    )
    return result.x.astype(float), cost, int(result.nfev), bool(result.success), str(result.message)


def _ga_fit(
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ref: ReferenceAirfoil,
    n_per_segment: int,
    maxiter: int,
    population_size: int,
    random_seed: int,
    rt_root_strategy: str,
    reference_seed: dict[str, float] | None = None,
) -> tuple[np.ndarray | None, float, int, bool, str]:
    """Differential Evolution fallback using the paper's GA-style settings."""
    rng = np.random.default_rng(random_seed)
    span = upper - lower
    population = start + rng.normal(0.0, 0.18, size=(population_size, len(start))) * span
    population = np.clip(population, lower, upper)
    population[0] = np.clip(start, lower, upper)

    def ga_objective(values: np.ndarray) -> float:
        return _objective(
            values,
            ref,
            n_per_segment,
            rt_root_strategy=rt_root_strategy,
            reference_seed=reference_seed,
        )

    result = optimize.differential_evolution(
        ga_objective,
        bounds=list(zip(lower, upper)),
        strategy="randtobest1bin",
        maxiter=maxiter,
        mutation=0.85,
        recombination=1.0,
        seed=random_seed,
        init=population,
        polish=False,
        updating="deferred",
        workers=1,
        tol=1e-7,
        atol=0.0,
    )
    try:
        generate_airfoil(
            BP3333Parameters.from_vector(result.x),
            n_per_segment=n_per_segment,
            rt_root_strategy=rt_root_strategy,
        )
    except Exception:
        return None, np.inf, int(result.nfev), False, str(result.message)
    cost = _objective(
        result.x,
        ref,
        n_per_segment,
        rt_root_strategy=rt_root_strategy,
        reference_seed=reference_seed,
    )
    return result.x.astype(float), cost, int(result.nfev), bool(result.success), str(result.message)


def _nonlinear_constraints(rt_root_strategy: str) -> list[dict[str, object]]:
    """Geometric BP3333 constraints expressed in SLSQP inequality form."""
    return [
        {
            "type": "ineq",
            "fun": lambda values: _thickness_control_monotonicity(values, rt_root_strategy),
        },
        {"type": "ineq", "fun": _camber_control_monotonicity},
    ]


def _thickness_control_monotonicity(values: np.ndarray, rt_root_strategy: str) -> np.ndarray:
    """Require BP3333 thickness control polygon x-coordinates to stay ordered."""
    try:
        from .model import thickness_control_points, solve_b9

        p = BP3333Parameters.from_vector(values)
        b9 = solve_b9(p.r_le, p.x_t, p.y_t, p.k_t, root_strategy=rt_root_strategy)
        x_le, y_le, x_te, y_te = thickness_control_points(p, b9)
        shoulder = y_le[1]
        return np.array(
            [
                b9,
                p.x_t - b9,
                x_te[1] - p.x_t,
                x_te[2] - x_te[1],
                1.0 - x_te[2],
                shoulder,
                p.y_t - shoulder,
                shoulder - p.dz_te,
            ],
            dtype=float,
        )
    except Exception:
        return np.full(8, -1.0)


def _camber_control_monotonicity(values: np.ndarray) -> np.ndarray:
    """Require BP3333 camber control polygon x-coordinates to stay ordered."""
    try:
        from .model import camber_control_points, solve_b1

        p = BP3333Parameters.from_vector(values)
        if abs(p.y_c) < 1e-6:
            return np.ones(6, dtype=float)
        b1 = solve_b1(p.gamma_le, p.y_c, p.k_c, p.z_te, p.alpha_te)
        x_le, _, x_te, _ = camber_control_points(p, b1)
        return np.array(
            [
                x_le[1],
                x_le[2] - x_le[1],
                p.x_c - x_le[2],
                x_te[1] - p.x_c,
                x_te[2] - x_te[1],
                1.0 - x_te[2],
            ],
            dtype=float,
        )
    except Exception:
        return np.full(6, -1.0)


def _surface_errors(geometry: dict[str, np.ndarray], ref: ReferenceAirfoil) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate reconstructed y-errors at the reference x stations."""
    upper_x, upper_y = _unique_sorted(geometry["upper_x"], geometry["upper_y"])
    lower_x, lower_y = _unique_sorted(geometry["lower_x"], geometry["lower_y"])

    if len(upper_x) < 8 or len(lower_x) < 8:
        raise ValueError("Generated surfaces have too few usable points.")

    upper_spline = interpolate.PchipInterpolator(upper_x, upper_y, extrapolate=True)
    lower_spline = interpolate.PchipInterpolator(lower_x, lower_y, extrapolate=True)
    x_eval = ref.x_eval
    upper_model = upper_spline(x_eval)
    lower_model = lower_spline(x_eval)
    if not (np.all(np.isfinite(upper_model)) and np.all(np.isfinite(lower_model))):
        raise ValueError("Generated surface interpolation failed.")
    return upper_model - ref.upper_y, lower_model - ref.lower_y


def _shape_penalties(
    values: np.ndarray,
    geometry: dict[str, np.ndarray],
    ref: ReferenceAirfoil,
    reference_seed: dict[str, float] | None = None,
) -> np.ndarray:
    """Soft penalties for known BP3333 feasibility preferences."""
    p = BP3333Parameters.from_vector(values)
    penalties: list[float] = []

    # Keep maxima near their reference definitions while allowing fitting freedom.
    seed = reference_seed if reference_seed is not None else extract_bp3333_seed(ref)
    penalties.append(0.05 * (p.x_t - seed["x_t"]))
    penalties.append(0.05 * (p.x_c - seed["x_c"]))

    # Discourage negative thickness and swapped surfaces with fixed-size terms.
    penalties.append(20.0 * min(float(np.min(geometry["thickness_y"])), 0.0))
    gap = geometry["upper_y"] - geometry["lower_y"]
    penalties.append(20.0 * min(float(np.min(gap)), 0.0))
    return np.asarray(penalties, dtype=float)


def _parameter_bounds(ref: ReferenceAirfoil, seed: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Construct broad but physical bounds for the BP3333 vector."""
    no_internal_camber_peak = abs(seed["y_c"]) <= 5e-5 and abs(seed["z_te"]) > 5e-5
    cambered = abs(seed["y_c"]) > 5e-5 or abs(seed["z_te"]) > 5e-5

    lower = np.array(
        [
            -0.12,
            0.04,
            max(1e-4, 0.55 * seed["y_t"]),
            -8.0,
            1e-4,
            -0.8,
            0.04,
            max(-0.35, min(-0.08, 1.5 * seed["y_c"] - 0.03)) if cambered else -1e-5,
            -8.0,
            -0.9,
            0.0,
            min(-0.08, seed["z_te"] - max(0.02, 0.5 * abs(seed["z_te"]))),
        ],
        dtype=float,
    )
    upper = np.array(
        [
            -1e-6,
            0.82,
            min(0.30, 1.55 * seed["y_t"] + 0.01),
            -1e-5,
            1.2,
            0.8,
            0.95,
            min(0.35, max(0.12, 1.5 * seed["y_c"] + 0.03)) if cambered else 1e-5,
            -1e-5,
            0.9,
            max(0.08, 2.0 * seed["dz_te"] + 0.01),
            max(0.08, seed["z_te"] + max(0.02, 0.5 * abs(seed["z_te"]))),
        ],
        dtype=float,
    )

    # Radius and curvature bounds follow the extracted profile more tightly.
    lower[0] = max(lower[0], 3.0 * seed["r_le"])
    upper[0] = max(upper[0], 0.05 * seed["r_le"])
    lower[3] = min(lower[3], 3.0 * seed["k_t"])
    upper[3] = min(upper[3], 0.2 * seed["k_t"])
    if seed["y_c"] < -5e-5:
        lower[8] = 1e-5
        upper[8] = max(8.0, 4.0 * seed["k_c"])
    else:
        lower[8] = min(lower[8], 4.0 * seed["k_c"])
        upper[8] = min(upper[8], -1e-5)

    if not cambered:
        lower[5] = -1e-5
        upper[5] = 1e-5
        lower[7] = -1e-7
        upper[7] = 1e-7
        lower[9] = -1e-5
        upper[9] = 1e-5
        lower[11] = -1e-6
        upper[11] = 1e-6
    elif no_internal_camber_peak:
        lower[7] = -1e-7
        upper[7] = 1e-7
        lower[8] = -0.05
        upper[8] = -1e-5

    return lower, upper


def _regularise_negative_camber_seed(seed: dict[str, float]) -> dict[str, float]:
    """Reduce only an infeasible negative-camber curvature seed.

    A local trough curvature can be much sharper than the globally constrained
    BP3333 control polygon permits. Positive-camber initialization is left
    untouched; for a negative trough we retain the extracted location, depth,
    and edge angles and choose the closest scaled curvature with ordered control
    points.
    """
    if seed["y_c"] >= -5e-5:
        return seed

    from .model import camber_control_points, solve_b1

    for scale in (1.0, 0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05):
        candidate = dict(seed)
        candidate["k_c"] = max(1e-4, scale * seed["k_c"])
        try:
            params = BP3333Parameters(**candidate)
            b1 = solve_b1(
                params.gamma_le,
                params.y_c,
                params.k_c,
                params.z_te,
                params.alpha_te,
            )
            x_le, _, x_te, _ = camber_control_points(params, b1)
            control_x = np.concatenate([x_le, x_te[1:]])
            if np.all(np.diff(control_x) > 1e-10):
                return candidate
        except ValueError:
            continue
    return seed


def _multistart_vectors(
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    ref: ReferenceAirfoil,
    extra_starts: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Deterministic starts around the extracted parameter vector."""
    starts = [x0]
    if extra_starts:
        starts.extend(np.clip(start, lower, upper) for start in extra_starts)
    perturbations = [
        {3: 0.7, 4: 0.7},
        {3: 1.3, 4: 1.25},
        {3: 6.0, 4: 0.8},
        {3: 10.0, 4: 0.8},
        {0: 0.08, 3: 6.0, 4: 0.8},
        {0: 0.04, 3: 10.0, 4: 0.8},
        {0: 0.7, 3: 0.8},
        {0: 1.3, 3: 1.2},
    ]
    if float(np.max(np.abs(ref.camber_y))) > 5e-5:
        perturbations.extend([{5: 0.7, 8: 0.75}, {5: 1.3, 8: 1.25}])

    for spec in perturbations:
        candidate = x0.copy()
        for idx, scale in spec.items():
            candidate[idx] *= scale
        starts.append(np.clip(candidate, lower, upper))

    absolute = x0.copy()
    absolute[0] = -5e-4
    absolute[1] = max(0.15, x0[1])
    absolute[3] = -8.0
    absolute[4] = 0.05
    starts.append(np.clip(absolute, lower, upper))

    gentle_camber = x0.copy()
    gentle_camber[5] = 0.01 if x0[5] >= 0.0 else -0.01
    gentle_camber[8] = max(0.02, x0[8]) if x0[7] < 0.0 else min(-0.02, x0[8])
    gentle_camber[9] = 0.01 if x0[9] >= 0.0 else -0.01
    starts.append(np.clip(gentle_camber, lower, upper))

    aft_loaded = x0.copy()
    aft_loaded[0] = -5e-4
    aft_loaded[1] = min(max(0.25, lower[1]), upper[1])
    aft_loaded[3] = -1.0
    camber_sign = -1.0 if x0[7] < 0.0 else 1.0
    aft_loaded[5] = 0.02 * camber_sign
    aft_loaded[8] = -0.5 * camber_sign
    aft_loaded[9] = 0.05 * camber_sign
    starts.append(np.clip(aft_loaded, lower, upper))

    return starts


def _unique_sorted(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort and deduplicate generated surface coordinates for interpolation."""
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    keep = np.concatenate([[True], np.diff(x_sorted) > 1e-10])
    return x_sorted[keep], y_sorted[keep]
