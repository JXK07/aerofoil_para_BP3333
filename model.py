"""BP3333 airfoil generation from physical parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import interpolate

from .bezier import cosine_sine_parameters, cubic_bezier


PARAMETER_NAMES = (
    "r_le",
    "x_t",
    "y_t",
    "k_t",
    "beta_te",
    "gamma_le",
    "x_c",
    "y_c",
    "k_c",
    "alpha_te",
    "dz_te",
    "z_te",
)


@dataclass(frozen=True)
class BP3333Parameters:
    """The twelve BP3333 parameters.

    Angles are in radians.  Curvatures and leading-edge radius follow the sign
    convention used by Derksen/Rogalsky: nose radius and maximum curvatures are
    negative; the curvature of an internal negative-camber minimum is positive.
    """

    r_le: float
    x_t: float
    y_t: float
    k_t: float
    beta_te: float
    gamma_le: float
    x_c: float
    y_c: float
    k_c: float
    alpha_te: float
    dz_te: float = 0.0
    z_te: float = 0.0

    @classmethod
    def from_vector(cls, values: np.ndarray) -> "BP3333Parameters":
        """Create parameters from the canonical vector order."""
        return cls(**dict(zip(PARAMETER_NAMES, np.asarray(values, dtype=float))))

    def to_vector(self) -> np.ndarray:
        """Return parameters in the canonical vector order."""
        data = asdict(self)
        return np.array([data[name] for name in PARAMETER_NAMES], dtype=float)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable dictionary."""
        return {key: float(value) for key, value in asdict(self).items()}


def generate_airfoil(
    params: BP3333Parameters | dict[str, float],
    n_per_segment: int = 220,
    rt_root_strategy: str = "smallest",
) -> dict[str, np.ndarray]:
    """Generate thickness, camber, and upper/lower surfaces for BP3333."""
    p = params if isinstance(params, BP3333Parameters) else BP3333Parameters(**params)

    x_t, y_t = thickness_distribution(p, n_per_segment, rt_root_strategy=rt_root_strategy)
    x_c, y_c = camber_distribution(p, n_per_segment, rt_root_strategy=rt_root_strategy)
    upper_x, upper_y, lower_x, lower_y = thickness_camber_to_surfaces(x_t, y_t, x_c, y_c)

    return {
        "thickness_x": x_t,
        "thickness_y": y_t,
        "camber_x": x_c,
        "camber_y": y_c,
        "upper_x": upper_x,
        "upper_y": upper_y,
        "lower_x": lower_x,
        "lower_y": lower_y,
    }


def thickness_distribution(
    params: BP3333Parameters,
    n_per_segment: int,
    rt_root_strategy: str = "smallest",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the BP3333 thickness distribution from two cubic Bezier curves."""
    b9 = solve_b9(
        params.r_le,
        params.x_t,
        params.y_t,
        params.k_t,
        root_strategy=rt_root_strategy,
    )
    leading_x, leading_y, trailing_x, trailing_y = thickness_control_points(params, b9)
    u_le, u_te = cosine_sine_parameters(n_per_segment)

    x_le = cubic_bezier(leading_x, u_le)
    y_le = cubic_bezier(leading_y, u_le)
    x_te = cubic_bezier(trailing_x, u_te[1:])
    y_te = cubic_bezier(trailing_y, u_te[1:])

    x = np.concatenate([x_le, x_te])
    y = np.concatenate([y_le, y_te])
    return _validate_distribution(x, y, "thickness")


def camber_distribution(
    params: BP3333Parameters,
    n_per_segment: int,
    rt_root_strategy: str = "smallest",
) -> tuple[np.ndarray, np.ndarray]:
    """Build the BP3333 camber distribution from two cubic Bezier curves."""
    if abs(params.y_c) < 1e-6 and abs(params.z_te) < 1e-8:
        x, _ = thickness_distribution(params, n_per_segment, rt_root_strategy=rt_root_strategy)
        return x, np.zeros_like(x)
    if abs(params.y_c) < 1e-6:
        raise ValueError("Nonzero trailing-edge camber requires a valid BP3333 internal camber crest.")

    b1 = solve_b1(params.gamma_le, params.y_c, params.k_c, params.z_te, params.alpha_te)
    leading_x, leading_y, trailing_x, trailing_y = camber_control_points(params, b1)
    u_le, u_te = cosine_sine_parameters(n_per_segment)

    x_le = cubic_bezier(leading_x, u_le)
    y_le = cubic_bezier(leading_y, u_le)
    x_te = cubic_bezier(trailing_x, u_te[1:])
    y_te = cubic_bezier(trailing_y, u_te[1:])

    x = np.concatenate([x_le, x_te])
    y = np.concatenate([y_le, y_te])
    return _validate_distribution(x, y, "camber")


def thickness_control_points(
    params: BP3333Parameters,
    b9: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return BP3333 cubic control points for thickness."""
    shoulder = 1.5 * params.k_t * (params.x_t - b9) ** 2 + params.y_t
    x_le = np.array([0.0, 0.0, b9, params.x_t])
    y_le = np.array([0.0, shoulder, params.y_t, params.y_t])
    x_te = np.array(
        [
            params.x_t,
            2.0 * params.x_t - b9,
            1.0 + (params.dz_te - shoulder) / np.tan(params.beta_te),
            1.0,
        ]
    )
    y_te = np.array([params.y_t, params.y_t, shoulder, params.dz_te])
    return x_le, y_le, x_te, y_te


def camber_control_points(
    params: BP3333Parameters,
    b1: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return BP3333 cubic control points for camber."""
    left_root = _positive_sqrt(2.0 * (b1 - params.y_c) / (3.0 * params.k_c))
    x_le = np.array([0.0, b1 / np.tan(params.gamma_le), params.x_c - left_root, params.x_c])
    y_le = np.array([0.0, b1, params.y_c, params.y_c])

    right_root = _positive_sqrt(2.0 * (b1 - params.y_c) / (3.0 * params.k_c))
    x_te = np.array(
        [
            params.x_c,
            params.x_c + right_root,
            1.0 + (params.z_te - b1) / np.tan(params.alpha_te),
            1.0,
        ]
    )
    y_te = np.array([params.y_c, params.y_c, b1, params.z_te])
    return x_le, y_le, x_te, y_te


def thickness_camber_to_surfaces(
    thickness_x: np.ndarray,
    thickness_y: np.ndarray,
    camber_x: np.ndarray,
    camber_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Offset the camber line by the local normal using the thickness curve."""
    thickness_spline = interpolate.PchipInterpolator(thickness_x, thickness_y, extrapolate=True)
    thickness_at_camber = thickness_spline(camber_x)

    camber_spline = interpolate.CubicSpline(camber_x, camber_y, bc_type="natural")
    theta = np.arctan(camber_spline.derivative()(camber_x))

    upper_x = camber_x - thickness_at_camber * np.sin(theta)
    upper_y = camber_y + thickness_at_camber * np.cos(theta)
    lower_x = camber_x + thickness_at_camber * np.sin(theta)
    lower_y = camber_y - thickness_at_camber * np.cos(theta)
    return upper_x, upper_y, lower_x, lower_y


def solve_b9(
    r_le: float,
    x_t: float,
    y_t: float,
    k_t: float,
    root_strategy: str = "smallest",
) -> float:
    """Solve the BP3333 r_t/b9 equation and select a bounded real root.

    ``smallest`` follows Derksen and Rogalsky. ``midpoint`` retains the former
    implementation's choice of the root closest to the admissible interval
    midpoint for comparison and reproducibility.
    """
    if not (r_le < 0.0 and x_t > 0.0 and y_t > 0.0 and k_t < 0.0):
        raise ValueError("Invalid thickness parameters for b9 solve.")
    if root_strategy not in {"smallest", "midpoint"}:
        raise ValueError("root_strategy must be 'smallest' or 'midpoint'.")

    coefficients = np.array(
        [
            (27.0 / 4.0) * k_t**2,
            -27.0 * k_t**2 * x_t,
            9.0 * k_t * y_t + (81.0 / 2.0) * k_t**2 * x_t**2,
            2.0 * r_le - 18.0 * k_t * x_t * y_t - 27.0 * k_t**2 * x_t**3,
            3.0 * y_t**2 + 9.0 * k_t * x_t**2 * y_t + (27.0 / 4.0) * k_t**2 * x_t**4,
        ],
        dtype=float,
    )

    lower = max(0.0, x_t - np.sqrt(max(-2.0 * y_t / (3.0 * k_t), 0.0)))
    upper = x_t
    root_tolerance = 1e-9
    roots = sorted(
        float(root.real)
        for root in np.roots(coefficients)
        if abs(float(root.imag)) <= root_tolerance * max(1.0, abs(float(root.real)))
        and lower + root_tolerance < float(root.real) < upper - root_tolerance
    )
    roots = [root for index, root in enumerate(roots) if index == 0 or abs(root - roots[index - 1]) > 1e-8]
    if not roots:
        raise ValueError("Could not solve BP3333 b9 thickness point.")
    if root_strategy == "smallest":
        return roots[0]
    midpoint = 0.5 * (lower + upper)
    return min(roots, key=lambda value: abs(value - midpoint))


def solve_b1(gamma_le: float, y_c: float, k_c: float, z_te: float, alpha_te: float) -> float:
    """Solve the BP3333 b1 camber control point equation."""
    camber_sign = 1.0 if y_c > 0.0 else -1.0
    gamma_work = camber_sign * gamma_le
    y_c_work = camber_sign * y_c
    k_c_work = camber_sign * k_c
    z_te_work = camber_sign * z_te
    alpha_work = camber_sign * alpha_te
    if abs(gamma_work) < 1e-7 or abs(alpha_work) < 1e-7 or k_c_work >= 0.0:
        raise ValueError("Invalid camber parameters for b1 solve.")

    cot_sum = 1.0 / np.tan(gamma_work) + 1.0 / np.tan(alpha_work)
    t1 = 3.0 * k_c_work * cot_sum**2
    t2 = 16.0 + 3.0 * k_c_work * cot_sum * (1.0 + z_te_work / np.tan(alpha_work))
    radicand = 16.0 + 6.0 * k_c_work * cot_sum * (
        1.0 - y_c_work * cot_sum + z_te_work / np.tan(alpha_work)
    )
    if radicand < 0.0 or abs(t1) < 1e-12:
        raise ValueError("Invalid BP3333 b1 radicand.")

    roots = [
        (t2 + 4.0 * np.sqrt(radicand)) / t1,
        (t2 - 4.0 * np.sqrt(radicand)) / t1,
    ]
    valid = [root for root in roots if np.isfinite(root) and 0.0 < root < y_c_work]
    if not valid:
        raise ValueError("Could not solve BP3333 b1 camber point.")
    return float(camber_sign * valid[0])


def _validate_distribution(x: np.ndarray, y: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Check monotonicity and finiteness before interpolation."""
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"{label} distribution contains non-finite values.")
    if np.any(np.diff(x) <= 1e-12):
        raise ValueError(f"{label} distribution is not monotonic in x.")
    if np.min(x) < -1e-8 or np.max(x) > 1.0 + 1e-8:
        raise ValueError(f"{label} distribution leaves the unit chord.")
    return x, y


def _positive_sqrt(value: float) -> float:
    """Square-root helper that treats negative radicands as infeasible."""
    if value < 0.0 or not np.isfinite(value):
        raise ValueError("BP3333 control point radicand is negative.")
    return float(np.sqrt(value))
