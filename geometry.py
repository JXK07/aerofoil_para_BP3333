"""Reference airfoil reading and geometry extraction.

The input airfoils in this workspace follow the common Selig-style contour:
trailing edge over the upper surface to the leading edge, then back along the
lower surface to the trailing edge.  This module converts that contour into
upper/lower splines, camber, thickness, and the physical BP3333 seed values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize


@dataclass(frozen=True)
class ReferenceAirfoil:
    """Surface and derived distributions for a reference airfoil."""

    name: str
    contour: np.ndarray
    x_eval: np.ndarray
    upper_y: np.ndarray
    lower_y: np.ndarray
    camber_y: np.ndarray
    thickness_y: np.ndarray
    camber_slope: np.ndarray
    thickness_slope: np.ndarray
    upper_spline: interpolate.PchipInterpolator
    lower_spline: interpolate.PchipInterpolator
    pairing_method: str
    chord_normalised: bool
    chord_le: tuple[float, float]
    chord_te: tuple[float, float]


def read_airfoil(
    path: str | Path,
    n_eval: int = 501,
    pairing_method: str = "auto",
    normalise_chord: bool | None = None,
    leading_edge: tuple[float, float] | np.ndarray | None = None,
    trailing_edge: tuple[float, float] | np.ndarray | None = None,
) -> ReferenceAirfoil:
    """Read a contour file and return regularly sampled reference geometry.

    ``pairing_method`` selects common-x interpolation (``"x"``), the native
    BP source-code station method (``"source"``), independent normalized-arc
    interpolation (``"arc"``), or geometry-based automatic selection.
    ``normalise_chord=None`` preserves coordinates already expressed on a unit,
    horizontal chord and normalizes all other inputs.
    """
    if pairing_method not in {"auto", "x", "source", "arc"}:
        raise ValueError("pairing_method must be 'auto', 'x', 'source', or 'arc'.")
    airfoil_path = Path(path)
    metadata_le, metadata_te = _read_chord_metadata(airfoil_path)
    if leading_edge is None:
        leading_edge = metadata_le
    if trailing_edge is None:
        trailing_edge = metadata_te
    if (leading_edge is None) != (trailing_edge is None):
        raise ValueError("leading_edge and trailing_edge must be supplied together.")
    if leading_edge is not None:
        leading_edge = np.asarray(leading_edge, dtype=float)
        trailing_edge = np.asarray(trailing_edge, dtype=float)
    contour = np.genfromtxt(airfoil_path, dtype=float, skip_header=1)
    contour = contour[np.all(np.isfinite(contour), axis=1)]
    if contour.ndim != 2 or contour.shape[1] < 2:
        raise ValueError(f"No coordinate data found in {airfoil_path}.")

    contour = contour[:, :2].astype(float)
    inferred_le, inferred_te, inferred_idx_le = _infer_chord_endpoints(contour)
    chord_le = leading_edge if leading_edge is not None else inferred_le
    chord_te = trailing_edge if trailing_edge is not None else inferred_te
    idx_le = (
        int(np.argmin(np.linalg.norm(contour - chord_le, axis=1)))
        if leading_edge is not None
        else inferred_idx_le
    )
    chord_normalised = (
        not _is_unit_horizontal_chord(chord_le, chord_te)
        if normalise_chord is None
        else bool(normalise_chord)
    )
    if chord_normalised:
        contour = _normalise_chord(contour, leading_edge=chord_le, trailing_edge=chord_te)
    contour[:, 1] -= contour[idx_le, 1]

    upper = contour[: idx_le + 1][::-1]
    lower = contour[idx_le:]
    beta = np.linspace(0.0, np.pi, n_eval)
    x_eval = 0.5 * (1.0 - np.cos(beta))

    resolved_pairing = _select_pairing_method(upper, lower) if pairing_method == "auto" else pairing_method
    if resolved_pairing == "arc":
        camber, thickness = _arc_reference_distributions(upper, lower, x_eval)
        camber_slope = _safe_derivative(x_eval, camber)
        theta = np.arctan(camber_slope)
        upper_geometry_x = x_eval - thickness * np.sin(theta)
        upper_geometry_y = camber + thickness * np.cos(theta)
        lower_geometry_x = x_eval + thickness * np.sin(theta)
        lower_geometry_y = camber - thickness * np.cos(theta)
        upper_x, upper_y = _unique_xy(upper_geometry_x, upper_geometry_y)
        lower_x, lower_y = _unique_xy(lower_geometry_x, lower_geometry_y)
    else:
        upper_x, upper_y = _unique_xy(upper[:, 0], upper[:, 1])
        lower_x, lower_y = _unique_xy(lower[:, 0], lower[:, 1])

    upper_spline = interpolate.PchipInterpolator(upper_x, upper_y, extrapolate=True)
    lower_spline = interpolate.PchipInterpolator(lower_x, lower_y, extrapolate=True)

    yu = upper_spline(x_eval)
    yl = lower_spline(x_eval)

    if resolved_pairing == "x":
        camber = 0.5 * (yu + yl)
        raw_half_thickness = 0.5 * (yu - yl)
        camber_slope = _safe_derivative(x_eval, camber)
        theta = np.arctan(camber_slope)
        thickness = raw_half_thickness / np.cos(theta)
    elif resolved_pairing == "source":
        camber, thickness = _source_reference_distributions(upper, lower, x_eval)
        camber_slope = _safe_derivative(x_eval, camber)
    thickness_slope = _safe_derivative(x_eval, thickness)

    return ReferenceAirfoil(
        name=airfoil_path.stem,
        contour=contour,
        x_eval=x_eval,
        upper_y=yu,
        lower_y=yl,
        camber_y=camber,
        thickness_y=thickness,
        camber_slope=camber_slope,
        thickness_slope=thickness_slope,
        upper_spline=upper_spline,
        lower_spline=lower_spline,
        pairing_method=resolved_pairing,
        chord_normalised=chord_normalised,
        chord_le=(float(chord_le[0]), float(chord_le[1])),
        chord_te=(float(chord_te[0]), float(chord_te[1])),
    )


def extract_bp3333_seed(ref: ReferenceAirfoil) -> dict[str, float]:
    """Estimate the twelve BP3333 parameters from reference distributions."""
    x = ref.x_eval
    t = ref.thickness_y
    c = ref.camber_y

    thickness_search = np.where(x > 0.01)[0]
    idx_t = int(thickness_search[np.argmax(t[thickness_search])]) if len(thickness_search) else int(np.argmax(t))
    x_t = float(x[idx_t])
    y_t = float(max(t[idx_t], 1e-5))
    k_t = float(min(_local_second_derivative(x, t, idx_t), -1e-4))

    idx_c_max = int(np.argmax(c))
    idx_c_min = int(np.argmin(c))
    idx_c = idx_c_min if abs(float(c[idx_c_min])) > abs(float(c[idx_c_max])) else idx_c_max
    x_c = float(x[idx_c])
    y_c = float(c[idx_c])
    raw_k_c = _local_second_derivative(x, c, idx_c)
    k_c = float(min(raw_k_c, -1e-4) if y_c >= 0.0 else max(raw_k_c, 1e-4))

    z_te = float(0.5 * (ref.upper_spline(1.0) + ref.lower_spline(1.0)))
    dz_te = float(max(0.5 * (ref.upper_spline(1.0) - ref.lower_spline(1.0)), 0.0))

    r_le = -abs(_leading_edge_radius(ref))
    beta_te = float(max(np.arctan(-_tail_slope(x, t)), 1e-4))
    gamma_le = float(np.arctan(_nose_slope(x, c)))
    alpha_te = float(np.arctan(-_tail_slope(x, c)))

    if abs(y_c) < 1e-6 or x_c > 0.97:
        x_c = 0.4
        y_c = 0.0
        k_c = -0.05
        if abs(z_te) < 1e-5:
            gamma_le = 0.0
            alpha_te = 0.0
        z_te = 0.0 if abs(z_te) < 1e-5 else z_te

    return {
        "r_le": r_le,
        "x_t": x_t,
        "y_t": y_t,
        "k_t": k_t,
        "beta_te": beta_te,
        "gamma_le": gamma_le,
        "x_c": x_c,
        "y_c": y_c,
        "k_c": k_c,
        "alpha_te": alpha_te,
        "dz_te": dz_te,
        "z_te": z_te,
    }


def _read_chord_metadata(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Read optional leading/trailing-edge coordinates from comment metadata."""
    leading_edge = None
    trailing_edge = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("# leading_edge "):
                leading_edge = np.asarray([float(value) for value in stripped.split()[-2:]], dtype=float)
            elif stripped.startswith("# trailing_edge "):
                trailing_edge = np.asarray([float(value) for value in stripped.split()[-2:]], dtype=float)
            elif stripped and not stripped.startswith("#") and leading_edge is not None and trailing_edge is not None:
                break
    if (leading_edge is None) != (trailing_edge is None):
        raise ValueError(f"Both leading_edge and trailing_edge metadata are required in {path}.")
    return leading_edge, trailing_edge


def _normalise_chord(
    points: np.ndarray,
    leading_edge: np.ndarray | None = None,
    trailing_edge: np.ndarray | None = None,
) -> np.ndarray:
    """Translate, rotate, and scale coordinates onto the unit chord."""
    if leading_edge is None or trailing_edge is None:
        inferred_le, inferred_te, _ = _infer_chord_endpoints(points)
        leading_edge = inferred_le
        trailing_edge = inferred_te
    chord_vector = trailing_edge - leading_edge
    chord = float(np.linalg.norm(chord_vector))
    if chord <= 0.0:
        raise ValueError("Airfoil chord length is zero.")
    chord_direction = chord_vector / chord
    normal_direction = np.array([-chord_direction[1], chord_direction[0]])
    relative = points - leading_edge
    return np.column_stack(
        [
            relative @ chord_direction / chord,
            relative @ normal_direction / chord,
        ]
    )


def _arc_reference_distributions(
    upper: np.ndarray,
    lower: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract camber/thickness after independent normalized-arc resampling."""
    n_pairs = max(len(upper), len(lower), 80)
    upper_paired = _resample_path(upper, n_pairs)
    lower_paired = _resample_path(lower, n_pairs)
    midpoints = 0.5 * (upper_paired + lower_paired)
    half_thickness = 0.5 * np.linalg.norm(upper_paired - lower_paired, axis=1)

    midpoint_x, camber_y = _unique_xy(midpoints[:, 0], midpoints[:, 1])
    thickness_x, thickness_y = _unique_xy(midpoints[:, 0], half_thickness)
    if midpoint_x[0] > 1e-8 or midpoint_x[-1] < 1.0 - 1e-8:
        raise ValueError("Paired full contour does not span the declared chord.")
    camber_spline = interpolate.PchipInterpolator(midpoint_x, camber_y, extrapolate=True)
    thickness_spline = interpolate.PchipInterpolator(thickness_x, thickness_y, extrapolate=True)
    camber = np.asarray(camber_spline(x_eval), dtype=float)
    thickness = np.maximum(np.asarray(thickness_spline(x_eval), dtype=float), 0.0)
    return camber, thickness


def _paired_reference_distributions(
    upper: np.ndarray,
    lower: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for normalized-arc surface pairing."""
    return _arc_reference_distributions(upper, lower, x_eval)


def _infer_chord_endpoints(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Infer LE and TE centre from an ordered upper-then-lower contour.

    Averaging the first and last coordinates recovers the centre of an ordinary
    blunt trailing edge without requiring a point at ``[1, 0]``.
    """
    if len(points) < 3:
        raise ValueError("At least three contour points are required.")
    trailing_edge = 0.5 * (points[0] + points[-1])
    # Match the original Parameterisations.py convention for an already
    # normalized contour. This matters for files such as uvblade.s1, where a
    # rounded nose contains a slightly negative-x surface point immediately
    # before the declared x=0 leading-edge station.
    if abs(float(trailing_edge[0]) - 1.0) <= 1.0e-2 and np.min(np.abs(points[:, 0])) <= 1.0e-2:
        idx_le = int(np.argmin(np.abs(points[:, 0])))
    else:
        idx_le = int(np.argmax(np.linalg.norm(points - trailing_edge, axis=1)))
    return points[idx_le].copy(), trailing_edge, idx_le


def _is_unit_horizontal_chord(
    leading_edge: np.ndarray,
    trailing_edge: np.ndarray,
    position_tolerance: float = 2.5e-6,
    length_tolerance: float = 1.0e-2,
    angle_tolerance: float = 1.0e-2,
) -> bool:
    """Return whether inferred chord coordinates already use the BP unit frame."""
    chord_vector = np.asarray(trailing_edge) - np.asarray(leading_edge)
    chord = float(np.linalg.norm(chord_vector))
    if chord <= 0.0:
        return False
    direction = chord_vector / chord
    return bool(
        abs(float(leading_edge[0])) <= position_tolerance
        and abs(float(trailing_edge[0]) - 1.0) <= position_tolerance
        and abs(chord - 1.0) <= length_tolerance
        and direction[0] > 0.0
        and abs(float(direction[1])) <= angle_tolerance
    )


def _select_pairing_method(upper: np.ndarray, lower: np.ndarray) -> str:
    """Use BP source stations when their rounded-LE pairing is well conditioned."""
    if not (_is_monotone_in_x(upper) and _is_monotone_in_x(lower)):
        return "arc"
    upper_step = _first_chordwise_step(upper)
    lower_step = _first_chordwise_step(lower)
    if upper_step is None or lower_step is None:
        return "arc"
    # The native BP method is reliable when the first upper/lower stations have
    # comparable chordwise resolution. A large mismatch creates either a
    # negative derived thickness x-coordinate or a near-singular camber angle.
    step_ratio = max(upper_step, lower_step) / min(upper_step, lower_step)
    return "arc" if step_ratio > 2.0 else "source"


def _is_monotone_in_x(surface: np.ndarray, tolerance: float = 1.0e-10) -> bool:
    """Check whether an LE-to-TE surface is a single-valued x-parametric path."""
    return bool(np.all(np.diff(surface[:, 0]) >= -tolerance))


def _first_chordwise_step(surface: np.ndarray, tolerance: float = 1.0e-12) -> float | None:
    """Return the first positive x increment along an LE-to-TE surface."""
    increments = np.asarray(surface[1:, 0] - surface[0, 0], dtype=float)
    positive = increments[increments > tolerance]
    return float(positive[0]) if positive.size else None


def _source_reference_distributions(
    upper: np.ndarray,
    lower: np.ndarray,
    x_eval: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the native-station extraction used by Parameterisations.py.

    Equal-size surfaces retain their original index correspondence and average
    the paired x coordinates. For unequal sizes, the longer side is interpolated
    onto the shorter side's native x stations. Derived thickness stations that
    fall ahead of the physical leading edge are discarded; the original plotting
    code hid those negative-x singular points by its axis limits, whereas dense
    extrapolation would move them into the visible unit-chord interval.
    """
    if len(upper) == len(lower):
        camber_x = 0.5 * (upper[:, 0] + lower[:, 0])
        upper_y = upper[:, 1]
        lower_y = lower[:, 1]
    elif len(upper) < len(lower):
        camber_x = upper[:, 0]
        upper_y = upper[:, 1]
        lower_x, lower_values = _unique_xy(lower[:, 0], lower[:, 1])
        lower_y = interpolate.PchipInterpolator(lower_x, lower_values, extrapolate=True)(camber_x)
    else:
        camber_x = lower[:, 0]
        lower_y = lower[:, 1]
        upper_x, upper_values = _unique_xy(upper[:, 0], upper[:, 1])
        upper_y = interpolate.PchipInterpolator(upper_x, upper_values, extrapolate=True)(camber_x)

    station_x = np.asarray(camber_x, dtype=float)
    camber_x, upper_at_camber = _unique_xy(station_x, upper_y)
    lower_camber_x, lower_at_camber = _unique_xy(station_x, lower_y)
    if not np.allclose(camber_x, lower_camber_x, rtol=0.0, atol=1.0e-12):
        raise ValueError("Upper/lower source stations could not be aligned.")
    camber = 0.5 * (upper_at_camber + lower_at_camber)

    theta = np.arctan(np.gradient(camber, camber_x))
    thickness = (upper_at_camber - lower_at_camber) / (2.0 * np.cos(theta))
    thickness_x = camber_x - thickness * np.sin(theta)
    valid = (
        np.isfinite(thickness_x)
        & np.isfinite(thickness)
        & (thickness_x >= -1.0e-10)
        & (thickness_x <= 1.0 + 1.0e-8)
        & (thickness >= -1.0e-10)
    )
    thickness_x = np.clip(thickness_x[valid], 0.0, 1.0)
    thickness = np.maximum(thickness[valid], 0.0)
    thickness_x, thickness = _unique_xy(thickness_x, thickness)
    if len(camber_x) < 2 or len(thickness_x) < 2:
        raise ValueError("Insufficient valid source stations for camber/thickness extraction.")

    camber_spline = interpolate.PchipInterpolator(camber_x, camber, extrapolate=True)
    thickness_spline = interpolate.PchipInterpolator(thickness_x, thickness, extrapolate=True)
    return (
        np.asarray(camber_spline(x_eval), dtype=float),
        np.maximum(np.asarray(thickness_spline(x_eval), dtype=float), 0.0),
    )


def _resample_path(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample one LE-to-TE surface path by normalized arc length."""
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(distances)])
    if arc[-1] <= 0.0:
        raise ValueError("Surface path length is zero.")
    arc /= arc[-1]
    target = np.linspace(0.0, 1.0, n_points)
    return np.column_stack(
        [
            np.interp(target, arc, points[:, 0]),
            np.interp(target, arc, points[:, 1]),
        ]
    )


def _unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort by x and average duplicate x locations."""
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]

    unique_x: list[float] = []
    unique_y: list[float] = []
    start = 0
    while start < len(x_sorted):
        stop = start + 1
        while stop < len(x_sorted) and abs(x_sorted[stop] - x_sorted[start]) < 1e-12:
            stop += 1
        unique_x.append(float(np.mean(x_sorted[start:stop])))
        unique_y.append(float(np.mean(y_sorted[start:stop])))
        start = stop

    return np.asarray(unique_x), np.asarray(unique_y)


def _safe_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Derivative helper that remains finite at clustered LE/TE samples."""
    spline = interpolate.PchipInterpolator(x, y, extrapolate=True)
    return np.asarray(spline.derivative()(x), dtype=float)


def _local_second_derivative(x: np.ndarray, y: np.ndarray, idx: int, half_window: int = 8) -> float:
    """Fit a local quadratic and return its second derivative."""
    lo = max(0, idx - half_window)
    hi = min(len(x), idx + half_window + 1)
    if hi - lo < 5:
        return -0.1

    coeff = np.polyfit(x[lo:hi] - x[idx], y[lo:hi], deg=2)
    return float(2.0 * coeff[0])


def _leading_edge_radius(ref: ReferenceAirfoil) -> float:
    """Estimate LE radius by fitting a circle to near-nose thickness data."""
    asymptotic_mask = (ref.x_eval > 1e-8) & (ref.x_eval <= 0.015)
    if np.count_nonzero(asymptotic_mask) >= 5:
        x = ref.x_eval[asymptotic_mask]
        y = ref.thickness_y[asymptotic_mask]
        # Near the leading edge, the thickness parabola satisfies y^2 ~= 2*r*x.
        radius = np.sum(x * y**2) / (2.0 * np.sum(x**2))
        if np.isfinite(radius) and radius > 0.0:
            return float(radius)

    mask = ref.x_eval <= 0.025
    x = ref.x_eval[mask]
    y = ref.thickness_y[mask]
    if len(x) < 5:
        return 0.01

    def residual(params: np.ndarray) -> np.ndarray:
        xc, yc, radius = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - abs(radius)

    guess = np.array([0.0, 0.0, max(float(np.max(y)), 1e-3)])
    result = optimize.least_squares(residual, guess, max_nfev=200)
    return float(max(abs(result.x[2]), 1e-6))


def _tail_slope(x: np.ndarray, y: np.ndarray, start: float = 0.90) -> float:
    """Fit a stable straight-line slope over the trailing-edge segment."""
    mask = x >= start
    if np.count_nonzero(mask) < 6:
        mask = np.arange(len(x)) >= len(x) - 8
    slope, _ = np.polyfit(x[mask], y[mask], deg=1)
    return float(slope)


def _nose_slope(x: np.ndarray, y: np.ndarray, stop: float = 0.04) -> float:
    """Fit a stable straight-line slope over the leading-edge camber segment."""
    mask = x <= stop
    if np.count_nonzero(mask) < 6:
        mask = np.arange(len(x)) < 8
    slope, _ = np.polyfit(x[mask], y[mask], deg=1)
    return float(slope)
