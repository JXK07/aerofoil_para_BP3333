"""BP3333 wrapper for the original ``python-par`` test workflow.

This module keeps the same high-level workflow used by
``Parameterisations.py``:

1. read a reference airfoil,
2. run an optimisation-based parameter fit,
3. reconstruct upper/lower surfaces,
4. report the same fixed-grid MAE/RMS/max-error metrics.

Only the analytic parameterisation is changed from BP3434 to BP3333.  The
actual BP3333 control-point equations are imported from the existing
``BP3333`` package in this repository to avoid duplicating formula code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from BP3333.fitting import fit_airfoil
from BP3333.model import BP3333Parameters, generate_airfoil
from Parameterisations import (
    _normalise_chord_for_error,
    _safe_surface_spline_for_error,
    _unique_xy_for_error,
)
from scipy import interpolate


class AirfoilParameterisationBP3333:
    """BP3333 parameterisation with a python-par-style public interface."""

    def __init__(self) -> None:
        self.last_result = None

    def FindInitialParameterisation(self, reference_file: Path) -> dict[str, float]:
        """Optimise BP3333 parameters for one reference airfoil.

        The optimiser backend and iteration budget follow the existing BP3333
        SLSQP defaults, which were written to mirror the BP3434 SLSQP style.
        """

        result = fit_airfoil(
            reference_file,
            max_nfev=900,
            apply_residual_correction=False,
            use_database_seed=False,
            optimizer="slsqp",
            allow_degenerate_camber=False,
        )
        self.last_result = result
        return result.params.to_dict()

    def ComputeProfileCoordinates(
        self,
        airfoil_params: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate upper and lower surfaces from BP3333 parameters."""

        geometry = generate_airfoil(
            BP3333Parameters(**airfoil_params),
            n_per_segment=260,
            allow_degenerate_camber=False,
        )
        return (
            geometry["upper_x"],
            geometry["upper_y"],
            geometry["lower_x"],
            geometry["lower_y"],
        )

    def ComputeReferenceGridError(
        self,
        airfoil_params: dict[str, float],
        reference_file: Path,
        n_eval: int = 701,
    ) -> dict[str, float]:
        """Compute the same fixed-grid vertical error metric as BP3434."""

        reference_coordinates = np.genfromtxt(reference_file, dtype=float, skip_header=1)
        reference_coordinates = reference_coordinates[np.all(np.isfinite(reference_coordinates), axis=1)]
        if reference_coordinates.ndim != 2 or reference_coordinates.shape[1] < 2:
            raise ValueError(f"No coordinate data found in {reference_file}.")

        reference_coordinates = _normalise_chord_for_error(reference_coordinates[:, :2])
        idx_le = int(np.argmin(np.abs(reference_coordinates[:, 0])))
        reference_coordinates[:, 1] -= reference_coordinates[idx_le, 1]

        upper_reference = reference_coordinates[: idx_le + 1][::-1]
        lower_reference = reference_coordinates[idx_le:]
        upper_ref_x, upper_ref_y = _unique_xy_for_error(upper_reference[:, 0], upper_reference[:, 1])
        lower_ref_x, lower_ref_y = _unique_xy_for_error(lower_reference[:, 0], lower_reference[:, 1])
        upper_ref_spline = interpolate.PchipInterpolator(upper_ref_x, upper_ref_y, extrapolate=True)
        lower_ref_spline = interpolate.PchipInterpolator(lower_ref_x, lower_ref_y, extrapolate=True)

        beta = np.linspace(0.0, np.pi, n_eval)
        x_eval = 0.5 * (1.0 - np.cos(beta))
        reference_upper_y = np.asarray(upper_ref_spline(x_eval), dtype=float)
        reference_lower_y = np.asarray(lower_ref_spline(x_eval), dtype=float)

        upper_x, upper_y, lower_x, lower_y = self.ComputeProfileCoordinates(airfoil_params)
        upper_spline = _safe_surface_spline_for_error(upper_x, upper_y)
        lower_spline = _safe_surface_spline_for_error(lower_x, lower_y)
        upper_error = upper_spline(x_eval) - reference_upper_y
        lower_error = lower_spline(x_eval) - reference_lower_y
        error = np.concatenate([upper_error, lower_error])

        return {
            "mae": float(np.mean(np.abs(error))),
            "rms": float(np.sqrt(np.mean(error**2))),
            "max_abs_error": float(np.max(np.abs(error))),
        }


if __name__ == "__main__":
    inputfile = Path(__file__).resolve().parent / "Test Airfoils/n0012.dat"
    call_class = AirfoilParameterisationBP3333()
    params = call_class.FindInitialParameterisation(inputfile)
    errors = call_class.ComputeReferenceGridError(params, inputfile)
    print("-----")
    print(params)
    print(
        f"{inputfile.name:14s}  MAE={errors['mae']:.6e}  "
        f"RMS={errors['rms']:.6e}  max={errors['max_abs_error']:.6e}",
        flush=True,
    )
