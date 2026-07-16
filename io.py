"""Output helpers for fitted BP3333 airfoils."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fitting import FitResult


def save_coordinates(result: FitResult, path: str | Path) -> None:
    """Save reconstructed contour coordinates in upper-then-lower order."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g = result.geometry
    upper = np.column_stack([g["upper_x"][::-1], g["upper_y"][::-1]])
    lower = np.column_stack([g["lower_x"], g["lower_y"]])
    contour = np.vstack([upper, lower])

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write(f"BP3333 reconstructed {result.airfoil}\n")
        for x, y in contour:
            handle.write(f"{x:.12e}  {y:.12e}\n")


def save_parameters(result: FitResult, path: str | Path) -> None:
    """Save fit parameters and error metrics as JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "airfoil": result.airfoil,
        "parameters": result.params.to_dict(),
        "error": {
            "mae": result.mae,
            "max_abs_error": result.max_abs_error,
            "rms": result.rms,
        },
        "optimizer": {
            "method": result.optimizer,
            "success": result.success,
            "message": result.message,
            "n_evaluations": result.n_evaluations,
            "rt_root_strategy": result.rt_root_strategy,
        },
        "reference_geometry": {
            "pairing_method": result.pairing_method,
            "chord_normalised": result.chord_normalised,
            "leading_edge": list(result.chord_le),
            "trailing_edge": list(result.chord_te),
        },
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
