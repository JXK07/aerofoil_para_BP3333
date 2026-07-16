"""Plotting helpers for BP3333 fitting results."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from .fitting import FitResult
from .geometry import read_airfoil
from .model import generate_airfoil


def plot_fit_result(
    result: FitResult,
    reference_path: str | Path,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Plot the pure BP3333 fit reconstructed from optimized parameters."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bp3333_matplotlib"))
    import matplotlib.pyplot as plt

    ref = read_airfoil(
        reference_path,
        pairing_method=result.pairing_method,
        normalise_chord=result.chord_normalised,
        leading_edge=result.chord_le,
        trailing_edge=result.chord_te,
    )
    pure = generate_airfoil(
        result.params,
        rt_root_strategy=result.rt_root_strategy,
    )

    # Figure 1: thickness & camber
    fig1, ax1 = plt.subplots(figsize=(9.0, 3.5), constrained_layout=True)
    _plot_thickness(ax1, ref, pure)
    fig1.suptitle(
        f"{result.airfoil} BP3333 fit: "
        f"MAE={result.mae:.2e}",
        fontsize=11,
    )

    # Figure 2: airfoil contour
    fig2, ax2 = plt.subplots(figsize=(9.0, 3.5), constrained_layout=True)
    _plot_airfoil(ax2, ref, pure)
    fig2.suptitle(
        f"{result.airfoil} BP3333 fit: "
        f"MAE={result.mae:.2e}",
        fontsize=11,
    )

    if save_path is not None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stem = out_path.stem
        suffix = out_path.suffix or ".png"
        parent = out_path.parent
        fig1.savefig(parent / f"{stem}_thickness{suffix}", dpi=220)
        fig2.savefig(parent / f"{stem}_airfoil{suffix}", dpi=220)
    if show:
        plt.show()


def _plot_thickness(ax, ref, pure: dict[str, np.ndarray]) -> None:
    """Draw reference and BP3333 half-thickness distributions."""
    ax.plot(ref.x_eval, ref.thickness_y, "k-", linewidth=1.4, label="Reference thickness")
    ax.plot(
        pure["thickness_x"],
        pure["thickness_y"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label="BP3333 thickness",
    )
    ax.plot(
        ref.x_eval,
        ref.camber_y,
        color="tab:orange",
        linestyle="-.",
        linewidth=1.0,
        label="Reference camber",
    )
    ax.plot(
        pure["camber_x"],
        pure["camber_y"],
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
        label="BP3333 camber",
    )
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    ax.set_title("Thickness and camber distributions")
    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.legend(loc="best", fontsize=8)

def _plot_airfoil(ax, ref, pure: dict[str, np.ndarray]) -> None:
    """Draw reference and pure BP3333 airfoil contours."""
    ax.plot(ref.contour[:, 0], ref.contour[:, 1], "k-", linewidth=1.2, label="Reference")
    ax.plot(
        pure["upper_x"],
        pure["upper_y"],
        color="tab:green",
        linestyle="--",
        linewidth=1.3,
        label="BP3333",
    )
    ax.plot(pure["lower_x"], pure["lower_y"], color="tab:green", linestyle="--", linewidth=1.3)
    ax.set_xlabel("x / c")
    ax.set_ylabel("y / c")
    # ax.set_title("Airfoil contour comparison")
    # ax.set_aspect("equal", adjustable="box")
    ax.grid(True, which="major", alpha=0.35)
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.5)
    ax.legend(loc="best", fontsize=8)
