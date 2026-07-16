"""Command line entry point for BP3333 fitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    # Spyder/%runfile executes this file as a plain script.  Add the repository
    # root to sys.path so package imports still resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from BP3333.fitting import fit_airfoil
    from BP3333.io import save_coordinates, save_parameters
    from BP3333.plotting import plot_fit_result
else:
    from .fitting import fit_airfoil
    from .io import save_coordinates, save_parameters
    from .plotting import plot_fit_result


def run_one(
    airfoil: str | Path,
    output_dir: str | Path,
    max_nfev: int,
    make_plot: bool = True,
    show_plot: bool = False,
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
) -> dict[str, float]:
    """Fit one airfoil and write coordinate/parameter outputs."""
    result = fit_airfoil(
        airfoil,
        max_nfev=max_nfev,
        use_database_seed=use_database_seed,
        optimizer=optimizer,
        rt_root_strategy=rt_root_strategy,
        pairing_method=pairing_method,
        normalise_chord=normalise_chord,
        leading_edge=leading_edge,
        trailing_edge=trailing_edge,
        ga_maxiter=ga_maxiter,
        ga_population=ga_population,
        random_seed=random_seed,
    )
    out_dir = Path(output_dir)
    save_coordinates(result, out_dir / f"{result.airfoil}_bp3333.dat")
    save_parameters(result, out_dir / f"{result.airfoil}_bp3333.json")
    if make_plot:
        plot_fit_result(
            result,
            reference_path=airfoil,
            save_path=out_dir / f"{result.airfoil}_bp3333.png",
            show=show_plot,
        )
    print(
        f"{result.airfoil:14s}  MAE={result.mae:.6e}  "
        f"RMS={result.rms:.6e}  max={result.max_abs_error:.6e}  "
        f"optimizer={result.optimizer}  root={result.rt_root_strategy}  "
        f"pairing={result.pairing_method}  normalised={result.chord_normalised}  "
        f"evals={result.n_evaluations}",
        flush=True,
    )
    return {"mae": result.mae, "rms": result.rms, "max_abs_error": result.max_abs_error}


def run_all(
    test_dir: str | Path,
    output_dir: str | Path,
    max_nfev: int,
    make_plot: bool = True,
    show_plot: bool = False,
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
) -> None:
    """Fit every supported airfoil in a Test Airfoils directory."""
    test_path = Path(test_dir)
    files = sorted(
        path
        for path in test_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".dat", ".s1", ".s6", ".s11"}
    )
    if not files:
        raise FileNotFoundError(f"No airfoil files found in {test_path}.")

    metrics = []
    failures = []
    for path in files:
        try:
            metrics.append(
                run_one(
                    path,
                    output_dir,
                    max_nfev=max_nfev,
                    make_plot=make_plot,
                    show_plot=show_plot,
                    use_database_seed=use_database_seed,
                    optimizer=optimizer,
                    rt_root_strategy=rt_root_strategy,
                    pairing_method=pairing_method,
                    normalise_chord=normalise_chord,
                    leading_edge=leading_edge,
                    trailing_edge=trailing_edge,
                    ga_maxiter=ga_maxiter,
                    ga_population=ga_population,
                    random_seed=random_seed,
                )
            )
        except Exception as exc:
            failures.append((path.name, str(exc)))
            print(f"{path.stem:14s}  FAILED: {exc}", flush=True)

    if not metrics:
        raise RuntimeError("No airfoil could be fitted.")

    mae_values = np.array([item["mae"] for item in metrics])
    print("-" * 72)
    print(f"Fitted {len(files)} airfoils")
    print(f"Mean MAE:   {float(np.mean(mae_values)):.6e}")
    print(f"Median MAE: {float(np.median(mae_values)):.6e}")
    print(f"Worst MAE:  {float(np.max(mae_values)):.6e}")
    if failures:
        print(f"Failures:   {len(failures)}")
        for name, message in failures:
            print(f"  {name}: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit BP3333 parameters to airfoil coordinates.")
    parser.add_argument("--airfoil", default="Test Airfoils/uvblade.s1", help="Path to one airfoil coordinate file.")
    parser.add_argument("--test-dir", default="Test Airfoils", help="Directory used with --all.")
    parser.add_argument("--output-dir", default="results", help="Output directory.")
    parser.add_argument("--all", action="store_true", help="Fit every airfoil in --test-dir.")
    parser.add_argument("--max-nfev", type=int, default=900, help="Optimiser iterations per start.")
    parser.add_argument("--no-plot", action="store_true", help="Do not save comparison plots.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    parser.add_argument(
        "--use-database-seed",
        action="store_true",
        help="Use optional BP3333 seeds from Airfoils_BP3333-main.zip when available.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("slsqp", "least_squares", "ga"),
        default="slsqp",
        help="Backend. SLSQP automatically falls back to Differential Evolution GA on failure.",
    )
    parser.add_argument(
        "--rt-root",
        choices=("smallest", "midpoint"),
        default="smallest",
        help="Selection rule for multiple admissible r_t roots.",
    )
    parser.add_argument(
        "--pairing",
        choices=("auto", "x", "source", "arc"),
        default="auto",
        help="Upper/lower matching: automatic, common x, BP source stations, or normalized arc length.",
    )
    parser.add_argument(
        "--normalise-chord",
        "--normalize-chord",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Transform the inferred chord to [0,0]-[1,0], skip it, or decide automatically.",
    )
    parser.add_argument(
        "--leading-edge",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Optional raw-coordinate leading edge; requires --trailing-edge.",
    )
    parser.add_argument(
        "--trailing-edge",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Optional raw-coordinate trailing-edge centre or virtual sharp trailing edge.",
    )
    parser.add_argument("--ga-maxiter", type=int, default=500, help="Maximum GA generations.")
    parser.add_argument("--ga-population", type=int, default=150, help="GA population size.")
    parser.add_argument("--random-seed", type=int, default=42, help="GA random seed.")
    args = parser.parse_args()
    normalise_chord = {"auto": None, "yes": True, "no": False}[args.normalise_chord]
    leading_edge = tuple(args.leading_edge) if args.leading_edge is not None else None
    trailing_edge = tuple(args.trailing_edge) if args.trailing_edge is not None else None

    if args.all or not args.airfoil:
        if not args.all:
            print("No --airfoil supplied; fitting all files in --test-dir.")
        run_all(
            args.test_dir,
            args.output_dir,
            max_nfev=args.max_nfev,
            make_plot=not args.no_plot,
            show_plot=args.show,
            use_database_seed=args.use_database_seed,
            optimizer=args.optimizer,
            rt_root_strategy=args.rt_root,
            pairing_method=args.pairing,
            normalise_chord=normalise_chord,
            leading_edge=leading_edge,
            trailing_edge=trailing_edge,
            ga_maxiter=args.ga_maxiter,
            ga_population=args.ga_population,
            random_seed=args.random_seed,
        )
        return
    run_one(
        args.airfoil,
        args.output_dir,
        max_nfev=args.max_nfev,
        make_plot=not args.no_plot,
        show_plot=args.show,
        use_database_seed=args.use_database_seed,
        optimizer=args.optimizer,
        rt_root_strategy=args.rt_root,
        pairing_method=args.pairing,
        normalise_chord=normalise_chord,
        leading_edge=leading_edge,
        trailing_edge=trailing_edge,
        ga_maxiter=args.ga_maxiter,
        ga_population=args.ga_population,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
