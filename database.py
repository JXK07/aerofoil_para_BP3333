"""Optional BP3333 parameter database support."""

from __future__ import annotations

import csv
import zipfile
from functools import lru_cache
from pathlib import Path


def database_seed(airfoil_name: str, root: str | Path = ".") -> dict[str, float] | None:
    """Return a BP3333 seed from Airfoils_BP3333-main.zip when available."""
    table = _load_database(Path(root))
    keys = _candidate_keys(airfoil_name)
    for key in keys:
        if key in table:
            params = dict(table[key])
            params["dz_te"] = 0.0
            params["z_te"] = 0.0
            return params
    return None


def _candidate_keys(name: str) -> list[str]:
    key = name.lower().replace("_", "").replace("-", "")
    keys = [key]
    if key.startswith("n") and key[1:].isdigit():
        keys.append(f"naca{key[1:]}")
    return keys


@lru_cache(maxsize=4)
def _load_database(root: Path) -> dict[str, dict[str, float]]:
    zip_path = root / "Airfoils_BP3333-main.zip"
    if not zip_path.exists():
        return {}

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("Airfoils_BP3333-main/parametersBP.csv") as handle:
            rows = csv.DictReader(line.decode("utf-8") for line in handle)
            output: dict[str, dict[str, float]] = {}
            for row in rows:
                key = row["airfoil"].lower().replace("_", "").replace("-", "")
                output[key] = {
                    "r_le": float(row["r_le"]),
                    "x_t": float(row["x_t"]),
                    "y_t": float(row["y_t"]),
                    "k_t": float(row["k_t"]),
                    "beta_te": float(row["beta_te"]),
                    "gamma_le": float(row["gamm_le"]),
                    "x_c": float(row["x_c"]),
                    "y_c": float(row["y_c"]),
                    "k_c": float(row["k_c"]),
                    "alpha_te": float(row["alpha_te"]),
                }
            return output
