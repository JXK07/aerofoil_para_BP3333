"""Modern BP3333 airfoil parameterisation package."""

from .model import BP3333Parameters, generate_airfoil
from .fitting import FitResult, fit_airfoil

__all__ = [
    "BP3333Parameters",
    "FitResult",
    "fit_airfoil",
    "generate_airfoil",
]
