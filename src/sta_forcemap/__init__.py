"""Interactive solvent-tip-approximation (STA) force maps from MD trajectories.

The STA force on a probe molecule at height z above a surface is

    F(x, y, z) = kB*T * d ln rho(x, y, z) / dz,

where rho is the probe (e.g. water-oxygen) number density measured relative
to the local surface height. See docs/method.md for details.
"""
__version__ = "0.1.0"

from .density import DensityHistogram, accumulate_density, local_surface_height
from .force import (
    KB_EV_PER_K,
    force_field,
    force_profile,
    resample_cartesian,
    slab_average,
    smooth_density,
)
from .pipeline import ForceMapResult, Settings, compute_force_map
from .selection import select_atoms

__all__ = [
    "__version__",
    "DensityHistogram",
    "accumulate_density",
    "local_surface_height",
    "KB_EV_PER_K",
    "force_field",
    "force_profile",
    "resample_cartesian",
    "slab_average",
    "smooth_density",
    "ForceMapResult",
    "Settings",
    "compute_force_map",
    "select_atoms",
]
