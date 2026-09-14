"""From density to STA force: F = kB*T * d ln(rho) / dz."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

KB_EV_PER_K = 8.617333262e-5
EV_PER_A_TO_PN = 1602.18
_EPS_RHO = 1e-6


def smooth_density(rho3d, lateral_sigma=1.0, z_sigma=1.5):
    """Gaussian smoothing of rho(z, a, b): periodic in (a, b), `nearest` edges
    in z. Sigmas are in bins; 0 disables that direction."""
    out = np.stack([gaussian_filter(layer, sigma=lateral_sigma, mode="wrap")
                    for layer in rho3d])
    if z_sigma > 0:
        # gaussian_filter1d divides by sigma**2, so sigma=0 must be skipped
        out = gaussian_filter1d(out, sigma=z_sigma, axis=0, mode="nearest")
    return out


def force_field(rho3d, dz, kT):
    """Per-pixel STA force F(z, a, b) = kT * d ln rho / dz (eV/Å)."""
    ln_rho = np.log(np.clip(rho3d, _EPS_RHO, None))
    return kT * np.gradient(ln_rho, dz, axis=0)


def force_profile(rho1d, dz, kT):
    """Laterally averaged STA force F(z) = kT * d ln <rho>/dz (eV/Å).

    `rho1d` must be the ARITHMETIC lateral mean of the density. Averaging the
    per-pixel force field instead differentiates the geometric mean of rho and
    overestimates the force wherever the layer is laterally structured (see
    docs/method.md).
    """
    return kT * np.gradient(np.log(np.clip(rho1d, _EPS_RHO, None)), dz)


def slab_average(f3d, z_mid, z0, half_width, valid=None):
    """Mean of f3d over the z-bins with centres in [z0 - hw, z0 + hw); the
    nearest bin if the window contains none.

    `valid` (bool per z-bin) excludes unsampled bins; if no valid bin is left,
    the result is all NaN.
    """
    mask = (z_mid >= z0 - half_width) & (z_mid < z0 + half_width)
    if valid is not None:
        mask &= valid
    if mask.any():
        return f3d[mask].mean(axis=0)
    nearest = int(np.argmin(np.abs(z_mid - z0)))
    if valid is None or valid[nearest]:
        return f3d[nearest]
    return np.full(f3d.shape[1:], np.nan)


class CartesianSampler:
    """Resamples periodic fields on a fractional (a, b) grid onto a lab-frame
    Cartesian (x, y) grid. The cell may be oblique; pixels are sampled with
    periodic wrap, so there is no seam at the cell boundary."""

    def __init__(self, cell2d, x, y, lateral_bins):
        A_inv = np.linalg.inv(np.asarray(cell2d).T)
        Xg, Yg = np.meshgrid(x, y, indexing="xy")
        frac = A_inv @ np.stack([Xg.ravel(), Yg.ravel()], axis=0)
        # bin i covers [i, i+1)/nb, so its centre -- where map_coordinates
        # places array element i -- is at fractional (i + 0.5)/nb
        self._coords = frac * lateral_bins - 0.5
        self.shape = (len(y), len(x))

    def __call__(self, field2d):
        f = map_coordinates(field2d, self._coords, order=1, mode="grid-wrap")
        return f.reshape(self.shape).astype(np.float32)


def resample_cartesian(field2d, cell2d, x, y):
    """One-off version of CartesianSampler: returns field[iy, ix] at (x[ix], y[iy])."""
    return CartesianSampler(cell2d, x, y, field2d.shape[0])(field2d)
