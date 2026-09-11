"""Trajectory -> force map, driven by a single Settings object."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from ase.io import iread, read

from .density import DensityHistogram, accumulate_density
from .force import (EV_PER_A_TO_PN, KB_EV_PER_K, CartesianSampler, force_field,
                    force_profile, slab_average, smooth_density)
from .overlay import Overlay, build_overlay
from .selection import select_atoms


@dataclass
class Settings:
    temperature: float                      # K, sets kT in F = kT d ln(rho)/dz
    # which atoms
    surface: str = "fixed"                  # selection defining the surface topography
    surface_zmin: Optional[float] = None
    surface_zmax: Optional[float] = None
    probe: str = "O"                        # selection whose density is measured
    cutoff: float = 2.5                     # Å, lateral radius for the local surface height
    # which frames
    start: int = 0
    stop: Optional[int] = None
    stride: int = 1
    format: Optional[str] = None            # ASE format name, if not guessable from the file
    # histogram + smoothing
    lateral_bins: int = 100                 # bins per cell vector
    dz: float = 0.3                         # Å
    z_max: float = 20.0                     # Å above the local surface
    lateral_smooth: float = 1.0             # Gaussian sigma, bins
    z_smooth: float = 1.5                   # Gaussian sigma, bins
    bulk_window: Optional[Tuple[float, float]] = None  # Å, only for reporting rho0
    # z slices shown on the map
    half_width: float = 1.0                 # Å, slab half-width averaged per slice
    z_default: Optional[float] = None       # Å, initial slice; None = first density peak
    z_step: float = 0.1                     # Å, slider step
    # Cartesian map
    patch_size: float = 80.0                # Å, side of the square window
    pixels: int = 200                       # pixels per side
    center: Optional[Tuple[float, float]] = None  # Å; None = centroid of surface atoms
    clim_percentiles: Tuple[float, float] = (0.2, 99.8)
    clim: Optional[Tuple[float, float]] = None     # eV/Å, overrides clim_percentiles
    # overlay
    overlay: bool = True
    overlay_file: Optional[str] = None      # structure to draw instead of the surface atoms
    bond_mult: float = 0.85                 # natural_cutoffs multiplier for bonds

    @property
    def kT(self):
        return KB_EV_PER_K * self.temperature


@dataclass
class ForceMapResult:
    settings: Settings
    histogram: DensityHistogram
    rho1d: np.ndarray           # (n_z,) arithmetic lateral mean of smoothed rho, Å^-3
    f1d: np.ndarray             # (n_z,) laterally averaged force, eV/Å
    f3d: np.ndarray             # (n_z, nb, nb) per-pixel force, eV/Å
    cell2d: np.ndarray          # (2, 2) rows = a, b lateral cell vectors, Å
    x: np.ndarray               # (pixels,) map pixel centres, Å
    y: np.ndarray
    z_values: np.ndarray        # slider z values, Å
    z_default: float            # requested initial slice (unsnapped)
    z_default_idx: int
    vmin: float
    vmax: float
    overlay: Optional[Overlay]
    rho0: Optional[float] = None
    _sampler: CartesianSampler = field(default=None, repr=False)

    @property
    def z_mid(self):
        return self.histogram.z_mid

    def slice_map(self, z0):
        """Cartesian force map F(x, y) of the slab centred at z0 (eV/Å)."""
        mat = slab_average(self.f3d, self.z_mid, z0, self.settings.half_width)
        return self._sampler(mat)


def compute_force_map(path, settings, verbose=True):
    s = settings
    log = print if verbose else (lambda *a, **k: None)

    ref = read(path, index=0, format=s.format)
    surface_idx = select_atoms(ref, s.surface, s.surface_zmin, s.surface_zmax)
    if len(surface_idx) == 0:
        raise ValueError(f"surface selection {s.surface!r} matched no atoms")
    probe_idx = np.setdiff1d(select_atoms(ref, s.probe), surface_idx)
    if len(probe_idx) == 0:
        raise ValueError(f"probe selection {s.probe!r} matched no (non-surface) atoms")
    sym = np.array(ref.get_chemical_symbols())
    log(f"Trajectory: {path}")
    log(f"Surface: {len(surface_idx)} atoms ({_composition(sym[surface_idx])}); "
        f"probe: {len(probe_idx)} atoms ({_composition(sym[probe_idx])})")

    frames = iread(path, index=slice(s.start, s.stop, s.stride), format=s.format)
    hist = accumulate_density(frames, surface_idx, probe_idx, cutoff=s.cutoff, dz=s.dz,
                              z_max=s.z_max, lateral_bins=s.lateral_bins,
                              progress_every=1000 if verbose else 0)
    log(f"{hist.n_frames} frames used (start={s.start}, stop={s.stop}, stride={s.stride})")

    rho = smooth_density(hist.density(), s.lateral_smooth, s.z_smooth)
    z_mid = hist.z_mid
    rho1d = rho.mean(axis=(1, 2))
    rho0 = None
    if s.bulk_window is not None:
        bulk = (z_mid >= s.bulk_window[0]) & (z_mid < s.bulk_window[1])
        if bulk.any():
            rho0 = float(rho1d[bulk].mean())
            log(f"Bulk density rho0 = {rho0:.5f} Å⁻³ over {s.bulk_window} Å")
        else:
            log(f"Warning: bulk window {s.bulk_window} Å lies outside 0-{s.z_max} Å")

    f3d = force_field(rho, hist.dz, s.kT)
    f1d = force_profile(rho1d, hist.dz, s.kT)
    peak = int(np.argmax(np.abs(f1d)))
    log(f"kT = {s.kT:.5f} eV; F(z) peak |F| = {abs(f1d[peak]):.5f} eV/Å at z = "
        f"{z_mid[peak]:.2f} Å ({abs(f1d[peak]) * EV_PER_A_TO_PN:.1f} pN)")

    cell2d = np.array(ref.cell[:2, :2])
    if s.center is not None:
        center = np.asarray(s.center, dtype=float)
    else:
        center = ref.positions[surface_idx, :2].mean(axis=0)
    half = s.patch_size / 2.0
    pix = s.patch_size / s.pixels
    x = center[0] - half + (np.arange(s.pixels) + 0.5) * pix
    y = center[1] - half + (np.arange(s.pixels) + 0.5) * pix
    sampler = CartesianSampler(cell2d, x, y, s.lateral_bins)

    z_values = np.arange(0.0, s.z_max + 1e-9, s.z_step)
    z_default = s.z_default if s.z_default is not None else float(z_mid[np.argmax(rho1d)])
    z_default_idx = int(np.argmin(np.abs(z_values - z_default)))

    result = ForceMapResult(
        settings=s, histogram=hist, rho1d=rho1d, f1d=f1d, f3d=f3d, cell2d=cell2d,
        x=x, y=y, z_values=z_values, z_default=z_default, z_default_idx=z_default_idx,
        vmin=0.0, vmax=0.0, overlay=None, rho0=rho0, _sampler=sampler)

    if s.clim is not None:
        result.vmin, result.vmax = map(float, s.clim)
    else:
        ref_map = result.slice_map(float(z_values[z_default_idx]))
        finite = ref_map[np.isfinite(ref_map)]
        result.vmin = float(np.percentile(finite, s.clim_percentiles[0]))
        result.vmax = float(np.percentile(finite, s.clim_percentiles[1]))
    log(f"Colour range (fixed for all slices, from z = {z_values[z_default_idx]:.2f} Å): "
        f"{result.vmin:.5f} to {result.vmax:.5f} eV/Å")

    if s.overlay:
        ov_atoms = read(s.overlay_file) if s.overlay_file else ref[surface_idx]
        result.overlay = build_overlay(ov_atoms, cell2d, s.bond_mult)
        log(f"Overlay: {len(ov_atoms)} atoms, {len(result.overlay.bonds)} bonds")
    return result


def _composition(symbols):
    u, c = np.unique(symbols, return_counts=True)
    return " ".join(f"{e}{n}" for e, n in zip(u, c))
