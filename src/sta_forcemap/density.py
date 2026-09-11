"""Histogramming probe positions into rho(z, a, b) relative to the local surface."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def local_surface_height(probe_xy, surf_xy, surf_z, cell2d, cutoff):
    """Local surface height under each probe atom.

    For each probe, the highest surface atom within `cutoff` (lateral,
    minimum-image) distance; if none is that close, the z of the laterally
    nearest surface atom.
    """
    inv_cell = np.linalg.inv(cell2d.T)
    diff = probe_xy[:, None, :] - surf_xy[None, :, :]
    frac = diff @ inv_cell.T
    frac -= np.round(frac)
    dist = np.linalg.norm(frac @ cell2d.T, axis=2)
    within = dist <= cutoff
    surf_height = np.where(within, surf_z[None, :], -np.inf).max(axis=1)
    missing = ~np.isfinite(surf_height)
    if missing.any():
        surf_height[missing] = surf_z[dist[missing].argmin(axis=1)]
    return surf_height


@dataclass
class DensityHistogram:
    """Probe counts binned in height above the local surface (z) and in
    fractional lateral cell coordinates (a, b)."""

    counts: np.ndarray        # (n_z, n_lateral, n_lateral)
    z_edges: np.ndarray       # (n_z + 1,) in Å
    n_frames: int
    mean_area: float          # mean lateral cell area over the frames, Å^2

    @property
    def dz(self):
        return float(self.z_edges[1] - self.z_edges[0])

    @property
    def z_mid(self):
        return 0.5 * (self.z_edges[:-1] + self.z_edges[1:])

    @property
    def lateral_bins(self):
        return self.counts.shape[1]

    def density(self):
        """Number density rho(z, a, b) in Å^-3."""
        bin_area = self.mean_area / self.lateral_bins**2
        return self.counts / (self.n_frames * bin_area * self.dz)


def check_cell(cell):
    """The method assumes a surface in the xy plane with the normal along z."""
    cell = np.asarray(cell)
    if abs(cell[0, 2]) > 1e-6 or abs(cell[1, 2]) > 1e-6:
        raise ValueError("cell vectors a and b must lie in the xy plane "
                         "(surface normal along z)")
    if abs(cell[2, 0]) > 1e-6 or abs(cell[2, 1]) > 1e-6:
        raise ValueError("cell vector c must be parallel to z")


def accumulate_density(frames, surface_idx, probe_idx, *, cutoff=2.5, dz=0.3,
                       z_max=20.0, lateral_bins=100, progress_every=0):
    """Histogram probe atoms over an iterable of ase.Atoms frames.

    Probe heights are measured from the local surface height (see
    local_surface_height); probes below the surface or above z_max are ignored.
    """
    surface_idx = np.asarray(surface_idx)
    probe_idx = np.asarray(probe_idx)
    nb = int(lateral_bins)
    z_edges = np.arange(0.0, z_max + dz, dz)
    n_zbins = len(z_edges) - 1

    counts = np.zeros((n_zbins, nb, nb))
    area_acc = 0.0
    n_frames = 0

    for atoms in frames:
        if n_frames == 0:
            check_cell(atoms.cell)
        cell2d = atoms.cell[:2, :2]
        surf_xy = atoms.positions[surface_idx, :2]
        surf_z = atoms.positions[surface_idx, 2]

        probe_pos = atoms.positions[probe_idx]
        probe_xy = probe_pos[:, :2]
        height = probe_pos[:, 2] - local_surface_height(probe_xy, surf_xy, surf_z,
                                                        cell2d, cutoff)

        inv_cell2d = np.linalg.inv(cell2d.T)
        frac = np.mod(probe_xy @ inv_cell2d.T, 1.0)

        zi = np.floor(height / dz).astype(int)
        valid = (zi >= 0) & (zi < n_zbins)
        if valid.any():
            ai = np.clip((frac[valid, 0] * nb).astype(int), 0, nb - 1)
            bi = np.clip((frac[valid, 1] * nb).astype(int), 0, nb - 1)
            np.add.at(counts, (zi[valid], ai, bi), 1)

        area_acc += np.linalg.norm(np.cross(atoms.cell[0], atoms.cell[1]))
        n_frames += 1
        if progress_every and n_frames % progress_every == 0:
            print(f"  {n_frames} frames processed", flush=True)

    if n_frames == 0:
        raise ValueError("no frames to analyse (check --start/--stop/--stride)")
    return DensityHistogram(counts=counts, z_edges=z_edges, n_frames=n_frames,
                            mean_area=area_acc / n_frames)
