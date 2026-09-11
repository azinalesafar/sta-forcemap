import numpy as np
import pytest
from conftest import CELL, SURFACE_Z, dln_layer_density

from sta_forcemap import (KB_EV_PER_K, accumulate_density, force_field, force_profile,
                          local_surface_height, resample_cartesian, smooth_density)

KT = KB_EV_PER_K * 300.0


def test_force_profile_recovers_analytic_force(frames):
    n_surf = 100
    hist = accumulate_density(frames, np.arange(n_surf), np.arange(n_surf, len(frames[0])),
                              dz=0.3, z_max=11.0, lateral_bins=20)
    rho1d = smooth_density(hist.density(), 0, 0).mean(axis=(1, 2))
    f = force_profile(rho1d, hist.dz, KT)
    z = hist.z_mid
    window = (z > 1.0) & (z < 6.0)
    expected = KT * dln_layer_density(z[window])
    rms = np.sqrt(np.mean((f[window] - expected) ** 2))
    assert rms < 0.1 * np.max(np.abs(expected))


def test_density_normalisation(frames):
    n_surf = 100
    hist = accumulate_density(frames[:10], np.arange(n_surf),
                              np.arange(n_surf, len(frames[0])), dz=0.3, z_max=13.0,
                              lateral_bins=10)
    area = np.linalg.norm(np.cross(CELL[0], CELL[1]))
    n_per_frame = hist.density().mean(axis=(1, 2)).sum() * area * hist.dz
    assert n_per_frame == pytest.approx(len(frames[0]) - n_surf)


def test_profile_uses_arithmetic_lateral_mean():
    """A laterally structured layer: averaging per-pixel forces (the geometric
    mean of rho) must NOT be what force_profile returns."""
    z = np.arange(0.15, 10, 0.3)
    g1 = 1 + 2.0 * np.exp(-(z - 3) ** 2)
    g2 = 1 + 0.1 * np.exp(-(z - 5) ** 2)
    rho = np.empty((len(z), 4, 4))
    rho[:, :2, :] = g1[:, None, None]
    rho[:, 2:, :] = g2[:, None, None]
    f1d = force_profile(rho.mean(axis=(1, 2)), 0.3, KT)
    expected = KT * np.gradient(np.log((g1 + g2) / 2), 0.3)
    np.testing.assert_allclose(f1d, expected)
    geometric = force_field(rho, 0.3, KT).mean(axis=(1, 2))
    assert np.max(np.abs(geometric - f1d)) > 0.1 * np.max(np.abs(f1d))


def test_local_surface_height_uses_highest_atom_within_cutoff():
    cell2d = CELL[:2, :2]
    surf_xy = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 8.0]])
    surf_z = np.array([10.0, 12.0, 11.0])
    probes = np.array([[0.5, 0.0],     # both first two atoms within 2.5 -> max z 12
                       [10.0, 8.5],    # only the third atom -> 11
                       [5.0, 4.0]])    # none within 2.5 -> nearest atom's z
    h = local_surface_height(probes, surf_xy, surf_z, cell2d, cutoff=2.5)
    d = np.linalg.norm(surf_xy - probes[2], axis=1)
    assert h[0] == 12.0 and h[1] == 11.0 and h[2] == surf_z[np.argmin(d)]


def test_local_surface_height_is_periodic():
    cell2d = CELL[:2, :2]
    surf_xy = np.array([[19.5, 0.0]])            # across the a boundary from the probe
    h = local_surface_height(np.array([[0.5, 0.0]]), surf_xy, np.array([10.0]),
                             cell2d, cutoff=2.0)
    assert h[0] == 10.0


def test_cartesian_resampling_on_oblique_cell():
    nb = 64
    centres = (np.arange(nb) + 0.5) / nb
    fa, fb = np.meshgrid(centres, centres, indexing="ij")
    field = np.cos(2 * np.pi * fa) + 0.5 * np.sin(2 * np.pi * fb)
    cell2d = CELL[:2, :2]
    # includes points outside the home cell, to exercise the periodic wrap
    x = np.linspace(-15, 40, 37)
    y = np.linspace(-10, 30, 29)
    out = resample_cartesian(field, cell2d, x, y)
    X, Y = np.meshgrid(x, y)
    frac = np.linalg.solve(cell2d.T, np.stack([X.ravel(), Y.ravel()]))
    exact = (np.cos(2 * np.pi * frac[0]) + 0.5 * np.sin(2 * np.pi * frac[1])).reshape(X.shape)
    assert np.max(np.abs(out - exact)) < 5e-3


def test_rejects_tilted_cell(frames):
    tilted = frames[0].copy()
    cell = tilted.cell.array.copy()
    cell[0, 2] = 1.0
    tilted.set_cell(cell)
    with pytest.raises(ValueError, match="xy plane"):
        accumulate_density([tilted], [0], [150])
