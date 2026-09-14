import numpy as np
import pytest
from ase import Atoms

from sta_forcemap import accumulate_density

CELL = np.diag([20.0, 20.0, 40.0])


def stepped_surface():
    """10 x 10 grid (2 Å spacing): low terrace z=10 for x<10, high terrace z=12."""
    g = np.arange(10) * 2.0 + 1.0
    X, Y = np.meshgrid(g, g)
    Z = np.where(X < 10, 10.0, 12.0)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])   # mean z = 11


def frame(surface, probes):
    pos = np.vstack([surface, probes])
    return Atoms("C" * len(surface) + "O" * len(probes), positions=pos, cell=CELL, pbc=True)


def heights(hist):
    """Bin-centre heights of the non-empty z bins."""
    return hist.z_mid[hist.counts.sum(axis=(1, 2)) > 0].tolist()


@pytest.fixture
def stepped():
    surf = stepped_surface()
    probes = np.array([[5.0, 5.0, 14.0],      # over the low terrace
                       [15.0, 5.0, 16.0]])    # over the high terrace
    atoms = frame(surf, probes)
    return atoms, np.arange(len(surf)), np.arange(len(surf), len(atoms))


KW = dict(dz=1.0, z_max=10.0, lateral_bins=4)


def test_local_follows_the_terraces(stepped):
    atoms, surf, probe = stepped
    hist = accumulate_density([atoms], surf, probe, reference="local", **KW)
    assert heights(hist) == [4.5]            # both probes 4 Å above their own terrace


def test_plane_uses_mean_surface_height(stepped):
    atoms, surf, probe = stepped
    hist = accumulate_density([atoms], surf, probe, reference="plane", **KW)
    assert heights(hist) == [3.5, 5.5]       # 14 - 11 and 16 - 11


def test_fixed_uses_given_plane(stepped):
    atoms, surf, probe = stepped
    hist = accumulate_density([atoms], surf, probe, reference="fixed", reference_z=9.0, **KW)
    assert heights(hist) == [5.5, 7.5]       # 14 - 9 and 16 - 9


def test_local_and_plane_agree_on_flat_surface(frames):
    kw = dict(dz=0.3, z_max=11.0, lateral_bins=20)
    surf, probe = np.arange(100), np.arange(100, len(frames[0]))
    local = accumulate_density(frames[:5], surf, probe, reference="local", **kw)
    plane = accumulate_density(frames[:5], surf, probe, reference="plane", **kw)
    np.testing.assert_array_equal(local.counts, plane.counts)
    assert local.n_fallback == 0


def test_plane_follows_a_drifting_slab(stepped):
    atoms, surf, probe = stepped
    shifted = atoms.copy()
    shifted.positions[:, 2] += 3.0
    both = accumulate_density([atoms, shifted], surf, probe, reference="plane", **KW)
    once = accumulate_density([atoms], surf, probe, reference="plane", **KW)
    np.testing.assert_array_equal(both.counts, 2 * once.counts)
    fixed = accumulate_density([atoms, shifted], surf, probe, reference="fixed",
                               reference_z=9.0, **KW)
    assert heights(fixed) == [5.5, 7.5, 8.5]  # the shifted frame lands 3 Å higher
    # (its high-terrace probe, at 10 Å, falls outside z_max)


def test_fallback_is_counted():
    surf = np.array([[5.0, 5.0, 10.0]])
    probes = np.array([[5.5, 5.0, 13.0],       # within the cutoff
                       [15.0, 15.0, 13.0]])    # ~14 Å away: fallback
    atoms = frame(surf, probes)
    hist = accumulate_density([atoms], [0], [1, 2], cutoff=2.5, **KW)
    assert (hist.n_counted, hist.n_fallback) == (2, 1)
    assert hist.fallback_fraction == 0.5


def test_unsampled_heights_are_blank_not_spikes(traj_file):
    """Flat surface at z=10 with probes from h=0.2 Å; measured from z=5, nothing
    is below h~5.2 Å. Smoothing leaks density into those bins, which must come
    out as NaN rather than as a large spurious force."""
    from sta_forcemap import Settings, compute_force_map
    s = Settings(temperature=300, reference="fixed", reference_z=5.0, z_max=14.0,
                 lateral_bins=20, pixels=16, patch_size=20, z_step=0.5, overlay=False)
    r = compute_force_map(traj_file, s, verbose=False)
    assert np.isnan(r.f1d[r.z_mid < 4.9]).all()
    assert np.isfinite(r.f1d[(r.z_mid > 6.5) & (r.z_mid < 12)]).all()
    assert np.isnan(r.slice_map(2.0)).all()
    assert np.isfinite(r.slice_map(9.0)).all()

    local = compute_force_map(traj_file, Settings(temperature=300, z_max=10.0, lateral_bins=20,
                                                  pixels=16, patch_size=20, z_step=0.5,
                                                  overlay=False), verbose=False)
    assert np.isfinite(local.f1d[local.z_mid > 0.5]).all()


@pytest.mark.parametrize("kw", [dict(reference="bogus"),
                                dict(reference="fixed"),
                                dict(reference="plane", reference_z=3.0)])
def test_invalid_reference_arguments(stepped, kw):
    atoms, surf, probe = stepped
    with pytest.raises(ValueError):
        accumulate_density([atoms], surf, probe, **kw)
