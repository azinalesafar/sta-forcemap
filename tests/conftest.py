import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

# Oblique cell, like a real hexagonal/low-symmetry surface cell
CELL = np.array([[20.0, 0.0, 0.0], [6.0, 17.0, 0.0], [0.0, 0.0, 40.0]])
SURFACE_Z = 10.0


def layer_density(h):
    """Unnormalised model hydration profile: one contact layer at h = 3 Å, broad
    enough (sigma = 1 Å) that 0.3 Å bins + central differences resolve it."""
    return 1.0 + 2.0 * np.exp(-(h - 3.0) ** 2 / 2.0)


def dln_layer_density(h):
    g = np.exp(-(h - 3.0) ** 2 / 2.0)
    return 2.0 * (-(h - 3.0)) * g / layer_density(h)


def _sample_heights(rng, n, hmin=0.2, hmax=12.0):
    out = np.empty(0)
    while out.size < n:
        h = rng.uniform(hmin, hmax, 2 * n)
        keep = rng.uniform(0, 3.0, h.size) < layer_density(h)
        out = np.concatenate([out, h[keep]])
    return out[:n]


def make_frames(n_frames=150, n_probe=2000, seed=0):
    """Flat surface layer of C at z=10 (FixAtoms) plus O probes whose height
    above it follows layer_density, laterally uniform."""
    rng = np.random.default_rng(seed)
    g = (np.arange(10) + 0.5) / 10
    fa, fb = np.meshgrid(g, g)
    surf = np.stack([fa.ravel(), fb.ravel()], 1) @ CELL[:2, :2]
    surf = np.column_stack([surf, np.full(len(surf), SURFACE_Z)])
    frames = []
    for _ in range(n_frames):
        frac = rng.uniform(0, 1, (n_probe, 2))
        xy = frac @ CELL[:2, :2]
        z = SURFACE_Z + _sample_heights(rng, n_probe)
        atoms = Atoms("C" * len(surf) + "O" * n_probe,
                      positions=np.vstack([surf, np.column_stack([xy, z])]),
                      cell=CELL, pbc=True)
        atoms.set_constraint(FixAtoms(indices=range(len(surf))))
        frames.append(atoms)
    return frames


@pytest.fixture(scope="session")
def frames():
    return make_frames()


@pytest.fixture(scope="session")
def traj_file(tmp_path_factory, frames):
    from ase.io import write
    path = tmp_path_factory.mktemp("data") / "synthetic.traj"
    write(path, frames[:40])
    return path
