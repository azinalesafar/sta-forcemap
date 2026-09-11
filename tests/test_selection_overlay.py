import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from sta_forcemap import select_atoms
from sta_forcemap.overlay import build_overlay


@pytest.fixture
def atoms():
    a = Atoms("C4N2O3", positions=[[0, 0, z] for z in (1, 2, 3, 4, 5, 6, 7, 8, 9)],
              cell=[10, 10, 20], pbc=True)
    a.set_constraint(FixAtoms(indices=[0, 1, 2, 3, 4]))
    return a


@pytest.mark.parametrize("spec, expected", [
    ("fixed", [0, 1, 2, 3, 4]),
    ("O", [6, 7, 8]),
    ("C,N", [0, 1, 2, 3, 4, 5]),
    ("2-4,8", [2, 3, 4, 8]),
    ("all", list(range(9))),
])
def test_select(atoms, spec, expected):
    assert select_atoms(atoms, spec).tolist() == expected


def test_select_z_window(atoms):
    assert select_atoms(atoms, "fixed", zmin=2.5).tolist() == [2, 3, 4]
    assert select_atoms(atoms, "all", zmin=2.5, zmax=4.5).tolist() == [2, 3]


@pytest.mark.parametrize("spec", ["Xx", "3-99", "12", ""])
def test_select_errors(atoms, spec):
    with pytest.raises(ValueError):
        select_atoms(atoms, spec)


def test_fixed_without_constraint():
    with pytest.raises(ValueError, match="FixAtoms"):
        select_atoms(Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]]), "fixed")


def test_overlay_keeps_molecule_whole_across_boundary():
    # C-C bond of 1.4 Å straddling the a boundary of an oblique cell
    cell = np.array([[20.0, 0, 0], [6.0, 17.0, 0], [0, 0, 30.0]])
    a = Atoms("C2", positions=[[0.5, 1.0, 10.0], [19.1, 1.0, 10.0]], cell=cell, pbc=True)
    ov = build_overlay(a, cell[:2, :2])
    assert ov.bonds == [(0, 1)]
    assert np.linalg.norm(ov.xy[0] - ov.xy[1]) == pytest.approx(1.4)
