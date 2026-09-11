"""Ball-and-stick geometry of the surface/adsorbate for the map overlay."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs


@dataclass
class Overlay:
    symbols: np.ndarray     # (n,)
    xy: np.ndarray          # (n, 2) lab-frame Å, molecules made whole
    bonds: list             # [(i, j), ...]


def build_overlay(atoms, cell2d, bond_mult=0.85):
    """Lateral positions + bonds of `atoms`, in the lab frame of `cell2d`.

    Atoms are wrapped into the cell, then each bonded fragment is made whole
    by walking the bond graph with minimum-image steps, so molecules that
    straddle the cell boundary are drawn in one piece rather than with bonds
    stretched across the cell.
    """
    n = len(atoms)
    A = np.asarray(cell2d).T
    frac = np.mod(atoms.positions[:, :2] @ np.linalg.inv(A).T, 1.0)

    cutoffs = natural_cutoffs(atoms, mult=bond_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    whole = frac.copy()
    visited = np.zeros(n, dtype=bool)
    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        while stack:
            i = stack.pop()
            for j in nl.get_neighbors(i)[0]:
                if visited[j]:
                    continue
                delta = frac[j] - frac[i]
                delta -= np.round(delta)
                whole[j] = whole[i] + delta
                visited[j] = True
                stack.append(j)

    bonds = sorted({(i, int(j)) for i in range(n) for j in nl.get_neighbors(i)[0] if j > i})
    return Overlay(symbols=np.array(atoms.get_chemical_symbols()),
                   xy=(A @ whole.T).T, bonds=bonds)
