"""Small atom-selection language used for --surface and --probe.

A selection is a comma-separated union of terms:

    fixed        atoms held by an ase.constraints.FixAtoms constraint
    all          every atom
    O, Pt, ...   every atom of that element
    12           a single atom index (0-based)
    0-127        an inclusive range of atom indices

Examples: "fixed", "O", "C,N", "0-127,300-310".
"""
from __future__ import annotations

import numpy as np
from ase.constraints import FixAtoms
from ase.data import chemical_symbols


def _fixed_indices(atoms):
    idx = []
    for c in atoms.constraints:
        if isinstance(c, FixAtoms):
            idx.extend(c.get_indices())
    if not idx:
        raise ValueError(
            "selection 'fixed' requested, but the reference frame has no FixAtoms "
            "constraint (formats such as plain .xyz do not store constraints); "
            "select the surface atoms by element or index instead")
    return idx


def select_atoms(atoms, spec, zmin=None, zmax=None):
    """Return the sorted indices of `atoms` matched by `spec`.

    zmin/zmax optionally keep only atoms whose z coordinate (in `atoms`) lies
    within [zmin, zmax] -- e.g. to drop a substrate layer lying underneath an
    adsorbate.
    """
    n = len(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    chosen = np.zeros(n, dtype=bool)
    terms = [t.strip() for t in str(spec).split(",") if t.strip()]
    if not terms:
        raise ValueError("empty atom selection")
    for term in terms:
        low = term.lower()
        if low == "fixed":
            chosen[_fixed_indices(atoms)] = True
        elif low == "all":
            chosen[:] = True
        elif term in chemical_symbols:
            chosen |= symbols == term
        elif "-" in term:
            lo, hi = term.split("-", 1)
            lo, hi = int(lo), int(hi)
            if not (0 <= lo <= hi < n):
                raise ValueError(f"index range {term!r} out of bounds for {n} atoms")
            chosen[lo:hi + 1] = True
        else:
            try:
                i = int(term)
            except ValueError:
                raise ValueError(f"unrecognised selection term {term!r}") from None
            if not 0 <= i < n:
                raise ValueError(f"atom index {i} out of bounds for {n} atoms")
            chosen[i] = True
    z = atoms.positions[:, 2]
    if zmin is not None:
        chosen &= z >= zmin
    if zmax is not None:
        chosen &= z <= zmax
    return np.flatnonzero(chosen)
