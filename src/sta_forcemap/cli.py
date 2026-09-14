"""Command-line entry point: sta-forcemap TRAJECTORY -T TEMPERATURE [options]"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from . import __version__
from .pipeline import Settings, compute_force_map
from .plot import write_html


def build_parser():
    d = Settings(temperature=0.0)  # defaults
    p = argparse.ArgumentParser(
        prog="sta-forcemap",
        description="Build an interactive solvent-tip-approximation (STA) force map, "
                    "F = kT d ln(rho)/dz, from an MD trajectory of a liquid on a surface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("trajectory", help="trajectory file, any format ASE can read")
    p.add_argument("-T", "--temperature", type=float, required=True,
                   help="simulation temperature in K (sets kT)")
    p.add_argument("-o", "--output", help="output HTML (default: <trajectory>_forcemap.html)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    g = p.add_argument_group("input")
    g.add_argument("--format", default=None, help="ASE format name, if not guessable")
    g.add_argument("--start", type=int, default=d.start,
                   help="first frame to use (skip equilibration)")
    g.add_argument("--stop", type=int, default=d.stop, help="stop before this frame")
    g.add_argument("--stride", type=int, default=d.stride, help="use every n-th frame")

    g = p.add_argument_group(
        "atom selection",
        "SPEC is a comma-separated union of: 'fixed' (FixAtoms-constrained atoms), "
        "'all', element symbols (O, Pt), indices (12) or inclusive ranges (0-127).")
    g.add_argument("--surface", default=d.surface, metavar="SPEC",
                   help="atoms whose topography defines the local surface height")
    g.add_argument("--surface-zmin", type=float, default=None, metavar="Z",
                   help="drop surface atoms below this z (Å), e.g. a substrate "
                        "underneath an adsorbate")
    g.add_argument("--surface-zmax", type=float, default=None, metavar="Z",
                   help="drop surface atoms above this z (Å)")
    g.add_argument("--probe", default=d.probe, metavar="SPEC",
                   help="atoms whose density is measured (surface atoms are excluded)")
    g.add_argument("--cutoff", type=float, default=d.cutoff,
                   help="lateral radius (Å) for the local surface height")

    g = p.add_argument_group(
        "height reference",
        "What probe heights are measured from. 'local': the highest surface atom within "
        "--cutoff of each probe, following a bumpy surface (e.g. non-planar adsorbates). "
        "'plane': the mean z of the surface atoms in each frame (bare substrates, planar "
        "adsorbates, constant-height AFM comparisons). 'fixed': the plane z = --reference-z.")
    # SUPPRESS keeps argparse from printing a misleading "(default: None)"
    g.add_argument("--reference", choices=["local", "plane", "fixed"], default=argparse.SUPPRESS,
                   help="height reference (default: local, or fixed if --reference-z is given)")
    g.add_argument("--reference-z", type=float, default=argparse.SUPPRESS, metavar="Z",
                   help="z (Å) of the reference plane; implies --reference fixed")

    g = p.add_argument_group("density grid")
    g.add_argument("--z-max", type=float, default=d.z_max,
                   help="max height above the surface (Å); keep below any "
                        "liquid-vapour interface")
    g.add_argument("--dz", type=float, default=d.dz, help="z bin width (Å)")
    g.add_argument("--lateral-bins", type=int, default=d.lateral_bins,
                   help="bins along each lateral cell vector")
    g.add_argument("--lateral-smooth", type=float, default=d.lateral_smooth,
                   help="lateral Gaussian smoothing sigma (bins, 0 = off)")
    g.add_argument("--z-smooth", type=float, default=d.z_smooth,
                   help="z Gaussian smoothing sigma (bins, 0 = off)")
    g.add_argument("--bulk-window", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="z range (Å) over which to report the bulk density")
    g.add_argument("--min-counts", type=int, default=d.min_counts,
                   help="height bins with fewer raw probe counts (summed over all frames) "
                        "are left blank instead of showing a spurious force")

    g = p.add_argument_group("map")
    g.add_argument("--half-width", type=float, default=d.half_width,
                   help="half-width (Å) of the z slab averaged for each map")
    g.add_argument("--z-default", type=float, default=None,
                   help="initial slice height (Å); sets the colour range "
                        "(default: height of the first density peak)")
    g.add_argument("--z-step", type=float, default=d.z_step, help="slider step (Å)")
    g.add_argument("--patch-size", type=float, default=d.patch_size,
                   help="side of the square map window (Å)")
    g.add_argument("--pixels", type=int, default=d.pixels, help="map pixels per side")
    g.add_argument("--center", type=float, nargs=2, default=None, metavar=("X", "Y"),
                   help="map centre (Å) (default: centroid of the surface atoms)")
    g.add_argument("--cmap", default="afmhot", help="any matplotlib colormap name")
    g.add_argument("--clim-percentiles", type=float, nargs=2, default=d.clim_percentiles,
                   metavar=("LO", "HI"),
                   help="colour range percentiles of the initial slice")
    g.add_argument("--clim", type=float, nargs=2, default=None, metavar=("VMIN", "VMAX"),
                   help="explicit colour range (eV/Å), overrides --clim-percentiles")

    g = p.add_argument_group("overlay")
    g.add_argument("--overlay", default=None, metavar="FILE",
                   help="structure to draw on the map (default: the surface atoms "
                        "of the first frame)")
    g.add_argument("--no-overlay", action="store_true", help="draw no structure")
    g.add_argument("--bond-mult", type=float, default=d.bond_mult,
                   help="multiplier on ASE natural cutoffs for drawing bonds")

    g = p.add_argument_group("output")
    g.add_argument("--title", default="lateral STA force", help="map panel title")
    g.add_argument("--plotlyjs", choices=["inline", "cdn"], default="inline",
                   help="embed plotly.js (works offline) or load it from a CDN "
                        "(smaller file)")
    g.add_argument("--save-npz", metavar="PATH",
                   help="also save z, rho(z), F(z) and the full F(z, a, b) field")
    g.add_argument("-q", "--quiet", action="store_true")
    return p


def resolve_reference(a):
    reference = getattr(a, "reference", None)
    if getattr(a, "reference_z", None) is not None:
        if reference not in (None, "fixed"):
            raise ValueError(f"--reference-z cannot be combined with --reference {reference}")
        return "fixed"
    if reference == "fixed":
        raise ValueError("--reference fixed needs --reference-z")
    return reference or "local"


def settings_from_args(a):
    return Settings(
        temperature=a.temperature,
        surface=a.surface, surface_zmin=a.surface_zmin, surface_zmax=a.surface_zmax,
        probe=a.probe, reference=resolve_reference(a),
        reference_z=getattr(a, "reference_z", None),
        cutoff=a.cutoff,
        start=a.start, stop=a.stop, stride=a.stride, format=a.format,
        lateral_bins=a.lateral_bins, dz=a.dz, z_max=a.z_max,
        lateral_smooth=a.lateral_smooth, z_smooth=a.z_smooth,
        bulk_window=tuple(a.bulk_window) if a.bulk_window else None,
        min_counts=a.min_counts,
        half_width=a.half_width, z_default=a.z_default, z_step=a.z_step,
        patch_size=a.patch_size, pixels=a.pixels,
        center=tuple(a.center) if a.center else None,
        clim_percentiles=tuple(a.clim_percentiles),
        clim=tuple(a.clim) if a.clim else None,
        overlay=not a.no_overlay, overlay_file=a.overlay, bond_mult=a.bond_mult,
    )


def save_npz(result, path):
    s = result.settings
    np.savez(path, z=result.z_mid, rho=result.rho1d, force_profile=result.f1d,
             force_field=result.f3d.astype(np.float32), cell_xy=result.cell2d,
             temperature=s.temperature, kT=s.kT, n_frames=result.histogram.n_frames,
             lateral_bins=s.lateral_bins, dz=result.histogram.dz, reference=s.reference,
             reference_z=np.nan if s.reference_z is None else s.reference_z)


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.temperature <= 0:
        sys.exit("error: --temperature must be positive")
    out = a.output or os.path.splitext(a.trajectory)[0] + "_forcemap.html"
    try:
        settings = settings_from_args(a)
        result = compute_force_map(a.trajectory, settings, verbose=not a.quiet)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"error: {e}")
    if a.save_npz:
        save_npz(result, a.save_npz)
        if not a.quiet:
            print(f"Wrote {a.save_npz}")
    size = write_html(result, out, title=a.title, cmap=a.cmap, plotlyjs=a.plotlyjs)
    if not a.quiet:
        print(f"Wrote {out} ({size / 1e6:.1f} MB, {len(result.z_values)} slices)")


if __name__ == "__main__":
    main()
