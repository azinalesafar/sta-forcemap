# sta-forcemap

Interactive **solvent-tip-approximation (STA) force maps** of interfacial liquids from
molecular-dynamics trajectories.

Given an MD trajectory of water (or another liquid) on a surface, `sta-forcemap` measures the
probe-molecule density relative to the local surface height and converts it to the force a
probe would feel.

$$F(x, y, z) = k_\mathrm{B}T \, \frac{\partial \ln \rho(x, y, z)}{\partial z},$$

the quantity usually compared with 3D-AFM / force-spectroscopy measurements of hydration
layers. The result is an HTML page:

* **left:** the laterally averaged force profile $F(z)$;
* **right:** the lateral force map $F(x, y)$ in true Cartesian geometry (oblique cells
  handled correctly), with the surface structure drawn on top as ball-and-stick;
* a **slider** to move the map through different heights $z$.

<!-- TODO: add a screenshot, e.g. docs/screenshot.png -->

## Installation

```bash
pip install sta-forcemap                                          # once released on PyPI
pip install git+https://github.com/azinalesafar/sta-forcemap      # latest version from GitHub
```

Requires Python ≥ 3.9 with numpy, scipy, ASE, plotly and matplotlib (installed automatically).

## Quick start

The repository contains a small example: 300 frames of water on a fixed organic adsorbate
(water hydrogens removed to keep the file small; only oxygens are used).

```bash
sta-forcemap examples/water_on_adsorbate.traj -T 350 --surface-zmin 13
```

This writes `examples/water_on_adsorbate_forcemap.html`. Open it in any browser.

## Using your own trajectory

```bash
sta-forcemap md.traj -T 300 --start 1000 --surface fixed --probe O --z-max 15
```

Any trajectory format ASE can read works (`.traj`, extended XYZ, LAMMPS dump, VASP XDATCAR, ...;
use `--format` if the format can't be guessed from the extension).

### What the tool assumes

* The surface lies in the **xy plane**. Cell vectors *a*, *b* are in-plane and *c* is along
  *z*. The liquid sits **above** the surface, at larger *z*. Probes below the surface are
  ignored.
* The surface atoms don't change identity during the run. They can be fixed or mobile: their
  positions are read in every frame.
* Use `--start N` to discard the first N frames while the system is equilibrating.

### Choosing atoms

`--surface` selects the atoms that define the local surface height. `--probe` selects the
atoms whose density is measured. Both take a comma-separated union of:

| term | meaning |
|---|---|
| `fixed` | atoms held by an ASE `FixAtoms` constraint (default for `--surface`) |
| `O`, `Pt`, ... | all atoms of an element (`O` is the default for `--probe`) |
| `12`, `0-127` | an atom index or an inclusive index range (0-based) |
| `all` | every atom |

`--surface-zmin` / `--surface-zmax` restrict the surface selection by height. For example, an
adsorbate lying on a fixed substrate is selected with `--surface fixed --surface-zmin <z between them>`.
Surface atoms are always removed from the probe selection. If the substrate itself contains
oxygen, select the water oxygens by index instead.

### Height reference

`--reference` chooses what heights are measured from:

| reference | height measured from | use for |
|---|---|---|
| `local` (default) | the highest surface atom within `--cutoff` Å (default 2.5 Å) laterally of each probe; the nearest surface atom if none is that close | bumpy, non-planar adsorbates |
| `plane` | the mean *z* of the surface atoms, recomputed every frame | bare substrates (select the top layer), planar adsorbates, constant-height AFM comparisons |
| `fixed` | a plane you give with `--reference-z Z` | a common zero for several systems on the same substrate |

```bash
sta-forcemap md.traj -T 300 --surface 0-63 --reference plane       # bare substrate, top layer = atoms 0-63
sta-forcemap md.traj -T 300 --reference-z 12.0                     # heights above z = 12 Å
```

See [docs/method.md](docs/method.md#2-height-reference) for details.

### Main options

| option | default | meaning |
|---|---|---|
| `-T/--temperature` | required | temperature in K; sets $k_\mathrm{B}T$ |
| `--start/--stop/--stride` | `0/end/1` | which frames to use |
| `--reference`, `--reference-z` | `local` | what heights are measured from (see above) |
| `--z-max` | 20 Å | maximum height; keep it below any liquid–vapour interface |
| `--dz`, `--lateral-bins` | 0.3 Å, 100 | density grid resolution |
| `--lateral-smooth`, `--z-smooth` | 1.0, 1.5 bins | Gaussian smoothing (0 = off) |
| `--min-counts` | 50 | heights with fewer probe counts are left blank |
| `--half-width` | 1.0 Å | half-thickness of the slab averaged for each map |
| `--z-default` | first density peak | initial slice; also sets the colour range |
| `--clim-percentiles` / `--clim` | 0.2 99.8 | colour range (fixed for all slices) |
| `--patch-size`, `--pixels`, `--center` | 80 Å, 200, surface centroid | map window |
| `--overlay FILE` / `--no-overlay` | surface atoms | structure drawn on the map |
| `--save-npz` | – | also save $z$, $\rho(z)$, $F(z)$ and the full $F(z, a, b)$ array |

Run `sta-forcemap --help` for the full list.

### Output size

Every slider position embeds one `pixels × pixels` map, so the file size grows as
`(z-max / z-step) × pixels²`. The defaults give ~50 MB. To make the file smaller, use
`--pixels 120`, a coarser `--z-step`, a smaller `--z-max`, or `--plotlyjs cdn` (saves ~4.5 MB
but needs internet access to display).

## Python API

```python
from sta_forcemap import Settings, compute_force_map
from sta_forcemap.plot import write_html

s = Settings(temperature=350, surface="fixed", surface_zmin=13, start=1000, z_max=15)
result = compute_force_map("md.traj", s)

result.z_mid, result.f1d          # F(z) profile, eV/Å
fmap = result.slice_map(2.6)      # F(x, y) at z = 2.6 ± half_width, on result.x / result.y
write_html(result, "map.html")
```

The building blocks (`accumulate_density`, `smooth_density`, `force_field`, `force_profile`,
`slab_average`, `resample_cartesian`) can also be used on their own.

## Method

See [docs/method.md](docs/method.md). In particular, it explains why $F(z)$ is computed from the
**arithmetic** lateral mean of the density, not by averaging the per-pixel force map.

## Citing

If you use this software, please cite it. See [CITATION.cff](CITATION.cff), or use GitHub's
"Cite this repository" button.

## License

MIT. See [LICENSE](LICENSE).
