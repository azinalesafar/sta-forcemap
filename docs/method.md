# Method

## 1. The STA force

In the solvent-tip approximation, an AFM tip is represented by a single probe molecule (here
a water oxygen). The potential of mean force on that probe is $W = -k_\mathrm{B}T \ln \rho$,
where $\rho$ is the equilibrium probe density. The force along the surface normal is therefore

$$F(x, y, z) = -\frac{\partial W}{\partial z} = k_\mathrm{B}T\,\frac{\partial \ln\rho(x,y,z)}{\partial z}.$$

**Sign convention:** positive $F$ pushes the probe away from the surface (+z).

## 2. Height reference

Every probe's height is $h = z_\text{probe} - z_\text{ref}$. The `reference` setting chooses
$z_\text{ref}$.

### `local` (default)

$z_\text{ref}$ is the local surface under each probe. It is the $z$ of the highest selected
surface atom within a lateral (minimum-image) distance `cutoff` of the probe. If none is that
close, it is the $z$ of the laterally nearest surface atom. The reference follows the
topography atom by atom (this matters for non-planar adsorbates). The profiles of rough or
adsorbate-covered surfaces then describe a tip–sample distance measured from the topmost atoms.

The tool reports how many counted probes needed the nearest-atom fallback. A large fraction
usually means part of the surface is not selected, e.g. bare substrate between adsorbates:
water above it is then measured from an adsorbate atom several Å away. Add those atoms to the
surface selection, or use `plane`.

### `plane`

$z_\text{ref}$ is the mean $z$ of the selected surface atoms, recomputed in every frame. Use it
for:

* **bare substrates**: select the top layer. In `local` mode, thermal vibrations make the
  "highest atom within the cutoff" fluctuate by ~0.1–0.2 Å, which blurs sharp hydration peaks.
  The mean over the whole layer is much steadier.
* **planar adsorbates**: `local` would switch between adsorbate and substrate at the
  adsorbate's edges.

Because the plane is recomputed per frame, a slab that drifts in $z$ is followed
automatically. The mean is taken over *all* selected atoms, so select one layer: mixing a
substrate with an adsorbate on top of it, would put the plane in between (a height where there's nothing at all).

### `fixed`

$z_\text{ref}$ is a constant `reference_z` that you supply. This gives several systems on the
same substrate a common zero, e.g. with and without an adsorbate. It assumes the slab does not
drift in $z$.

### Which to choose

| surface | reference |
|---|---|
| bumpy / non-planar adsorbate | `local` |
| planar adsorbate | `plane` (select the adsorbate) or `local` |
| bare substrate | `plane` (select the top layer) |
| comparing systems on one substrate | `fixed` |

For a laterally uniform flat surface, all three give the same profile up to a constant shift of
$h$.

With `plane` and `fixed`, the lowest heights can contain no probes at all, e.g. the space
occupied by an adsorbate, or the gap between a fixed plane and the first water layer. See the
note on unsampled heights in section 4.

## 3. Density histogram

Probe positions are binned in $h$ (bin width `dz`, from 0 to `z_max`) and in the fractional
lateral cell coordinates $(a, b)$ (`lateral_bins` bins each). They are accumulated over the
selected frames and normalised to a number density in Å⁻³ using the mean lateral cell area.

The histogram is smoothed with Gaussians: periodic in $(a, b)$ (`lateral_smooth`) and
nearest-edge in $h$ (`z_smooth`). The smoothing tames the noise that the $z$-derivative
amplifies.

## 4. Force field and force profile

* **Force field** (right panel): $F(h, a, b) = k_\mathrm{B}T\,\partial_h \ln\rho(h,a,b)$ per
  pixel, using central finite differences.
* **Force profile** (left panel): $F(h) = k_\mathrm{B}T\,\partial_h \ln\langle\rho\rangle(h)$,
  where $\langle\rho\rangle$ is the *arithmetic* lateral mean of the density.

The order of averaging matters. The number of molecules in a slab scales with the arithmetic
mean of $\rho$ over the area, so that is the density whose logarithm gives the laterally
averaged PMF. Averaging the per-pixel force instead gives
$\langle \partial_h \ln \rho\rangle = \partial_h \ln (\prod \rho)^{1/N}$, i.e. it
differentiates the *geometric* mean. By Jensen's inequality, geometric ≤ arithmetic, with
equality only for a laterally uniform layer. In a structured first hydration layer the two
can differ severalfold. The per-pixel average is also dominated by nearly empty pixels, where
$\ln\rho$ diverges, so it depends strongly on any density floor used to mask them. The
arithmetic-mean profile has neither problem.

**Unsampled heights.** Height bins where fewer than `min_counts` probes were found over the whole
run (default 50) have no measurable force. The z-smoothing still leaks a tiny density into them,
and its logarithm would produce a large, meaningless spike. Such bins, and the bins next to them
(which central differences use), are therefore left blank: NaN in F(z), and an empty map for
slices that contain no other bins. Near the onset of the liquid, the first remaining bins still
depend on `z_smooth`. Check them with `--z-smooth 0` before interpreting a force there.

## 5. Slices and the Cartesian map

The map at slider position $z_0$ is the mean of the force field over the $h$-bins with
centres in $[z_0 - w, z_0 + w)$, where $w$ is `half_width`.

The field lives on the fractional $(a, b)$ grid, and plotting that grid directly would
distort oblique or non-square cells. Instead, each pixel of a square Cartesian window
(`patch_size`, `pixels`) is converted to fractional coordinates, and the field is sampled
there by periodic bilinear interpolation. The map therefore has true lengths and angles and
no seam at cell boundaries. Unit cells that don't fit evenly in the window are simply cut
off at its edges.

The colour range is computed once, from percentiles of the initial slice (`z_default`), and
held fixed while you move through $z$. This way, changes in contrast between slices are real.

## 6. Overlay

The overlay is drawn from the surface atoms of the first frame, or from `--overlay FILE`
(which must be in the same coordinate frame). Bonds are found with ASE's natural covalent
cutoffs × `bond_mult`. Each bonded fragment is made whole across the periodic boundary
before drawing.

## 7. Limitations

* The STA treats the tip as a single solvent molecule. It ignores the tip's own structure and
  its perturbation of the liquid.
* $F$ is a derivative of a histogram, so it needs good sampling. Check convergence by
  comparing halves of the trajectory, or with `--start/--stop`. Slow interfacial dynamics
  make the effective number of independent samples much smaller than the number of frames.
