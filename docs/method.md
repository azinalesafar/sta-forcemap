# Method

## 1. The STA force

In the solvent-tip approximation, an AFM tip is represented by a single probe molecule (here
a water oxygen). The potential of mean force on that probe is $W = -k_\mathrm{B}T \ln \rho$,
where $\rho$ is the equilibrium probe density. The force along the surface normal is therefore

$$F(x, y, z) = -\frac{\partial W}{\partial z} = k_\mathrm{B}T\,\frac{\partial \ln\rho(x,y,z)}{\partial z}.$$

**Sign convention:** positive $F$ pushes the probe away from the surface (+z).

## 2. Height above the local surface

Heights are measured from the local surface under each probe, not from a flat reference
plane. For each probe atom, the local surface height is the $z$ of the highest selected
surface atom within a lateral (minimum-image) distance `cutoff`. If none is that close, it is
the $z$ of the laterally nearest surface atom. The probe's height is then
$h = z_\text{probe} - z_\text{surface}$.
This makes the profiles of rough or adsorbate-covered surfaces comparable to a
tip-sample distance measured from the topmost atoms.

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

## 7. Caveats

* The STA treats the tip as a single solvent molecule. It ignores the tip's own structure and
  its perturbation of the liquid.
* $F$ is a derivative of a histogram, so it needs good sampling. Check convergence by
  comparing halves of the trajectory, or with `--start/--stop`. Slow interfacial dynamics
  make the effective number of independent samples much smaller than the number of frames.
* Smoothing, bin widths and slab half-width all change the apparent contrast. Report them
  alongside any published map.
