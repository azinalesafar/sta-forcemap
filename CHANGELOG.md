# Changelog

## 0.2.0 (2026-09-14)

* New `--reference {local,plane,fixed}` and `--reference-z` options choose what probe heights
  are measured from:
  * `local` (default, unchanged behaviour): the local surface under each probe;
  * `plane`: the mean z of the surface atoms, per frame;
  * `fixed`: a constant plane.

  `plane` suits bare substrates, planar adsorbates and constant-height AFM comparisons.
* In `local` mode, the tool now reports the fraction of probes measured from the nearest
  surface atom because none was within `--cutoff`, and warns above 10%.
* Heights with fewer than `--min-counts` probe counts (default 50) are now left blank, in both
  F(z) and the maps. Smoothing previously leaked a tiny density into such empty regions, where
  its logarithm gave large spurious forces. This matters with `plane`/`fixed` references, where
  the lowest heights can be empty. Results are unchanged wherever every height is sampled.
* The F(z) axis label names the height reference.
* `--save-npz` also stores `reference` and `reference_z`.

## 0.1.0 (2026-09-11)

* First release.
