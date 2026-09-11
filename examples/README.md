# Example data

`water_on_adsorbate.traj`: 300 frames (every 20th frame after 1000 frames of equilibration) of
an MD run at 350 K of liquid water on a carbon substrate carrying a fixed C/N/H organic
adsorbate, simulated with a MACE machine-learning potential.

* atoms 0–127: substrate carbon (z = 12 Å, fixed)
* atoms 128–253: adsorbate, 56 C + 8 N + 62 H (fixed)
* atoms 254–864: water oxygens (water hydrogens were removed to keep the file small)

The substrate lies under the adsorbate, so the surface is selected as "fixed atoms above
z = 13 Å":

```bash
sta-forcemap water_on_adsorbate.traj -T 350 --surface fixed --surface-zmin 13
```
