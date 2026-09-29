# HOI

The last of the series, and the one where the HOCl/HOBr picture may break.

For HOCl and HOBr the visible band is a triplet (ã ³A″) borrowing a little
intensity from the singlet above it through spin-orbit coupling: one
spin-free surface, a borrowed transition dipole, and the dynamics as for any
single state. Iodine's spin-orbit constant is 2.1x bromine's, and the
measurements say more than that changes.

## What the measurements say

`02_band_model/10_obs_bands.py` (log: `logs/obs_bands.log`) fits Gaussians
in energy to the cached spectra in `data/obs/` and integrates each band:

| | visible band | f | UV band | f | vis/UV |
| --- | --- | --- | --- | --- | --- |
| HOCl (Barnes 1998, JPL) | 368 nm | 3.4e-5 | 310 nm | 3.0e-4 | 0.11 |
| HOBr (Ingham, Barnes 1996) | 437-457 nm | 4.7-6.2e-5 | 352 nm | 6.5e-4 | 0.07-0.10 |
| HOI (Bauer, JPL, Rowley) | 407 nm | 0.96-1.17e-3 | 340 nm | 1.85e-3 | 0.52-0.64 |

HOI's visible band is ~20x HOBr's; perturbation theory from HOBr's computed
f, scaled by the spin-orbit constant squared, gives ~4x. And the band is
bluer than HOBr's, not redder. Both suggest a strongly mixed state.

Spectra (MPI-Mainz UV/VIS Spectral Atlas): Bauer et al., J. Phys. Chem. A
102, 2857 (1998), doi:10.1021/jp9804300; Rowley et al., J. Atmos. Chem. 34,
137 (1999), doi:10.1023/A:1006210322389; Jenkin (1991); the IUPAC 2007 and
JPL-2010 recommendations.

## The gate

`HOBr/01_method/04_soc_vertical.py --molecule HOI` -- the HOBr script, made
generic. Its report ends in a verdict from a two-state estimate of the
singlet admixture (c² = triplet SOC shift / gap; HOBr 2.0%):

- **PERTURBATIVE** (c² < 5%): HOBr's pipeline carries over as is.
- **BORDERLINE** (5-20%): same pipeline, but spin-orbit coupling must enter
  the surface.
- **STRONGLY MIXED** (> 20%): coupled spin-orbit states in the dynamics.

**Basis:** cc-pvtz-dk has no iodine anywhere. The default is x2c-tzvpall,
which is all-electron and contracted for X2C. A same-basis HOBr run bridges
the change.

**Geometry:** recalled for now; `HOBr/01_method/01_geometry.py --molecule
HOI` computes it (CSV to `data/`).
