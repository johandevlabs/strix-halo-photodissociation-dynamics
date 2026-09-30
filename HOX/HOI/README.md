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

`01_method/04_soc_vertical.py` registers HOI in HOBr's `04` and runs it,
as HOBr's scripts load HOCl's. The report ends in a verdict from a two-state estimate of the
singlet admixture (c² = triplet SOC shift / gap; HOBr 2.0%):

- **PERTURBATIVE** (c² < 5%): HOBr's pipeline carries over as is.
- **BORDERLINE** (5-20%): same pipeline, but spin-orbit coupling must enter
  the surface.
- **STRONGLY MIXED** (> 20%): coupled spin-orbit states in the dynamics.

**Basis:** cc-pvtz-dk, as for HOBr. (An earlier version of this file said
it has no iodine. That was wrong; the first gate run used x2c-tzvpall
because of it. A same-basis HOBr run showed the change is harmless.)

**Geometry:** `01_method/01_geometry.py`, CCSD(T): r(O-I) 1.9907 Å, r(O-H)
0.9694 Å, 104.65°.

## Result so far: HOI's triplet band has not been measured

EOM-CCSD across the series (`01_method/05_eom_vertical.py`), calibrated on
HOCl and HOBr, assigns the measured bands. **407 nm is the 1A″ band, with
³A′ under it, and 340 nm is 1A′**: HOBr's 352 and 284 nm bands, red-shifted.
The offsets are +0.11 and +0.17 eV, as for HOBr. Calling 407 nm the
triplet band would need −0.63 eV. This matches Minaev, J. Phys. Chem. A
103, 7294 (1999), doi:10.1021/jp990203d.

So the "visible band 20× HOBr's" in the table above compares different
states. HOI's ³A″ band is computed at 516 nm (f 1.5e-4, singlet admixture
~20%). Calibrated, that is ~530–560 nm, beyond every measurement (Bauer
stops at 490 nm, and its red end is fixed by an I₂ subtraction scaled at
500 nm). If the band is there, J(HOI) is 20–50% larger at high sun
(`05_atmosphere/15_j_hoi_triplet.py`, log in `logs/`).

The one real constraint is Bauer's 532 nm photolysis, which found no OH:
σ(532) < ~1e-20 cm², but only if HO + I is open at 532 nm. It is:
`01_method/06_d0.py` gives D₀(HO–I) = 2.07–2.18 eV (CCSD(T)/CBS, spin-orbit
and zero-point corrected, calibrated on HOCl's measured D₀). That is a
threshold of 570–600 nm, and it agrees with Bauer's 582 nm, not IUPAC's
507 nm.

**So the calculations and the 532 nm null disagree by 3–4×.** Ours
(f 1.5e-4) and Minaev's (1.6–2.5e-4) both put 3–4e-20 cm² at 532 nm.
Meeting Bauer's bound needs f ≲ 4e-5, less than HOBr's measured triplet
band. An OH action spectrum over 500–600 nm, as Barnes did for HOBr, would
settle it, and it avoids the I₂ subtraction.

**The computed band** (`02_band_model/11_band_soc_1d.py`, a 1D model on the
spin-orbit states along O–I) peaks at 514 nm raw, or about 555 nm after
calibrating on Bauer's two measured bands (the calculation is 0.17–0.19 eV
too blue on both). Its FWHM is 0.35 eV and its f is 1.45e-4, giving σ(532)
= 2.8–3.6e-20, 3× Bauer's bound. The same model overestimates the measured
singlet bands' f by 2.4–4.3×, which suggested the triplet might be
overestimated in the same way. The active-space benchmark
(`01_method/08_cas_bench.py`) rules that out. A larger CAS brings the
singlet bands to within 1.1–1.7× of measurement, but leaves the triplet f
unchanged (1.3–1.5e-4). For HOCl and HOBr the same method underestimates
the triplet (30–50× and 3–5×). So the calculation is more likely low than
high for HOI. Every estimate (ours, Minaev's 1.6–2.5e-4, and an
extrapolation from HOBr's error) puts σ(532) at 1.5–6e-20, which is 1.5–6×
over Bauer's bound. J(HOI) at the computed f is 18–32% higher at SZA 0–60°,
with photolysis cut at the 585 nm threshold.

On the way: along this cut the spin-free a³A″ has a shallow well near
2.4 Å, which spin-orbit coupling mostly removes. The triplet's dipole is
strongly non-Condon (f falls 3× from 1.75 Å to equilibrium). Details are in `TASKS.md`; paper
notes are in `../references/README.md`.

Offline test: `python 01_method/tests/test_hoi_report.py`.
