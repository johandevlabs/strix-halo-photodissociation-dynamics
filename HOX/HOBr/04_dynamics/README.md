# 04_dynamics — HOBr

| script | descends from | what |
| --- | --- | --- |
| `11_jacobi.py` | water's `07_jacobi.py` | valence → Jacobi (R, r, γ); continues the surface beyond its raster |
| `12_relax.py` | water's `08_relax.py` | vibrational states by imaginary time (imports water's `Propagator`) |
| `13_propagate.py` | water's `09_propagate.py` | 3D propagation, σ(λ, T) (imports water's `RTProp`) |

**Production runs are on the EVO.** Grid 256 × 48 × 48 (590k points), four
states (v0, ν₃, ν₂, 2ν₃). What ran on the laptop was a 256 × 24 × 24 test grid
with two states, to validate the pipeline before sending it.

Water's propagators are imported rather than copied. What is HOBr's own:

- **The transform's continuation of the surface.** The raster is a box in all
  three coordinates and the Jacobi grid is not. Beyond r(O-H) the free OH
  curve (Morse fit to `08`'s fragments) is added; beyond the angle range a
  stiff quadratic rise; r(O-Br) held flat, which past 6.25 Å is the asymptote
  anyway.
- **One absorber.** HOBr has one channel, Br + OH. Water needed absorbers in
  both radial coordinates because either OH bond can break.
- **The cross section.** Water's `09` carries the factor 2 in σ (see
  `../02_band_model/README.md`); `13` has it fixed, and checks the exact sum
  rule ∫(σ/E) dE = 4π²|μ|²/3c on every run.

## Two things the test grid caught

- **A grid too coarse in R looks like physics.** A 96-point R grid carries
  momenta up to π/dR = 30 bohr⁻¹; heavy Br sliding off the wall needs ~50.
  The momentum aliased and the packet looked trapped: S(t) never decayed and
  the band collapsed to a 0.03 eV spike. `11` now checks k_max against the
  energy the packet picks up and says what nR is needed.
- **The sum rule written with f needs an energy**, and `⟨V_ex⟩ − E₀` misses
  χ₀'s kinetic energy — 6% in 3D. The exact form ∫(σ/E) dE = 4π²|μ|²/3c needs
  none, and gives 1.0000 in both the 1D and 3D codes.

## First look, from the test grid (superseded below)

| | peak | FWHM | red HWHM |
| --- | --- | --- | --- |
| 3D, test grid | 443 nm | 0.575 eV | 0.272 eV |
| 1D (`03`) | 437 nm | 0.570 | 0.268 |
| Barnes 1996 | 437 nm | 0.548 | 0.282 |
| Ingham 1998 / JPL / IUPAC | 457 nm | 0.281 | 0.152 |

The bend and O-H stretch add ~1% to the width: the 3D band stays with Barnes.
ν₃ from the 3D ground surface: 618.9 cm⁻¹ against 620.2 observed; ν₂ (coarse
grid) 1171.8 against 1162.6.

## Production run, EVO, 2026-09-29

256 × 48 × 48, four states: ν₃ 618.9, ν₂ 1171.8, 2ν₃ 1230.1 cm⁻¹ (observed
620.2, 1162.6). Band at 298 K: **443 nm, FWHM 0.575 eV** — identical to the
test grid. Scaled to Barnes's integrated f, its peak is Barnes's measured
8.9 × 10⁻²¹ cm²: position, width and height all match Barnes 1996.

σ(298)/σ(220), Condon, stretch and bend hot bands:

| nm | 440 | 457 | 480 | 500 | 520 | 550 |
| --- | --- | --- | --- | --- | --- | --- |
| as computed | 0.972 | 0.968 | 0.985 | 1.029 | 1.120 | 1.427 |
| Ingham-aligned | 0.981 | 0.970 | 0.971 | 0.990 | 1.036 | 1.194 |

−3% to +3% over 440-500 nm — negligible by `TASKS.md`'s threshold — and
+12% to +43% in the red wing, where σ is a fifth of the peak and less.
