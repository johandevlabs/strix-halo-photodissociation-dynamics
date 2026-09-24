# HOBr — the actual target

HOCl was the validation case. HOBr is the contribution: its visible band
carries up to 50% of the HOBr photolysis rate at high solar zenith angles, and
**no temperature dependence has ever been measured or computed** for any
hypohalous acid band.

Measured, to aim at (Ingham et al. 1998): λmax 457 nm, σ = 2.3 ± 0.2 × 10⁻²⁰
cm². Impact threshold from [`../TASKS.md`](../TASKS.md): if σ over 440-500 nm
changes by ≥10-15% between 220 K and 298 K it materially affects polar
J(HOBr); below 5% the honest conclusion is that it is negligible.

## What HOBr inherits, and what it cannot

Most of `HOCl/01_method/` does not need repeating — those runs settled
*method* questions, and the answers carry over. What does not carry over is
anything that depends on the halogen being light:

| settled for HOCl | carries to HOBr? |
| --- | --- |
| the two states are each the lowest of their spin manifold | yes — a symmetry argument |
| no interior avoided crossing; one smooth 3A" surface | **no** — must be rechecked, the Br SOC is 4× larger |
| EOM-CCSD on the CCSD(T) ground state, right slope, smoothest | **no** — heavier atom, must be rechecked |
| the band needs the surface only to ~2.3 Å | likely, but the 1D model should be rerun on HOBr's own curve |
| raster + NEVPT2 shell + fragment asymptote, spliced | yes — the recipe, not the numbers |
| def2-TZVP | **no** — see below |

## The basis and the SOC operator: settled, 2026-09-22

Two sweeps at the atom, Br and Cl, over 12 candidate all-electron sets and
both SOC Hamiltonians ([`logs/br_basis_sweep.log`](logs/br_basis_sweep.log),
[`logs/cl_basis_sweep.log`](logs/cl_basis_sweep.log)). Errors against the
observed 2P splittings, 3685.24 cm⁻¹ for Br and 882.35 for Cl:

| basis | Br nao | Cl DKH1 | Cl BP | Br DKH1 | Br BP | |
| --- | --- | --- | --- | --- | --- | --- |
| def2-tzvp | 48 | −4.3% | −3.1% | −16.6% | −11.9% | non-rel. contracted |
| def2-qzvp | 75 | −6.6% | −5.4% | −16.1% | −11.3% | non-rel. contracted |
| jorge-tzp-dkh | 65 | −2.1% | −0.8% | −2.3% | +4.9% | unconverged TZ |
| x2c-tzvpall | 48 | −3.8% | −2.5% | −4.9% | +1.4% | unconverged TZ |
| sapporo-dkh3-tzp | 43 | — | — | −5.2% | +1.5% | unconverged TZ |
| **cc-pvtz-dk** | **43** | **−6.5%** | −5.4% | **−7.1%** | −0.7% | **converged** |
| aug-cc-pvtz-dk | 59 | −6.7% | −5.5% | −7.2% | −0.8% | converged |
| x2c-qzvpall | 83 | −5.9% | −4.5% | −7.3% | −0.5% | converged |
| unc-def2-tzvp | 103 | −6.5% | −5.3% | −7.6% | −1.7% | converged |
| ano-rcc | 109 | −6.3% | −5.1% | −7.0% | −0.0% | converged |
| dyall-v3z | 128 | −6.2% | −4.9% | −7.1% | −0.3% | converged |

**Use `cc-pvtz-dk` with DKH1**, and carry the deficit as an error bar.

### Contraction, not size

def2-TZVP → def2-QZVP adds 27 functions and buys 0.5 points on Br.
Decontracting def2-TZVP — the same functions — buys 9.0. `cc-pvtz-dk` is the
*smallest* set in the sweep, smaller than the def2-TZVP that is 16.6% wrong,
and matches the 128-function dyall set to 0.1 points. What the contraction
does near the nucleus is nearly everything; how many functions there are is
nearly nothing.

### Agreeing better is not converging

Six independent families cluster tightly once converged: Br DKH1 −7.2 ± 0.3%,
Cl DKH1 −6.4 ± 0.4%. The sets that agree *better* with experiment —
jorge-tzp-dkh, x2c-tzvpall, sapporo, and def2-TZVP on Cl — are all
unconverged triple-zeta, and the x2c family gives the clean within-family
control:

| | TZ | QZ |
| --- | --- | --- |
| Cl, x2c-*all, DKH1 | −3.8% | −5.9% |
| Br, x2c-*all, DKH1 | −4.9% | −7.3% |

**Improving the basis moves away from experiment**, towards the converged
value. So `02`'s Phase 0.5 headline — Cl −4.3% on def2-TZVP, "toolchain is
sound" — was an unconverged number flattered by cancellation. The converged
figure is −6.4%. The toolchain *is* sound (degeneracy 4+2, correct inverted
ordering); it was the accuracy claim that was optimistic.

### Breit-Pauli's Br result is cancellation, as suspected

The prediction recorded before the Cl run was that Cl would show the same
pattern, smaller, with both operators inside ~2%. **That was wrong**, and
wrong in the informative direction. At the basis limit:

| | Cl (Z=17) | Br (Z=35) |
| --- | --- | --- |
| DKH1 | −6.4% | −7.2% |
| Breit-Pauli | −5.1% | −0.7% |
| BP − DKH1 | +1.2 points | +6.6 points |

DKH1's deficit is nearly **Z-independent**. Breit-Pauli adds a shift that
grows 5.3× from Cl to Br — which is how the known Breit-Pauli overestimate
behaves, since it lacks the relativistic damping of the wavefunction near the
nucleus and that failure scales steeply with Z. For Br it happens to cancel
the deficit almost exactly; for Cl it does not. Adopting Breit-Pauli would
have meant adopting a Z-dependent artefact that is only calibrated at one
value of Z — and HOI, the eventual Phase 4 target, is Z=53.

### What the remaining ~7% probably is

Not the basis (converged) and not the operator (nearly Z-independent, so not
relativistic in origin). The most likely candidate is the reference:
CAS(5e,3o) + QD-NEVPT2 with only the np shell active captures no core-valence
correlation and no core polarisation, and the SOC constant is sensitive to the
radial density near the nucleus. A minimal active space would underestimate it
by roughly a fixed fraction for both atoms, which is what is seen.

That is a hypothesis, not a result, and it has a cheap test: enlarge the
active space at the atom and see whether the deficit shrinks for Cl and Br
together. Worth doing, because the molecular calculation uses a similarly
minimal active space.

**Consequence for the band:** f scales as |H_SO|², so a 7% SOC deficit is 14%
in the intensity. Real, worth quoting, and nowhere near HOCl's 10-25×
shortfall — that remains unexplained and elsewhere.

## Layout

| | |
| --- | --- |
| [`01_method/`](01_method/) | geometry, force field, and the validity checks that must be redone for a heavier halogen |
| [`02_band_model/`](02_band_model/) | the 1D band: how much surface it needs, the basis measured on the band, a first look at σ(298)/σ(220) |
| [`03_surfaces/`](03_surfaces/) | the 3D raster, and later the NEVPT2 shell, fragments and splice |
| `01_method/tests/` | offline tests, run before anything goes to the EVO |
| `data/`, `logs/` | as in `HOCl/` |

Numbering restarts at `01` here rather than continuing HOCl's, because the
sequence genuinely differs — HOBr inherits the method decisions and needs
different checks. Each script's docstring names the HOCl script it descends
from.

## Order of work

- [x] **Br basis and SOC operator**, at the atom. Settled: `cc-pvtz-dk` with
      DKH1, carrying a ~7% SOC deficit. See above.
- [x] **The Cl cross-check.** Done; it falsified the prediction and settled
      the operator question against Breit-Pauli.
- [x] `01_method/01_geometry.py` validated on HOCl with `cc-pvtz-dk`:
      frequencies 769.4 / 1279.8 / 3817.9 cm⁻¹ against fundamentals 724.36 /
      1238.62 / 3609.48, ratios 1.062 / 1.033 / 1.058 — all in the expected
      band, harmonic above fundamental. T1 ≤ 0.0075, fit residual 0.145 meV.
      The chain works.

      Geometry came out r(O-Cl) 1.7069, r(O-H) 0.9648, ∠ 101.88°, against the
      literature 1.6891 / 0.9644 / 102.96 used throughout the HOCl work. The
      +0.018 Å on r(O-Cl) is the one notable deviation and is worth
      understanding before the same method places HOBr's minimum.

      **No def2-TZVP comparison exists yet**, despite a log that claims one —
      see the cache bug below.
- [x] **HOCl under def2-tzvp**: better heavy-atom stretch than cc-pvtz-dk
      for Cl (ratio 1.039 against 1.062), plausibly cc-pVTZ's missing tight d
      on a second-row atom. A Cl issue; HOBr unaffected.
- [x] **HOBr's geometry**: r(O-Br) 1.8357 Å, r(O-H) 0.9646 Å, ∠ 102.02°;
      ω 632 / 1199 / 3869 cm⁻¹, ratios to the fundamentals 1.020 / 1.032 /
      1.070. v=1 of the O-Br stretch: 1.60% at 220 K, 4.72% at 298 K.
- [x] **`02_obr_cut.py`**: the crossing stays ~0.7 Å out (3A′ lowest from
      2.55 Å); EOM trustworthy to ~2.40 Å by single-excitation weight;
      vertical 432 nm against 457 measured; **aug- does not matter where the
      bond breaks** (exit channel flat to 9 meV) and costs 4.1×. Raster in
      `cc-pvtz-dk`. Details in `../TASKS.md`.
- [x] **`02_band_model/03_band_1d.py`**: the band needs the surface to
      ~2.0 Å (HOCl's offsets exactly), so the raster to 2.40 Å has 0.4 Å of
      margin; aug- moves the band 0.9 nm; implied f ≥ 1.3e-4 (after a factor-2 fix, see there); and σ(298)/σ(220)
      is −1 to −3% over 440-500 nm but +20% at 550 nm. See
      `02_band_model/README.md`.
- [x] **`01_method/04_soc_vertical.py`**: f = 1.5e-5, 17x HOCl's (the
      SOC-squared scaling exactly) and ~8x short of the 1.3e-4 needed --
      HOCl was 10-25x short, so a common systematic. SOC shifts the band by
      -1 meV: it does NOT close the 20 nm gap. CASSCF unconverged at 100
      cycles.
- [ ] **The f deficit**: `04 --nroots 10/16 --max-cycle 300`, then a larger
      active space. CAS(12,7) has one virtual, so every state it can borrow
      from is n/pi -> sigma*.
- [x] **`03_surfaces/05_pes_raster.py`**: 2470 points, 0 failed, 9.7 h;
      `06_raster_check.py` finds it smooth (<= 5.4 meV), warnings only at
      >= 2.30 A, and the cut reproduced to 0.02 meV.
- [x] **EOM-CCSD validity**: along O-Br by `02` (the crossing stayed out).
      Across O-H and the bend -- HOCl's `11` -- the raster checks it at every
      point instead (T1, single-excitation weight, A'/A" label), so a separate
      run is not needed; read it off the raster.
- [ ] Outer NEVPT2 shell, fragments, splice -- `HOCl/03_surfaces/`'s `13`-`15`.
- [ ] Relaxation, Boltzmann average, propagation, σ(λ, T) over 200-300 K.

## Why the force field is not a detail here

σ(λ, T) is a Boltzmann average over initial vibrational states, and the
populations are exponential in the frequencies. At 298 K the O-Br stretch
(620 cm⁻¹ observed) holds ~5.0% of the population and the bend (1163) ~0.4%;
at 220 K, 1.7% and 0.05%. A frequency 10% off moves the population that
carries the entire temperature effect by ~15% of itself. That is why
`01_geometry.py` computes the force field and checks it against the observed
fundamentals rather than assuming the surface will come out right.

It also means the **O-X stretch, not the bend, is the mode that carries the
temperature dependence** — and since the stretch is the dissociation
coordinate, it is the one coordinate the model treats most carefully.
