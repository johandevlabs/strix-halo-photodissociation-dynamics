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

## The basis: settled, 2026-09-22

Run at the atom, where it costs seconds
(`../toolchain/02_soc_atoms.py --sweep --atoms Br --soc DKH1 breit-pauli`,
log in [`logs/br_basis_sweep.log`](logs/br_basis_sweep.log)). Against the
observed Br 2P splitting of 3685.24 cm⁻¹:

| basis | nao | contraction | DKH1 | breit-pauli |
| --- | --- | --- | --- | --- |
| def2-tzvp | 48 | non-relativistic | −16.6% | −11.9% |
| def2-qzvp | 75 | non-relativistic | −16.1% | −11.3% |
| unc-def2-tzvp | 103 | decontracted | −7.6% | −1.7% |
| **cc-pvtz-dk** | **43** | **DK** | **−7.1%** | **−0.7%** |
| aug-cc-pvtz-dk | 59 | DK | −7.2% | −0.8% |
| ano-rcc | 109 | relativistic ANO | −7.0% | −0.0% |
| dyall-v3z | 128 | relativistic | −7.1% | −0.3% |

**Use `cc-pvtz-dk`.** It is the *smallest* basis in the sweep — smaller than
the def2-TZVP that is 16.6% wrong — and it matches the 128-function dyall set
to 0.1 points. `aug-` adds nothing (−0.8 against −0.7), so diffuse functions
are not where this lives.

**Contraction, not size.** def2-TZVP → def2-QZVP adds 27 functions and buys
0.5 points. Decontracting def2-TZVP — the same functions — buys 9.0 points.
What the contraction does near the nucleus is everything; how many functions
there are is almost irrelevant.

**The SOC operator is a second, independent error.** Across the five
relativistically contracted bases, spanning a 3× range in size, DKH1 sits at
−7.0 to −7.6%: converged, and wrong. Breit-Pauli on the same bases sits at
−0.0 to −1.7%. So basis convergence cannot fix DKH1 — it is already
converged. Phase 0.5 concluded "the SOC treatment was never the problem";
that was true of the def2 contraction error and false in general, and the
operator error was hidden underneath it.

**Breit-Pauli is not adopted yet.** Better agreement is not the same as more
correct: ano-rcc/BP lands 0.56 cm⁻¹ from experiment, which is far better than
CAS(5,3) + QD-NEVPT2 deserves and must be partly cancellation, and BP is a
first-order reduction generally held to be *less* reliable than DKH/X2C as Z
grows. The test is Cl, where `02` gives an independent anchor of −4.3% on
def2-TZVP/DKH1. Prediction: Cl shows the same pattern, smaller — cc-pvtz-dk
better than def2-tzvp, BP better again, both inside ~2%. If BP instead
*overshoots* on Cl, the Br agreement is cancellation and the right choice is
DKH1 carrying an explicit −7% error bar on the intensity.

**None of this rescues HOCl's intensity.** f scales as |H_SO|², so even a 7%
SOC error is 14% in f, against the 10-25× by which HOCl's f fell short of
measurement. That discrepancy is elsewhere.

## Layout

| | |
| --- | --- |
| [`01_method/`](01_method/) | geometry, force field, and the validity checks that must be redone for a heavier halogen |
| `01_method/tests/` | offline tests, run before anything goes to the EVO |
| `data/`, `logs/` | as in `HOCl/` |

Numbering restarts at `01` here rather than continuing HOCl's, because the
sequence genuinely differs — HOBr inherits the method decisions and needs
different checks. Each script's docstring names the HOCl script it descends
from.

## Order of work

- [x] **Br basis**, at the atom. Settled: `cc-pvtz-dk`. See above.
- [ ] **The Cl cross-check**, which decides DKH1 vs breit-pauli:
      `python ../toolchain/02_soc_atoms.py --sweep --atoms Cl --soc DKH1 breit-pauli`
- [ ] `01_method/01_geometry.py` — equilibrium geometry and harmonic force
      field from CCSD(T). Run `--molecule HOCl` first: its fundamentals are
      known, so it validates the whole chain against measurement before HOBr
      is asked for a number nobody can check. Worth running it under both
      `--basis def2-tzvp` (like-for-like with the HOCl surface work) and
      `--basis cc-pvtz-dk`, since that also tests whether the basis chosen for
      SOC is equally good for a force field, which the atomic sweep cannot
      say.
- [ ] **Molecular SOC and the borrowed intensity**, the analogue of
      `HOCl/01_method/03`. This is where the basis choice is confirmed on a
      molecule rather than a free atom, and where f is compared with Ingham's
      measured σ.
- [ ] **EOM-CCSD validity** across the Franck-Condon region, the analogue of
      `HOCl/01_method/11`, and the triplet manifold along O-Br, the analogue
      of `06`. Both must be redone: Br's SOC is 4× Cl's, and the 3A'/3A"
      crossing that sat harmlessly outside HOCl's band may not stay there.
- [ ] Raster, outer shell, fragments, splice — `HOCl/03_surfaces/`'s recipe.
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
