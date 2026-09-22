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

**The basis is the first gate.** def2-TZVP underestimates the Br 2P splitting
by 16.6%; decontracting it recovers more than half (−7.6%). It is
all-electron for Br but non-relativistically contracted, and the SOC operator
samples exactly the near-nuclear region that contraction gets wrong. Since the
HOBr band exists *only* through SOC borrowing, that error goes straight into
the intensity. Settle it at the atom:

```
python ../toolchain/02_soc_atoms.py --sweep --atoms Br --soc DKH1 breit-pauli
```

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

- [ ] **Br basis**, at the atom, via `toolchain/02_soc_atoms.py --sweep`.
      Gates everything: every script below takes `--basis`.
- [ ] `01_method/01_geometry.py` — equilibrium geometry and harmonic force
      field from CCSD(T). Run `--molecule HOCl` first: its fundamentals are
      known, so it validates the whole chain against measurement before HOBr
      is asked for a number nobody can check.
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
