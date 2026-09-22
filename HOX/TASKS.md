# HOX — tasks

Target: the **visible (triplet) absorption band of HOBr**, and its
temperature dependence, from 3D quantum wavepacket dynamics on an ab initio
a 3A'' surface with spin-orbit coupling.

Background and citations: `docs/deep-review-claude.md`.

**The premise is not yet verified.** Phase 0 exists to kill this project
cheaply if the gap is already filled. Do Phase 0 before writing any code.

---

## Why this target

- HOBr has three bands: 284, 351 and 457 nm. Ingham et al. (1998) report the
  457 nm band is responsible for **up to 50% of total J(HOBr)** at high solar
  zenith angles near the surface, and at least 25% elsewhere down to SZA 40°.
- That band is a singlet→triplet transition (a 3A'' ← X 1A') borrowing
  intensity through Br spin-orbit coupling. Assigned by Francisco et al.
  (1996), intensity computed by Minaev et al. (1999).
- Published 3D wavepacket dynamics on HOBr covers the **singlets only**
  (Balint-Kurti/Szalay 1999; Zhang & Jiang 2020).
- **No temperature dependence has been measured or computed for any HOX
  band.** JPL and IUPAC recommendations rest on ~295 K data.
- HOBr photolysis is the light-driven branch of the bromine explosion cycle
  driving polar boundary-layer ozone depletion.

The gap as stated: *a 3D ab initio triplet surface, propagated quantum
mechanically, giving σ(λ, T) for the band that dominates J(HOBr).*

---

## Phase 0 — verify the premise (do this first)

Cheap, and it either hardens the claim or saves months.

- [ ] Read **Füsti-Molnár, Szalay, Balint-Kurti**, *J. Chem. Phys.* **110**,
      8448 (1999) and its Part II in full. Confirm singlet-only.
- [ ] Read **Zhang & Jiang**, *Chin. J. Chem. Phys.* **33**, 173 (2020),
      doi:10.1063/1674-0068/cjcp1911214. Confirm the three states are
      singlets and no triplet surface is constructed.
- [ ] Read **Minaev et al.**, *J. Phys. Chem. A* (1999),
      doi:10.1021/jp990203d, and the THEOCHEM companion. Confirm these are
      response-theory transition probabilities, not dynamics on a surface.
      Extract their predicted band positions and cross sections for
      comparison later.
- [ ] Read the 2016 nonadiabatic HOBr study (doi pii S1010603016300740).
      Confirm it is TDDFT surface-hopping / nuclear-ensemble, not quantum
      wavepacket on a converged surface.
- [ ] Open **JPL 19-5**, BrOx Photochemistry, reaction G6 in a browser
      (the site blocks automated fetch) and confirm no temperature
      parameterisation for HOBr.
- [ ] Search specifically for HOI 3D wavepacket dynamics on any surface.
      The review found none, but the search was not exhaustive.

### SOC route — resolved on paper, 2026-09-13

The open question was whether the PySCF toolchain can produce SOC matrix
elements and SOC-borrowed transition dipoles at all. It can, via two
open-source PySCF plugins:

- **Prism** (sokolov-group/prism) — NEVPT2 and QD-NEVPT2 with
  *state-interaction spin-orbit coupling*, Breit-Pauli and X2C-DKH
  Hamiltonians, plus a cheaper SOMF-QDNEVPT2 variant. The method paper is
  Sokolov and co-workers, "Simulating Spin-Orbit Coupling with Quasidegenerate
  N-Electron Valence Perturbation Theory", *J. Phys. Chem. A* (2023),
  arXiv:2211.06466, with a second-order follow-up arXiv:2404.04716.
- **socutils** (xubwa/socutils) — the SOC integrals (X2CAMF, optional Gaunt
  and Breit), bundled with a quaternion spinor SCF eigensolver. Prism
  requires it for SOC.

This matters because SO-QDNEVPT2 reports **oscillator strengths from the
spin-orbit-mixed states**, which *is* the borrowed intensity the ã 3A" band
needs — not a separate piece of machinery to build. It also sits directly on
the SA-CASSCF/NEVPT2 stack `water/` already uses, so the single-toolchain
property survives.

**Confirmed on the EVO, 2026-09-13.** Both plugins build and import against
conda-forge PySCF 2.14 (the build needed three fixes, all the same
pip-vs-conda layout split; see `00_setup_soc.sh`). The API, from the shipped
`prism/examples/soc/01-qdnevpt2-SOC.py`:

```python
interface = prism.interface.PYSCF(mf, mc, backend='opt_einsum')
nevpt = prism.nevpt.NEVPT(interface)
nevpt.method = "nevpt2"; nevpt.method_type = "qd"
nevpt.soc = "DKH1"          # or breit-pauli
e_tot, e_corr, osc = nevpt.kernel()
```

That last line is from `04-qdnevpt2-SUS.py` and it is the important one:
**`kernel()` returns oscillator strengths**, so the SOC-borrowed intensity is
read straight out rather than assembled by hand. There is also a cheaper
CASSCF-level route, `interface.run_soc("x2c-1")`, with no perturbation
correction.

**Validated against experiment, 2026-09-13.** `02_soc_atoms.py` on Cl,
def2-TZVP, SA-CASSCF(5e,3o)/3 roots, DKH1-QD-NEVPT2:

| check | result |
| --- | --- |
| degeneracy pattern | **[4, 2]** — correct for a 2P term |
| ordering | 4-fold at 0, 2-fold at +844.54 cm-1: **inverted**, correct for p^5 |
| 2P_1/2 − 2P_3/2 (Cl) | **844.54 vs 882.35 cm-1 observed, −4.3%** |
| 2P_1/2 − 2P_3/2 (Br) | **3072.71 vs 3685.24 cm-1 observed, −16.6%** |
| oscillator strengths | returned, ~1e-24, i.e. zero as required within a term |
| cost | Cl 5.3 s, Br 9.5 s; SOC step 1.1 and 3.2 s wall (~30x parallel) |

The inversion is the check that matters most: a normal ordering would have been
a sign error wearing a plausible magnitude. Toolchain is sound.

**Br: the basis, confirmed.** def2-TZVP gave −16.6%; simply decontracting it
(`--basis unc-def2-tzvp`) gave **−7.6%**, recovering more than half the error
for 14.9 s instead of 9.5 s (nao 48 -> 103). So the SOC treatment was never the
problem: def2-TZVP is all-electron for Br but *non-relativistically
contracted*, and the SOC operator samples precisely the near-nuclear region
that contraction gets wrong.

Follow-ups, in order of expected value:

- [ ] A basis built for this, rather than one repaired by decontraction:
      x2c-TZVPall, ANO-RCC, or dyall (`pip install basis-set-exchange`).
      Expect better than −7.6% at lower cost than full decontraction.
- [ ] Compare `--soc breit-pauli` against DKH1 on the same basis. Cheap, and
      it bounds how much of the residual belongs to the SOC operator.
- [ ] Whatever basis wins here is the one HOBr should use. Fixing this at the
      atom is far cheaper than discovering it in a 3D raster.

Remaining caveats:

- **Only a light atom, and only an atom.** Cl SOC is small and a free atom has
  no bonding to get wrong. Br is the real scaling test, and the molecular case
  adds the part HOX actually needs — SOC *borrowing* between states of
  different multiplicity, which the atomic test does not exercise at all.
- **Prism's NEVPT2 is not PySCF's.** `water/` used PySCF's native strongly
  contracted `mrpt.NEVPT2`; Prism is a separate fully internally contracted
  implementation. This is a new dependency, not a flag on existing code.
- **Cost per geometry is the real open question.** A 3D raster is thousands of
  points, and SO-QDNEVPT2 over several roots is far heavier than the SC-NEVPT2
  used for water. If it is too slow, SOMF-QDNEVPT2 or a geometry-independent
  (constant) SOC approximation are the fallbacks.

### HOCl molecular test — intensity borrowing works, 2026-09-13

`03_soc_hocl_vertical.py`, def2-TZVP, AVAS CAS(12e,7o), 6 roots in the Ms=0
space, DKH1-QD-NEVPT2. Prism logged `Apply S_plus due to Ms=0...`, confirming
it recognises the mixed-multiplicity reference and builds the missing Ms
components itself. Three singlets + three triplets gave 3(1) + 3(3) = **12 SOC
states**, exactly as the microstate count requires.

| group | n | dE / eV | dE / nm | sum f | |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0.0000 | — | — | singlet, ground |
| 1 | **3** | **3.4477** | **359.6** | **8.92e-07** | **a 3A" — borrowed** |
| 2 | 1 | 4.3620 | 284.2 | 9.03e-04 | singlet |
| 3 | 3 | 4.4220 | 280.6 | 1.97e-04 | triplet |
| 4 | 1 | 5.3837 | 230.3 | 5.92e-03 | singlet |
| 5 | 3 | 7.2025 | 172.1 | 1.55e-06 | triplet |

**The borrowing works.** The a 3A" band is dark by spin selection and comes out
with f = 8.9e-7 — small, nonzero, and split into three components by ~4 cm-1.
Against the measured band at 380 nm: vertical is **+0.185 eV (+5.7%)**, good
for this level. The intensity is roughly 10-25x low depending on the assumed
band width, which is the same *kind* of failure water had with mu (low by a
constant 1.29), though larger.

### Cost: the blocker for Phase 1

One point cost 155 s wall **using the entire machine** — htop showed all 32
threads saturated, and the SOC step alone reported a 30x parallel factor. The
first estimate in the script divided wall time by 30 processes and got 4.7 h,
which is simply wrong: 30 concurrent copies cannot each have all 16 cores.

The honest measure is total CPU work over the machine's throughput. The SOC
step alone was 414 CPU-seconds; with the reference included a point is
plausibly 2500-4000 CPU-seconds, so a 3289-point raster is roughly
**2300-3700 CPU-hours, or 6-10 days on 16 cores.** For scale, water's entire
raster was 41 minutes. `03` now measures CPU time directly and reports this
properly, so the next run replaces that range with a number.

**Measured, 2026-09-13: 5783 CPU-seconds per point.** A 3289-point raster is
**5283 CPU-hours, 330 h (~14 days) on 16 cores**, against 41 minutes for the
whole of `water/`. The observed parallel factor was 31.7x over 32 logical CPUs,
so the machine was fully committed — there is no headroom to reclaim by
rearranging the job.

`--nroots 4` made it **worse**, not better: wall 155 -> 183 s, because the
CASSCF stopped converging (hit the cycle limit). Averaging over more states
evidently stabilises it. The unconverged reference also moved f from 8.92e-07
to 4.71e-07, nearly 2x, while the vertical energy barely moved (3.4477 ->
3.4495 eV). **f is the sensitive quantity; the energy is not.** Any future
economy has to be judged on f, not on the excitation energy.

So the full multi-state raster is off the table at this level of theory.

### Phase 1, restructured — state-specific surfaces

Johan's observation, and it is the way out: the two states HOX needs are each
the **lowest of their own spin manifold**, so neither requires excited-state
theory at all.

| quantity | method | where | cost |
| --- | --- | --- | --- |
| V(X 1A') | CCSD(T), closed shell | bound region | water's `13_gs_well.py` |
| V(a 3A") | ~~state-specific, spin=2~~ **SA-CASSCF + SC-NEVPT2 over 3A" roots** — see `06` below | full 3D raster | 624 CPU-s/pt |
| mu_SOC(R) | 6-state QD-NEVPT2 + SOC | **FC window only** | 5783 CPU-s/pt |

The intensity is the one thing this does not give: a 3A" <- X 1A' is
spin-forbidden and borrows entirely from bright singlets, so f needs the
multi-state treatment. But only across the Franck-Condon window, exactly as in
`water/09_propagate.py`: *"The dipole enters only at t = 0, which is why mu(R)
was only ever needed across the Franck-Condon window."*

- [x] `04_statespecific_check.py` — **dCCSD(T) gives 3.4142 eV against the
      multi-state 3.4477, off by 0.034 eV**, converged, `<S^2>` = 2.0000
      exactly, 594 CPU-s. The state-specific route reproduces the expensive
      one at equilibrium.

      dCASSCF+NEVPT2 is a dead end here: unconverged even at 200 macro cycles
      and 14245 CPU-s, 24x dCCSD(T) and 2.5x the full multi-state route it was
      meant to undercut. *(Later overturned by `06`: the problem was the
      single-root CASSCF. Averaged over several 3A" roots it converges at
      every point and costs 624 CPU-s.)* It was also redundant — `03`'s QD-NEVPT2+SOC and
      `04`'s dCCSD(T) are already two unrelated methods (multireference
      perturbation vs single-reference coupled cluster) agreeing to 0.034 eV,
      which is exactly the cross-check water got from NEVPT2 vs UCCSD(T) on
      the OH curve.

- [x] `05_triplet_scan.py`, O-Cl from 1.4 to 4.0 A — **found a defect.** The
      triplet surface is not smooth between roughly 2.2 and 2.8 A:

      | r / A | E(T) / Ha | T1(T) | |
      | --- | --- | --- | --- |
      | 2.00 | −536.71711832 | 0.0290 | |
      | 2.20 | −536.72000708 | 0.0411 | minimum |
      | 2.40 | −536.71561692 | **0.0952** | rises |
      | 2.60 | −536.71913365 | **0.0995** | **falls again — unphysical** |
      | 2.80 | −536.71781644 | 0.0182 | T1 recovers |

      Past its minimum a dissociating state must rise monotonically to the
      asymptote. It does not: a 0.096 eV drop at 2.60 A, with second
      differences 89x and 54x the median post-minimum curvature at 2.40 and
      2.60. The reference changes character through that window and leaves a
      kink in the potential. `<S^2>` is 2.0000 throughout, so it is not spin
      contamination.

      **Two mitigating facts.** The Franck-Condon window is clean — T1(T) is
      0.024-0.030 around equilibrium, below the open-shell threshold — and the
      band position and width follow from that region by the reflection
      principle. And size consistency passed: the triplet at 4.0 A sits 28 meV
      from separately computed OH + Cl.

      But the packet travels through 2.2-2.8 A on its way out, so this has to
      be repaired before any propagation. A wavepacket scattering off an
      artefact is precisely the class of quiet wrong answer this repo
      catalogues.

- [x] **Rerun with `--symmetry Cs` — changed nothing, and could not have.**
      Energies matched the unsymmetric scan to ~3e-7 Ha at every point,
      including 2.40 and 2.60 A. That is not evidence against a state switch;
      it shows the test was the wrong one. In water, forcing the point group
      worked because state-averaged CASSCF *labels its roots*. A state-specific
      SCF with symmetry on still fills orbitals by aufbau — it only makes them
      symmetry-pure, and for a planar molecule they already were. What pins a
      single determinant to 3A" is `irrep_nelec`, the alpha/beta count per
      irrep. **The water gotcha needs a different mechanism for state-specific
      references**, and that belongs in the gotchas list once confirmed.

      The fragment asymptote also failed under Cs (OH is C∞v, Cl an atom); it
      now always runs without symmetry.

- [x] **Rerun with `--pin-irrep` — no state switch.** Equilibrium triplet
      occupation {A': (11,10), A": (3,2)} = 3A". Unconstrained aufbau kept it
      at every point from 1.4 to 4.0 A, and the pinned energies match the
      unpinned ones to ~1e-7 Ha, kink included. So the kink is not an A'/A"
      flip. Whatever crowds our state is itself 3A": a 3A' would genuinely
      cross, aufbau would have jumped to it, and this run would have said so.

- [x] **Is it an avoided crossing?** Johan's reading of the plots: a higher
      3A" state appears to come down and meet ours near 2.5 A. It fits
      everything so far:
      - same-symmetry states repel rather than cross, and the pin run shows
        any partner must be 3A";
      - T1 settles at a *different* level on each side (0.024-0.030 inside,
        0.014-0.018 outside), like a change of dominant configuration rather
        than a method steadily failing;
      - OH(2Pi) + Cl(2P) gives 6 spatial states, singlet and triplet each, so
        several 3A" states converge on one asymptote.

      Earlier wording in `05` called the drop at 2.60 A "unphysical for a
      dissociating state". That overstated it: a real avoided crossing can put
      a hump into the lower adiabat. What `05` shows is that a single
      determinant cannot follow a change of configuration, not what the true
      curve does. Reworded.

      `06_triplet_manifold.py` computes several 3A" roots together along the
      same cut (SA-CASSCF, symmetry forced to A", S = 1, then SC-NEVPT2 per
      root, AVAS rebuilt per point as in `water/03_tdm_check.py`) and reports
      three measures that do not depend on active-orbital ordering, which
      changes along a scan: the root 1 - root 0 gap, a localised dip in each
      root's dominant-configuration weight, and whether the roots *exchange
      dipole moments* across the gap minimum. Its verdict logic was checked on
      synthetic weak, strong and no-crossing cases before being sent to the
      EVO.

      **Result, 2026-09-14: no avoided crossing — the higher 3A" states converge
      on ours asymptotically instead.** Four runs, 4 roots, AVAS [Cl 3p, O 2p]
      giving CAS(10e,6o) from 1.8 A outward, def2-TZVP, all points converged:
      CASSCF 1.7-3.2 A, SC-NEVPT2 1.7-3.2 A, then SC-NEVPT2 at 0.025 A
      spacing over 3.0-3.4 and 3.2-3.6 A.

      | r / A | root 0 NEVPT2, eV rel. 2.0 A | UCCSD(T), same | gap 1-0 / eV | gap 2-1 / eV | w0 | w1 | T1(T) |
      | --- | --- | --- | --- | --- | --- | --- | --- |
      | 1.8 | +0.436 | +0.514 | 2.848 | 1.783 | 0.82 | 0.71 | 0.024 |
      | 2.0 | 0.000 | 0.000 | 1.752 | 1.230 | 0.64 | 0.54 | 0.029 |
      | 2.2 | **−0.076** | **−0.079** | 1.029 | 0.751 | 0.45 | 0.37 | 0.041 |
      | 2.4 | −0.038 | +0.041 | 0.583 | 0.441 | 0.41 | 0.39 | 0.095 |
      | 2.6 | +0.012 | −0.055 | 0.329 | 0.257 | 0.43 | 0.45 | 0.100 |
      | 2.8 | +0.052 | −0.019 | 0.191 | 0.150 | 0.43 | 0.48 | 0.018 |
      | 3.2 | +0.102 | +0.013 | 0.078 | 0.054 | 0.44 | 0.53 | 0.015 |

      - **The multireference lower curve is smooth.** NEVPT2 root 0 has a
        shallow minimum at 2.20 A and rises monotonically after it, with no
        drops. Relative to 2.0 A it agrees with UCCSD(T) at the minimum to
        **3 meV**, then the two separate by 0.07-0.09 eV from 2.4 A outward.
      - **The gap never has an interior minimum.** Root 1 - root 0 closes
        monotonically from 2.85 eV at 1.8 A to 0.042 eV at 3.6 A. At 3.6 A
        roots 0, 1 and 2 sit at 0, +0.042 and +0.063 eV while root 3 is at
        +4.15 eV: three 3A" states converging, which is exactly the count
        OH(2Pi) x Cl(2P) predicts (a'/a" x a'/a'/a" gives three A").
      - **No exchange of character.** In the CASSCF run the root 0 and root 1
        dipoles converge smoothly to ~1.83 D together.
      - **Why UCCSD(T) fails there.** The weight of root 0's dominant
        configuration falls from 0.82 at 1.8 A through 0.64 at 2.0 to 0.45 at
        2.2 and 0.39 at 2.3 A, then stays near 0.43. The T1 spike (0.041 at
        2.2, 0.095 at 2.4) coincides with the state becoming strongly
        multiconfigurational as the neighbouring 3A" states close in. A single
        determinant cannot represent a state whose largest configuration is
        ~40% of it. No crossing is needed to explain the kink.

      So Johan's eye was right that higher 3A" states come down to meet ours:
      two of them do. They converge on a shared asymptote rather than crossing.

- [x] **What to propagate on.** With no interior crossing, Landau-Zener does
      not apply. The band σ(E, T) is set in the Franck-Condon region on a
      femtosecond timescale, on a single smooth 3A" surface. The manifold only
      becomes coupled beyond ~3 A, where the 3A" gaps (0.12 eV at 3.0 A,
      falling) drop below the Cl 2P spin-orbit splitting (882 cm-1, 0.11 eV).
      There SOC reorganises the three states, which is where the
      Cl(2P3/2) vs Cl(2P1/2) product branching is decided. That matters for
      product fine structure, not for the absorption band, so it is optional
      scope rather than a blocker.

- [ ] **Build the triplet surface multireference, not dCCSD(T).**
      SA-CASSCF(10,6) over 4 3A" roots + SC-NEVPT2 cost **624 CPU-s/point**
      and converged at all 16 points, against 830-922 CPU-s/point for the
      dCCSD(T) scan. That is *cheaper* than the single-reference route, smooth
      where UCCSD(T) is not, and it yields the neighbouring 3A" states as a
      by-product. About 570 CPU-hours, or ~36 h on 16 cores, for 3289 points.
      The ground-state surface stays CCSD(T) over the bound region, where
      T1(S) stays below 0.02 out to 2.4 A (0.018) and passes it by 2.6 A.
- [ ] **Fix the active space inside the Franck-Condon window first.** AVAS
      gives CAS(12e,7o) at 1.7 A (and at 1.6891 A in `04`) but CAS(10e,6o)
      from 1.8 A outward, so the switch falls right where the band is decided.
      A raster needs one size throughout. Compare the two at equilibrium and
      pick one, via an AVAS threshold or an explicit orbital count.
- [ ] **Check the vertical energy on this footing**: singlet and triplet with
      the same active space and SC-NEVPT2, against the multi-state 3.4477 eV
      and dCCSD(T) 3.4142 eV.

      Both are `07_fc_active_space.py`, on a uniform grid r_eq + k x 0.05 A,
      k = -3..4 (1.54-1.89 A, equilibrium included):
      1. prints the AVAS projection eigenvalues and the atomic character of
         the borderline orbitals at each point, to show whether the switch is
         the 6th occupied orbital drifting through the 0.2 threshold;
      2. selects FIXED sizes by reimplementing AVAS's projector with an
         orbital count instead of a threshold, and checks against a real
         `avas.avas` call at every point that the reimplementation agrees;
      3. for CAS(10,6) and CAS(12,7), computes singlet (CASSCF + NEVPT2) and
         triplet (4-root 3A" SA-CASSCF + NEVPT2 on root 0) with identical
         starting orbitals, and reports convergence, roughness against a smooth
         quartic, and the equilibrium vertical energy against both references.

      Decision rule: a space is usable only if it converges everywhere in the
      window without a kink. If both are usable and their equilibrium vertical
      energies agree within 0.05 eV, take the smaller (cheaper, and already
      validated to 3.6 A in `06`). If they differ by more, the extra orbital
      matters and the larger space has to be re-validated along the whole
      dissociation cut first.

      Checked before sending, on synthetic inputs: the block assembly of the
      fixed selection, and all four verdict outcomes (agree, spread,
      unconverged, kink). The first kink test used a ratio of second
      differences, which flagged every nearly straight curve as kinked because
      the median is numerical noise. It was replaced with a deviation in meV
      from a quartic fit. Measured sensitivity: noise gives 0.2-0.4 meV, and a
      50 meV step gives 13 meV. Steps below ~40 meV would pass unflagged.

      **Result, 2026-09-15 — use CAS(10,6).** 58 min wall (estimated 6; see
      cost below).

      *Why AVAS switches.* Occupied #6's projection eigenvalue falls steadily
      through the window: 0.34, 0.31, 0.27, 0.24 at equilibrium, 0.21, then
      0.19 at 1.79 A, crossing the 0.2 threshold between 1.74 and 1.79 A. Its
      character is **O 2s 0.50-0.53 + Cl 3s 0.35-0.41**, with Cl 3p only
      0.02-0.05: an s-type lone-pair combination that is barely one of the
      requested orbitals at all. A threshold artefact, not physics. Virtual #1
      is sigma*(O-Cl), Cl 3p 0.6 + O 2p 0.4. The fixed-count selection matched
      a real `avas.avas` call at all 8 points.

      *The extra orbital is a spectator.* At equilibrium, adding it lowers the
      CASSCF energies by only 6.3 meV (singlet) and 4.9 meV (triplet): it is
      essentially doubly occupied. The CASSCF vertical energy moves by
      1.4 meV and the NEVPT2 vertical by 35 meV, below the 0.05 eV threshold.
      (NEVPT2 *totals* shift by 0.23-0.26 eV, because moving an orbital from
      core to active changes how NEVPT2 partitions the correlation, but that
      cancels in the difference.)

      | space | converged | roughness E(T) / dE, meV | dE eq NEVPT2 | vs 03 | vs 04 |
      | --- | --- | --- | --- | --- | --- |
      | (10,6) | 8/8 | 0.2 / 0.4 | 3.5501 eV | +0.102 | +0.136 |
      | (12,7) | 4/8 (triplet) | 0.3 / 0.4 | 3.5854 eV | +0.138 | +0.171 |

      CAS(10,6) converges everywhere, is smooth, and is the space `06`
      validated out to 3.6 A. **Decision: CAS(10,6) for the triplet surface.**

      **Two open problems from this run.**

- [x] **The cost was mostly my bug — no, it was the singlet.** CAS(10,6) cost 5430 CPU-s per point
      here against 624 in `06` for the same space. That would be the
      difference between ~310 h and ~36 h for a raster. `06` used PySCF's AVAS,
      which by default semicanonicalises each orbital block (Fock matrix per
      block, then `dmet_cas.symmetrize`); confirmed from `avas.py`. My fixed
      selection did not, so the "core" orbitals were arbitrary mixtures of deep
      Cl 1s/2p and valence orbitals, and CASSCF's orbital optimiser, which
      preconditions with orbital energies, had to work much harder. The four
      unconverged CAS(12,7) triplets may share that cause, though the extra
      orbital is also nearly doubly occupied, a known source of flat rotations.
      Fixed by copying AVAS's canonicalisation (tested: off-diagonal Fock
      elements drop from O(1) to 1e-15 per block, spans unchanged). Per-stage
      timings are now recorded so a slow point shows where it was slow.

      **Checked on the EVO, equilibrium only, and the hypothesis was wrong.**
      Per-stage wall time with 4 triplet roots: singlet CASSCF **103 s**,
      singlet NEVPT2 1 s, triplet SCF 2 s, triplet SA-CASSCF 24 s, CASCI +
      NEVPT2 2 s, total 131 s. With 2 roots: 105 s of 153 s. The cost is the
      closed-shell singlet CASSCF, which `06` never ran. Its active space is
      five nearly doubly occupied orbitals plus a nearly empty sigma*, which
      gives flat core/active rotations and slow convergence; it does converge,
      slowly. Canonicalisation did not change a single energy (triplet NEVPT2
      -536.61238216 vs -536.61238225 before). Whether it sped up the triplet
      cannot be told, because the first run recorded no per-stage times.

      **It does not block the raster.** The ground-state surface is CCSD(T)
      over the bound region, not CASSCF, so the triplet surface costs what
      its own stages cost here: about 28 s wall, roughly 900 CPU-s per point
      at the observed ~32x parallel factor, so ~50 h on 16 cores for 3289
      points. That is one geometry, so treat it as rough.

      `--nroots-triplet 1` crashed. PySCF's state-average wrapper assumes
      more than one root: with one, the solver returns a scalar and the
      wrapper's `einsum('i,i->')` fails. The singlet path was guarded for
      this and the triplet path was not. Fixed.

- [ ] **The vertical energy is 0.10-0.14 eV above both references.**
      CAS(10,6) SC-NEVPT2 gives 3.5501 eV (349 nm) against 3.4477 (03) and
      3.4142 (04). NEVPT2 is doing a lot of work here, lowering the CASSCF
      4.049 eV by 0.50 eV. Candidate causes, cheapest to test first:
      - state-averaging imbalance, triplet over 4 roots and singlet over 1,
        which biases the vertical energy upward. Test by varying the triplet
        root count at equilibrium: 4, 2, 1.
      - SC-NEVPT2 against 03's QD-NEVPT2; a method difference, not a bug.
      If the imbalance does not account for it, a hybrid is natural and has
      precedent in `water/06_splice.py`: the triplet is single-reference in
      the FC window (T1 0.024-0.030) where dCCSD(T) is good, and needs NEVPT2
      only beyond ~2.1 A. `06` showed the two methods agree in shape from 2.0
      to 2.2 A to 3 meV, which is exactly where a splice would blend them.

      **Imbalance ruled out.** At equilibrium, 4 -> 2 triplet roots: the
      CASSCF vertical energy drops 57 meV (4.049 -> 3.992 eV), as expected
      when the orbitals serve fewer states, but the NEVPT2 vertical energy
      *rises* by 15 meV (3.5502 -> 3.5650 eV). That is the wrong direction
      for the imbalance explanation and a tenth of the size of the gap. The
      1-root run crashed (above), but these two points already rule it out.

- [ ] **Leading explanation: how NEVPT2 is contracted.** PySCF's
      `mrpt.NEVPT`, used in `06` and `07`, is strongly contracted (SC).
      Prism, used in `03`, is fully internally contracted (its log: "Internal
      contraction: Full (= Partial)"). The two independent references agree
      with each other (FIC QD-NEVPT2 3.4477, dCCSD(T) 3.4142, 34 meV apart),
      and SC-NEVPT2 is the outlier: 3.5502 in CAS(10,6), and 3.5854 in the
      same CAS(12,7) that `03` used. SC-NEVPT2 being less accurate than the
      partially/fully contracted variants for excitation energies is known
      behaviour, but `03` also differed in other ways (QD, six roots, mixed
      spin), so this is not yet a controlled comparison.

      Controlled test: Prism SC vs FIC NEVPT2 on *identical* CAS(10,6)
      references (Prism ships `examples/nevpt/04-nevpt2_sc_vs_fic.py`), at
      equilibrium and at ±0.05 A. That answers two things at once:
      - the offset: does FIC land near 3.41-3.45 eV?
      - the slope across the FC window, which sets the band *width*. If SC
        and FIC differ only by a constant there, SC-NEVPT2 is fine for the
        surface shape, and the absolute position can come from a dCCSD(T)
        splice. If the slope differs too, the surface needs FIC, and its cost
        per point has to be measured before a raster.

      Written as `08_nevpt2_contraction.py`. Prism's own
      `examples/nevpt/04-nevpt2_sc_vs_fic.py` does the same comparison: FIC
      from Prism, SC from `pyscf.mrpt.NEVPT`, same orbitals. Design choices,
      each from something already learned in this project:
      - one set of orbitals for both states, the 4-root 3A" SA-CASSCF from
        `07`'s pipeline; the singlet is a CASCI on them, which skips the
        103 s singlet CASSCF and keeps orbital relaxation out of the SC/FIC
        difference;
      - Cs only for optimising the orbitals. Every Prism example runs without
        symmetry and its symmetry handling could not be inspected, so the
        CASCI references both NEVPT2 variants read are rebuilt on a C1 copy of
        the molecule. Three fatal checks: the AO overlaps match, the C1 triplet
        root 0 reproduces the symmetry-forced 3A" root 0 to 1e-6 Ha, and
        Prism's reference energy reproduces the CASCI it was given;
      - separate CASCI objects for Prism and PySCF, since PySCF's NEVPT
        rewrites the CI vector in place (the `06` dipole bug).
      Reports the equilibrium offset for CASCI, SC and FIC; the triplet-energy
      slope over r_eq ± 0.05 A with a 3% tolerance for "same shape"; and
      per-stage cost. Verdicts checked on synthetic data: offset explained,
      not explained, partial, and slopes differing by 10%.

      **Result, 2026-09-15: the contraction is not the explanation.** All three
      checks passed at all three geometries (same-state deviation <= 3e-12 Ha,
      Prism reference deviation 0), triplets converged, ~35 s per geometry.

      | r / A | dE CASCI | dE SC | dE FIC | FIC - SC |
      | --- | --- | --- | --- | --- |
      | 1.6391 | 3.8359 | 3.8099 | 3.7982 | −11.7 meV |
      | 1.6891 | 3.4796 | 3.4946 | 3.4827 | −12.0 meV |
      | 1.7391 | 3.1256 | 3.1764 | 3.1652 | −11.3 meV |

      - **SC and FIC differ by a constant 12 meV** and their triplet slopes
        agree to 1.1% (−5.977 vs −5.912 eV/A). PySCF's SC-NEVPT2 is fine; the
        raster does not need Prism.
      - **56 meV of the offset was the singlet's orbitals.** The same SC-NEVPT2
        gives 3.5502 eV with the singlet on its own CASSCF orbitals (`07`) and
        3.4946 eV as a CASCI on the triplet's. On common orbitals the CASCI
        vertical, 3.480 eV, is already within 15 meV of NEVPT2. So the
        −0.50 eV NEVPT2 correction in `07` was mostly repairing the orbital
        imbalance between two separately optimised states.
      - Remaining gap to the references: FIC +0.035 eV from `03`, +0.069 eV
        from `04`.
      - Cost is small: FIC 3 s and SC 2 s for both states at a geometry.

- [ ] **The offset was the wrong target; the slope is the right one.**
      Two things came out of looking at these numbers against `05` and `07`:

      *The geometry is not the minimum.* The CCSD(T) ground state has a slope of
      −0.81 eV/A at the "equilibrium" 1.6891 A, which was quoted from memory.
      Fits to `05`'s 0.2 A grid put the CCSD(T)/def2-TZVP O-Cl minimum at
      1.70-1.72 A along this cut (cubic and quartic fits disagree; the grid is
      too coarse to say better), with OH and the angle still fixed. The
      vertical energy falls at ~6.7 eV/A, so a 0.03 A geometry error alone
      moves it 0.2 eV, more than every method difference chased in `07` and
      `08`. A vertical energy at a fixed, unoptimised geometry cannot be
      compared to 0.05 eV. The pipeline does not need to: the band centre
      comes from chi_0 on the method's own ground-state surface.

      *The triplet slopes disagree between methods.* The slope at 1.6891 A
      sets the band width by the reflection principle, and unlike an offset
      it cannot be fixed by a shift or a splice:

      | method | d(E_triplet)/dr at 1.6891 A |
      | --- | --- |
      | SC-NEVPT2, `07` quartic over 8 points | −5.954 eV/A |
      | SC-NEVPT2, `08` over ±0.05 A | −5.977 eV/A |
      | FIC-NEVPT2, `08` over ±0.05 A | −5.912 eV/A |
      | UCCSD(T), `05`, cubic / quartic on the 0.2 A grid | −6.931 / −6.960 eV/A |

      NEVPT2 is internally consistent to 1%, while UCCSD(T) is **~16%
      steeper**. The vertical-energy slopes differ less (−6.74 dCCSD(T) vs
      −6.33 SC-NEVPT2, 6%), because the ground states also disagree in slope
      there (CCSD(T) −0.81, NEVPT2 −0.38 or +0.36 depending on orbitals).
      That pattern fits the two methods placing the O-Cl bond ~0.02-0.03 A
      apart, rather than one of them being badly wrong. `06` found the two
      methods' relative curves agreeing to 3 meV from 2.0 to 2.2 A, so the
      disagreement sits at the Franck-Condon region, which is the part that
      matters for the band.

      A 16% slope difference would be a ~16% difference in band width. For
      scale, water's computed band width matched measurement to 1.4%, using
      the same mix of methods: CCSD(T) ground state, NEVPT2 excited state.

- [x] **Next: pin the UCCSD(T) slopes on the same fine grid.** The 16% rests
      on a 0.2 A grid. No new code is needed; `05` on `07`'s grid:
      `python 05_triplet_scan.py --rmin 1.5391 --rmax 1.8891 --npoints 8
      --no-asymptote --csv hocl_scan_fc.csv --png hocl_scan_fc.png`.
      That gives both methods' V_S, V_T and dE slopes, and both minima, at
      identical points. If the slope gap survives, the band width becomes the
      main Phase 1 uncertainty. Measurement is the natural referee, as it was
      for water, but HOCl's triplet band sits on the tail of the 300 nm
      singlet band, which will make its width harder to extract.

      **Result, 2026-09-15: the gap is real, and it is not the bond length.**
      Identical 8-point grid (1.5391-1.8891 A), CCSD(T)/UCCSD(T) from `05`
      against CAS(10,6) SC-NEVPT2 from `07` (singlet on its own orbitals):

      | | dV_T/dr at 1.6891 | dV_S/dr | d(dE)/dr | singlet r_min | k_S, eV/A^2 | dE at r_min |
      | --- | --- | --- | --- | --- | --- | --- |
      | CCSD(T) | **−6.999** | −0.217 | −6.783 | 1.6981 A | 23.5 | 3.355 eV |
      | NEVPT2 | **−5.954** | −0.375 | −5.579 | 1.7037 A | 24.6 | 3.469 eV |

      - **Triplet slope ratio 1.18 at 1.6891 A, and 1.20 with each method at
        its own minimum.** Robust: CCSD(T) gives −6.97 to −7.06 eV/A across a
        central difference, fits of degree 3-5 and leave-one-out; NEVPT2 gives
        −5.95 to −5.98.
      - **The ground states agree.** Minima 6 mA apart, curvatures within 5%.
        So last round's explanation, the two methods placing the O-Cl bond
        0.02-0.03 A apart, is **wrong**: the disagreement is in the triplet
        surface itself. The 6% vertical-slope figure quoted there was also
        wrong, because it compared `08`'s CASCI-singlet vertical against
        dCCSD(T); on a consistent footing the vertical-slope ratio is 1.22.
      - The CCSD(T)/def2-TZVP O-Cl minimum along this cut is **1.698 A**, not
        1.6891 (still with OH and the angle fixed at the from-memory values).
      - **The UCCSD(T) triplet is slightly uneven near equilibrium.** T1 goes
        0.030, 0.035 at 1.639, then 0.018 at 1.689 A. The curve is 20x rougher
        than NEVPT2's (4.2 vs 0.2 meV from a quartic), with second differences
        zigzagging 83, 57, 73 meV where NEVPT2's fall smoothly, 71, 64, 58.
        That is a 10-15 meV wobble, likely a small shift in the ROHF reference.
        It cannot account for the gap, which adds up to 105 meV over 0.1 A.

      **Which slope is right is now the main open question for the band
      width**, and the difference is ~18%. One data point is already in hand:
      07's CAS(12,7), using only its converged points at 1.6391 and 1.7391 A,
      gives a NEVPT2 triplet slope of **−6.15 eV/A**, 3% steeper than
      CAS(10,6) and in the direction of CCSD(T). So the NEVPT2 slope is not
      converged in active space.

- [x] **Does the NEVPT2 slope move towards CCSD(T) with a better reference?**
      Two runs of `08`, no new code, three geometries each:
      - `--nroots-triplet 1`: a state-specific triplet, the like-for-like
        counterpart of state-specific UCCSD(T). Tests whether averaging the
        orbitals over four 3A" roots flattens the slope.
      - `--n-occ 6`: CAS(12,7) with `08`'s cleaner setup (canonicalised
        orbitals, singlet as a CASCI, so no singlet convergence problem).
      If the slope moves towards −7 eV/A, the reference was the limitation:
      trust CCSD(T) in the Franck-Condon window, which is the splice design
      anyway (CCSD(T) inside ~2.1 A, NEVPT2 outside). If it stays at −6, it is
      a genuine method disagreement, and needs a third, independent method
      (EOM-CCSD triplet energies, as `water/12_vertical.py` used) or the
      measured band.

      **Result, 2026-09-15.**

      *State-specific triplet (`--nroots-triplet 1`): unusable.* The CASSCF
      failed to converge at 1.639 and 1.689 A (132 s each), as the single-root
      triplet did in `04`, even with canonicalised starting orbitals. On those
      references SC-NEVPT2 broke down: vertical energies 2.22, 2.34, 1.80 eV,
      non-monotonic and 1.2-1.6 eV below FIC on the *same* reference, while
      FIC stayed sensible (3.80, 3.50, 3.18). Even the converged point at
      1.739 A had SC and FIC 1.37 eV apart, so a single-root reference is not
      safe for SC-NEVPT2 here whether or not it converges. `08`'s verdict
      accepted these points and reported a 173% slope difference; it now
      excludes unconverged points and any point where SC and FIC differ by
      more than 0.2 eV. Replayed on this run it reports "no usable geometry".

      *CAS(12,7): clean, and the slope moves.* All three triplets converged
      this time; `07`'s CAS(12,7) failures at these geometries came before the
      canonicalisation fix. Consistent with that fix mattering for this space,
      though not proof. Triplet slope **SC −6.152, FIC −6.185 eV/A** (FIC/SC
      1.005), matching the −6.15 estimated from `07`'s converged points
      exactly.

      | method | triplet slope at 1.6891 A |
      | --- | --- |
      | SC-NEVPT2 CAS(10,6) | −5.95 eV/A |
      | SC-NEVPT2 CAS(12,7) | −6.15 eV/A |
      | UCCSD(T) | −7.00 eV/A |

      The gap narrows from 18% to ~13% after adding one orbital, which is
      suggestive of active-space incompleteness in NEVPT2, but two points are
      not a trend.

- [x] **Two runs to settle the slope.**
      - *Full-valence NEVPT2, CAS(14,9)*: every valence orbital including
        O 2s, Cl 3s and sigma/sigma*(O-H). The natural end of the active-space
        series; `08` supports it without new code:
        `python 08_nevpt2_contraction.py --avas-aos "Cl 3s" "Cl 3p" "O 2s"
        "O 2p" "H 1s" --n-occ 7 --n-vir 2 --csv hocl_nevpt2_fv.csv`
      - *EOM-CCSD triplet*, `09_eom_triplet.py`, on `07`'s 8-point grid. It
        reaches the triplet from the closed-shell CCSD ground state (T1(S)
        ~0.008), so there is no ROHF triplet reference to wobble and no active
        space to choose. That removes the suspect on each side. It is still
        coupled cluster, so agreement with UCCSD(T) is less independent than
        agreement with NEVPT2 would be. Reports the slope of
        E_CCSD(T) + omega_T against UCCSD(T) and both NEVPT2 spaces read from
        their CSVs, and checks the lowest-triplet assignment through the gap
        to the next root. Tested on synthetic EOM data with the real
        comparison CSVs: it reproduces −6.999, −5.954 and −6.152, and gives
        the right verdict for EOM siding with UCCSD(T), with NEVPT2, and
        landing between them.

      **Result, 2026-09-15: settled. Coupled cluster has the right slope, and
      NEVPT2 converges to it as the active space grows.**

      | method | triplet slope at 1.6891 A | gap to UCCSD(T) |
      | --- | --- | --- |
      | SC-NEVPT2 CAS(10,6) | −5.954 eV/A | 14.9% |
      | SC-NEVPT2 CAS(12,7) | −6.152 eV/A | 12.1% |
      | SC-NEVPT2 CAS(14,9), full valence | −6.610 eV/A | 5.6% |
      | FIC-NEVPT2 CAS(14,9) | −6.339 eV/A | 9.4% |
      | EOM-CCSD triplet on the CCSD(T) ground state | −6.965 eV/A | 0.5% |
      | UCCSD(T), state-specific | −6.999 eV/A | — |

      (Earlier entries quoted the gap as 18%, i.e. as a fraction of the NEVPT2
      slope. Here it is a fraction of the UCCSD(T) slope, which is the
      referee.)

      - **Two routes to the triplet agree**: UCCSD(T) from an ROHF triplet
        reference, and EOM-CCSD from the closed-shell ground state, within
        0.5%. The vertical-energy slopes agree too: EOM −6.748, dCCSD(T)
        −6.783 eV/A.
      - **NEVPT2 converges towards them with the active space**, 15% → 12% →
        6%, and its vertical-energy slope at full valence (SC −6.673, FIC
        −6.604) is within 1-2.5% of coupled cluster. Independent corroboration:
        the earlier NEVPT2 slope was active-space incompleteness, not a
        coupled-cluster artefact.
      - **The ground-state correlation level matters for V_T's slope.** EOM on
        the CCSD ground state gives −6.670; on the CCSD(T) ground state
        −6.965. The (T) correction accounts for 4%.
      - **EOM-CCSD gives the cleanest FC-region triplet.** Roughness against a
        quartic is 0.1 meV, against UCCSD(T)'s 4.2 meV, and its second
        differences fall smoothly (75, 71, 66, 60, 53, 46 meV). EOM V_T minus
        UCCSD(T) is constant to ±2 meV across the window except at 1.6391 A,
        where UCCSD(T) is 8 meV off: the point where its T1 jumped to 0.035.
        So the UCCSD(T) wobble is a single glitch, located.
      - Lowest-triplet assignment is safe: 0.94-1.06 eV to the next EOM
        triplet throughout.
      - EOM-CCSD vertical energy at 1.6891 A: **3.432 eV**, between `03`
        (3.448) and `04` (3.414). Full-valence NEVPT2: SC 3.506, FIC 3.469.
      - In full valence SC and FIC slopes differ by 4%, and SC is the closer
        to coupled cluster. `08`'s verdict said "the surface needs FIC" there;
        that rule compared the two contractions only with each other, and the
        wording now asks for an external referee instead.
      - Cost: EOM-CCSD ~18 s wall per point for the singlet CCSD(T) and the
        triplet together. Full-valence NEVPT2 ~55 s per geometry (orbitals
        32 s, FIC 13 s, SC 3 s).

- [ ] **Surface recipe this points to (proposal, not yet validated):**
      - V_S: CCSD(T) over the bound region, as planned.
      - V_T near equilibrium: **E_CCSD(T) + omega(EOM-CCSD triplet)**. It
        comes out of the same calculation as V_S at ~570 CPU-s per point, has
        the coupled-cluster slope, and has neither the ROHF wobble nor an
        active space to choose.
      - V_T at longer range: EOM-CCSD from a closed-shell reference degrades
        as the ground state turns diradical (T1(S) 0.018 at 2.4 A, above 0.02
        by 2.6 A). Beyond that, multireference NEVPT2, which `06` showed smooth
        and which also gives the neighbouring 3A" states. Full valence is the
        better candidate (6% from coupled cluster in the FC window, against
        15% for CAS(10,6)), but it is untested at dissociation, where
        CAS(10,6) is the one `06` validated to 3.6 A.
      - Splice them where both are valid, water's `06_splice.py` pattern.

- [x] **Next: find the splice window.** Both scripts already accept a grid,
      so no new code, on the same outward grid r_eq + 0.1k A, k = 1..11
      (1.79-2.79 A):
      - `09 --step 0.1 --kmin 1 --kmax 11`: where does EOM-CCSD stop
        tracking? Watch T1(S) and the smoothness of V_T.
      - `08` full valence, same grid: does CAS(14,9) stay converged and smooth
        at dissociation, and where do its shape and EOM's agree?
      The built-in verdicts of both scripts assume the grid includes
      equilibrium, so they are not meaningful for these runs; the CSVs are
      what gets compared. `08`'s same-state check will also stop any point
      where the lowest C1 triplet stops being 3A", which marks where the
      3A'/3A" near-degeneracy begins.

      **Result, 2026-09-15.**

      *EOM-CCSD is reliable through 2.29 A and breaks at 2.39 A.*

      | r / A | T1(S) | omega0 / eV | gap to next triplet | 2nd diff of V_T, meV |
      | --- | --- | --- | --- | --- |
      | 1.89 | 0.009 | 2.198 | 0.936 | 159 |
      | 1.99 | 0.010 | 1.710 | 0.853 | 109 |
      | 2.09 | 0.012 | 1.312 | 0.740 | 68 |
      | 2.19 | 0.013 | 0.995 | 0.577 | 36 |
      | 2.29 | 0.015 | 0.747 | 0.332 | 13 |
      | 2.39 | 0.017 | 0.555 | **0.055** | **−206** |
      | 2.49 | 0.020 | 0.205 | 0.204 | −33 |
      | 2.59 | 0.022 | **−0.132** | 0.431 | 18 |

      Up to 2.29 A, V_T is smooth and passes through the shallow minimum near
      2.2 A that `05` and `06` also found. At 2.39 A the gap to the next EOM
      triplet collapses to 55 meV and V_T kinks. Beyond it, root 0 falls by
      ~0.2 eV per 0.1 A and drops below the singlet from 2.59 A, where T1(S)
      also passes 0.02. **A second triplet crosses the lowest one at about
      2.35-2.4 A.** `06`'s 3A"-only NEVPT2 has its root 1-root 0 gap at
      0.58 eV at 2.4 A and no such state, so the crossing state is most
      likely **3A'**, which Cs allows to cross. That identification is
      inference, not a calculation.

      *Full-valence NEVPT2 agrees where it passes, and is not robust.* Only
      3 of 11 points passed `08`'s same-state check:

      | r / A | SC vertical | FIC vertical | EOM omega0 |
      | --- | --- | --- | --- |
      | 1.79 | 2.850 | 2.821 | 2.779 |
      | 1.89 | 2.264 | 2.241 | 2.198 |
      | 2.29 | 0.748 | 0.748 | **0.747** |

      - From 2.39 A outward the C1 triplet lies 6-15 mHa (0.16-0.41 eV) below
        the symmetry-forced 3A" root: consistent with the same crossing
        state.
      - At 1.99-2.19 A it lies **24-36 mHa (0.65-0.97 eV) below**, and that is
        not physics. EOM-CCSD's lowest triplet there continues smoothly from
        the equilibrium region, where it matched UCCSD(T) to ±2 meV, with the
        next triplet 0.6-0.85 eV above, not below. The likely reading is that
        the full-valence SA-CASSCF lands on a different orbital solution at
        those points; 2.29 A passing between failures fits that. Full valence
        is not usable for a raster as it stands.

      Both built-in verdicts are meaningless for these off-equilibrium grids,
      as expected (09 reports a −9.6 eV/A "slope" by extrapolating a fit).

- [ ] **Scope decision the result forces.** The triplet surfaces are sound out
      to ~2.3 A with CCSD(T) + EOM-CCSD, and the region beyond holds a
      probable 3A'/3A" crossing near 2.4 A that needs a multistate treatment,
      plus SOC wherever the triplets are within the Cl 882 cm-1 splitting.
      - **For the absorption band sigma(lambda, T)**, which is the HOX
        contribution, the surface beyond ~2.3 A may not matter. The
        autocorrelation decays as the packet leaves the Franck-Condon region.
        With ~1 eV of kinetic energy and mu(OH-Cl) = 11.4 amu, the packet
        needs of order 10-20 fs to reach 2.3 A, by which point S(t) should
        already be gone. An absorbing potential from ~2.3 A would then remove
        the flux before the crossing, and CCSD(T) + EOM-CCSD is enough. This
        is an estimate, and has to be checked by varying where the absorber
        starts, as `water/09_propagate.py`'s R_abs allows.
      - **For product branching**, Cl(2P3/2) vs Cl(2P1/2), the crossing
        region is exactly where it is decided. That needs the multistate
        treatment with SOC, and it is optional scope, not a blocker for the
        band.

      **Decision, 2026-09-15: band first.** The crossing is left for product
      branching.

- [x] **1D scoping model, `10_band_1d.py`.** Built from data already in the
      repo, NumPy only, 13 s on the laptop. V_S from CCSD(T); V_T from
      E_CCSD(T) + omega(EOM-CCSD) to 2.29 A, anchored at 1.40 A by UCCSD(T)
      (moved onto EOM's scale, +16 meV); three tails beyond 2.29 A (06's
      3A" NEVPT2 shifted, flat, linear). Split-operator propagation and
      `water/09_propagate.py`'s cross-section formula. Condon.

      **Result: the band depends on V_T only out to ~2 A.** Positive controls
      pass, so the test can detect sensitivity:

      | change | band change |
      | --- | --- |
      | absorber from 1.6 A (inside chi_0) | **13.2%** |
      | absorber from 1.7 A | 1.5% |
      | absorber from 1.8 A | 0.05% |
      | absorber from 1.9 to 2.6 A | 0.00% |
      | V_T flattened beyond 1.78 A | **99%** |
      | V_T flattened beyond 1.85 A | 11.3% |
      | V_T flattened beyond 1.95 A | 0.7% |
      | V_T flattened beyond 2.10 A | 0.1% |
      | any tail, absorber strength or ramp length, from 2.3 A out | < 0.02% |

      CCSD(T) + EOM-CCSD, trusted to 2.29 A, covers the part of the surface
      the band depends on with ~0.3 A to spare. The 3A'/3A" crossing near
      2.4 A does not reach the band in this model.

      Orientation, not results:
      - 1D ground well: minimum 1.6986 A, O-Cl stretch **736 cm-1** (v=1<-0)
        against ~725 cm-1 measured (quoted from memory, verify). A useful
        check on the ground surface and the reduced mass.
      - Band peak **373 nm**, FWHM 0.72 eV, against the measured ~380 nm.
      - Peak sigma 2.5e-22 cm2 against ~4e-21 measured: **16x low**, in line
        with the earlier estimate that `03`'s f is 10-25x too small. Absolute
        intensity is the weakest number here.
      - **Temperature dependence, O-Cl stretch hot bands only:**
        sigma(298 K)/sigma(200 K) = 0.98 at 380 nm, 1.01 at 420 nm, **1.12 at
        450 nm, 1.42 at 480 nm**. Negligible at the peak, and growing into
        the red tail. That is the kind of effect HOX is after for HOBr at
        440-500 nm, but 1D omits the bend (~1240 cm-1) and OH hot bands
        entirely, so the size must come from 3D.

      Three mistakes caught on the way, all in the test rather than the
      physics:
      - The first absorber ramped as the cube over the whole remaining grid,
        so an absorber "from 2.0 A" was at 1e-4 of its strength 0.1 A later
        and really absorbed from ~2.6 A. Replaced with a fixed 0.4 A ramp.
      - The first "positive control", an absorber from 1.8 A, was not one: it
        sits outside chi_0, so it changed the band by 0.05%, and the verdict
        correctly refused to conclude. Real controls are an absorber inside
        chi_0 (1.6 A) and cutting the surface on the Franck-Condon slope
        (1.78 A).
      - A plot scaled each chi by max() rather than |chi|.max(); eigenvector
        signs are arbitrary, so a negative chi got divided by ~1e-10 tail
        noise and showed a spurious jag. The wavefunctions were checked
        directly (amplitude beyond 2.25 A <= 4e-10 of peak), and no number was
        affected.

- [ ] **Phase 1 proper, for the band:**
      - [x] **First, check EOM-CCSD across the other two coordinates** in the
            Franck-Condon region, the O-H stretch and the bend: T1(S), the gap
            to the next triplet, and smoothness. The 1D model only
            established the O-Cl direction. Cheap, and a 3D raster depends on
            it.

            `11_eom_coordinates.py`: O-H stretch 0.75-1.25 A, bend 75-135 deg,
            and 8 corner points combining all three coordinates, in Cs. Every
            EOM root is labelled by the irrep of its dominant single
            excitation, so the surface value is the lowest **3A"**, not merely
            the lowest triplet: along the bend a 3A' could drop below with a
            comfortable gap and 09's rule would not notice. If labelling
            fails, it falls back to 09's lowest-root rule and says so.
            Verdict gates: convergence, T1(S) < 0.02, next 3A" > 0.3 eV above,
            V_T within 10 meV of a quartic along each cut, and reproducing
            09's 3.4320 eV at the shared geometry. A 3A' below the 3A" is
            reported but not disqualifying in Cs without SOC. Verdict checked
            on synthetic cuts: clean, a 30 meV bend step (flagged at 11.9 meV,
            so steps under ~25 meV would pass), T1 reaching 0.023, and a 3A'
            dropping below at wide angles.

            **Result, 2026-09-21: cleared.** 34 points, ~25 s each, all
            converged.

            | cut | max T1(S) | min gap to next 3A" | roughness omega / V_T |
            | --- | --- | --- | --- |
            | O-H stretch, 0.75-1.25 A | 0.0091 | 2.06 eV | 0.1 / 1.4 meV |
            | bend, 75-135 deg | 0.0083 | 3.52 eV | 0.1 / 0.2 meV |
            | 8 corners | 0.0100 | 2.38 eV | - |

            The lowest triplet is **3A" at every one of the 34 geometries**,
            so no 3A' drops below anywhere in the Franck-Condon region; the
            gap to the nearest root of any symmetry narrows to 0.67 eV only at
            135 deg. T1(S) stays at or below 0.010 everywhere, half the
            threshold. omega at the shared geometry reproduces 09's 3.4320 eV
            exactly. Single excitations carry 94-95% of each root.

            The roughness check needed fixing, not the surface. At degree 4 it
            reported 9.5 meV on the O-H cut against a 10 meV threshold -- a
            near-miss that was a fitting artefact: that cut spans 2.06 eV of a
            Morse-shaped curve, and its second differences are perfectly
            smooth (384, 275, 198, 142, 101, 70, 47, 29, 16 meV, same sign,
            monotone). Degree 5 gives 1.4 meV, degree 6 gives 0.2. Calibrated
            on a synthetic Morse curve on this grid: degree 5 leaves 0.4 meV
            when smooth and 10.3 meV with a 30 meV step injected, so the check
            is now degree 5 against a 5 meV threshold, which catches steps of
            ~15 meV and up while accepting real curvature -- strictly better
            than the quartic, which both flagged smooth curvature and would
            have missed that step.
      - [ ] 3D raster of CCSD(T) + EOM-CCSD over r(O-Cl) <= ~2.4 A and the
            bound range of r(O-H) and the angle; one calculation gives both
            surfaces. At ~570 CPU-s per point, a grid of roughly
            21 x 7 x 11 points is ~16 h on 16 cores (rough).

            `12_pes_raster.py`. Grid r(O-Cl) 1.40-2.40 A, r(O-H) 0.80-1.25 A,
            angle 75-135 deg, steps 0.05 A and 5 deg: **2730 points, ~432
            CPU-hours, ~14 h on 30 workers** at the measured 570 CPU-s/point.
            `--rocl-max 2.30` trims it to 2470 points and ~13 h; `10` showed
            the band needs the surface only to ~2.0-2.3 A, and the extra
            0.1 A is margin for the Jacobi transform.

            Built on `water/04_pes_grid.py`: environment variables set before
            numpy and pyscf load, OpenMP pinning cleared (water's first
            gotcha), one single-threaded process per geometry, per-worker CPU
            affinity, and rows appended as they finish so an interrupt costs
            nothing. `--pilot N` runs N points spread over the grid plus
            equilibrium **into the same CSV**, so the pilot's points count
            towards the full raster instead of being thrown away, and it
            reports the measured CPU-s/point for a corrected estimate.

            Every root is labelled A' or A" as in `11`, so what is stored is
            the lowest **3A"**, not merely the lowest triplet; the labelling
            is duplicated rather than imported, since a worker process should
            not depend on another script's import side effects. Status per
            row separates ok / warn (T1 >= 0.02, unconverged EOM) / fail.

            Tested locally without pyscf on a fake calculation: pilot then
            full then a third run gives 6 + 66 + 0 points, 72 unique rows, no
            duplicates, header intact. That test also showed the pool needs
            an explicit fork context -- fork is the default on the EVO's
            Python 3.12, but 3.14 defaults to forkserver, where each worker
            re-imports the module and re-runs the thread-limiting environment
            setup this design depends on.

            **Pilot, 2026-09-22: passed, and 5x cheaper than assumed.**
            60 points in 3.7 min, 57 ok, 3 warned, 0 failed.
            **111 CPU-s per point, not 570**, so the full 2730-point raster is
            **84 CPU-hours, ~2.8 h on 30 workers**. The 570 came from
            multiplying 09's wall time by its parallel factor, which counts
            threading overhead as work; single-threaded workers avoid it.
            Another instance of water's finding that the job-array pattern
            beats threading at this problem size.

            | r(O-Cl) band | points | max T1(S) |
            | --- | --- | --- |
            | 1.40-1.80 A | 24 | 0.0102 |
            | 1.80-2.10 A | 16 | 0.0123 |
            | 2.10-2.30 A | 11 | 0.0165 |
            | 2.30-2.45 A | 9 | **0.0251** |

            All three T1 warnings sit at r(O-Cl) >= 2.35 A *and* r(O-H) >=
            1.10 A: the far corner, beyond the ~2.3 A the band needs and
            inside where the absorber will be. Within the band-relevant
            region the worst T1 is 0.0165.

            **The symmetry labelling earned its place.** At four points the
            lowest triplet is 3A', not 3A": (2.30, 0.80, 135), (2.35, 0.85,
            125), (2.40, 0.90, 115) and (2.40, 1.25, 135). That is the
            crossing from `06`/`09` arriving, a little earlier at wide angles.
            The raster stored the lowest 3A" at each, as designed; without the
            labelling a 3A' energy would have gone into the surface unnoticed.
            Those points now carry their own `warn:Ap_below` flag so the full
            run counts them, and single-excitation weight falls to 0.875 at
            2.40 A, consistent with the same picture.

            Surface construction should drop or replace the warned corner
            points rather than interpolate through them; they lie inside the
            absorber, so nothing the band depends on is lost.
      - [ ] Jacobi transform, relaxation and propagation from `water/`, with
            an absorber from ~2.3 A along the dissociation coordinate, and
            the absorber-position check repeated in 3D.
      - [ ] Transition dipole: Condon first; then mu_SOC over the
            Franck-Condon window from `03`'s machinery, with
            `water/10_mu_sensitivity.py`'s test of how much its variation
            matters.
      - [ ] sigma(lambda, T) with bend and O-H hot bands included, against
            the measured 380 nm band.

      **Three bugs in `06`, found in these runs and fixed:**
      - *Dipoles after NEVPT2 were wrong.* PySCF's `NEVPT` copies the CASCI
        object's attributes by reference and its kernel replaces
        `ci[root]` with a vector in rotated natural orbitals, while the CASCI
        keeps the old orbitals. Dipoles computed afterwards rose linearly to
        6 D at 3.6 A. **The dipole columns in the three NEVPT2 CSVs
        (`_cas2`, `_cas3`, `_cas4`) are invalid.** Energies, weights and
        `<S^2>` are unaffected. Dipoles are now computed before any NEVPT2.
      - *A gap minimum on the grid edge was reported as "inconclusive" with a
        suggestion to rescan around it.* That only moves the edge, and it
        cost two follow-up runs chasing it from 3.2 to 3.6 A. An edge minimum
        is now reported as monotonic convergence, not a crossing.
      - *The 1.7 A point, in a different active space, was included in the
        analysis* and supplied the starting energy for Landau-Zener, so the
        P = 0.39 and 0.22 in the first two logs mean nothing. Analysis now
        uses only the points sharing the most common active space.
      Replaying the real CSVs through the fixed verdict gives "no avoided
      crossing, roots 0-2 converging" for both runs, and the synthetic weak
      crossing is still detected.
- [ ] Cost so far: 830 CPU-s/point averaged over the scan (worse than the
      594 at equilibrium, as expected away from it) = **47 h for 3289 points
      on 16 cores**, against 330 h for the full multi-state route.
- [ ] **Force Cs for the raster.** "Lowest triplet" is only well defined if
      a 3A' cannot overtake a 3A" along the dissociation coordinate; within
      the A" irrep it is the ground state of its block. water/README.md's
      "force the point group, don't detect it" stops being tidiness here — a
      state-specific solver with symmetry off would follow the lower adiabat
      through a crossing and produce a kinked surface with no error message.
- [ ] **Sample mu_SOC coarsely.** `water/10_mu_sensitivity.py` measured what
      freezing mu costs: peak moved 4 nm, band narrowed 15%. So mu matters but
      does not need the full raster grid — a coarse FC-window scan plus
      interpolation is the likely answer.

Still-open fallbacks if the state-specific route fails:

- [ ] **Geometry-independent SOC.** Compute the coupling once near equilibrium
      and apply it across the surface — defensible, since it is dominated by
      the halogen core and varies weakly with bond length.
- [ ] **Orbital reuse between geometries**, with water's warning about seeding
      active spaces along a scan (negative excitation energies, 66 eV roots).
- [ ] **SOMF-QDNEVPT2**, a smaller active space, or a smaller basis.

### Phase 0.5 — validate the SOC toolchain (cheap, do with Phase 0)

Scripts are written; run them on the EVO in order, tee'ing into `logs/`.

- [ ] `00_setup_soc.sh` — install Prism + socutils into the `qc` env. socutils
      needs a `make` at its repo root against BLAS/LAPACK, so run it inside
      the activated env.
- [ ] `01_soc_probe.py` — diagnostic, not a calculation. Dumps the Prism API
      surface and prints the shipped `examples/soc/*.py` verbatim, which is
      the real documentation. Every probe is independently guarded so one
      missing piece does not hide the rest.
- [ ] `02_soc_atoms.py` — halogen fine structure (2P_1/2 − 2P_3/2) against
      NIST. Chosen over reproducing a paper table because the answer is known
      to six figures, it is cheap, and the atomic SOC on the halogen is
      precisely what lends the ã 3A" band its intensity. Three checks in one
      run: the 4+2 degeneracy pattern, the *inverted* multiplet ordering, and
      the magnitude.
- [ ] `03_soc_hocl_vertical.py` — **not yet written.** Molecular singlet–triplet
      with an SOC-borrowed oscillator strength at the HOCl equilibrium
      geometry, plus the per-point timing that sets the raster cost. Deliberately
      deferred until `01` reveals the real API, rather than guessing twice.

The one piece of `02` written without having seen the API is `run_soc()`,
flagged in place; everything else in it is API-independent.

**Kill criteria.** If any of the above turns out to be a full 3D quantum
wavepacket treatment on an ab initio triplet surface, stop and pivot to HOI
(see Phase 4). If a temperature-dependent HOX cross section exists anywhere,
the second half of the contribution is gone and the project is much weaker.

---

## Phase 1 — HOCl, validating the SOC extension

HOCl has a **measured** triplet band at 380 nm, σ ≈ 4 × 10⁻²¹ cm², tail to
480 nm. Light atom, small SOC, known answer. Build here, not on HOBr.

- [ ] Port the `water/` pipeline to HOCl: Jacobi coordinates with Cl + OH
      arrangement, same raster/splice/relax/propagate chain. The geometry
      transform in `water/07_jacobi.py` generalises directly; only masses
      and the asymptotic fragment change.
- [ ] Ground-state surface: CCSD(T)/aug-cc-pVTZ over the bound region, as in
      `water/13_gs_well.py`. Validate against known HOCl fundamentals.
- [ ] Add spin-orbit coupling. **Leading route: Prism + socutils**, which keeps
      everything in the PySCF toolchain (see "SOC route" below). Fallbacks if
      it does not work out:
      - OpenMolcas RASSI-SO (atomic mean-field SOC over CASSCF/RASSCF)
      - MOLPRO state-interaction SOC over MRCI
- [ ] Build the a 3A'' surface and the SOC-borrowed transition dipole
      surface.
- [ ] Propagate. Compare against the measured 380 nm band.

**Success threshold:** band peak within ~15 nm and integrated cross section
within a factor ~2 of experiment. If this fails, the SOC-borrowed-dipole
model is inadequate and a full 4-state (X, A, B, a) coupled propagation is
needed before touching HOBr.

---

## Phase 2 — HOBr, the contribution

- [ ] Repeat Phase 1 for HOBr. Scalar-relativistic treatment required
      (ECP or x2c/DKH; aug-cc-pVnZ-PP or ANO-RCC basis).
- [ ] Ground-state surface: consider using **Peterson's global MRCI PES**
      (*J. Chem. Phys.* **113**, 4598, 2000, doi:10.1063/1.1288913) rather
      than rebuilding, if obtainable.
- [ ] Compute the a 3A'' band profile. Compare against Ingham et al. (1998):
      λmax 457 nm, σ = 2.3 ± 0.2 × 10⁻²⁰ cm².
- [ ] **Temperature dependence.** Relax the lowest several vibrational states
      (`water/08_relax.py` works unchanged), Boltzmann-weight, propagate each,
      sum. HOBr fundamentals: ν3 (O-Br stretch) 620.23, ν2 (bend) 1162.57,
      ν1 (OH stretch) 3614.90 cm⁻¹ (NIST WebBook). At 298 K that gives ~5.0%
      population in v3=1 and ~0.36% in v2=1; less at 220 K.
- [ ] Report σ(λ, T) over 200-300 K.

**Impact threshold:** if σ in the 440-500 nm region changes by ≥10-15%
between 220 K and 298 K, it materially affects polar J(HOBr) and the result
matters. If the change is <5%, the honest conclusion is that the temperature
dependence is negligible — still publishable, still useful, less interesting.

---

## Phase 3 — atmospheric consequence

Turns a spectroscopy result into an atmospheric one.

- [ ] Feed σ(λ, T) into a TUV or box-model J-value calculation, as Ingham
      et al. did. Quantify ΔJ(HOBr) vs the current recommendation.
- [ ] Optionally propagate through a polar-ODE box model to show sensitivity
      of bromine-explosion timing / ozone-depletion onset.
- [ ] No one has quantified the fraction of J(HOBr) uncertainty attributable
      specifically to the triplet band. That is a publishable sub-result on
      its own.

---

## Phase 4 — HOI (optional, higher novelty)

Marine iodine chemistry and new particle formation. Per Minaev, the triplet
**dominates** the visible absorption in HOI, so SOC is not a perturbation.
Highest novelty of the HOX set, largest relativistic burden. Only after HOBr.

---

## Known risks

- **SOC machinery: route identified, cost unknown.** Prism + socutils gives
  state-interaction SOC over QD-NEVPT2 with oscillator strengths, inside the
  PySCF toolchain (see "SOC route" above), so the single-toolchain property
  looks safe. The unknown has moved from *can it be done* to *what does it
  cost per geometry* — SO-QDNEVPT2 over several roots at thousands of raster
  points is a different proposition from the SC-NEVPT2 water used. Phase 0.5
  answers this before anything else is built.
- **The triplet surface may need more states.** If the a 3A'' band borrows
  intensity from several singlets, a two-state model will not suffice.
- **HOBr ground-state PES quality.** CAS(8,6)-equivalent active spaces were
  adequate for water; bromine may need more.
- **The gap may close underneath us.** Monitor JCP, JPCA, PCCP, ACP.

## Carry over from `water/`

Read `water/README.md` Gotchas before writing propagation code. The ones
that will bite again: absorbing boundaries needed in *every* coordinate the
packet can leave along; `fci.addons.fix_spin_` when using CASSCF (and note
that for a triplet target the spin constraint changes); AVAS rebuilt per
geometry rather than seeded; symmetric hole-filling; forcing the point group
rather than detecting it; and the propagator performance traps (cached
exponentials, threaded FFT, OpenMP pinning vs multiprocessing).
