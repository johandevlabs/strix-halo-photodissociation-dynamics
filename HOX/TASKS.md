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
| V(a 3A") | state-specific, spin=2 | full 3D raster | to be measured |
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
      meant to undercut. It was also redundant — `03`'s QD-NEVPT2+SOC and
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

- [ ] **Rerun with `--pin-irrep`.** Pins each spin's equilibrium occupation
      along the whole scan, and at every point also runs an unconstrained SCF
      to report whether aufbau would have left that occupation. One run
      separates the two explanations:
      - aufbau switches near 2.2-2.8 A **and** the pinned curve is smooth: an
        A'/A" occupation flip, fixed for free by `irrep_nelec`.
      - no switch, or still kinked when pinned: a same-symmetry problem (a
        3A" avoided crossing, or a different local SCF solution) that pinning
        cannot reach.
- [ ] If pinning does not fix it, the region genuinely needs a multireference
      treatment: either patch 2.2-2.8 A with CASSCF/NEVPT2 and splice, or
      accept the multireference route for the triplet surface as a whole.
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
