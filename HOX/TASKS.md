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
