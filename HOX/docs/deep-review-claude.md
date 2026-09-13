# Literature Review: Photodissociation Dynamics of HOCl, HOBr, HOI — Is There a Genuine Gap for 3D Quantum Wavepacket Cross Sections on the Triplet (a 3A'') Surface?

## TL;DR
- **The specific gap is GENUINELY UNFILLED for the triplet band.** No one has built a full 3D ab initio a 3A'' potential energy surface for HOBr (or HOCl/HOI) and run quantum wavepacket dynamics to produce the visible triplet-band absorption cross section. Existing triplet work stops at vertical excitation / MCSCF response-theory oscillator strengths (Francisco 1996; Minaev 1999/2004); existing 3D wavepacket dynamics (Balint-Kurti/Szalay 1999; Zhang & Jiang 2020) cover only the two/three lowest SINGLET states.
- **Temperature dependence is a second, wide-open sub-gap.** JPL and IUPAC recommended HOBr/HOCl/HOI cross sections are all single-temperature (~295 K) measurements; no temperature dependence has ever been measured or computed for any hypohalous-acid band, least of all the actinically critical 457 nm triplet band.
- **The atmospheric leverage is real but modest in magnitude:** per Ingham et al. (1998), the visible band "is responsible for up to 50% of the total photolysis rate at high zenith angles close to the surface and at least 25% at all other altitudes and zenith angles down to 40°," and HOBr photolysis is the light-driven step competing with heterogeneous recycling in the bromine explosion / polar ozone depletion cycle. A first-principles temperature-dependent triplet cross section is a legitimate niche contribution, though it requires adding spin-orbit coupling to the user's pipeline.

## Key Findings

1. **Singlet 3D wavepacket dynamics of HOBr is done (twice).** Balint-Kurti and co-workers produced ab initio PESs and time-dependent wavepacket cross sections for the two lowest singlet bands (1 1A'', 2 1A'); Zhang & Jiang (2020) produced the first full 3D MRCI PESs for the three lowest singlets with real-wavepacket state-to-state dynamics. Neither treats the triplet.

2. **Triplet HOBr work is entirely electronic-structure/response-theory, not dynamics.** Francisco et al. (1996) predicted the ~477 nm a 3A'' absorption from vertical excitation energies/oscillator strengths. Minaev et al. (1999) and the HOCl/HOBr/HOI paper computed S-T transition probabilities with the full Breit-Pauli SOC operator — but these are NOT wavepacket dynamics and do NOT produce a properly dynamics-broadened band profile from a global surface.

3. **The only "dynamics" on the triplet is semiclassical.** The nonadiabatic surface-hopping / nuclear-ensemble TDDFT study (2016) is category (c), not (d): it uses on-the-fly TDDFT M06-2X and a Wigner nuclear ensemble, not a converged 3D ab initio surface with quantum wavepacket propagation.

4. **Temperature dependence: not measured, not computed, anywhere in the HOX family.** JPL and IUPAC both recommend cross sections at ~295 K only (primary source Ingham et al. 1998 for HOBr, at 295 K; Bauer et al. 1998 for HOI, room temperature). No temperature parameterization exists for any band.

5. **Hot-band contributions plausibly matter and have never been examined.** HOBr fundamentals are low — the measured gas-phase values are ν3 (O-Br stretch) = 620.23 cm⁻¹ and ν2 (bend) = 1162.57 cm⁻¹ (with ν1 OH stretch = 3614.90 cm⁻¹), per the NIST Chemistry WebBook (compiler M.E. Jacox), citing McRae & Cohen, J. Mol. Spectrosc. 139, 369 (1990) for the bend and Cohen, McRae et al., J. Mol. Spectrosc. 173, 55 (1995) for ν1/ν3. Using ν3 = 620.23 cm⁻¹ and kT(298 K) ≈ 207 cm⁻¹, the first O-Br stretch level lies ~3.0 kT above ground, giving a Boltzmann population of exp(−620.23/207) ≈ 5.0%; the bend at 1162.57 cm⁻¹ gives exp(−1162.57/207) ≈ 0.36%. Because the triplet band sits on the red tail near the actinic-flux peak, hot bands preferentially enhance the long-wavelength absorption that dominates J(HOBr) — a temperature-dependent wavepacket calculation (initial states weighted by Boltzmann population) is the natural tool and has never been applied here.

## Details

### Question 1 — Has anyone done 3D quantum wavepacket dynamics on an ab initio triplet HOX surface?

**No.** Categorizing the literature by the user's own taxonomy:

- **(a) Vertical excitation / oscillator strengths:** Francisco, Hand, Williams, "Ab Initio Study of the Electronic Spectrum of HOBr," J. Phys. Chem. 1996, 100(22), 9250–9253, doi:10.1021/jp9529782. Verbatim: "The calculations predict another absorption with a maximum at about 477 nm due to the ã ³A″ state," and the 350 and 280 nm bands are assigned to the Ã ¹A″ and B̃ ¹A′ states. Vertical excitation energies and oscillator strengths only.

- **(b) Response-theory / S-T transition probabilities:** Minaev, "Ab initio study of the singlet-triplet transitions in hypobromous acid," J. Mol. Struct. THEOCHEM 1999, doi pii S0166128099000846 — MCSCF with linear/quadratic response and the complete Breit-Pauli SOC operator; a 3A'' ← X 1A' responsible for the 440-650 nm band; transition polarized along O-Br, consistent with Barnes et al. OH Doppler profiles. Companion: "The Singlet-Triplet Absorption and Photodissociation of the HOCl, HOBr, and HOI Molecules Calculated by the MCSCF Quadratic Response Method," J. Phys. Chem. A 1999, doi:10.1021/jp990203d — reports S-T bands for the whole HOX series at **380, 450 and 460 nm for X = Cl, Br, I, with peak cross sections of about 4, 15 and 24 × 10⁻²¹ cm² respectively**. For HOI the second S-T transition (3A' ← X 1A') even dominates the visible absorption. These are transition-probability calculations, not wavepacket dynamics.

- **(c) Semiclassical / surface-hopping nonadiabatic dynamics:** "Photochemical dissociation of HOBr. A nonadiabatic dynamics study," J. Photochem. Photobiol. A 2016, doi pii S1010603016300740 — EOM-CCSD/TDDFT M06-2X, 10 excited states, nuclear-ensemble (Wigner, 500 points) spectrum plus surface-hopping trajectories; HO + Br elimination ~67% of processes. Category (c). Not a converged 3D surface; not quantum wavepacket.

- **(d) Full 3D quantum wavepacket dynamics on an ab initio surface — ONLY for singlets:**
  - Füsti-Molnár, Szalay, Balint-Kurti, "Photodissociation of HOBr. I. Ab initio potential energy surfaces for the three lowest electronic states…," J. Chem. Phys. 1999, 110(17), 8448–8460 — ab initio PESs for the three lowest SINGLET states (MR-AQCC, TZ2P) plus transition dipole surfaces; grid-based rovibrational states. Part II computed cross sections and product rotational distributions for the first two UV (singlet, 1 1A''/2 1A') bands.
  - Zhang & Jiang, "A Quantum Wavepacket Study of State-to-State Photodissociation Dynamics of HOBr/DOBr," Chinese J. Chem. Phys. 2020, 33(2), 173–182, doi:10.1063/1674-0068/cjcp1911214 — first full 3D MRCI PESs for the three lowest SINGLET states; real-wavepacket dynamics giving absorption spectra, internal-state and angular distributions. Explicitly singlet only.
  - Peterson, "An accurate global ab initio potential energy surface for the X 1A' electronic state of HOBr," J. Chem. Phys. 2000, 113(11), 4598–4612, doi:10.1063/1.1288913 — ground-state global PES (MRCI+CBS), 708 bound vibrational levels; the natural ground-state surface for the initial wavepacket.

**Verdict on Q1: The triplet 3D-wavepacket gap is genuinely open for all three HOX molecules.** The a 3A'' surface has never been globally mapped ab initio and propagated on. This is precisely category (d) applied to the triplet.

### Question 2 — Temperature dependence of the cross section; JPL/IUPAC basis

- **Primary HOBr data:** Ingham, Bauer, Landgraf, Crowley, "Ultraviolet-Visible Absorption Cross Sections of Gaseous HOBr," J. Phys. Chem. A 1998, 102(19), 3293–3298, doi:10.1021/jp980272c — three bands, λmax at 284 nm (σ = 25.0 ± 1.2 × 10⁻²⁰ cm²), 351 nm (12.4 ± 0.6 × 10⁻²⁰), and 457 nm (2.3 ± 0.2 × 10⁻²⁰), measured **at 295 K only**. Verbatim: absorption into the third band "is responsible for up to 50% of the total photolysis rate at high zenith angles close to the surface and at least 25% at all other altitudes and zenith angles down to 40°"; total J(HOBr) ranges 3.2 × 10⁻³ s⁻¹ at 15 km to 2.2 × 10⁻³ s⁻¹ at the surface (lifetimes 5.2 and 7.6 min).
- **Earlier detection:** Barnes, Lock, Coleman, Sinha, "Observation of a New Absorption Band of HOBr and Its Atmospheric Implications," J. Phys. Chem. 1996, 100(2), 453, doi:10.1021/jp952445t — new band near 440 nm; verbatim "σmax ∼ 9 × 10⁻²¹ cm²… inclusion of absorption by this new band system will shorten the photochemical lifetime of tropospheric HOBr in the polar regions by a factor of 2 compared to the recently recommended value based on the near-UV absorption bands alone."
- **IUPAC:** Task Group datasheet III.A5.114 "HOBr + hν → products" (Atkinson et al., ACP 7, 981–1191, 2007) gives "Absorption cross sections of HOBr at 295 K," dissociation thresholds 578 nm (HO+Br) and 445 nm (HBr+O(3P)), quantum yield ~1, with NO temperature-dependence expression.
- **JPL:** JPL 15-10 (Evaluation 18, Burkholder et al. 2015) and JPL 19-5 (Evaluation 19), BrOx Photochemistry, reaction G6 "HOBr + hν → products" — recommend the room-temperature cross sections (based on Ingham et al.) with quantum yield 1 and no temperature parameterization. (See caveat: JPL 19-5 note not verifiable verbatim due to the site's automated-access block.)
- **HOI:** Bauer, Ingham, Carl, Moortgat, Crowley, "Ultraviolet-Visible Absorption Cross Sections of Gaseous HOI and Its Photolysis at 355 nm," J. Phys. Chem. A 1998, doi:10.1021/jp9804300 — bands at 340.4 nm (3.85 ± 0.4 × 10⁻¹⁹) and 406.4 nm (3.30 ± 0.3 × 10⁻¹⁹), room temperature; OH yield ~1 at 355 nm. Also Rowley et al., J. Atmos. Chem. (HOI at 298 K).
- **HOCl triplet:** "Assessing the Contribution of the Lowest Triplet State to the Near-UV Absorption Spectrum of HOCl," J. Phys. Chem. A 1998, doi:10.1021/jp9835869 — triplet feature at 380 nm, σ = 4 × 10⁻²¹ cm², tail to 480 nm; estimated to shorten stratospheric HOCl lifetime by 10-20% and reduce abundance 11-12%.

**Verdict on Q2: No temperature dependence has been measured or computed for any HOX band.** This is a clean, defensible gap that a temperature-dependent wavepacket calculation (Boltzmann-weighted initial vibrational states) could fill — exactly what the user's imaginary-time-relaxation + autocorrelation pipeline is built to do.

### Question 3 — Uncertainty in J(HOBr) and modelled sensitivity

- The 457 nm band alone contributes up to 50% of J(HOBr) (Ingham et al. 1998), so any error in the triplet band propagates almost directly into J(HOBr).
- HOBr photolysis is the light-driven branch competing with heterogeneous HOBr + H⁺ + Br⁻ → Br2 recycling that powers the bromine explosion (Falk & Sinnhuber, GMD 11, 1115, 2018; Sinnhuber et al., ACP 25, 15653, 2025; Cao et al., ACP 22, 3875, 2022). Concentration-sensitivity analyses (Cao et al., KINAL model) identify HOBr photolysis as a major ODE-decelerating reaction.
- The JPL-assigned uncertainty factor for the HOBr cross section is comparatively large (factor ~2 at λ < 350 nm in JPL 15). No study isolates the fraction of J(HOBr) uncertainty attributable specifically to the triplet-band cross-section shape or its temperature dependence — that quantification does not appear to exist and is itself a publishable sub-result.

**Verdict on Q3: The sensitivity is real and acknowledged qualitatively, but has not been quantified with respect to the triplet-band cross section or its T-dependence.**

### Question 4 — Hot bands at atmospheric temperatures

No published treatment exists. With ν3 (O-Br stretch) = 620.23 cm⁻¹, the first excited stretch level carries ~5.0% Boltzmann population at 298 K (fewer at 220 K); the bend (1162.57 cm⁻¹) adds ~0.36%. Because hot bands red-shift absorption and the triplet band already lies on the actinic-flux-rich red side, even a few-percent hot population can measurably raise long-wavelength absorption and hence J(HOBr) — temperature-dependently (more hot-band absorption at 300 K than at 200 K). This is directly analogous to the documented vibrational-hot-band temperature dependence of CO2 (VUV) and O3 (Huggins) cross sections. It has never been considered for HOBr.

### Question 5 — Competing/alternative targets (ranked)

1. **HOBr a 3A'' triplet band (temperature-dependent 3D wavepacket + SOC): TOP TARGET.** Gap genuinely open; atmospheric leverage high (bromine explosion, polar ODEs); tractable (triatomic, single triplet surface + SOC coupling). Requires adding SOC — see Q6.

2. **HOI triplet/visible band (marine iodine, new particle formation): STRONG SECOND.** Gap open (no 3D wavepacket dynamics on any HOI surface, singlet or triplet; only room-T experiments and Minaev response theory). Atmospheric importance rising fast (iodine NPF, tropospheric ozone). Tractability slightly worse: iodine SOC is large, relativistic treatment (ECP/x2c) mandatory, and per Minaev the triplet dominates the visible absorption, so SOC is not a perturbation. Highest scientific novelty of the HOX set.

3. **HOCl triplet band: WEAKER but ideal validation.** Gap technically open for 3D triplet wavepacket dynamics, but atmospheric payoff smaller (triplet shortens HOCl lifetime only ~10-20%) and HOCl singlet dynamics are already extensively studied. Best used as a light-atom, small-SOC stepping-stone before HOBr/HOI (a measured 380 nm benchmark band exists).

4. **CH3Br / CH3Cl photolysis isotope effects (13C, 81Br, 37Cl source apportionment): PARTIALLY FILLED.** CH3Br A-band wavepacket dynamics on coupled ab initio SOC surfaces already exist (J. Chem. Phys. 2009, 130, 244305) and CH3I is well studied (Amatatsu SOCI 3D). Isotopologue-resolved, temperature-dependent fractionation cross sections are less complete, and the N2O isotopomer wavepacket work (Schinke et al., JPCA 2010, doi:10.1021/jp101691r) is a proven template — but this is a 5-atom system straining a triatomic Jacobi pipeline.

5. **BrO / IO / OIO photodissociation: MODERATE gap, HARDER.** Open-shell radicals with dense SOC-split manifolds, strong multireference character and predissociation; OIO is atmospherically important (iodine NPF) but less clean for a first triplet-band demonstration.

6. **Br2O, BrONO2, BrOCl: LOWER priority.** Larger, more electrons, more coupled surfaces; poor fit to the described pipeline.

### Question 6 — State of the art for SOC in photodissociation wavepacket dynamics (Br/I)

- **Electronic structure + SOC:** MOLPRO (state-interaction SOC over MRCI/CASSCF), OpenMolcas RASSI-SO / MPSSI (atomic mean-field one-electron SOC over CASSCF/RASSCF with CASPT2 energies; biorthogonal handling of separately-optimized singlet/triplet orbitals), and PySCF (x2c / SOC) are all standard. For Br and especially I, scalar-relativistic + SOC (ECP or x2c/DKH; ANO-RCC or aug-cc-pVnZ-PP basis) is required. This aligns with the user's SA-CASSCF/NEVPT2 stack (NEVPT2 in ORCA/PySCF plus an SOC state-interaction step is the natural extension).
- **Dynamics with SOC:** Two proven routes. (i) Diabatize the spin-free states, add SOC as (near-constant asymptotically) coupling, and propagate coupled surfaces — used for CH3Br A-band (JCP 2009), CH3I (Amatatsu SOCI, 3D quantum), and HBr (spin-orbit-coupled wavepacket, PubMed 16623464). (ii) MCTDH with an SOC-augmented vibronic-coupling Hamiltonian; SHARC for trajectory-based SOC dynamics.
- **Tractability of an SOC 3D wavepacket HOX triplet band:** Highly tractable. HOBr is a triatomic; the triplet band is weak (borrows intensity via SOC from nearby singlets), so a reasonable model is: ground X 1A' (Peterson PES) as the initial-state surface, a global a 3A'' surface, and SOC coupling matrix elements (constant or geometry-dependent) to the bright singlet(s), with an SOC-borrowed transition dipole to the triplet. Split-operator propagation of the (Boltzmann-weighted) initial wavepacket promoted onto the a 3A'' surface, with the autocorrelation giving the band. The only genuinely new machinery for the user is (1) computing SOC matrix elements (add RASSI-SO or MOLPRO SOC) and (2) an SOC-borrowed transition dipole. Both are incremental additions, not a rewrite.

## Recommendations

**Stage 1 (validate the SOC extension on HOCl):** Build a 3D a 3A'' surface for HOCl (SA-CASSCF/NEVPT2 + SOC via RASSI-SO or MOLPRO), reuse the existing ground-state pipeline, and reproduce the known 380 nm triplet band (σ = 4 × 10⁻²¹ cm², tail to 480 nm). Light-atom SOC is small and the experimental band is a benchmark. Success threshold: band peak within ~15 nm and integrated cross section within a factor ~2 of experiment.

**Stage 2 (the contribution — HOBr):** Extend to HOBr a 3A''. Produce (a) the visible triplet band profile from 3D wavepacket dynamics on the ab initio surface, and (b) its temperature dependence by Boltzmann-weighting the initial vibrational states (ground, O-Br stretch at 5.0% at 298 K, bend at 0.36%) from imaginary-time relaxation, propagating each, and summing. Report σ(λ, T) over 200-300 K and the resulting fractional change in J(HOBr). This is the publishable core: the first first-principles triplet cross section AND the first temperature dependence for any HOX band. Impact threshold: if σ at 440-500 nm changes by ≥10-15% between 220 K and 298 K, it materially affects polar J(HOBr).

**Stage 3 (amplify significance):** Feed σ(λ, T) into a TUV/box-model J-value calculation (as Ingham did) to quantify ΔJ(HOBr), and ideally propagate through a polar-ODE box model (KINAL-type) to show sensitivity of bromine-explosion timing / ozone-depletion onset. This converts a spectroscopy paper into an atmospheric-impact paper — squarely in the user's wheelhouse (Jacob-group atmospheric halogen background).

**Stage 4 (optional high-novelty follow-on):** Repeat for HOI, where the triplet dominates the visible absorption and iodine NPF gives high topical relevance. Highest-novelty single result but largest SOC/relativistic burden.

**Benchmarks that would change the plan:** If a full-3D SOC triplet HOBr wavepacket study appears during scoping (monitor JCP, JPCA, PCCP, ACP), pivot to HOI immediately. If HOCl Stage 1 cannot reproduce the 380 nm band, the SOC-borrowed-dipole model is inadequate and a full 4-state (X, A, B, a) SOC-coupled propagation is needed before proceeding.

## Caveats

- **Could not verify verbatim the JPL 19-5 HOBr note text** (jpldataeval.jpl.nasa.gov blocks automated fetch). The conclusion that JPL recommends single-temperature HOBr data with no T-dependence is robust but rests on JPL 15-10 content plus the absence of any newer HOBr measurement; confirm by opening the JPL 19-5 BrOx PDF manually.
- **No source states explicitly "the temperature dependence of HOBr σ has not been measured."** The gap is established implicitly but consistently: every recommendation and every primary study (Barnes 1996, Ingham 1998, Bauer 1998, HOCl triplet study) reports room-temperature-only data with no temperature parameterization. A referee may want this stated carefully.
- **Papers to read in full (paywalled) to settle the question definitively:** Füsti-Molnár/Szalay/Balint-Kurti JCP 1999 Part I and the Part II cross-section paper (confirm singlet-only); Zhang & Jiang CJCP 2020 (confirm no triplet); Minaev THEOCHEM 1999 and JPCA 1999 (confirm response-theory not dynamics, and get exact predicted band shapes); the 2016 nonadiabatic-dynamics paper (confirm TDDFT/surface-hopping, category c).
- **HOI singlet dynamics status not fully confirmed** — I found no 3D wavepacket study of HOI on any surface, but a targeted search of the Jiang/Guo/Balint-Kurti quantum-dynamics groups is advisable before committing to HOI as "fully open."
- **The magnitude of the ultimate atmospheric payoff is modest**: the triplet band matters most at high SZA / polar surface conditions, and the absolute σ is small (~2.3 × 10⁻²⁰ cm² at 457 nm). This is a genuine but niche contribution — consistent with the user's stated goal of a real, even if niche, addition to atmospheric halogen photochemistry.

## Full Citations (key papers)

- Francisco, Hand, Williams, "Ab Initio Study of the Electronic Spectrum of HOBr," J. Phys. Chem. 1996, 100(22), 9250–9253. doi:10.1021/jp9529782.
- Barnes, Lock, Coleman, Sinha, "Observation of a New Absorption Band of HOBr and Its Atmospheric Implications," J. Phys. Chem. 1996, 100(2), 453. doi:10.1021/jp952445t.
- Ingham, Bauer, Landgraf, Crowley, "Ultraviolet–Visible Absorption Cross Sections of Gaseous HOBr," J. Phys. Chem. A 1998, 102(19), 3293–3298. doi:10.1021/jp980272c.
- Bauer, Ingham, Carl, Moortgat, Crowley, "Ultraviolet–Visible Absorption Cross Sections of Gaseous HOI and Its Photolysis at 355 nm," J. Phys. Chem. A 1998, 102(17), 2857–2864. doi:10.1021/jp9804300.
- (HOCl triplet) "Assessing the Contribution of the Lowest Triplet State to the Near-UV Absorption Spectrum of HOCl," J. Phys. Chem. A 1998. doi:10.1021/jp9835869.
- Minaev, "Ab initio study of the singlet–triplet transitions in hypobromous acid," J. Mol. Struct. THEOCHEM 1999. doi pii S0166128099000846.
- Minaev et al., "The Singlet–Triplet Absorption and Photodissociation of the HOCl, HOBr, and HOI Molecules Calculated by the MCSCF Quadratic Response Method," J. Phys. Chem. A 1999. doi:10.1021/jp990203d.
- Füsti-Molnár, Szalay, Balint-Kurti, "Photodissociation of HOBr. I. Ab initio potential energy surfaces…," J. Chem. Phys. 1999, 110(17), 8448–8460.
- Peterson, "An accurate global ab initio potential energy surface for the X 1A′ electronic state of HOBr," J. Chem. Phys. 2000, 113(11), 4598–4612. doi:10.1063/1.1288913.
- Zhang, Jiang, "A Quantum Wavepacket Study of State-to-State Photodissociation Dynamics of HOBr/DOBr," Chinese J. Chem. Phys. 2020, 33(2), 173–182. doi:10.1063/1674-0068/cjcp1911214.
- "Photochemical dissociation of HOBr. A nonadiabatic dynamics study," J. Photochem. Photobiol. A 2016. doi pii S1010603016300740.
- Atkinson et al., "Evaluated kinetic and photochemical data for atmospheric chemistry: Vol. III – gas phase reactions of inorganic halogens," Atmos. Chem. Phys. 2007, 7, 981–1191 (IUPAC datasheet III.A5.114, HOBr).
- Burkholder et al., "Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies," JPL Publication 15-10 (Eval. 18) and 19-5 (Eval. 19), BrOx Photochemistry, reaction G6.
- HOBr vibrational fundamentals: NIST Chemistry WebBook (compiler M.E. Jacox), ν3 = 620.23 cm⁻¹, ν2 = 1162.57 cm⁻¹, ν1 = 3614.90 cm⁻¹; from McRae & Cohen, J. Mol. Spectrosc. 139, 369 (1990) and Cohen, McRae et al., J. Mol. Spectrosc. 173, 55 (1995).
- Atmospheric context: Falk & Sinnhuber, Geosci. Model Dev. 2018, 11, 1115; Sinnhuber et al., Atmos. Chem. Phys. 2025, 25, 15653; Cao et al., Atmos. Chem. Phys. 2022, 22, 3875.
