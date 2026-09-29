# references

Literature filed in the repo. Only open-licence items are committed.
Publisher-copyrighted papers go in `licensed/`, which is gitignored: the
copies exist on the machine they were put on and nowhere else, and the notes
below (ours) are what the repo carries. Get them by DOI.

| file | reference | licence |
| --- | --- | --- |
| `Atkinson2007_IUPAC_VolIII_acp-7-981-2007.pdf` | R. Atkinson, D. L. Baulch, R. A. Cox, J. N. Crowley, R. F. Hampson, R. G. Hynes, M. E. Jenkin, M. J. Rossi and J. Troe, "Evaluated kinetic and photochemical data for atmospheric chemistry: Volume III – gas phase reactions of inorganic halogens", *Atmos. Chem. Phys.* **7**, 981-1191 (2007). doi:[10.5194/acp-7-981-2007](https://doi.org/10.5194/acp-7-981-2007) | Creative Commons (stated on p. 981) |
| `licensed/Bauer1998_JPCA_102_2857.pdf` | D. Bauer, T. Ingham, S. A. Carl, G. K. Moortgat and J. N. Crowley, "Ultraviolet-Visible Absorption Cross Sections of Gaseous HOI and Its Photolysis at 355 nm", *J. Phys. Chem. A* **102**, 2857-2864 (1998). doi:[10.1021/jp9804300](https://doi.org/10.1021/jp9804300) | ACS, not committed |
| `licensed/Minaev1999_JPCA_103_7294.pdf` | B. F. Minaev, "The Singlet-Triplet Absorption and Photodissociation of the HOCl, HOBr, and HOI Molecules Calculated by the MCSCF Quadratic Response Method", *J. Phys. Chem. A* **103**, 7294-7309 (1999). doi:[10.1021/jp990203d](https://doi.org/10.1021/jp990203d) | ACS, not committed |

## Atkinson et al. 2007 — what bears on this project

**HOBr photolysis, datasheet III.A5.114 (pp. 1159-1161).** The IUPAC
recommendation for HOBr, and the evaluators' reasoning on the visible band:

- The spectrum is "three broad bands with maxima at ∼284 nm, ∼351 nm, and
  ∼457 nm". The ~457 nm band was predicted by theory (Francisco et al. 1996;
  Minaev 1999) and is seen by Rattigan et al. (1996) and Ingham et al.
  (1998); Barnes et al.'s OH action spectrum over 400-600 nm is cited as
  "further evidence" for the band's *existence*.
- Why Ingham was chosen: in the other absorption studies HOBr came from
  Br₂O + H₂O equilibrium mixtures, and "it is likely that this scatter, and
  the difficulty in detecting the long wavelength band, are largely due to
  the presence of impurities such as Br₂O and Br₂". Ingham made HOBr in situ
  (laser photolysis of H₂O₂–Br₂) and calibrated against Br₂, which "appears to
  have been the most successful in avoiding interference from impurities".
  The preferred table is Ingham's three-Gaussian fit, extended slightly.
- Barnes's action spectrum is **not** used for the cross sections — it
  supplies no absolute σ and no band shape in the evaluation's reasoning.
  So the Barnes-vs-Ingham *shape* disagreement our calculation lands on is not
  discussed there at all.
- The evaluators themselves say "there remain significant uncertainties in
  the values of the absorption cross sections", with differences up to a
  factor of three near 350 nm.
- Quantum yield: only Benter et al. (1995), > 0.95 at 363 nm; "confirmation
  of unit quantum yield is desirable". Nothing measured in the visible band.
- Lifetime (Ingham's cross sections): ~5 min in the lower stratosphere at
  SZA 40°, ~30 min at the surface at high SZA.

**A citation error in it:** the Barnes reference is given as "J. Phys.
Chem., 100, 817, 1994". Crossref has it as *J. Phys. Chem.* **100**, 453-457
(1996), doi:10.1021/jp952445t (volume 100 is 1996). Cite the Crossref form.

**HOCl, III.A5.103 (pp. 1142-1143):** preferred σ from the Barnes et al.
(1998) expression, with the weak band centred at ≈ 370 nm extending to
500 nm; quantum yield Φ(HO + Cl) = 1 for λ > 200 nm. This is the HOCl
comparison target.

Also relevant: Br₂ (III.A5.121) and BrCl (III.A5.120) cross sections — the
species that contaminate or calibrate the HOBr measurements.

**HOI, III.A5.123 (pp. 1178-1179):** preferred σ = Bauer et al. (1998),
280-490 nm, Φ(HO + I) = 1 over 280-490 nm. Mössinger et al. (1999) agree
except for systematically higher values in the 405 nm band. The datasheet's
threshold for HOI → HO + I is **507 nm** (ΔH° 236 kJ/mol). Bauer et al.
used 582 ± 20 nm. The difference matters: see Bauer below.

## Bauer et al. 1998 — how the HOI spectrum was made

- HOI from OH + I₂ in situ (248 nm photolysis of H₂O₂, then OH + I₂ →
  HOI + I in microseconds). Gated diode array 280-500 nm, plus a
  photomultiplier at selected wavelengths.
- **The red end is fixed by construction.** The I₂-loss contribution was
  found by scaling an I₂ reference spectrum to the post-flash absorption *at
  500 nm* and subtracting it (p. 2860). Any HOI absorption at 500 nm is
  therefore booked as I₂ loss, and near 500 nm the retrieved spectrum is
  σ_HOI(λ) − σ_HOI(500)·σ_I₂(λ)/σ_I₂(500). HOI cross sections were
  calibrated at the 436 nm isosbestic point (σ_I₂ = 1.41e-19 cm²).
- Above 500 nm, I₂ absorption was not linear in concentration, so "a
  reliable measurement of any absorption due to HOI above 500 nm was not
  possible". The authors expected a triplet band there by analogy with HOBr
  (p. 2863).
- **The published table is the fit**, a two-band semilogarithmic Gaussian
  (eq ii, Table 1), not the data. The MPI-Mainz atlas file we cache
  (`HOI/data/obs/HOI_Bauer(1998)_…`) reproduces that fit exactly, including
  at 490-500 nm. So Bauer's "σ(490) = 8.6e-22" is a model tail, not a
  measured bound.
- **532 nm photolysis:** no OH at 1e18 photons cm⁻². The authors conclude σ(532) ≲ 1e-20 cm²
  *if* Φ(OH) = 1, taking their threshold of 582 ± 20 nm. With IUPAC's 507 nm
  threshold, 532 nm lies below it and the null says nothing about
  absorption. For a band on a purely repulsive state there is no absorption
  below D₀ at all, from v = 0, so D₀(HO–I) decides whether this is a
  constraint.
- Φ(OH) at 355 nm = 1.05 (+0.13/−0.34). Temperature dependence is
  semi-empirical: +5% at both maxima at 220 K. Lifetime 100-170 s at the
  surface, SZA 30-70°.

## Minaev 1999 — the theory the HOX series has to be compared with

- MCSCF quadratic response, full one- plus two-electron SOC operator,
  CAS(12,9) with three virtuals. Small bases: HOCl aug-cc-pVDZ, HOBr
  Sadlej/Ahlrichs VDZ, **HOI 3-21G only**, at geometries that for HOI put
  r(O-I) at 2.104 Å (optimised) against 1.991 Å (experiment).
- a³A″ ← X oscillator strengths: HOCl 4-6e-6, HOBr 7-8e-5 (4.9e-5 in
  3-21G), HOI 1.6-2.5e-4. For comparison, this project (QD-NEVPT2 state
  interaction, CAS(12,7)) gets HOCl 8.9e-7, HOBr 1.5e-5, HOI 1.5e-4, and the
  measured values are HOCl 3.3e-5 and HOBr 4.7-6.2e-5. For HOBr, Minaev is
  close to measurement and we are 3-4× low; for HOI the two calculations agree.
- **The mechanism:** a³A″ (T_z sublevel, polarised along O-X) borrows mostly
  from a *high* singlet, 4¹A′ at ~9 eV with M ≈ 0.7-1 au (σ* ← σ),
  plus triplet-triplet terms. In quadratic response the sum over states is
  complete within the CAS, and orbital relaxation brings in configurations
  outside it. Our state interaction only sees the roots we compute. Adding
  roots (HOBr: 6, 10 and 16 roots, to 10.7 eV) does not raise f. So if this
  is our f deficit, the missing lenders lie outside CAS(12,7), not above its
  highest root.
- HOI assignment (Table 9 and p. 7307): the 410 nm band is 1¹A″ plus
  1³A′, with ³A′ ← X the larger (f 3e-3 in 3-21G). The a³A″ band is
  "overlapped by" it. His estimates: a³A″ σ = 7.6e-20 cm² at 21 400 cm⁻¹
  (467 nm); ³A′ σ = 1.8e-19 at 25 000 cm⁻¹ (400 nm). He concludes that
  3-21G probably overestimates ³A′'s growth.
- HOBr: ³A′ ← X contributes to the 350 nm band as much as 1¹A″ ← X does.
  Also HOCl's 304 nm band, less so.
