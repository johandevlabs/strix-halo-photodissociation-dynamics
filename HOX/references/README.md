# references

Literature filed in the repo. Only open-licence items belong here.

| file | reference | licence |
| --- | --- | --- |
| `Atkinson2007_IUPAC_VolIII_acp-7-981-2007.pdf` | R. Atkinson, D. L. Baulch, R. A. Cox, J. N. Crowley, R. F. Hampson, R. G. Hynes, M. E. Jenkin, M. J. Rossi and J. Troe, "Evaluated kinetic and photochemical data for atmospheric chemistry: Volume III – gas phase reactions of inorganic halogens", *Atmos. Chem. Phys.* **7**, 981-1191 (2007). doi:[10.5194/acp-7-981-2007](https://doi.org/10.5194/acp-7-981-2007) | Creative Commons (stated on p. 981) |

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
