# HOCl — the validation case

HOCl is not the target; HOBr is. HOCl is here because its ã 3A" band **has
been measured** (~380 nm), so it is where the method can be shown to work
before it is trusted on a molecule where no answer exists to check against.

Success threshold, set in [`../TASKS.md`](../TASKS.md): band peak within ~15 nm
and integrated cross section within a factor of ~2.

## Layout

| | |
| --- | --- |
| [`01_method/`](01_method/) | choosing and validating the electronic structure: SOC borrowing, the active space, the triplet manifold, which method has the right slope |
| [`02_band_model/`](02_band_model/) | a 1D model that asks how much surface the band actually needs, before paying for a 3D one |
| [`03_surfaces/`](03_surfaces/) | the raster, the multireference outer shell, the fragments, and the splice |
| `data/` | every CSV, NPZ and PNG produced, all git-tracked |
| `logs/` | the EVO's stdout for each run |

Scripts keep their original numeric prefixes (`03`…`15`). The numbers are
chronological and are referenced throughout `TASKS.md` and the commit history,
so they are worth more than a tidier renumbering would be. Data paths inside
each script resolve against `HOCl/data/`, not the working directory.

## Where it stands

**The surface is built.** `data/hocl_surfaces.npz` holds V_S over the bound
region and V_T on r(O-Cl) 1.40-6.00 Å × r(O-H) 0.80-1.25 Å × 75-135°, assembled
from three sources with checks that can genuinely fail (see `03_surfaces/`).

**The physics that matters for the band**, established along the way:

- the ã 3A" ← X 1A' band is dark by spin selection and borrows f = 8.9e-7 from
  bright singlets; vertical 3.448 eV against the measured 380 nm (+5.7%)
- the band is set in the Franck-Condon region on a femtosecond timescale, on a
  **single smooth 3A" surface** — there is no interior avoided crossing
- EOM-CCSD on the CCSD(T) ground state is the cleanest triplet in that window:
  slope within 0.5% of UCCSD(T), roughness 0.1 meV against 4.2
- the lowest triplet is 3A" everywhere in the Franck-Condon region, and 3A'
  arrives only past ~2.3 Å at wide angles, inside where the absorber will sit

## What is left

- [ ] Jacobi transform and relaxation, from `water/`
- [ ] 3D propagation with an absorber from ~2.3 Å, repeating water's
      absorber-position check in 3D
- [ ] the transition dipole: Condon first, then μ_SOC over the Franck-Condon
      window using `01_method/03`'s machinery, with
      `water/10_mu_sensitivity.py`'s test of whether its variation matters
- [ ] σ(λ, T) against the measured band

Optional, deliberately deferred: the 3A'/3A" crossing beyond ~3 Å, where the
3A" gaps fall below the Cl 2P splitting (882 cm-1) and SOC decides
Cl(2P_3/2) vs Cl(2P_1/2) branching. That is product fine structure, not the
absorption band.
