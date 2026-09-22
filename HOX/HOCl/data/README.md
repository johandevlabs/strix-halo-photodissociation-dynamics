# data — what each file is and what wrote it

All git-tracked, so any analysis can be rerun on the laptop without touching
the EVO. Scripts resolve these paths from their own location, so
`python HOCl/03_surfaces/15_splice.py` works from anywhere in the repo.
The matching stdout for each run is in [`../logs/`](../logs/).

## The surface

| file | from | contents |
| --- | --- | --- |
| `hocl_surfaces.npz` `.png` | `03_surfaces/15` | **the product**: V_S over the bound region, V_T on 1.40-6.00 Å × 0.80-1.25 Å × 75-135° |
| `hocl_pes_raster.csv` | `03_surfaces/12` | 2730 pts, CCSD(T) + EOM-CCSD, r(O-Cl) ≤ 2.40 Å |
| `hocl_outer_shell.csv` | `03_surfaces/13` | 1050 pts, SA-CASSCF + SC-NEVPT2, 2.20-3.60 Å |
| `hocl_fragments.csv` | `03_surfaces/14` | V_OH(r) and E(Cl) in both method families |

## Method selection

| file | from | what it settled |
| --- | --- | --- |
| `hocl_scan.csv` `.png` | `01_method/05` | the original O-Cl scan, where the 2.2-2.8 Å kink appeared |
| `hocl_scan_cs.csv` `.png` | `01_method/05 --symmetry Cs` | the test that changed nothing and could not have |
| `hocl_scan_pin.csv` `.png` | `01_method/05 --pin-irrep` | no A'/A" flip: the crowding state is itself 3A" |
| `hocl_scan_fc.csv` `.png` | `01_method/05`, fine FC grid | the UCCSD(T) reference slope |
| `hocl_triplet_manifold_cas*.csv` `.png` | `01_method/06` | multi-root 3A" along the cut; no avoided crossing |
| `hocl_fc_active_space.csv` `.png` | `01_method/07` | CAS(10,6) vs CAS(12,7) across the FC window |
| `hocl_fc_eq_t{1,2,4}.csv` `.png` | `01_method/07` | the same at equilibrium, 1/2/4 triplet roots |
| `hocl_nevpt2_contraction.csv` | `01_method/08` | SC vs FIC on identical references |
| `hocl_nevpt2_cas127.csv` | `01_method/08` | CAS(12,7) |
| `hocl_nevpt2_fv.csv`, `_fv_outer.csv` | `01_method/08` | full valence CAS(14,9), FC window and outward |
| `hocl_nevpt2_t1.csv` | `01_method/08` | single-root variant |
| `hocl_eom_triplet.csv` | `01_method/09` | EOM-CCSD slopes — the referee that settled the slope |
| `hocl_eom_outer.csv` | `01_method/09` | EOM out to where it breaks (~2.39 Å) |
| `hocl_eom_coordinates.csv` | `01_method/11` | O-H stretch, bend and corners: 3A" lowest everywhere |
| `hocl_band_1d.csv` `.png` | `02_band_model/10` | σ(E) from the 1D model, and the sensitivity scan |

`hocl_outer_shell.csv.bak`, a snapshot kept by `13 --retry-failed` while the
41 `fail:rohf` points were recomputed, was deleted once the shell was complete
at 1050/1050.
