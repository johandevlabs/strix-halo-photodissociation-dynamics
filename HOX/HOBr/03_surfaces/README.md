# 03_surfaces — HOBr

| script | descends from | what |
| --- | --- | --- |
| `05_pes_raster.py` | HOCl's `12_pes_raster.py` | 3D CCSD(T) + EOM-CCSD raster: V_S and V_T from one calculation per point |
| `06_raster_check.py` | HOCl's hand check in `TASKS.md` | warnings and where, smoothness, agreement with `02`'s cut, the FC window |

HOCl's `12` ran 2730 points with zero failures, so `05` is that script with
HOBr's numbers in it rather than a rewrite: geometry, grid, basis, roots and
defaults changed; the calculation, the A′/A″ root labelling, the forced Cs
and the job-array machinery unchanged.

| | HOBr | HOCl | from |
| --- | --- | --- | --- |
| r(O-X) | 1.55-2.45 Å (r_eq −0.29 … +0.61) | 1.40-2.40 Å | `03`: band needs r_eq + 0.16; `02`: EOM to ~2.40, ³A′ from 2.55 |
| r(O-H), angle | 0.80-1.25 Å, 75-135° | same | |
| points | 2470 | 2730 | |
| basis | cc-pvtz-dk | def2-tzvp | toolchain sweep, `02`, `03` |
| EOM roots | 5 | 4 | one spare where ³A′ drops below at wide angles |
| cost | ~480 CPU-s/pt, ~11 h on 30 workers | 101 CPU-s/pt | `02`'s cut: 532 with 6 roots |

Two changes beyond the numbers, both from lessons in this directory tree:
the data directory is created if missing (`01` lost a round of points to it),
and the per-point cost estimate counts only the workers a small pilot actually
uses.

**All-electron, as `01` and `02` were**, so their conclusions transfer.
Correlating Br's 28 core electrons in a valence basis is most of the 5× cost
over HOCl and is not especially balanced; a frozen-core comparison on `02`'s
cut would settle whether to switch. Open, not done.

`tests/test_raster.py`: pilot then full then resume on a stubbed calculation,
into a directory that does not exist yet — 2470 points, no duplicates,
equilibrium included in the pilot, nothing re-run.

## The run, 2026-09-24

2470 points in 9.7 h, 0 failed, 432 CPU-s/point. `06_raster_check.py`:

| check | HOBr | HOCl |
| --- | --- | --- |
| warnings | 45, all at r ≥ 2.30 Å | 181, none below 2.25 Å |
| max T1(S) | 0.0163 | 0.0251 |
| max cubic residual, inner wall / elsewhere | 5.4 / ≤ 3.6 meV | 10.1 / ≤ 3.6 meV |
| vs the independent 1D cut | 0.02 meV at 19 radii | — |
| vertical at the minimum | 2.867 eV = 432 nm | 3.432 eV |
| V_T across χ₀'s angles at the FC radius | 0.20 eV | 0.26 eV |

The cut agreement is not the same data compared twice: the cut's spectators
(0.961 Å, 102.3°) sit between grid points, the nearest of which is up to
14 meV off. Interpolation over 0.011 Å and 2.3° recovering an independent
6-root calculation to 0.02 meV says the surface is smooth in the spectators
and the labelling found the same state everywhere.
