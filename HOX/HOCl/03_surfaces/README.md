# 03_surfaces — building the ã 3A" surface

Four scripts turn the method decisions from `01_method/` into
`../data/hocl_surfaces.npz`. The surface is assembled from three sources
because no single method is sound everywhere: coupled cluster inside, where the
band is decided; multireference outside, where the ground state turns
diradical; and the exact fragment asymptote beyond that.

| script | region | method | cost |
| --- | --- | --- | --- |
| `12_pes_raster.py` | r(O-Cl) 1.40-2.40 Å | CCSD(T) + EOM-CCSD, one calculation gives **both** surfaces | 2730 pts, 149 min on 30 workers |
| `13_outer_shell.py` | 2.20-3.60 Å | SA-CASSCF(10,6)/4 roots + SC-NEVPT2 | 1050 pts, 2.3 min |
| `14_fragments.py` | asymptote | V_OH(r) and E(Cl) in two method families | minutes |
| `15_splice.py` | all | joins them; local, no PySCF | seconds |

The shell **starts at 2.20 Å, inside the raster**. NEVPT2 and CCSD(T)+EOM sit
on different absolute energy scales, and an overlap is the only thing that can
fix the offset between them; abutting two grids would hide the disagreement
instead of measuring it.

## The splice, and which of its checks mean anything

`15` prints several numbers. Two of them can fail:

- **Is each column's offset constant across the overlap radii?** Median 10 meV,
  max 42 meV. An offset can absorb a constant difference; it cannot absorb a
  disagreement in *shape*, so this is a real test. (Water's analogous splice
  held to 13 meV.)
- **Do the slopes agree at the seam?** At 2.30 Å, interior to both grids:
  raster +0.091, shell +0.102 eV/Å, average mismatch 0.011 eV/Å, worst column
  0.181 eV/Å = 36 meV over the 0.20 Å blend.

The "local residual 0.0 meV" figures are **not** evidence. Everything on the
output grid is spline-interpolated and cosine-blended, so that number measures
the interpolation, which is smooth by construction.

Per-column offsets were necessary, not fastidious: the offsets span 330 meV
across (r(O-H), angle), so a single global offset would misplace columns by up
to 198 meV.

## What each run found

**`12`, the raster.** 2730 points, **0 failures**, 101 CPU-s/point. 2549 ok,
181 warned, and the warnings are confined to the outer edge — none below
2.25 Å, and within the band-relevant region (≤ 2.30 Å) only 36 of 2470 points,
1.5%, worst T1 0.0233. Smoothness, as deviation from a cubic through four
neighbours: no point exceeds 20 meV on any axis; the two non-trivial numbers
(10.1 meV on the steep repulsive wall, 3.6 meV along O-H in *every* band) are
curvature a 4-point cubic cannot follow, not scatter.

**The symmetry labelling earned its place.** Every EOM root is labelled by the
irrep of its dominant single excitation (`orbsym[i] XOR orbsym[nocc+a]` in Cs),
so what is stored is the lowest **3A"**, not merely the lowest triplet. At 112
points the lowest triplet is 3A' — the crossing arriving, earlier at wide
angles — and without the labelling a 3A' energy would have entered the surface
unnoticed. Those points carry `warn:Ap_below`.

**`13`, the shell.** 4 CPU-s/point, not the 200 assumed — SA-CASSCF(10,6) +
SC-NEVPT2 on 74 basis functions is genuinely seconds of serial work. 41 points
failed as `fail:rohf`, all at ≥ 2.4 Å where the fragments become two open-shell
radicals and the default guess is poor. Fixed by seeding the triplet SCF from
the converged **closed-shell density of the same geometry**; all 41 recovered
as plain `ok`. That is a local guess, not orbitals chained along a scan —
water's warning is about propagating an *active space*, which is still rebuilt
per point.

**`14`, the fragments.** OH r_e 0.9782 Å (UCCSD(T)) and 0.9803 (NEVPT2) against
0.9697 observed; ω_e 3727 and 3768 against 3738. Two unrelated methods agreeing
with each other and with measurement at ~1% is what made water's diatomic
trustworthy. NEVPT2 − UCCSD(T) runs 0.791-0.855 eV across the curve, spread
36 meV over the 0.80-1.25 Å actually used, varying smoothly rather than
scattering — the constancy is what the splice needs. Two adaptations from
water: **x2c everywhere**, or the fragments sit on a different energy scale
from the surfaces they anchor, and no point group for the diatomic, since
PySCF's Abelian subgroup of C∞v splits the degenerate 2Π across irreps.

## Cost estimates were wrong twice, both times high

`12` was budgeted at 570 CPU-s/point and came in at 101; `13` at 200 and came
in at 4. Both estimates were made by multiplying an earlier script's wall time
by its parallel factor, which counts threading overhead as work. Johan caught
the first: *"I saw all 32 threads in htop being at close to 100%"* — 32 threads
thrashing on matrices far too small to parallelise. This is water's job-array
finding again, and more extreme here than there.

Consequently both scripts follow water's job-array pattern: thread-limiting
environment variables set **before** numpy and pyscf load (BLAS reads them at
library load time), OpenMP pinning cleared explicitly (or every worker binds to
place 0 and piles onto one physical core while the load average looks healthy),
one single-threaded process per geometry, per-worker CPU affinity, rows
appended as they finish so an interrupt costs nothing, and an explicit `fork`
context — fork is the default on the EVO's Python 3.12, but 3.14 defaults to
forkserver, where each worker re-imports the module and re-runs the environment
setup the design depends on.

## Output

`../data/hocl_surfaces.npz` — V_S over the bound region (for χ₀) and V_T on
r(O-Cl) 1.40-6.00 Å × r(O-H) 0.80-1.25 Å × 75-135°.
`../data/hocl_surfaces.png` — contours: the ã 3A" surface is weakly dependent
on angle and steeply repulsive along O-Cl, as an n → σ* should be; the ground
state's well sits at r(O-Cl) 1.70, r(O-H) 0.97; and the cut through both seams
shows no kink.
