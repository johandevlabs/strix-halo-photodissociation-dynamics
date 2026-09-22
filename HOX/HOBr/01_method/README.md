# 01_method — HOBr

| script | descends from | question |
| --- | --- | --- |
| `01_geometry.py` | — (new) | where is the minimum, and is the force field right? |
| `02_obr_cut.py` | HOCl's `06`, `09`, `11` | is EOM-CCSD valid for HOBr, where does ³A′ cross, and does aug- matter in the bond-breaking region? |

## `01_geometry.py`

CCSD(T) with x2c on a 3×3×3 grid in (r(O-Br), r(O-H), ∠HOBr), then a full
quadratic fit — 10 coefficients from 27 points — which gives the stationary
point and the internal-coordinate Hessian together. Frequencies follow from
`H_cart = Bᵀ H_int B`, evaluated at the fitted minimum where that identity is
exact (the neglected term carries ∂E/∂q, which vanishes there).

**B is finite-differenced, not written out.** The analytic B matrix —
equivalently the Wilson G matrix — for a bent triatomic is precisely the kind
of recalled formula this repo keeps getting caught by. `internals()` is the
only definition of the coordinates, and B is differenced from it, so the two
cannot disagree.

**Run `--molecule HOCl` first.** Its fundamentals are known (724, 1239,
3609 cm⁻¹) and it is cheap, so it tests the whole chain against measurement
before HOBr is asked to produce a number nobody can check. Harmonic
frequencies should land a few percent *above* the observed fundamentals;
anharmonicity is negative. A ratio below 1.00 means something is wrong rather
than merely approximate.

### Tests, in `tests/`

Run both before sending anything to the EVO; neither needs PySCF.

- `test_geom.py` — the mathematics. The fit recovers a known force field
  exactly, and the frequencies are checked against a **direct Cartesian
  finite-difference Hessian** of the same analytic surface. That second route
  shares no code with the first: it never touches B and never converts
  degrees. It exists because the unit chain (Å→bohr, degree→radian, amu→mₑ)
  is a place where a wrong answer looks entirely plausible — and it caught two
  real conversion errors in the first draft, one of them off by 10⁷.

  The synthetic force constants are calibrated to HOBr's observed frequencies
  rather than picked arbitrarily, so the conditioning is exercised in the
  regime the script actually runs in.

  Note what is *not* a test: `BᵀHB` has rank ≤ 3, so six of the nine
  eigenvalues are zero by construction. Their vanishing checks nothing.

- `test_run.py` — the machinery that is not chemistry, with a fake quadratic
  calculation standing in for CCSD(T): the fork pool, the resumable CSV, the
  grid/CSV key agreement, and `--refine`. An earlier version had the grid
  rounding θ to 5 decimals and the CSV to 4, which would have silently
  recomputed every point on every rerun while the resume logic looked like it
  worked; both now go through one `key()`.

## `02_obr_cut.py`

One 1D cut along O-Br answers four questions that would otherwise be four
runs, and a second basis adds a fifth for the cost of doubling a cheap job.

1. **Is EOM-CCSD valid for HOBr, out to what radius?** For HOCl it held to
   2.29 Å, T1(S) passing 0.02 by 2.6 Å. Br is heavier with 4× the SOC, so
   none of that transfers. The raster's outer edge depends on the answer.
2. **Where does ³A′ cross ³A″?** HOCl's raster found ³A′ below at 112 of 2730
   points; the symmetry labelling is the only reason a ³A′ energy did not
   enter the surface unnoticed.
3. **Band position and width to first order** — vertical and slope at the FC
   geometry, against Ingham's 457 nm.
4. **Does aug- matter where the bond breaks?**

On (4), the atomic sweep is no evidence either way: SOC is a near-nuclear
property, so `aug-cc-pvtz-dk` matching `cc-pvtz-dk` to 0.1 points on the ²P
splitting says nothing about 2-3 Å. The expectation is that it does little —
ã³A″ is n→σ* dissociating to OH(²Π) + Br(²P), valence throughout, with no
Rydberg, anion or charge-transfer character. Two things could still differ:

- **BSSE**, an attractive error that peaks where the fragments are close but
  separating. Its signature is a difference that **grows with r**, so the
  script fits the trend and reports that, not the offset — a constant offset
  is two bases describing the same state at different completeness and
  cancels out of a band shape.
- **Spurious diffuse roots**, the way aug- could make things *worse*. The
  raster stores whatever its labelling picks, and diffuse functions add
  low-lying roots to sort through exactly where the level density is already
  high. So all six roots and their irreps are stored, and the report counts
  how many sit below the valence ³A″ in each basis.

### Tests

`tests/test_cut.py`, no PySCF. A synthetic manifold with a ³A′ crossing, a
T1 threshold and a basis difference of known slope, checking that each is
*detected* rather than merely storable — plus a control where the basis
difference is a constant offset, which must **not** be reported as a trend.

Expectations in that test are derived from the synthetic surface rather than
written down. The first draft hard-coded the crossing radius and the T1
threshold by eye, both wrong, and the script's correct answers looked like
failures.

## The cache bug, 2026-09-22

The first two runs produced logs headed `CCSD(T)/def2-tzvp` and
`CCSD(T)/cc-pvtz-dk` containing **bit-identical numbers**, down to a 0.145 meV
fit residual. Two different bases cannot do that.

`key()` identified a grid point by geometry alone, so the second run found all
27 points already present, computed nothing, and reported the first run's rows
under its own header. Only one set of energies was ever computed — `nbf` 78
and the wall times identify them as `cc-pvtz-dk` — and the def2-TZVP log is
simply mislabelled. The stored CSV has been annotated accordingly.

A cache keyed on less than the thing it caches does not fail, it answers. The
basis is now part of the key and a `basis` column of the CSV, `--refine`
filters to one basis, and `tests/test_run.py` runs two bases into the same
file and checks that the second is not served from the first's rows and that
the reported frequencies differ.

What made it visible was reading two logs side by side and noticing that
agreement was *too* good. Nothing in the run would have flagged it.
