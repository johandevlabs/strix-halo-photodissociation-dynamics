# 01_method — HOBr

| script | descends from | question |
| --- | --- | --- |
| `01_geometry.py` | — (new) | where is the minimum, and is the force field right? |

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
