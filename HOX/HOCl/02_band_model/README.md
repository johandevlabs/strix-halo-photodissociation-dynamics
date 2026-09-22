# 02_band_model — how much surface does the band need?

One script, `10_band_1d.py`, and it is the cheapest thing in the project: pure
NumPy/SciPy on data already computed, no PySCF, runs on the laptop in seconds.
It exists to answer a question that would otherwise be settled by paying for a
3D raster and hoping.

A 1D model along O-Cl with the other coordinates frozen: relax χ₀ on V_S, hang
it on V_T, propagate with the split-operator method, window the
autocorrelation, and read σ(E) = (4πE/3c)·2Re∫e^{i(E_v+E)t}S(t)w(t)dt — water's
machinery, reused.

## Result: the band does not see past ~2.3 Å

| V_T flattened beyond | change in the band |
| --- | --- |
| 2.10 Å | 0.10% |
| 1.95 Å | 0.73% |
| 1.85 Å | 11.3% |
| 1.78 Å | 99.3% |

So CCSD(T) + EOM-CCSD out to ~2.3 Å with an absorber beyond is enough for
σ(λ, T). This is what let `03_surfaces/12` stop at 2.40 Å — inside the 3A'/3A"
crossing rather than through it — and it is what makes the crossing optional
scope (product branching) rather than a blocker.

**The controls are the point.** A sensitivity test that reports "no
sensitivity" is worthless unless it can be shown to detect the sensitivity it
is looking for. Here an absorber placed at 1.6 Å changes the band by 13% and a
surface cut at 1.78 Å changes it by 99%, so the test can see. The band starts
to respond (≥1%) to an absorber from 1.7 Å and to a cut at 1.85 Å.

Both controls had to be repaired before they meant anything: the first absorber
was a cubic ramp over the whole grid rather than a fixed length, and the first
"positive control" was placed at 1.8 Å, outside χ₀ entirely — it could not have
moved the answer whatever the surface did.

The peak, 373 nm against the measured ~380, is **orientation only**: frozen OH
and bend, Condon, and an absolute σ sized from `01_method/03`'s f, which is
itself uncertain by 10-25×.

(One plotting bug worth remembering: χ was normalised by `max()` rather than
`abs().max()`, so a negative eigenvector got divided by ~1e-10 of tail noise.)
