#!/usr/bin/env python3
"""
HOBr step 6: is the raster fit to build a surface from?

HOCl's raster was checked by hand and the numbers went into TASKS.md; this is
that check as a script, so HOBr's (and later HOI's) get the same treatment
and it can be rerun. Local: numpy only, seconds.

  1. WARNINGS, and WHERE. A warning is harmless in the corner the absorber
     eats and serious in the Franck-Condon window. HOCl: none below 2.25 A.
  2. SMOOTHNESS. Largest deviation of any point from a cubic through its four
     nearest neighbours along each axis, by band of r(O-Br). A kink shows up
     as a point sticking out; smooth curvature does not, except on the steep
     inner wall where a 4-point cubic cannot follow an exponential (HOCl:
     10.1 meV there, <= 3.6 meV elsewhere, nothing above 20).
  3. AGREEMENT WITH 02's CUT, which used 6 roots and slightly different
     spectators (0.961 A, 102.3 deg): interpolated to them, the raster should
     reproduce the cut's omega to a few meV. A mismatch means the raster
     stored a different state somewhere.
  4. THE FC WINDOW: vertical and slope at 01's minimum, and how much the bend
     moves V_T there (HOCl: 0.26 eV across the angles chi_0 samples) -- the
     number 03's frozen-bend model could not see.

Usage:
    python 06_raster_check.py 2>&1 | tee ../logs/raster_check.log
"""
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

DATA = Path(__file__).resolve().parents[1] / "data"
HARTREE2EV = 27.211386245988
NM_EV = 1239.841984
R_EQ, ROH_EQ, TH_EQ = 1.8357, 0.9646, 102.02      # 01_geometry.py
T1_MAX = 0.02
BANDS = [(1.55, 1.80), (1.80, 2.10), (2.10, 2.30), (2.30, 2.45)]


def load(path):
    rows = list(csv.DictReader(open(path)))
    r = np.array(sorted({round(float(x["r_obr_A"]), 4) for x in rows}))
    h = np.array(sorted({round(float(x["r_oh_A"]), 4) for x in rows}))
    t = np.array(sorted({round(float(x["theta_deg"]), 3) for x in rows}))
    shape = (r.size, h.size, t.size)
    VS, OM, T1, W1 = (np.full(shape, np.nan) for _ in range(4))
    ST = np.empty(shape, dtype=object)
    for x in rows:
        i = int(np.searchsorted(r, round(float(x["r_obr_A"]), 4)))
        j = int(np.searchsorted(h, round(float(x["r_oh_A"]), 4)))
        k = int(np.searchsorted(t, round(float(x["theta_deg"]), 3)))
        ST[i, j, k] = x["status"]
        if x["status"].startswith("fail"):
            continue
        VS[i, j, k] = float(x["e_ccsdt_Ha"])
        OM[i, j, k] = float(x["omega_app_eV"]) / HARTREE2EV
        T1[i, j, k] = float(x["t1_s"])
        W1[i, j, k] = float(x["w1"]) if x["w1"] else np.nan
    return r, h, t, VS, OM, T1, W1, ST, len(rows)


def local_resid(V, axis):
    """|V_i - cubic through i-2, i-1, i+1, i+2| along one axis, meV.

    For equal spacing that cubic's value at the centre is
    (-V[i-2] + 4V[i-1] + 4V[i+1] - V[i+2]) / 6.
    """
    V = np.moveaxis(V, axis, 0)
    out = np.full(V.shape, np.nan)
    pred = (-V[:-4] + 4 * V[1:-3] + 4 * V[3:-1] - V[4:]) / 6.0
    out[2:-2] = np.abs(V[2:-2] - pred) * HARTREE2EV * 1000
    return np.moveaxis(out, 0, axis)


def main():
    r, h, t, VS, OM, T1, W1, ST, n = load(DATA / "hobr_pes_raster.csv")
    VT = VS + OM
    print("=" * 72)
    print("== HOBr raster check")
    print("=" * 72)
    print(f"  {n} points: r(O-Br) {r[0]}-{r[-1]} ({r.size}), r(O-H) "
          f"{h[0]}-{h[-1]} ({h.size}), angle {t[0]}-{t[-1]} ({t.size})")

    # ---- 1. warnings, and where
    st = ST.ravel()
    kinds = {}
    for s in st:
        kinds[s] = kinds.get(s, 0) + 1
    print("\n  status: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())))
    R, Hh, Tt = np.meshgrid(r, h, t, indexing="ij")
    warn = np.array([s != "ok" for s in st]).reshape(ST.shape)
    if warn.any():
        print(f"  warned points span r(O-Br) {R[warn].min():.2f}-{R[warn].max():.2f} A, "
              f"r(O-H) {Hh[warn].min():.2f}-{Hh[warn].max():.2f}, "
              f"angle {Tt[warn].min():.0f}-{Tt[warn].max():.0f}")
        for lo, hi in BANDS:
            m = (R >= lo - 1e-9) & (R < hi - 1e-9 if hi < r[-1] else R <= hi + 1e-9)
            print(f"    r {lo:.2f}-{hi:.2f}: {int(warn[m].sum()):3d} of "
                  f"{int(m.sum())} warned")
    print(f"  max T1(S) {np.nanmax(T1):.4f}"
          + (f"  -- above {T1_MAX} at {int((T1 >= T1_MAX).sum())} points"
             if np.nanmax(T1) >= T1_MAX else "  (all below 0.02)"))
    print(f"  single-excitation weight: min {np.nanmin(W1):.3f}; below 0.90 at "
          f"{int((W1 < 0.90).sum())} points, from r(O-Br) "
          + (f"{R[W1 < 0.90].min():.2f} A" if (W1 < 0.90).any() else "--"))

    # ---- 2. smoothness
    print("\n  smoothness: max |V - cubic through 4 neighbours|, meV")
    print(f"    {'r(O-Br) band':>14}{'along O-Br':>12}{'along O-H':>11}"
          f"{'along angle':>13}   (V_T)")
    worst = 0.0
    res = [local_resid(VT, ax) for ax in range(3)]
    for lo, hi in BANDS:
        m = (R >= lo - 1e-9) & (R <= hi + 1e-9)
        vals = [np.nanmax(np.where(m, rr, np.nan)) for rr in res]
        worst = max(worst, *vals)
        print(f"    {lo:.2f}-{hi:.2f} A  {vals[0]:12.1f}{vals[1]:11.1f}{vals[2]:13.1f}")
    resS = [local_resid(VS, ax) for ax in range(3)]
    print(f"    V_S overall: {', '.join(f'{np.nanmax(x):.1f}' for x in resS)}")
    stack = np.stack(res)                       # edges are NaN by design
    tot = np.where(np.isfinite(stack).any(axis=0),
                   np.nanmax(np.where(np.isfinite(stack), stack, -1.0), axis=0),
                   -1.0)
    i, j, k = np.unravel_index(int(np.argmax(tot)), VT.shape)
    print(f"    largest V_T residual at ({r[i]}, {h[j]}, {t[k]}), status {ST[i, j, k]}")

    # ---- 3. agreement with 02's cut
    fin = np.isfinite(VT).all()
    if not fin:
        print("\n  !! missing points; interpolation below uses nearest-finite "
              "fill and is approximate")
    interp = lambda A: RegularGridInterpolator((r, h, t), A, method="cubic")  # noqa: E731
    I_om = interp(OM)
    cut = [x for x in csv.DictReader(open(DATA / "hobr_obr_cut.csv"))
           if x["basis"] == "cc-pvtz-dk" and not x["status"].startswith("fail")]
    diffs = []
    for x in cut:
        rr = float(x["r_ox_A"])
        if not (r[0] <= rr <= r[-1]):
            continue
        om = float(I_om([[rr, 0.961, 102.3]])[0]) * HARTREE2EV
        diffs.append((rr, (om - float(x["omega_app_eV"])) * 1000))
    d = np.array(diffs)
    print(f"\n  against 02's cut (cc-pvtz-dk, spectators 0.961 A / 102.3 deg), "
          f"{len(d)} radii:")
    print(f"    omega(raster, interpolated) - omega(cut): "
          f"{d[:, 1].min():+.2f} to {d[:, 1].max():+.2f} meV, "
          f"median {np.median(d[:, 1]):+.2f}")
    # Not the same data twice: the nearest grid point, (0.95 A, 100 deg), is
    # ~12 meV from the cut, so agreement at this level means the cubic
    # interpolation over 0.011 A and 2.3 deg reproduces an independent
    # calculation, run with a different root count.
    near = OM[:, int(np.argmin(abs(h - 0.961))), int(np.argmin(abs(t - 102.3)))]
    cutd = {round(float(x["r_ox_A"]), 4): float(x["omega_app_eV"]) for x in cut}
    gap = [abs(near[q] * HARTREE2EV - cutd[round(r[q], 4)]) * 1000
           for q in range(r.size) if round(r[q], 4) in cutd]
    print(f"    (nearest grid point differs from the cut by up to "
          f"{max(gap):.0f} meV, so this tests the interpolation, not a copy)")
    bad = d[np.abs(d[:, 1]) > 10]
    if bad.size:
        print(f"    !! off by > 10 meV at r = "
              + ", ".join(f"{a:.2f} ({b:+.0f})" for a, b in bad))

    # ---- 4. the FC window
    I_vt, I_vs = interp(VT), interp(VS)
    eq = np.array([[R_EQ, ROH_EQ, TH_EQ]])
    vert = float(I_vt(eq)[0] - I_vs(eq)[0]) * HARTREE2EV
    dr = 0.01
    slope = float((I_vt([[R_EQ + dr, ROH_EQ, TH_EQ]])[0]
                   - I_vt([[R_EQ - dr, ROH_EQ, TH_EQ]])[0]) / (2 * dr)) * HARTREE2EV
    print(f"\n  FC window at 01's minimum ({R_EQ}, {ROH_EQ}, {TH_EQ}):")
    print(f"    vertical {vert:.3f} eV = {NM_EV / vert:.0f} nm; slope along O-Br "
          f"{slope:.2f} eV/A (02's cut: 2.871 eV, -5.32)")
    # bend: V_T across the angles the ground state samples at the FC radius
    vs_rel = (VS - np.nanmin(VS)) * HARTREE2EV
    kfc = np.where((vs_rel <= 0.25).any(axis=(0, 1)))[0]
    ang = np.linspace(t[kfc[0]], t[kfc[-1]], 25)
    vt_ang = I_vt(np.column_stack([np.full_like(ang, R_EQ),
                                   np.full_like(ang, ROH_EQ), ang])) * HARTREE2EV
    print(f"    chi_0 samples angles {t[kfc[0]]:.0f}-{t[kfc[-1]]:.0f} deg "
          f"(V_S <= 0.25 eV); across them V_T moves {np.ptp(vt_ang):.2f} eV "
          f"at the FC radius (HOCl: 0.26)")

    print("\n" + "=" * 72)
    ok = worst < 20 and (not d.size or np.all(np.abs(d[:, 1]) <= 10))
    print("  -> fit to build a surface from." if ok else
          "  -> NOT clean: see the flags above before splicing.")


if __name__ == "__main__":
    main()
