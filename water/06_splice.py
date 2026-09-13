#!/usr/bin/env python3
"""
Tier 2d: assemble the raster and the OH curve into the surface the
propagator reads.

Four jobs:

  1. MIRROR. The raster computes only the r1 >= r2 triangle. V(r1,r2,th) ==
     V(r2,r1,th) fills the rest exactly, rather than approximately from
     independently converged points.

  2. REPAIR. Points with status != ok (typically scf_unconverged at large r1
     and tight theta) are holes. They are filled from their mirror partner
     where possible, otherwise interpolated.

  3. EXTEND. The raster stops at r1 = 3.4 A. The dynamics grid needs to run
     out to ~6 A so the wavepacket can dissociate into an absorbing boundary.
     Beyond the raster the surface is E_H + V_OH(r2) from 05_oh_diatomic.py.

  4. BLEND. Over a switching window the two are mixed with a smooth (C1)
     function, after shifting the asymptotic form onto the raster. Because
     SC-NEVPT2 is size consistent that shift should come out near zero, which
     is itself a check on the whole construction.

The both-bonds-long corner is O + 2H, a channel the photochemistry never
visits and which the asymptotic form does not describe. It is left as
computed and flagged; keep the dynamics grid away from it.

Usage:
    python 06_splice.py --grid pes_grid.csv --oh oh_curve.csv
"""
import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline, RegularGridInterpolator

HARTREE2EV = 27.211386245988


def load_raster(path, surface):
    """Read the CSV, keep ok rows, return sorted axes and filled 3D arrays."""
    rows = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} is empty")

    col_gs = f"e_gs_{surface}_Ha"
    col_ex = f"e_ex_{surface}_Ha"

    r_vals = sorted({round(float(x["r1_A"]), 6) for x in rows} |
                    {round(float(x["r2_A"]), 6) for x in rows})
    th_vals = sorted({round(float(x["theta_deg"]), 4) for x in rows})
    ri = {v: i for i, v in enumerate(r_vals)}
    ti = {v: i for i, v in enumerate(th_vals)}
    shape = (len(r_vals), len(r_vals), len(th_vals))

    gs = np.full(shape, np.nan)
    ex = np.full(shape, np.nan)
    mu = np.full(shape, np.nan)

    n_ok = n_bad = 0
    for row in rows:
        if row.get("status") != "ok":
            n_bad += 1
            continue
        try:
            i = ri[round(float(row["r1_A"]), 6)]
            j = ri[round(float(row["r2_A"]), 6)]
            k = ti[round(float(row["theta_deg"]), 4)]
            g, e, m = (float(row[col_gs]), float(row[col_ex]),
                       float(row["mu_abs"]))
        except (KeyError, ValueError):
            n_bad += 1
            continue
        if not np.isfinite([g, e, m]).all():
            n_bad += 1
            continue
        # mirror as we go: the triangle and its transpose
        for a, b in ((i, j), (j, i)):
            gs[a, b, k], ex[a, b, k], mu[a, b, k] = g, e, m
        n_ok += 1

    print(f"raster   {n_ok} usable points, {n_bad} skipped")
    print(f"         grid {len(r_vals)} x {len(r_vals)} x {len(th_vals)}"
          f"  ({np.isnan(gs).sum()} holes after mirroring)")
    return np.array(r_vals), np.array(th_vals), gs, ex, mu


def _fill_axis0(a, axis_r):
    """Interpolate NaNs along axis 0 at fixed (axis1, axis2)."""
    out = a.copy()
    for j in range(a.shape[1]):
        for k in range(a.shape[2]):
            col = out[:, j, k]
            bad = np.isnan(col)
            if not bad.any() or bad.all():
                continue
            good = ~bad
            col[bad] = np.interp(axis_r[bad], axis_r[good], col[good])
    return out


def fill_holes(a, axis_r, label=""):
    """
    Fill NaNs, then enforce V(r1,r2,th) == V(r2,r1,th) EXACTLY.

    Filling along r1 alone silently destroys the symmetry that mirroring
    established: the hole at (i,j) sees different neighbours than (j,i), so
    they get different values. Observed cost was a 1.6 eV asymmetry in the
    ground-state surface. Note that mirroring cannot repair these holes at
    all -- the raster computes each pair once, so a failed row empties both
    (i,j) and (j,i).

    So: interpolate along r1, independently along r2, average the two, and
    symmetrize. The average alone is nearly symmetric; the explicit
    symmetrization makes it exact.
    """
    n_holes = int(np.isnan(a).sum())
    f1 = _fill_axis0(a, axis_r)
    f2 = _fill_axis0(np.ascontiguousarray(a.transpose(1, 0, 2)),
                     axis_r).transpose(1, 0, 2)
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.stack([f1, f2]), axis=0)
    out = 0.5 * (out + out.transpose(1, 0, 2))
    left = int(np.isnan(out).sum())
    if n_holes:
        print(f"         {label}: filled {n_holes} holes "
              f"(r1 and r2 interpolation, then symmetrized)"
              + (f", {left} still empty" if left else ""))
    return out


def load_oh(path):
    """V_OH(r) spline and E_H, both from 05_oh_diatomic.py."""
    r, e, e_h = [], [], None
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            v = float(row["e_nevpt2_Ha"])
            if np.isfinite(v):
                r.append(float(row["r_A"]))
                e.append(v)
            if e_h is None and row.get("e_H_Ha"):
                e_h = float(row["e_H_Ha"])
    order = np.argsort(r)
    r, e = np.array(r)[order], np.array(e)[order]
    print(f"OH curve {len(r)} points, {r[0]:.3f}-{r[-1]:.3f} A, "
          f"E_H = {e_h:.8f} Ha")
    return CubicSpline(r, e, extrapolate=True), e_h, r


def switch(x, lo, hi):
    """Smooth C1 step, 0 below lo and 1 above hi."""
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", default="pes_grid.csv")
    p.add_argument("--oh", default="oh_curve.csv")
    p.add_argument("--surface", default="nevpt2", choices=["nevpt2", "casscf"])
    p.add_argument("--r-out-max", type=float, default=6.0,
                   help="outer edge of the dynamics grid")
    p.add_argument("--nr-out", type=int, default=160)
    p.add_argument("--ntheta-out", type=int, default=48)
    p.add_argument("--blend", type=float, nargs=2, default=[2.5, 3.3],
                   metavar=("LO", "HI"),
                   help="switching window in r; ab initio below LO, "
                        "asymptotic above HI")
    p.add_argument("--out", default="pes_surface.npz")
    p.add_argument("--png", default="pes_surface.png")
    args = p.parse_args()

    r_ax, th_ax, gs, ex, mu = load_raster(args.grid, args.surface)
    gs = fill_holes(gs, r_ax, "V_gs")
    ex = fill_holes(ex, r_ax, "V_ex")
    mu = fill_holes(mu, r_ax, "mu")

    v_oh, e_h, oh_r = load_oh(args.oh)

    # --- reference energies ------------------------------------------------
    # ground state minimum, used as the zero of energy throughout
    e_min = np.nanmin(gs)
    idx = np.unravel_index(np.nanargmin(gs), gs.shape)
    print(f"\nV_gs minimum {e_min:.8f} Ha at r1={r_ax[idx[0]]:.3f} "
          f"r2={r_ax[idx[1]]:.3f} theta={th_ax[idx[2]]:.1f}")

    # --- size-consistency check on the splice ------------------------------
    # At the largest r1 in the raster the ground state should already equal
    # E_H + V_OH(r2). SC-NEVPT2 is size consistent so the offset should be
    # near zero; a large value means something is wrong upstream.
    i_edge = len(r_ax) - 1
    k_ref = int(np.argmin(np.abs(th_ax - 104.48)))
    r2_probe = r_ax[(r_ax > 0.85) & (r_ax < 1.30)]
    offs = []
    for r2 in r2_probe:
        j = int(np.argmin(np.abs(r_ax - r2)))
        if np.isfinite(gs[i_edge, j, k_ref]):
            offs.append(gs[i_edge, j, k_ref] - (e_h + v_oh(r2)))
    offs = np.array(offs)
    shift = float(np.mean(offs)) if len(offs) else 0.0
    if len(offs):
        spread = (offs.max() - offs.min()) * HARTREE2EV * 1000
        print(f"splice   offset at r1={r_ax[i_edge]:.2f} A over {len(offs)} "
              f"probe points in r2:")
        print(f"           mean   {shift * HARTREE2EV * 1000:9.2f} meV  "
              f"(absorbs the basis difference between the raster and the "
              f"OH curve; magnitude is not diagnostic)")
        print(f"           spread {spread:9.2f} meV  "
              f"(THIS is the check: a size-consistent method should give a "
              f"CONSTANT offset)")
        if len(offs) < 3:
            print("           warning: fewer than 3 probe points, "
                  "the spread is not meaningful")
        elif spread > 50:
            print("           warning: offset varies with r2 by >50 meV; "
                  "the raster has not reached its asymptote, or the two "
                  "calculations disagree. Widen --r-max on the raster.")
    else:
        print("splice   no probe points available")

    # Degeneracy check on the AB INITIO surfaces at the inner edge of the
    # blend. After blending both states share one asymptote by construction,
    # so comparing them there is tautological; this is the real test that the
    # raster reaches the H + OH(X 2Pi) limit where the splice takes over.
    i_lo = int(np.argmin(np.abs(r_ax - args.blend[0])))
    split = np.abs(ex[i_lo, :, :] - gs[i_lo, :, :])
    with np.errstate(invalid="ignore"):
        print(f"         A'/A\" splitting at r1={r_ax[i_lo]:.2f} A "
              f"(ab initio, pre-blend): "
              f"max {np.nanmax(split) * HARTREE2EV:.4f} eV, "
              f"median {np.nanmedian(split) * HARTREE2EV:.4f} eV")

    # --- output grid -------------------------------------------------------
    r_out = np.linspace(r_ax[0], args.r_out_max, args.nr_out)
    th_out = np.linspace(th_ax[0], th_ax[-1], args.ntheta_out)

    interp_gs = RegularGridInterpolator((r_ax, r_ax, th_ax), gs,
                                        bounds_error=False, fill_value=None)
    interp_ex = RegularGridInterpolator((r_ax, r_ax, th_ax), ex,
                                        bounds_error=False, fill_value=None)
    interp_mu = RegularGridInterpolator((r_ax, r_ax, th_ax), mu,
                                        bounds_error=False, fill_value=None)

    R1, R2, TH = np.meshgrid(r_out, r_out, th_out, indexing="ij")
    pts = np.stack([np.clip(R1, r_ax[0], r_ax[-1]),
                    np.clip(R2, r_ax[0], r_ax[-1]),
                    np.clip(TH, th_ax[0], th_ax[-1])], axis=-1)

    V_ab_gs = interp_gs(pts)
    V_ab_ex = interp_ex(pts)
    MU = interp_mu(pts)

    # asymptotic forms, shifted onto the raster
    lo, hi = args.blend
    A1 = e_h + v_oh(np.clip(R2, oh_r[0], oh_r[-1])) + shift  # r1 long
    A2 = e_h + v_oh(np.clip(R1, oh_r[0], oh_r[-1])) + shift  # r2 long
    w1 = switch(R1, lo, hi)
    w2 = switch(R2, lo, hi)

    # both long is O + 2H: not described by either asymptote. Flag it.
    both = (w1 > 0.5) & (w2 > 0.5)

    denom = np.clip(w1 + w2, 1e-12, None)
    V_as = (w1 * A1 + w2 * A2) / denom
    w = np.clip(w1 + w2 - w1 * w2, 0.0, 1.0)

    V_gs = (1.0 - w) * V_ab_gs + w * V_as
    # the excited surface is degenerate with the ground state at the
    # asymptote (both go to H + OH(X 2Pi)), so it shares the same limit
    V_ex = (1.0 - w) * V_ab_ex + w * V_as
    MU = (1.0 - w) * MU  # transition dipole dies as the fragments separate

    V_gs -= e_min
    V_ex -= e_min

    print(f"\noutput   {args.nr_out} x {args.nr_out} x {args.ntheta_out}"
          f" on r = {r_out[0]:.2f}-{r_out[-1]:.2f} A")
    print(f"         blend window {lo}-{hi} A")
    print(f"         {both.sum()} points in the O+2H corner (flagged)")


    # symmetry must survive the blend as well
    for name, V in (("V_gs", V_gs), ("V_ex", V_ex), ("mu", MU)):
        asym = np.abs(V - V.transpose(1, 0, 2)).max()
        unit = HARTREE2EV if name != "mu" else 1.0
        tag = "eV" if name != "mu" else "a.u."
        flag = "  <-- NOT SYMMETRIC" if asym * unit > 1e-6 else ""
        print(f"         symmetry {name}: max |V - V^T| = "
              f"{asym * unit:.3e} {tag}{flag}")

    np.savez_compressed(
        args.out, r=r_out, theta_deg=th_out,
        V_gs=V_gs, V_ex=V_ex, mu=MU, o2h_mask=both,
        e_min_Ha=e_min, e_H_Ha=e_h, splice_shift_Ha=shift,
        blend_lo=lo, blend_hi=hi, surface=args.surface)
    print(f"\nwrote {args.out}")

    # --- diagnostics -------------------------------------------------------
    k = int(np.argmin(np.abs(th_out - 104.48)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    lv = np.linspace(0, 12, 25)
    for a, V, name in ((ax[0], V_gs, "ground $\\tilde{X}$"),
                       (ax[1], V_ex, "excited $\\tilde{A}$")):
        c = a.contourf(r_out, r_out, V[:, :, k].T * HARTREE2EV,
                       levels=lv, extend="max")
        a.axvline(lo, ls=":", c="w", lw=1)
        a.axvline(hi, ls="--", c="w", lw=1)
        a.set_xlabel("r1 / A")
        a.set_ylabel("r2 / A")
        a.set_title(f"{name}, theta={th_out[k]:.0f} deg")
        fig.colorbar(c, ax=a, label="eV")

    j = int(np.argmin(np.abs(r_out - 0.97)))
    ax[2].plot(r_out, V_gs[:, j, k] * HARTREE2EV, label="$\\tilde{X}$")
    ax[2].plot(r_out, V_ex[:, j, k] * HARTREE2EV, label="$\\tilde{A}$")
    ax[2].axvspan(lo, hi, color="0.85", label="blend")
    ax[2].set_xlabel("r1 / A")
    ax[2].set_ylabel("V / eV")
    ax[2].set_title(f"cut at r2={r_out[j]:.2f} A")
    ax[2].legend()
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
