#!/usr/bin/env python3
"""
Phase 1, step 4: splice raster + outer shell + asymptote into one a 3A"
surface, and plot it.

Three pieces, each valid where the others are not:

  12  CCSD(T) + EOM-CCSD raster, 1.40-2.40 A. The accurate part, and the only
      part the band can see (10): trustworthy to 2.29 A, where a 3A' crosses.
  13  symmetry-forced 3A" SA-CASSCF(10,6) + SC-NEVPT2 shell, 2.20-3.60 A,
      overlapping the raster on purpose.
  14  E(Cl, 2P) + V_OH(r_OH), the exact asymptote, in both method families.

JOINING THE SCALES. NEVPT2 recovers less correlation, so the shell sits above
the raster by a large constant -- irrelevant on its own. What matters is
whether that offset is CONSTANT. 14 measured the same offset on the fragments
alone: 0.791-0.855 eV across the O-H curve, spread 64 meV over the full range
and 36 meV over the 0.80-1.25 A used here, varying smoothly rather than
scattering. Water's analogous offset held to 13 meV.

So this aligns the shell to the raster with an offset fitted PER (r_OH, angle)
over the overlap region, not one global number, which absorbs that smooth
variation exactly. Both are computed and compared: if the per-column offsets
have a small spread the distinction does not matter, and the report says so.
The spread of each column's offset ACROSS the overlap radii is the real test
of whether the two methods agree on the shape, since a shape disagreement
cannot be absorbed by any offset.

Then the shifted shell is blended onto E(Cl) + V_OH(r_OH) at long range, on
the raster's (UCCSD(T)) energy scale, and the residual at the join is the
analogue of water's splice check.

Blends are cosine switches, so the result is continuous and has no kink at
either seam -- the propagation would scatter off one.

Outputs: an NPZ with V_S (bound region, for chi_0) and V_T on an extended
grid, plus contour plots.

Usage:
    python 15_splice.py 2>&1 | tee logs/splice.log
"""
import argparse
import csv

import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path

# Data files live in HOCl/data/, one level up from this approach directory,
# so the defaults below work no matter where the script is invoked from.
DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2EV = 27.211386245988


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def grid_from(rows, keys):
    return [np.array(sorted({round(float(r[k]), 4) for r in rows})) for k in keys]


def fill(rows, axes, keys, value, ok=lambda r: True):
    idx = [{v: i for i, v in enumerate(a)} for a in axes]
    A = np.full([len(a) for a in axes], np.nan)
    for r in rows:
        if not ok(r):
            continue
        try:
            ijk = tuple(idx[n][round(float(r[k]), 4)] for n, k in enumerate(keys))
            A[ijk] = value(r)
        except (KeyError, ValueError):
            continue
    return A


def cosine_blend(x, lo, hi):
    """0 below lo, 1 above hi, smooth in between."""
    s = np.clip((np.asarray(x, float) - lo) / (hi - lo), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def interp_axis(A, src, dst, axis):
    """Cubic-spline A from src to dst along one axis."""
    A = np.moveaxis(A, axis, 0)
    out = np.empty((len(dst),) + A.shape[1:])
    flat = A.reshape(len(src), -1)
    res = np.empty((len(dst), flat.shape[1]))
    for c in range(flat.shape[1]):
        res[:, c] = CubicSpline(src, flat[:, c], bc_type="natural")(dst)
    out = res.reshape((len(dst),) + A.shape[1:])
    return np.moveaxis(out, 0, axis)


def local_resid_mev(A, axis):
    """|y_i - cubic through its 4 neighbours| along one axis, meV."""
    A = np.moveaxis(A, axis, 0)
    out = np.full(A.shape, np.nan)
    for i in range(2, A.shape[0] - 2):
        out[i] = np.abs(A[i] - (-A[i - 2] + 4 * A[i - 1] + 4 * A[i + 1]
                                - A[i + 2]) / 6.0) * 1000
    return np.moveaxis(out, 0, axis)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raster", default=str(DATA / "hocl_pes_raster.csv"))
    p.add_argument("--shell", default=str(DATA / "hocl_outer_shell.csv"))
    p.add_argument("--fragments", default=str(DATA / "hocl_fragments.csv"))
    p.add_argument("--e-cl-cc", type=float, default=-461.00908095,
                   help="E(Cl) UCCSD(T)/x2c, from 14's log")
    p.add_argument("--join-in", type=float, default=2.20,
                   help="raster->shell blend starts here")
    p.add_argument("--join-out", type=float, default=2.40,
                   help="...and ends here; the raster's last point")
    p.add_argument("--asym-in", type=float, default=3.20,
                   help="shell->asymptote blend starts here")
    p.add_argument("--asym-out", type=float, default=3.60,
                   help="...and ends at the shell's last point")
    p.add_argument("--rmax", type=float, default=6.00,
                   help="extend the surface to here on the asymptote")
    p.add_argument("--npz", default=str(DATA / "hocl_surfaces.npz"))
    p.add_argument("--png", default=str(DATA / "hocl_surfaces.png"))
    args = p.parse_args()

    ras = read_csv(args.raster)
    she = read_csv(args.shell)
    fra = read_csv(args.fragments)
    keys = ("r_ocl_A", "r_oh_A", "theta_deg")

    r_ras, roh, th = grid_from(ras, keys)
    r_she, roh_s, th_s = grid_from(she, keys)
    print("=" * 78)
    print('== HOX Phase 1 -- splicing the a 3A" surface')
    print("=" * 78)
    print(f"  raster {len(r_ras)}x{len(roh)}x{len(th)} over r(O-Cl) "
          f"{r_ras[0]}-{r_ras[-1]} A")
    print(f"  shell  {len(r_she)}x{len(roh_s)}x{len(th_s)} over "
          f"{r_she[0]}-{r_she[-1]} A")
    if not np.array_equal(roh, roh_s):
        raise SystemExit("raster and shell disagree on the r(O-H) grid")

    VS = fill(ras, (r_ras, roh, th), keys,
              lambda r: float(r["e_ccsdt_Ha"]))
    VT_in = fill(ras, (r_ras, roh, th), keys,
                 lambda r: float(r["e_ccsdt_Ha"]) + float(r["omega_app_eV"])
                 / HARTREE2EV)
    VT_sh = fill(she, (r_she, roh, th_s), keys,
                 lambda r: float(r["e_nevpt2_Ha"]))
    print(f"  missing: raster {np.isnan(VT_in).sum()}, "
          f"shell {np.isnan(VT_sh).sum()}")

    # shell onto the raster's angular grid
    VT_sh = interp_axis(VT_sh, th_s, th, axis=2)

    # ---- offsets over the overlap, per (r_OH, angle) and globally
    ov = [r for r in r_ras if r >= args.join_in - 1e-9 and r in set(r_she)]
    print(f"\n  overlap radii used for the offset: "
          f"{', '.join(f'{v:.2f}' for v in ov)} A")
    i_ras = [int(np.argmin(abs(r_ras - v))) for v in ov]
    i_she = [int(np.argmin(abs(r_she - v))) for v in ov]
    dif = VT_in[i_ras] - VT_sh[i_she]                     # (nov, nroh, nth)
    off_col = dif.mean(axis=0)                            # per (r_OH, angle)
    spread_R = (dif.max(axis=0) - dif.min(axis=0)) * HARTREE2EV * 1000
    print(f"  offset (raster - shell): {off_col.min() * HARTREE2EV:.3f} to "
          f"{off_col.max() * HARTREE2EV:.3f} eV, "
          f"column spread {(off_col.max() - off_col.min()) * HARTREE2EV * 1000:.0f} meV")
    print(f"  spread of each column ACROSS the overlap radii: max "
          f"{np.nanmax(spread_R):.0f} meV, median {np.nanmedian(spread_R):.0f} meV")
    print("  (that second number is the shape test: an offset cannot absorb a "
          "shape disagreement)")
    glob = float(np.nanmean(off_col))
    d_glob = float(np.nanmax(np.abs(off_col - glob)) * HARTREE2EV * 1000)
    print(f"  a single global offset would misplace columns by up to "
          f"{d_glob:.0f} meV; using per-column offsets")

    VT_sh_al = VT_sh + off_col[None, :, :]

    # ---- asymptote on the raster's scale, from 14
    fr = np.array([[float(r["r_A"]), float(r["e_uccsdt_Ha"])] for r in fra])
    v_oh = CubicSpline(fr[:, 0], fr[:, 1], bc_type="natural")(roh)
    asym = args.e_cl_cc + v_oh                             # (nroh,)
    resid = (VT_sh_al[-1] - asym[:, None]) * HARTREE2EV * 1000
    print(f"\n  shell (aligned) at {r_she[-1]:.2f} A minus E(Cl)+V_OH: "
          f"{np.nanmin(resid):.0f} to {np.nanmax(resid):.0f} meV")
    print("  (water's analogous splice check held to 13 meV; here the shell is "
          "still on its\n   van der Waals tail at 3.6 A, so a residual of tens "
          "of meV is expected)")

    # ---- assemble on an extended radial grid
    step = float(r_ras[1] - r_ras[0])
    r_out = np.round(np.arange(r_ras[0], args.rmax + 0.5 * step, step), 4)
    V = np.empty((len(r_out), len(roh), len(th)))
    ras_s = interp_axis(VT_in, r_ras, np.clip(r_out, r_ras[0], r_ras[-1]), 0)
    she_s = interp_axis(VT_sh_al, r_she, np.clip(r_out, r_she[0], r_she[-1]), 0)
    w1 = cosine_blend(r_out, args.join_in, args.join_out)[:, None, None]
    V = (1 - w1) * ras_s + w1 * she_s
    w2 = cosine_blend(r_out, args.asym_in, args.asym_out)[:, None, None]
    V = (1 - w2) * V + w2 * np.broadcast_to(asym[None, :, None], V.shape)

    print(f"\n  assembled grid {V.shape} over r(O-Cl) {r_out[0]}-{r_out[-1]} A")
    for name, lo, hi in (("raster->shell", args.join_in, args.join_out),
                         ("shell->asymptote", args.asym_in, args.asym_out)):
        m = (r_out >= lo - 2 * step) & (r_out <= hi + 2 * step)
        res = local_resid_mev(V, 0)[m]
        print(f"  {name:18s} blend {lo}-{hi} A: max local residual "
              f"{np.nanmax(res):.1f} meV")
    res_all = local_resid_mev(V, 0)
    band = r_out <= 2.30
    print(f"  whole surface: max local residual {np.nanmax(res_all):.1f} meV; "
          f"within 2.30 A {np.nanmax(res_all[band]):.1f} meV")
    print("  (those residuals mostly measure the spline interpolation and the")
    print("   cosine blends, which are smooth by construction. The test that")
    print("   can actually fail is next.)")

    # Do the two methods agree on the SLOPE where they are joined? A blend
    # hides a slope mismatch by spreading it over the window, so compare the
    # pieces directly, before blending.
    print()
    print("  slope agreement at the seam (the check a blend cannot fake):")
    # np.gradient, not a central difference by hand: both seam radii are at
    # an array edge (2.20 starts the shell, 2.40 ends the raster), and the
    # hand-rolled version skipped exactly those points and printed nothing.
    g_ras = np.gradient(VT_in, r_ras, axis=0) * HARTREE2EV
    g_she = np.gradient(VT_sh_al, r_she, axis=0) * HARTREE2EV
    width = args.join_out - args.join_in
    for rj in ov:
        ia = int(np.argmin(abs(r_ras - rj)))
        ib = int(np.argmin(abs(r_she - rj)))
        edge = (ia in (0, len(r_ras) - 1)) or (ib in (0, len(r_she) - 1))
        s_ras, s_she = g_ras[ia], g_she[ib]
        d = s_ras - s_she
        worst = np.nanmax(np.abs(d))
        print(f"    at {rj:.2f} A: raster {np.nanmean(s_ras):+.3f}, shell "
              f"{np.nanmean(s_she):+.3f} eV/A; worst column mismatch "
              f"{worst:.3f} eV/A = {worst * width * 1000:.0f} meV over the "
              f"{width:.2f} A blend"
              + ("   (one-sided derivative at a grid edge)" if edge else ""))

    np.savez_compressed(args.npz, r_ocl_A=r_out, r_oh_A=roh, theta_deg=th,
                        V_T_Ha=V, r_ocl_bound_A=r_ras, V_S_Ha=VS,
                        asymptote_Ha=asym, offset_Ha=off_col,
                        e_cl_cc_Ha=args.e_cl_cc)
    print(f"\n  wrote {args.npz}")

    # ---- plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        j = int(np.argmin(abs(roh - 0.9644)))
        k = int(np.argmin(abs(th - 105.0)))
        ev = lambda A: (A - np.nanmin(VS)) * HARTREE2EV
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        # The triplet's repulsive wall runs to ~7 eV and swamps everything the
        # band samples, so clip the radial range and the levels to the region
        # that matters, and draw labelled contour lines over the fill.
        top = r_out <= 4.0
        lv = np.arange(1.5, 5.01, 0.125)

        c = ax[0, 0].contourf(th, r_out[top], ev(V[top][:, j, :]), levels=lv,
                              extend="both")
        cl = ax[0, 0].contour(th, r_out[top], ev(V[top][:, j, :]),
                              levels=np.arange(2.0, 5.01, 0.5), colors="k",
                              linewidths=0.4)
        ax[0, 0].clabel(cl, fmt="%.1f", fontsize=6)
        ax[0, 0].set_title(f'a 3A" at r(O-H) = {roh[j]:.3f} A')
        ax[0, 0].set_xlabel("angle / deg"); ax[0, 0].set_ylabel("r(O-Cl) / A")
        fig.colorbar(c, ax=ax[0, 0], label="E / eV")

        c = ax[0, 1].contourf(roh, r_out[top], ev(V[top][:, :, k]), levels=lv,
                              extend="both")
        cl = ax[0, 1].contour(roh, r_out[top], ev(V[top][:, :, k]),
                              levels=np.arange(2.0, 5.01, 0.5), colors="k",
                              linewidths=0.4)
        ax[0, 1].clabel(cl, fmt="%.1f", fontsize=6)
        ax[0, 1].plot([roh[j]], [r_ras[int(np.argmin(VS[:, j, k]))]], "r*",
                      ms=9, label="ground-state minimum")
        ax[0, 1].legend(fontsize=7, loc="upper right")
        ax[0, 1].set_title(f'a 3A" at angle = {th[k]:.0f} deg')
        ax[0, 1].set_xlabel("r(O-H) / A"); ax[0, 1].set_ylabel("r(O-Cl) / A")
        fig.colorbar(c, ax=ax[0, 1], label="E / eV")

        c = ax[1, 0].contourf(roh, r_ras, ev(VS[:, :, k]),
                              levels=np.arange(0, 3.01, 0.1), extend="max")
        cl = ax[1, 0].contour(roh, r_ras, ev(VS[:, :, k]),
                              levels=np.arange(0.25, 3.01, 0.5), colors="k",
                              linewidths=0.4)
        ax[1, 0].clabel(cl, fmt="%.2f", fontsize=6)
        ax[1, 0].set_title(f"X 1A' (ground) at angle = {th[k]:.0f} deg")
        ax[1, 0].set_xlabel("r(O-H) / A"); ax[1, 0].set_ylabel("r(O-Cl) / A")
        fig.colorbar(c, ax=ax[1, 0], label="E / eV")

        ax[1, 1].plot(r_ras, ev(VT_in[:, j, k]), "o-", ms=3, label="raster (CC+EOM)")
        ax[1, 1].plot(r_she, ev(VT_sh_al[:, j, k]), "s-", ms=3,
                      label="shell (NEVPT2, aligned)")
        ax[1, 1].plot(r_out, ev(V[:, j, k]), "k-", lw=1, label="spliced")
        ax[1, 1].axhline(ev(asym[j]), color="grey", ls=":", label="E(Cl)+V_OH")
        for v in (args.join_in, args.join_out, args.asym_in, args.asym_out):
            ax[1, 1].axvline(v, color="grey", lw=0.5)
        ax[1, 1].set_xlabel("r(O-Cl) / A"); ax[1, 1].set_ylabel("E / eV")
        ax[1, 1].set_title("cut through the seams")
        ax[1, 1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.png, dpi=140)
        print(f"  wrote {args.png}")
    except Exception as exc:
        print(f"  plot skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
