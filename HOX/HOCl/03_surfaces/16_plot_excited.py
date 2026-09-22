#!/usr/bin/env python3
"""
The a 3A" surface on its own, plotted properly.

15's figure has two triplet panels, but they share one colour scale with the
repulsive wall, which spans 5 eV and flattens everything else -- the angular
panel there comes out nearly featureless. This script plots only the excited
state, with a scale per panel, and adds the two things that figure does not
show: the shape along the dissociation coordinate, and the vertical gap
V_T - V_S, which is what the band actually maps.

ENERGY ZERO is the ground-state minimum throughout, so every contour reads
directly as an excitation energy: the ~3.4 eV vertical that 03, 04 and 09 all
agree on should appear right at the ground-state minimum marker.

Local: numpy + matplotlib, no pyscf. Reads 15's output.

Usage:
    python 16_plot_excited.py
    python 16_plot_excited.py --rmax 4.0 --angle 105
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
HARTREE2EV = 27.211386245988


def nearest(grid, value):
    """Index of the grid point closest to value."""
    return int(np.argmin(np.abs(np.asarray(grid) - value)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", default=str(DATA / "hocl_surfaces.npz"))
    p.add_argument("--png", default=str(DATA / "hocl_excited_pes.png"))
    p.add_argument("--rmax", type=float, default=4.0,
                   help="outer r(O-Cl) shown in the contour panels; the grid "
                        "runs to 6.0 A but it is flat past ~4")
    p.add_argument("--angle", type=float, default=105.0)
    p.add_argument("--roh", type=float, default=0.95)
    args = p.parse_args()

    d = np.load(args.npz)
    r_t, roh, th = d["r_ocl_A"], d["r_oh_A"], d["theta_deg"]
    r_s = d["r_ocl_bound_A"]
    VT, VS = d["V_T_Ha"], d["V_S_Ha"]
    asym = d["asymptote_Ha"]

    # Ground-state minimum: the zero for every panel.
    i0, j0, k0 = np.unravel_index(np.argmin(VS), VS.shape)
    e0 = VS[i0, j0, k0]
    ev = lambda a: (a - e0) * HARTREE2EV           # noqa: E731
    print(f"  ground-state minimum at r(O-Cl) = {r_s[i0]:.3f} A, "
          f"r(O-H) = {roh[j0]:.3f} A, angle = {th[k0]:.0f} deg")

    # The vertical gap: the one number this figure must reproduce.
    it = nearest(r_t, r_s[i0])
    # 09 quotes 3.432 eV at r(O-Cl) = 1.6891 A. The grid minimum is at 1.700,
    # and the surface is steep there, so the two are only comparable once that
    # displacement is taken out.
    vert = ev(VT[it, j0, k0])
    slope0 = np.gradient(ev(VT[:, j0, k0]), r_t)[it]
    print(f"  vertical a 3A\" <- X 1A' there: {vert:.3f} eV; "
          f"09 gives 3.432 at 1.6891 A")
    print(f"  the 0.011 A between those geometries is worth "
          f"{abs(slope0) * (r_s[i0] - 1.6891) * 1000:.0f} meV at "
          f"{slope0:.2f} eV/A, which is the difference")

    ka = nearest(th, args.angle)
    jh = nearest(roh, args.roh)
    imax = nearest(r_t, args.rmax)
    sl = slice(0, imax + 1)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('HOCl  a 3A"  (energies relative to the X 1A\' minimum)',
                 fontsize=13)

    # --- A: the dissociation plane, r(O-Cl) x r(O-H) --------------------
    # Clipped at 6 eV: the wall reaches 9 eV at 1.40 A and would otherwise
    # take most of the colour range for a region no wavepacket visits.
    lv = np.arange(1.5, 6.01, 0.25)
    a = ax[0, 0]
    c = a.contourf(roh, r_t[sl], ev(VT[sl][:, :, ka]), levels=lv,
                   cmap="viridis", extend="both")
    cs = a.contour(roh, r_t[sl], ev(VT[sl][:, :, ka]),
                   levels=np.arange(2.0, 6.01, 0.5), colors="k", linewidths=0.5)
    a.clabel(cs, fmt="%.1f", fontsize=7)
    # Where chi_0 lives, so the Franck-Condon window is visible rather than
    # implied: the V_S = 0.25 eV contour, roughly the zero-point turning region.
    a.contour(roh, r_s, ev(VS[:, :, ka]) - ev(VS).min(), levels=[0.25],
              colors="w", linewidths=1.6, linestyles="--")
    a.plot(roh[j0], r_s[i0], "r*", ms=13, zorder=5)
    a.set_xlabel("r(O-H) / A")
    a.set_ylabel("r(O-Cl) / A")
    a.set_title(f"dissociation plane, angle = {th[ka]:.0f} deg\n"
                "white dashed: ground-state V = 0.25 eV (where chi_0 sits)",
                fontsize=10)
    fig.colorbar(c, ax=a, label="E / eV")

    # --- B: the angular dependence, as a DEVIATION -------------------
    # Plotting V_T itself here fails: its angular spread is a few tenths of an
    # eV against a 5 eV wall in the same panel, so a shared range paints a
    # blank square -- which is what 15's figure shows, and reading that blank
    # as "the surface barely depends on angle" is how a wrong claim got into
    # the notes. What is wanted is how much the bend MOVES the surface, so
    # plot V_T(angle) - V_T(105 deg) on a range set by the region the packet
    # crosses, not by the inner wall.
    a = ax[0, 1]
    kref = nearest(th, 105.0)
    D = ev(VT[sl][:, jh, :]) - ev(VT[sl][:, jh, kref])[:, None]
    m = float(np.abs(D[nearest(r_t[sl], 1.60):]).max())
    c = a.contourf(th, r_t[sl], np.clip(D, -m, m),
                   levels=np.linspace(-m, m, 25), cmap="RdBu_r",
                   extend="both")
    cs = a.contour(th, r_t[sl], D, levels=np.linspace(-m, m, 9),
                   colors="k", linewidths=0.4)
    a.clabel(cs, fmt="%.2f", fontsize=7)
    a.axhline(r_s[i0], color="k", lw=0.9, ls=":")
    # The angles chi_0 actually samples, which is the span that matters --
    # the full 75-135 deg grid is much wider than the ground state explores.
    kfc = np.where((ev(VS) <= 0.25).any(axis=(0, 1)))[0]
    for x in (th[kfc[0]], th[kfc[-1]]):
        a.axvline(x, color="k", lw=0.9, ls="--", alpha=0.6)
    a.set_xlabel("angle / deg")
    a.set_ylabel("r(O-Cl) / A")
    ifc = nearest(r_t[sl], r_s[i0])
    spread_fc = float(np.ptp(ev(VT[sl][ifc, j0, kfc])))
    spread_out = float(np.ptp(ev(VT[nearest(r_t, 3.0), j0, :])))
    a.set_title(f"bend sensitivity: V_T(angle) - V_T(105 deg)\n"
                f"{spread_fc:.2f} eV across the angles chi_0 samples "
                f"(dashed), {spread_out:.2f} eV by 3 A", fontsize=10)
    fig.colorbar(c, ax=a, label="dE / eV")

    a.text(0.5, 0.955,
           "flat past 3.6 A: there the surface IS E(Cl) + V_OH, angle-free",
           transform=a.transAxes, ha="center", fontsize=7.5, color="0.35")

    # --- C: cuts along the dissociation coordinate ----------------------
    a = ax[1, 0]
    for k in (0, nearest(th, 105.0), len(th) - 1):
        a.plot(r_t, ev(VT[:, jh, k]), lw=1.6, label=f"{th[k]:.0f} deg")
    e_as = ev(asym[jh])
    a.axhline(e_as, color="0.4", ls=":", lw=1.2)
    a.annotate(f"E(Cl) + V_OH({roh[jh]:.2f} A) = {e_as:.2f} eV",
               xy=(4.3, e_as), xytext=(0, 7), textcoords="offset points",
               fontsize=8, color="0.3")
    # The exit-channel well is real, not a splice artefact: 06 found NEVPT2 and
    # UCCSD(T) agreeing on its depth to 3 meV relative to 2.0 A.
    cut = ev(VT[:, jh, nearest(th, 105.0)])
    iw = int(np.argmin(np.where(r_t > 2.0, cut, np.inf)))
    a.annotate(f"shallow exit-channel well,\n{e_as - cut[iw]:.2f} eV deep "
               f"at {r_t[iw]:.2f} A",
               xy=(r_t[iw], cut[iw]), xytext=(3.4, 3.3), fontsize=8,
               color="0.25",
               arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    a.axvspan(1.4, 2.4, color="C0", alpha=0.07)
    a.axvspan(2.4, 3.6, color="C1", alpha=0.07)
    a.text(1.9, 6.0, "CCSD(T)+EOM\nraster", ha="center", va="top",
           fontsize=8, color="C0")
    a.text(3.0, 6.0, "NEVPT2\nshell", ha="center", va="top",
           fontsize=8, color="C1")
    a.plot(r_s[i0], ev(VT[it, j0, k0]), "r*", ms=13, zorder=5)
    a.set_xlim(1.4, 6.0)
    a.set_ylim(1.8, 6.3)
    a.set_xlabel("r(O-Cl) / A")
    a.set_ylabel("E / eV")
    a.set_title(f"cuts at r(O-H) = {roh[jh]:.2f} A: steeply repulsive out of "
                f"the\nFranck-Condon region (star), one dissociation channel",
                fontsize=10)
    a.legend(fontsize=8, title="H-O-Cl", title_fontsize=8, loc="center right")
    a.grid(alpha=0.25)

    # --- D: the vertical gap, which is what the band maps ---------------
    # Reflection principle: chi_0's spatial spread is carried into the band by
    # V_T - V_S, so this panel is the closest thing to the spectrum itself.
    a = ax[1, 1]
    it_s = np.array([nearest(r_t, r) for r in r_s])
    gap = (VT[it_s][:, :, ka] - VS[:, :, ka]) * HARTREE2EV
    c = a.contourf(roh, r_s, gap, levels=24, cmap="magma")
    cs = a.contour(roh, r_s, gap, levels=10, colors="w", linewidths=0.5)
    a.clabel(cs, fmt="%.1f", fontsize=7)
    a.contour(roh, r_s, ev(VS[:, :, ka]) - ev(VS).min(), levels=[0.25],
              colors="c", linewidths=1.6, linestyles="--")
    a.plot(roh[j0], r_s[i0], "r*", ms=13, zorder=5)
    a.set_xlabel("r(O-H) / A")
    a.set_ylabel("r(O-Cl) / A")
    a.set_title("vertical gap V_T - V_S (the band maps this)\n"
                f"{gap.min():.2f} to {gap.max():.2f} eV over the bound region",
                fontsize=10)
    fig.colorbar(c, ax=a, label="dE / eV")

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(args.png, dpi=140)
    print(f"  wrote {args.png}")

    # Numbers worth having next to the picture.
    print(f"\n  a 3A\" at the ground-state geometry: {ev(VT[it, j0, k0]):.3f} eV")
    print(f"  a 3A\" asymptote  E(Cl) + V_OH(r_eq): "
          f"{ev(asym[nearest(roh, 0.95)]):.3f} eV")
    print(f"  so at the FC geometry the triplet sits "
          f"{ev(VT[it, j0, k0]) - ev(asym[nearest(roh, 0.95)]):.3f} eV ABOVE "
          f"its own asymptote -- unbound there, and that excess is what goes "
          f"into fragment recoil")
    slope = np.gradient(ev(VT[:, j0, ka]), r_t)[it]
    print(f"  slope there: {slope:.3f} eV/A, by central difference on the "
          f"0.05 A grid at r(O-H) = {roh[j0]:.2f} A and {th[ka]:.0f} deg.")
    print(f"  09's -6.965 was a fitted slope at the true equilibrium "
          f"(1.6891 A, 0.9644 A, 102.96 deg), so these are close but not the "
          f"same measurement.")


if __name__ == "__main__":
    main()
