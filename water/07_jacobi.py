#!/usr/bin/env python3
"""
Tier 3a: valence (r1, r2, theta) -> Jacobi (R, r, gamma).

The valence coordinates were right for building the surface and wrong for
propagating on it: the kinetic energy operator carries cross terms between
the two stretches, so it does not split. In Jacobi coordinates with J = 0,

    T = -1/(2 mu_R) d2/dR2 - 1/(2 mu_r) d2/dr2
        + j^2(gamma) * [ 1/(2 mu_R R^2) + 1/(2 mu_r r^2) ]

which is exactly separable for a split-operator scheme: FFT on R and r, and
a Legendre DVR on gamma where j^2 is diagonal with eigenvalue j(j+1).

GEOMETRY. OH centre of mass at the origin, OH axis along z,
d = m_H/(m_O+m_H) * r the O-to-COM distance, R the departing H measured
from the OH centre of mass at angle gamma from the OH axis:

    r2     = r
    r1     = sqrt(R^2 + d^2 + 2 R d cos gamma)
    cos th = (R cos gamma + d) / r1

Everything in and out of this script is atomic units (bohr, hartree); the
valence surface arrives in Angstrom and is converted on read.

ISOTOPOLOGUES. --m-spectator and --m-departing set which hydrogen is which.
For HDO the two choices are physically different calculations on the SAME
electronic surface, which is where the H/D branching comes from.

Usage:
    python 07_jacobi.py --surface pes_surface_v2.npz --oh oh_curve.csv
    python 07_jacobi.py --m-departing 2.0141018 --out jacobi_D.npz   # D + OH
"""
import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline, RegularGridInterpolator

BOHR = 0.529177210903          # Angstrom per bohr
HARTREE2EV = 27.211386245988
AMU = 1822.888486209           # electron masses per amu

M_H = 1.00782503207
M_D = 2.01410177812
M_O = 15.9949146196


def load_valence(path):
    d = np.load(path)
    r_ang = d["r"]
    th_deg = d["theta_deg"]
    print(f"valence  {len(r_ang)}x{len(r_ang)}x{len(th_deg)} grid, "
          f"r = {r_ang[0]:.2f}-{r_ang[-1]:.2f} A, "
          f"theta = {th_deg[0]:.0f}-{th_deg[-1]:.0f} deg")
    mk = lambda a: RegularGridInterpolator(  # noqa: E731
        (r_ang, r_ang, th_deg), a, bounds_error=False, fill_value=None)
    return (mk(d["V_gs"]), mk(d["V_ex"]), mk(d["mu"]),
            r_ang, th_deg, float(d["e_min_Ha"]), float(d["splice_shift_Ha"]))


def load_oh_far(path, e_min, shift):
    """
    V_OH(r) for the far field, where the Jacobi grid reaches beyond the
    valence raster in r1. Same curve and same shift the splice used, so the
    two descriptions agree where they overlap.
    """
    r, e, e_h = [], [], None
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            v = float(row["e_nevpt2_Ha"])
            if np.isfinite(v):
                r.append(float(row["r_A"]))
                e.append(v)
            if e_h is None and row.get("e_H_Ha"):
                e_h = float(row["e_H_Ha"])
    o = np.argsort(r)
    sp = CubicSpline(np.array(r)[o], np.array(e)[o], extrapolate=True)
    return lambda r_ang: e_h + sp(np.clip(r_ang, min(r), max(r))) + shift - e_min


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surface", default="pes_surface_v2.npz")
    p.add_argument("--oh", default="oh_curve.csv")
    p.add_argument("--R-min", type=float, default=1.2, help="bohr")
    p.add_argument("--R-max", type=float, default=14.0, help="bohr")
    p.add_argument("--nR", type=int, default=256)
    p.add_argument("--r-min", type=float, default=1.0, help="bohr")
    p.add_argument("--r-max", type=float, default=6.0, help="bohr")
    p.add_argument("--nr", type=int, default=128)
    p.add_argument("--ngamma", type=int, default=36,
                   help="Gauss-Legendre points in cos(gamma)")
    p.add_argument("--m-spectator", type=float, default=M_H,
                   help="mass of the H bound to O in the OH fragment, amu")
    p.add_argument("--m-departing", type=float, default=M_H,
                   help="mass of the leaving atom, amu")
    p.add_argument("--wall", type=float, default=1.5,
                   help="hartree; potential imposed where the transform "
                        "leaves the raster's theta range")
    p.add_argument("--out", default="jacobi_surface.npz")
    p.add_argument("--png", default="jacobi_surface.png")
    args = p.parse_args()

    v_gs, v_ex, v_mu, r_ang_ax, th_ax, e_min, shift = load_valence(args.surface)
    far = load_oh_far(args.oh, e_min, shift)

    mA, mB, mO = args.m_departing, args.m_spectator, M_O
    mu_r = mB * mO / (mB + mO) * AMU
    mu_R = mA * (mB + mO) / (mA + mB + mO) * AMU
    print(f"masses   departing {mA:.5f}, spectator {mB:.5f}, O {mO:.5f} amu")
    print(f"         mu_R = {mu_R / AMU:.5f} amu, mu_r = {mu_r / AMU:.5f} amu")

    # uniform in R and r for the FFT; Gauss-Legendre in cos(gamma) for the DVR
    R = np.linspace(args.R_min, args.R_max, args.nR)
    r = np.linspace(args.r_min, args.r_max, args.nr)
    x_gl, w_gl = np.polynomial.legendre.leggauss(args.ngamma)
    gamma = np.arccos(x_gl)

    RR, rr, GG = np.meshgrid(R, r, gamma, indexing="ij")

    # --- the transform ----------------------------------------------------
    d = (mB / (mB + mO)) * rr                      # O to OH centre of mass
    r1 = np.sqrt(RR**2 + d**2 + 2.0 * RR * d * np.cos(GG))
    r2 = rr
    cos_th = np.clip((RR * np.cos(GG) + d) / np.maximum(r1, 1e-12), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_th))

    r1_ang, r2_ang = r1 * BOHR, r2 * BOHR

    # Region the raster never covered: gamma -> 0 at small R drives theta
    # below the 55 deg edge, which is the two hydrogens on top of each other.
    # Steeply repulsive, never visited, and the interpolator would happily
    # extrapolate nonsense there. Wall it off instead.
    outside = (theta < th_ax[0]) | (theta > th_ax[-1])

    pts = np.stack([np.clip(r1_ang, r_ang_ax[0], r_ang_ax[-1]),
                    np.clip(r2_ang, r_ang_ax[0], r_ang_ax[-1]),
                    np.clip(theta, th_ax[0], th_ax[-1])], axis=-1)

    V_gs = v_gs(pts)
    V_ex = v_ex(pts)
    MU = v_mu(pts)

    # Beyond the valence box in r1 the surface is the analytic asymptote.
    # This MUST take precedence over the theta wall below: gamma -> 0 sends
    # theta -> 0 at ANY R, including large R where the geometry is simply H
    # approaching OH from the hydrogen end. That is an ordinary asymptotic
    # point, not a repulsive one, and walling it would put a 1.5 Ha cliff in
    # the middle of the dissociation channel.
    beyond = r1_ang > r_ang_ax[-1]
    if beyond.any():
        V_gs[beyond] = far(r2_ang[beyond])
        V_ex[beyond] = far(r2_ang[beyond])
        MU[beyond] = 0.0

    # Wall only where the transform leaves the raster AND the fragments are
    # still close: the two hydrogens on top of each other, steeply repulsive
    # and never visited by the wavepacket.
    walled = outside & ~beyond
    for V in (V_gs, V_ex):
        V[walled] = args.wall
    MU[walled] = 0.0

    print(f"transform {100 * beyond.mean():5.1f}% beyond the valence box "
          f"in r1 -> analytic E_H + V_OH(r)")
    print(f"          {100 * outside.mean():5.1f}% outside the raster's "
          f"theta range, of which")
    print(f"          {100 * walled.mean():5.1f}% walled at "
          f"{args.wall:.1f} Ha (close-in, repulsive)")
    print(f"          {100 * (outside & beyond).mean():5.1f}% covered by the "
          f"asymptote instead")

    print(f"\nV_gs min {V_gs.min() * HARTREE2EV:.4f} eV at "
          f"R={R[np.unravel_index(V_gs.argmin(), V_gs.shape)[0]]:.3f} "
          f"r={r[np.unravel_index(V_gs.argmin(), V_gs.shape)[1]]:.3f} bohr")
    print(f"V_ex - V_gs at the V_gs minimum: "
          f"{(V_ex - V_gs).ravel()[V_gs.argmin()] * HARTREE2EV:.4f} eV")

    np.savez_compressed(
        args.out, R=R, r=r, gamma=gamma, x_gl=x_gl, w_gl=w_gl,
        V_gs=V_gs, V_ex=V_ex, mu=MU, wall_mask=outside,
        mu_R=mu_R, mu_r=mu_r,
        m_departing=mA, m_spectator=mB, m_O=mO, e_min_Ha=e_min)
    print(f"\nwrote {args.out}")

    # --- diagnostics ------------------------------------------------------
    kg = int(np.argmin(np.abs(np.degrees(gamma) - 180.0)))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for a, V, nm in ((ax[0], V_gs, "$\\tilde{X}$"), (ax[1], V_ex, "$\\tilde{A}$")):
        c = a.contourf(R, r, np.clip(V[:, :, kg].T * HARTREE2EV, 0, 12),
                       levels=25)
        a.set_xlabel("R / bohr")
        a.set_ylabel("r / bohr")
        a.set_title(f"{nm}, gamma={np.degrees(gamma[kg]):.0f} deg")
        fig.colorbar(c, ax=a, label="eV")
    jr = int(np.argmin(np.abs(r - 1.83)))
    ax[2].plot(R, V_gs[:, jr, kg] * HARTREE2EV, label="$\\tilde{X}$")
    ax[2].plot(R, V_ex[:, jr, kg] * HARTREE2EV, label="$\\tilde{A}$")
    ax[2].set_xlabel("R / bohr")
    ax[2].set_ylabel("V / eV")
    ax[2].set_ylim(-1, 12)
    ax[2].set_title(f"cut at r={r[jr]:.2f} bohr")
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
