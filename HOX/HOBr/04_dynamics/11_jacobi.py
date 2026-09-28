#!/usr/bin/env python3
"""
HOBr step 11: valence (r_OBr, r_OH, theta) -> Jacobi (R, r, gamma).

Descends from water/07_jacobi.py, and uses its geometry exactly: the OH
centre of mass at the origin, OH along z, R the departing atom (here Br)
from the OH centre of mass at angle gamma to the OH axis, d = m_H/(m_O+m_H) r:

    r_OX   = sqrt(R^2 + d^2 + 2 R d cos gamma)
    cos th = (R cos gamma + d) / r_OX

Water breaks H-OH and HOBr breaks Br-OH, so this is water's transform with
the departing mass changed -- one channel, not two. What does NOT carry over
is how the surface is continued beyond its raster, because HOBr's raster is
a box in all three coordinates and the Jacobi grid is not:

  r(O-Br)  held flat beyond each end. Inside 1.55 A V_T is ~5.7 eV above
           the ground minimum and V_S ~1.5 eV -- never visited. Beyond 6.25 A
           the spliced surface already IS E(Br) + V_OH, flat in r(O-Br).
           V_S is only defined over the bound raster, 1.55-2.45 A; outside it
           is held at its edge value, >= 1.38 eV, a confining plateau for
           vibrational states below ~0.2 eV.
  r(O-H)   continued with the free OH curve: V(r_OH) = V(clipped r_OH) +
           V_OH(r_OH) - V_OH(clipped), V_OH a Morse fit to 08's UCCSD(T)
           fragment curve. At the raster's O-H edges V_S is already >= 1.0 eV.
  angle    a stiff quadratic rise from each edge, --theta-curv Ha/deg^2. At
           the raster's 75 and 135 deg edges V_S is already 0.57 and 0.66 eV,
           so chi_0 is negligible there; the diagnostics report how much
           norm sits within 5 deg of an edge, which is the thing to watch.
  cap      nothing above --wall Ha.

Energies are shifted so min(V_S) = 0. The transition dipole is Condon
(mu = 1 on the grid): 10_compare_obs showed the lenders are right and the
borrowed f is ~3-4x low, and a constant mu cancels in every temperature
RATIO, so it is scaled in 13, not here.

Everything out is atomic units, with water's keys, so water's Propagator
and RTProp read the file unchanged.

Usage:
    python 11_jacobi.py 2>&1 | tee ../logs/jacobi.log
    python 11_jacobi.py --nR 96 --nr 24 --ngamma 24 --out ../data/jacobi_coarse.npz
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parents[1] / "data"
BOHR = 0.529177210903
HARTREE2EV = 27.211386245988
AMU = 1822.888486209
M_H, M_O, M_BR = 1.00782503207, 15.9949146196, 78.9183376


def morse(r, de, a, re, e0):
    return e0 + de * (1.0 - np.exp(-a * (r - re))) ** 2


def load_oh(path):
    """Morse fit to 08's UCCSD(T) OH curve: V_OH(r in A) in hartree."""
    rows = list(csv.DictReader(open(path)))
    r = np.array([float(x["r_A"]) for x in rows])
    e = np.array([float(x["e_uccsdt_Ha"]) for x in rows])
    ok = np.isfinite(e)
    p, _ = curve_fit(morse, r[ok], e[ok], p0=[0.17, 2.3, 0.97, e[ok].min()],
                     maxfev=20000)
    resid = np.abs(morse(r[ok], *p) - e[ok]).max() * HARTREE2EV * 1000
    return (lambda x: morse(x, *p)), p, resid


def continued(interp, h_axis, t_axis, x_lo, x_hi, v_oh, theta_curv, wall):
    """Evaluate a valence surface anywhere, continued beyond its box."""
    def f(r_ox, r_oh, th):
        rc = np.clip(r_ox, x_lo, x_hi)
        hc = np.clip(r_oh, h_axis[0], h_axis[-1])
        tc = np.clip(th, t_axis[0], t_axis[-1])
        v = interp(np.stack([rc, hc, tc], axis=-1))
        v = v + (v_oh(r_oh) - v_oh(hc))                 # free OH beyond r_OH
        v = v + theta_curv * (th - tc) ** 2             # stiff rise in angle
        return np.minimum(v, wall)
    return f


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surface", default=str(DATA / "hobr_surfaces.npz"))
    p.add_argument("--fragments", default=str(DATA / "hobr_fragments.csv"))
    p.add_argument("--R-min", type=float, default=2.6, help="bohr")
    p.add_argument("--R-max", type=float, default=12.5, help="bohr")
    p.add_argument("--nR", type=int, default=256)
    p.add_argument("--r-min", type=float, default=1.2, help="bohr")
    p.add_argument("--r-max", type=float, default=3.4, help="bohr")
    p.add_argument("--nr", type=int, default=48)
    p.add_argument("--ngamma", type=int, default=48)
    p.add_argument("--theta-curv", type=float, default=5e-5,
                   help="Ha/deg^2 beyond the raster's angle range: 0.5 Ha "
                        "30 deg past an edge")
    p.add_argument("--wall", type=float, default=1.5, help="hartree cap")
    p.add_argument("--out", default=str(DATA / "jacobi_hobr.npz"))
    args = p.parse_args()

    d = np.load(args.surface)
    rt, h, t = d["r_ox_A"], d["r_oh_A"], d["theta_deg"]
    rb = d["r_ox_bound_A"]
    VT, VS = d["V_T_Ha"], d["V_S_Ha"]
    e0 = float(VS.min())
    VT, VS = VT - e0, VS - e0

    v_oh, pm, res = load_oh(args.fragments)
    print(f"OH Morse fit to 08's UCCSD(T) curve: De {pm[0] * HARTREE2EV:.2f} eV,"
          f" a {pm[1]:.3f} /A, re {pm[2]:.4f} A; max residual {res:.1f} meV")

    iT = RegularGridInterpolator((rt, h, t), VT, method="cubic")
    iS = RegularGridInterpolator((rb, h, t), VS, method="cubic")
    fT = continued(iT, h, t, rt[0], rt[-1], v_oh, args.theta_curv, args.wall)
    fS = continued(iS, h, t, rb[0], rb[-1], v_oh, args.theta_curv, args.wall)

    mA, mB, mO = M_BR, M_H, M_O
    mu_r = mB * mO / (mB + mO) * AMU
    mu_R = mA * (mB + mO) / (mA + mB + mO) * AMU
    print(f"masses   departing 79Br, spectator H: mu_R {mu_R / AMU:.4f} amu, "
          f"mu_r {mu_r / AMU:.4f} amu")

    R = np.linspace(args.R_min, args.R_max, args.nR)
    r = np.linspace(args.r_min, args.r_max, args.nr)
    x_gl, w_gl = np.polynomial.legendre.leggauss(args.ngamma)
    gamma = np.arccos(x_gl)
    RR, rr, GG = np.meshgrid(R, r, gamma, indexing="ij")
    dd = (mB / (mB + mO)) * rr
    r1 = np.sqrt(RR ** 2 + dd ** 2 + 2.0 * RR * dd * np.cos(GG))
    cos_th = np.clip((RR * np.cos(GG) + dd) / np.maximum(r1, 1e-12), -1, 1)
    th = np.degrees(np.arccos(cos_th))
    r_ox, r_oh = r1 * BOHR, rr * BOHR

    V_ex = fT(r_ox, r_oh, th)
    V_gs = fS(r_ox, r_oh, th)
    mu = np.ones_like(V_gs)

    inside = ((r_ox >= rt[0]) & (r_ox <= rt[-1]) & (r_oh >= h[0])
              & (r_oh <= h[-1]) & (th >= t[0]) & (th <= t[-1]))
    print(f"grid     {args.nR} x {args.nr} x {args.ngamma} = {R.size * r.size * gamma.size} "
          f"points; R {R[0]}-{R[-1]} bohr, r {r[0]}-{r[-1]} bohr")
    print(f"         {100 * inside.mean():.1f}% of points inside the valence raster; "
          f"{100 * (V_gs >= args.wall).mean():.1f}% of V_gs and "
          f"{100 * (V_ex >= args.wall).mean():.1f}% of V_ex at the cap")
    i = np.unravel_index(V_gs.argmin(), V_gs.shape)
    print(f"V_gs min {V_gs.min() * HARTREE2EV:.4f} eV at R {R[i[0]]:.3f}, "
          f"r {r[i[1]]:.3f} bohr, gamma {np.degrees(gamma[i[2]]):.1f} deg "
          f"(r_OBr {r_ox[i]:.4f} A, r_OH {r_oh[i]:.4f} A, theta {th[i]:.2f} deg)")
    print(f"V_ex - V_gs there {(V_ex[i] - V_gs[i]) * HARTREE2EV:.3f} eV "
          f"(06's check: 2.867 eV at 01's minimum)")

    # Can the R grid carry the packet's momentum? Br sliding down the wall
    # from the Franck-Condon region picks up roughly V_ex(FC) - min(V_ex) of
    # kinetic energy along R. A grid whose k_max = pi/dR cannot represent it
    # ALIASES, and the packet looks trapped: S(t) never decays and the band
    # collapses to a spike. A 96-point test grid did exactly that -- it looks
    # like physics, which is why this is checked rather than left to notice.
    dR = R[1] - R[0]
    e_kmax = (np.pi / dR) ** 2 / (2.0 * mu_R)
    e_need = float(V_ex[i] - V_ex.min())
    print(f"R grid   dR {dR:.4f} bohr: k_max {np.pi / dR:.0f} /bohr carries "
          f"{e_kmax * HARTREE2EV:.2f} eV along R; the packet needs ~"
          f"{e_need * HARTREE2EV:.2f} eV")
    if e_kmax < 1.5 * e_need:
        print(f"  !! the R grid is too coarse: the packet's momentum will "
              f"alias and it will look trapped.\n     Use nR >= "
              f"{int(np.ceil((args.R_max - args.R_min) * np.sqrt(2 * mu_R * 1.5 * e_need) / np.pi)) + 1}.")

    np.savez_compressed(
        args.out, R=R, r=r, gamma=gamma, x_gl=x_gl, w_gl=w_gl,
        V_gs=V_gs, V_ex=V_ex, mu=mu, inside_raster=inside,
        r_ox_A=r_ox, r_oh_A=r_oh, theta_deg=th,
        mu_R=mu_R, mu_r=mu_r, m_departing=mA, m_spectator=mB, m_O=mO,
        e_min_Ha=e0, asymptote_Ha=d["asymptote_Ha"] - e0, halogen="Br")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
