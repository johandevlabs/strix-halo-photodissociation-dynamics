#!/usr/bin/env python3
"""
HOBr step 15: what the visible band's shape and temperature do to J(HOBr).

A FIRST LOOK, not a J-value. J = Int sigma(lambda) phi F(lambda, SZA) dlambda
needs the actinic flux F at the ground, which properly comes from a radiative
transfer model (NCAR TUV) with diffuse light, surface albedo -- high over
polar snow, exactly where HOBr photolysis matters -- aerosol and a real
ozone profile. This uses a transparent clear-sky model instead, and brackets
the one thing it cannot do, the diffuse field:

  direct   F = F0 exp(-tau_R m_R - tau_O3 m_O3): the direct beam only. The
           REDDEST case, since every Rayleigh-scattered photon is lost.
  noRay    F = F0 exp(-tau_O3 m_O3): no net Rayleigh loss, as if every
           scattered photon still arrives -- roughly what a bright snow
           surface does. The BLUEST case.
  The truth is between them. What is reported is mostly RATIOS -- J with one
  visible band against J with another -- which depend on the spectral shape
  of F far more than on its size, and which this model gets roughly right.

Inputs, all cached in HOBr/data/:
  F0      Chance & Kurucz (2010), JQSRT 111, 1289 -- the 'sao2010' reference
          TUV ships (from NCAR's tuv-x repository), smoothed to 1 nm
  O3      Serdyuchenko et al. (2014), AMT 7, 625, 293 K
  Rayleigh  Hansen & Travis (1974) optical depth at 1013 hPa
  air mass  Kasten & Young (1989) for Rayleigh; a 22 km shell for ozone
  HOBr    the UV bands (284, 352 nm) from 10_compare_obs's decomposition of
          Ingham (1998), with the visible band swapped:
            JPL       the JPL-2010 recommendation itself (Ingham's band)
            Ingham    Ingham's fitted visible Gaussian
            Barnes    Barnes (1996)'s reported Gaussian
            computed  this work's 3D non-Condon band, scaled to Ingham's or
                      Barnes's integrated f
          phi = 1 (JPL: HOBr + hv -> OH + Br).

Usage:
    python 15_j_hobr.py 2>&1 | tee ../logs/j_hobr.log
    python 15_j_hobr.py --o3-du 150        # ozone-hole column
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve()
DATA = HERE.parents[1] / "data"
NM_EV = 1239.841984
H, C = 6.62607015e-34, 2.99792458e8
DU = 2.6867e16                                     # molecules cm-2 per DU
R_EARTH, H_O3 = 6371.0, 22.0                       # km

_s = importlib.util.spec_from_file_location("c10", HERE.parents[1] / "02_band_model" / "10_compare_obs.py")
c10 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(c10)


def solar(lam):
    """Extraterrestrial actinic flux, photons cm-2 s-1 nm-1, at 1 nm bins."""
    d = np.loadtxt(DATA / "radiation" / "sao2010.solref.converted")
    wl, e = d[:, 0], d[:, 1]                        # nm, W m-2 nm-1
    ph = e * (wl * 1e-9) / (H * C) * 1e-4          # photons cm-2 s-1 nm-1
    edges = np.concatenate([lam - 0.5, [lam[-1] + 0.5]])
    idx = np.digitize(wl, edges) - 1
    out = np.array([ph[idx == i].mean() if np.any(idx == i) else np.nan
                    for i in range(lam.size)])
    return out


def ozone(lam):
    d = np.loadtxt(DATA / "radiation" / "O3_Serdyuchenko(2014)_293K_213-1100nm(2013 version).txt",
                   usecols=(0, 1))
    return np.interp(lam, d[:, 0], d[:, 1])


def tau_rayleigh(lam_nm):
    x = lam_nm / 1000.0
    return 0.008569 * x ** -4 * (1 + 0.0113 * x ** -2 + 0.00013 * x ** -4)


def m_rayleigh(sza):
    z = np.asarray(sza, float)
    return 1.0 / (np.cos(np.radians(z)) + 0.50572 * (96.07995 - z) ** -1.6364)


def m_ozone(sza):
    c = np.cos(np.radians(sza))
    h = H_O3 / R_EARTH
    return (1 + h) / np.sqrt(c ** 2 + 2 * h)


def cross_sections(lam):
    """The HOBr sigma variants on lam, 298 K; and the computed T-ratio."""
    obs = DATA / "obs"
    ing = np.loadtxt(obs / c10.FILES["Ingham(1998)"])
    E = NM_EV / ing[:, 0]
    g3 = lambda E, *p: sum(c10.gauss(E, *p[3 * k:3 * k + 3]) for k in range(3))  # noqa: E731
    p, _ = curve_fit(g3, E, ing[:, 1], p0=[3e-19, 4.4, 0.6, 1.5e-19, 3.5, 0.5,
                                          2.3e-20, 2.7, 0.5], maxfev=40000)
    b = sorted([tuple(p[3 * k:3 * k + 3]) for k in range(3)], key=lambda x: x[1])
    El = NM_EV / lam
    uv = c10.gauss(El, *b[1]) + c10.gauss(El, *b[2])
    vis_ing = c10.gauss(El, *b[0])
    f_ing = c10.f_of(b[0][0], b[0][2])

    bar = np.loadtxt(obs / c10.FILES["Barnes(1996)"])
    pb, _ = curve_fit(c10.gauss, NM_EV / bar[:, 0], bar[:, 1], p0=[9e-21, 2.85, 0.5])
    vis_bar = c10.gauss(El, *pb)
    f_bar = c10.f_of(pb[0], pb[2])

    jpl = np.loadtxt(obs / c10.FILES["JPL-2010"])
    s_jpl = np.interp(lam, jpl[:, 0], jpl[:, 1], left=0.0, right=0.0)

    z = np.load(DATA / "sigma_hobr_3d_noncondon.npz")
    Ec, s298, s220 = z["E_eV"], z["sigma_298K"], z["sigma_220K"]
    o = np.argsort(Ec)
    comp298 = np.interp(El, Ec[o], s298[o], left=0.0, right=0.0)
    comp220 = np.interp(El, Ec[o], s220[o], left=0.0, right=0.0)
    # scale to a measured integrated f; f = 1.1296e12 Int sigma dnu
    nu = 1e7 / lam
    f_comp = 1.1296e12 * abs(np.trapezoid(comp298, nu))
    return dict(uv=uv, jpl=s_jpl, vis={
        "Ingham (fit)": vis_ing,
        "Barnes": vis_bar,
        "computed, f=Ingham": comp298 * f_ing / f_comp,
        "computed, f=Barnes": comp298 * f_bar / f_comp,
    }, ratio_220=np.where(comp298 > 0, comp220 / np.maximum(comp298, 1e-40), 1.0),
        f=dict(Ingham=f_ing, Barnes=f_bar))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--o3-du", type=float, default=300.0)
    p.add_argument("--sza", type=float, nargs="+",
                   default=[0, 30, 60, 70, 80, 85, 88, 90])
    p.add_argument("--png", default=str(DATA / "j_hobr.png"))
    args = p.parse_args()

    lam = np.arange(250.0, 700.5, 1.0)
    F0 = solar(lam)
    tR, sO3 = tau_rayleigh(lam), ozone(lam)
    X = cross_sections(lam)
    print("=" * 78)
    print(f"== J(HOBr), clear sky at the surface, O3 {args.o3_du:.0f} DU, phi = 1")
    print("=" * 78)
    print(f"  visible band f: Ingham {X['f']['Ingham']:.2e}, Barnes {X['f']['Barnes']:.2e}")

    def flux(sza, mode):
        tO3 = sO3 * args.o3_du * DU
        t = tO3 * m_ozone(sza)
        if mode == "direct":
            t = t + tR * m_rayleigh(sza)
        return F0 * np.exp(-t)

    def J(sig, F):
        return float(np.trapezoid(sig * F, lam))

    variants = {"JPL-2010": X["jpl"]}
    variants.update({k: X["uv"] + v for k, v in X["vis"].items()})
    for mode in ("direct", "noRay"):
        print(f"\n  --- {mode}: "
              + ("direct beam only (reddest)" if mode == "direct"
                 else "no net Rayleigh loss (bluest)") + " ---")
        print(f"  {'SZA':>4}" + "".join(f"{k:>20}" for k in variants)
              + f"{'vis share (JPL)':>17}")
        for z in args.sza:
            F = flux(z, mode)
            js = {k: J(s, F) for k, s in variants.items()}
            vis_share = 1 - J(X["uv"], F) / js["JPL-2010"]
            print(f"  {z:4.0f}" + "".join(f"{v:20.3e}" for v in js.values())
                  + f"{vis_share:17.0%}")
        print(f"\n  ratio to JPL-2010:")
        print(f"  {'SZA':>4}" + "".join(f"{k:>20}" for k in list(variants)[1:]))
        for z in args.sza:
            F = flux(z, mode)
            j0 = J(X["jpl"], F)
            print(f"  {z:4.0f}" + "".join(f"{J(s, F) / j0:20.3f}"
                                          for s in list(variants.values())[1:]))

    # temperature: the computed visible band at 220 K against 298 K, UV fixed
    print("\n  temperature: J(220 K) / J(298 K), only the visible band's T-dependence"
          " (this work);\n  the UV bands' is not known here and is held fixed.")
    print(f"  {'SZA':>4}{'direct':>12}{'noRay':>12}   (visible band = computed, f=Barnes)")
    vis = X["vis"]["computed, f=Barnes"]
    for z in args.sza:
        r = []
        for mode in ("direct", "noRay"):
            F = flux(z, mode)
            r.append(J(X["uv"] + vis * X["ratio_220"], F) / J(X["uv"] + vis, F))
        print(f"  {z:4.0f}{r[0]:12.3f}{r[1]:12.3f}")

    # where J comes from, at high sun and low sun
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        for k, s in variants.items():
            ax[0].semilogy(lam, np.maximum(s, 1e-26), label=k, lw=1.2)
        ax[0].set_xlim(300, 650)
        ax[0].set_ylim(1e-24, 5e-19)
        ax[0].set_xlabel("wavelength / nm")
        ax[0].set_ylabel("sigma / cm2")
        ax[0].set_title("HOBr cross sections used")
        ax[0].legend(fontsize=7)
        for i, z in enumerate((60, 85)):
            F = flux(z, "direct")
            for k, s in variants.items():
                ax[1 + i].plot(lam, s * F / J(X["jpl"], F), label=k, lw=1.2)
            ax[1 + i].set_xlim(300, 650)
            ax[1 + i].set_xlabel("wavelength / nm")
            ax[1 + i].set_ylabel("dJ/dlambda / J(JPL)  (1/nm)")
            ax[1 + i].set_title(f"where J comes from, SZA {z}, direct beam")
            ax[1 + i].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.png, dpi=140)
        print(f"\n  wrote {args.png}")
    except Exception as exc:
        print(f"  plot skipped: {exc}")


if __name__ == "__main__":
    main()
