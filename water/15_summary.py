#!/usr/bin/env python3
"""
Tier 3h: collect every isotopologue run into one table.

Scans runs/<LABEL>/ for vib_states.npz and cross_section.npz and reports the
vibrational levels and the band parameters side by side, against measured
values where they are known.

Two consistency checks it applies automatically:

  HDO vs HDOd   the same molecule in two different Jacobi coordinate sets.
                The TOTAL cross section must agree; a difference means one
                of the grids does not cover its awkward channel, not physics.

  isotope shift deuteration lowers the ZPE, and hv = E_ex - E_v, so every
                deuterated band must lie at HIGHER photon energy (shorter
                wavelength) than H2O. A red shift means the masses went in
                backwards somewhere.

The observed fundamentals below are standard gas-phase values but are
embedded from memory; check them against a spectroscopic source before
quoting any comparison.

Usage:
    python 15_summary.py
    python 15_summary.py --runs runs --temperature 200
"""
import argparse
import glob
import os

import numpy as np

HARTREE2CM = 219474.6313702
HARTREE2EV = 27.211386245988
HC_NM_EV = 1239.841984

# fundamentals in cm-1; VERIFY before quoting
OBS_FUND = {
    "H2O": {"bend": 1594.7, "2bend": 3151.6, "str1": 3657.1, "str2": 3755.9},
    "HDO": {"bend": 1403.5, "OD str": 2723.7, "OH str": 3707.5},
    "HDOd": {"bend": 1403.5, "OD str": 2723.7, "OH str": 3707.5},
    "D2O": {"bend": 1178.4, "str1": 2671.6, "str2": 2788.0},
}
OBS_ZPE = {"H2O": 4638.0}          # only this one am I confident of


def band_stats(lam_nm, sig):
    i = int(np.argmax(sig))
    m = sig >= sig.max() / 2.0
    lo, hi = lam_nm[m].min(), lam_nm[m].max()
    nu = HC_NM_EV / lam_nm / 1.239841984e-4
    k = np.argsort(nu)
    f = 1.1296e12 * np.trapezoid(sig[k], nu[k])
    return lam_nm[i], sig.max(), abs(HC_NM_EV/lo - HC_NM_EV/hi), f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--temperature", type=float, default=200.0)
    args = p.parse_args()

    labels = sorted(os.path.basename(os.path.dirname(d))
                    for d in glob.glob(os.path.join(args.runs, "*", "")))
    if not labels:
        raise SystemExit(f"no run directories under {args.runs}/")

    vib, band = {}, {}
    for lab in labels:
        vp = os.path.join(args.runs, lab, "vib_states.npz")
        cp = os.path.join(args.runs, lab, "cross_section.npz")
        if os.path.exists(vp):
            vib[lab] = np.load(vp)["energies"]
        if os.path.exists(cp):
            d = np.load(cp)
            key = f"sigma_{args.temperature:.0f}K"
            if key not in d.files:
                key = next(k for k in d.files
                           if k.startswith("sigma_") and k.endswith("K"))
            lam = d["wavelength_nm"]
            o = np.argsort(lam)
            band[lab] = band_stats(lam[o], d[key][o])

    print("=== vibrational levels, cm-1 above v=0 ===\n")
    print(f"{'label':>8}{'v=1':>10}{'v=2':>10}{'v=3':>10}{'v=4':>10}")
    for lab in labels:
        if lab not in vib:
            continue
        e = (vib[lab] - vib[lab][0]) * HARTREE2CM
        print(f"{lab:>8}" + "".join(f"{x:10.1f}" for x in e[1:5]))
        obs = OBS_FUND.get(lab)
        if obs:
            print(f"{'observed':>8}" +
                  "".join(f"{v:10.1f}" for v in list(obs.values())[:4]) +
                  "   <- assignment is by ENERGY ORDER, check the wavefunctions")

    print(f"\n=== band parameters, {args.temperature:.0f} K ===\n")
    print(f"{'label':>8}{'peak/nm':>10}{'peak/eV':>10}{'sigma/cm2':>13}"
          f"{'FWHM/eV':>10}{'f':>9}")
    for lab in labels:
        if lab not in band:
            continue
        pk, sm, fw, f = band[lab]
        print(f"{lab:>8}{pk:10.1f}{HC_NM_EV/pk:10.3f}{sm:13.3e}{fw:10.3f}{f:9.4f}")

    # --- consistency checks ---------------------------------------------
    print("\n=== checks ===")
    if "HDO" in band and "HDOd" in band:
        a, b = band["HDO"], band["HDOd"]
        print(f"HDO vs HDOd (same molecule, two coordinate sets):")
        print(f"   peak  {a[0]:.2f} vs {b[0]:.2f} nm   "
              f"({HC_NM_EV/a[0]-HC_NM_EV/b[0]:+.4f} eV)")
        print(f"   sigma {a[1]:.3e} vs {b[1]:.3e}  ({100*(a[1]/b[1]-1):+.1f}%)")
        print(f"   f     {a[3]:.4f} vs {b[3]:.4f}  ({100*(a[3]/b[3]-1):+.1f}%)")
        if abs(a[3]/b[3] - 1) > 0.05:
            print("   -> differ by >5%: one grid is not covering its "
                  "awkward channel.")
        else:
            print("   -> consistent, as required.")

    if "H2O" in band:
        e0 = HC_NM_EV / band["H2O"][0]
        print(f"\nisotope shift relative to H2O (deuteration lowers the ZPE, "
              f"so hv must RISE):")
        for lab in labels:
            if lab == "H2O" or lab not in band:
                continue
            e = HC_NM_EV / band[lab][0]
            flag = "" if e >= e0 else "   <- RED shift, check the masses"
            print(f"   {lab:>6}  {e - e0:+.4f} eV  "
                  f"({band[lab][0] - band['H2O'][0]:+.2f} nm){flag}")

    if "H2O" in vib:
        zpe_note = OBS_ZPE.get("H2O")
        print(f"\nZPE is reported by 08_relax.py per run; H2O observed "
              f"{zpe_note} cm-1.")
        print("Observed ZPEs for HDO and D2O are not embedded here -- "
              "look them up\nrather than trusting a value I did not verify.")


if __name__ == "__main__":
    main()
