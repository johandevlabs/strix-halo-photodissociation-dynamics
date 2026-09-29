#!/usr/bin/env python3
"""
HOBr step 10: the computed bands against every measured HOBr spectrum.

The comparison that should have come before chasing a "10x f deficit" for
three rounds. The target it was measured against -- f >= 1.3e-4 -- came from
Ingham's peak sigma times the 1D MODEL's band width, and the model's width
turns out to be exactly what is in dispute. This script replaces the
estimate with the measurements themselves, band by band.

Data: the MPI-Mainz UV/VIS Spectral Atlas (Keller-Rudek et al., Earth Syst.
Sci. Data 5, 365 (2013)), the same source water/11_compare_obs.py used,
cached in HOBr/data/obs/:

  Ingham (1998)      260-600 nm, three resolved bands
  JPL-2010, IUPAC    recommendations, which follow Ingham in the visible
  Barnes (1996)      380-600 nm; reported as a single Gaussian band
  Rattigan (1996)    240-510 nm; the atlas's 5-nm averages show no separate
                     peak, though IUPAC 2007 says Rattigan observed the band

What is compared:

  1. f per band. Ingham's spectrum is decomposed into three Gaussians in
     energy and each integrated, f = 1.1296e12 Int sigma dnu. Against the
     SOC-QD-NEVPT2 f of the states under each band (04's log). The singlet
     bands test the LENDERS' transition dipoles, which is what decides
     whether a weak borrowed f is the lenders' fault or the coupling's.
  2. The visible band's shape, fit-free where possible: its red half-width
     read straight off each dataset, on the side where nothing else absorbs.
  3. Against 03's 1D band: position and width.

Usage:
    python 10_compare_obs.py 2>&1 | tee ../logs/compare_obs.log
"""
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parents[1] / "data"
LOGS = Path(__file__).resolve().parents[1] / "logs"
NM_EV = 1239.841984
EV_CM = 8065.544
F_CONST = 1.1296e12       # f = F_CONST * Int sigma dnu~  (sigma cm2, nu~ cm-1)
FILES = {
    "Ingham(1998)": "HOBr_Ingham(1998)_298K_260-600nm.txt",
    "JPL-2010": "HOBr_JPL-2010(2011)_295K_250-550nm(rec).txt",
    "IUPAC(2007)": "HOBr_IUPAC(2007)_295K_250-545nm(rec).txt",
    "Barnes(1996)": "HOBr_Barnes(1996)_298K_380-600nm.txt",
    "Rattigan(1996)": "HOBr_Rattigan(1996)_298K_240-510nm.txt",
}


def gauss(E, a, e0, w):
    return a * np.exp(-4 * np.log(2) * ((E - e0) / w) ** 2)


def f_of(a, w):
    """f of a Gaussian band in energy with peak a (cm2) and FWHM w (eV)."""
    return F_CONST * a * abs(w) * np.sqrt(np.pi / (4 * np.log(2))) * EV_CM


def computed_states(path):
    """(dE_eV, f) of every excited SOC state in 04's log."""
    out, on = [], False
    for line in open(path):
        if "spin-orbit states:" in line:
            on = True
            continue
        if on:
            m = re.match(r"\s+(\d+)\s+-?\d+\.\d+\s+(\d+\.\d+)\s+\S+\s+([\d.eE+-]+)\s*$", line)
            if m:
                out.append((float(m.group(2)), float(m.group(3))))
            elif out and not line.strip():
                break
    return out


def main():
    print("=" * 76)
    print("== HOBr: computed bands against the measured spectra")
    print("=" * 76)
    sets = {k: np.loadtxt(DATA / "obs" / v) for k, v in FILES.items()}

    # ---- 1. f per band, Ingham decomposed
    lam, sig = sets["Ingham(1998)"][:, 0], sets["Ingham(1998)"][:, 1]
    E = NM_EV / lam
    g3 = lambda E, *p: sum(gauss(E, *p[3 * k:3 * k + 3]) for k in range(3))  # noqa: E731
    p, _ = curve_fit(g3, E, sig, p0=[3e-19, 4.4, 0.6, 1.5e-19, 3.5, 0.5,
                                     2.3e-20, 2.7, 0.5], maxfev=40000)
    # ascending energy: visible, 350 nm, 280 nm -- the order of `windows`
    bands = sorted([tuple(p[3 * k:3 * k + 3]) for k in range(3)],
                   key=lambda b: b[1])
    rms = np.sqrt(np.mean((sig - g3(E, *p)) ** 2))
    comp = computed_states(LOGS / "soc_vertical.log")
    print(f"\n  Ingham (1998), three Gaussians in energy (rms {rms:.1e} cm2):")
    print(f"  {'band':>9}{'E/eV':>8}{'peak/cm2':>11}{'FWHM/eV':>9}{'f obs':>10}"
          f"{'f calc':>10}{'calc/obs':>10}   computed states under it")
    windows = [(2.3, 3.3), (3.3, 4.2), (4.2, 5.0)]    # visible, 350, 280 nm
    for (a, e0, w), (lo, hi) in zip(bands, windows):
        fo = f_of(a, w)
        under = [(e, f) for e, f in comp if lo <= e < hi]
        fc = sum(f for _, f in under)
        print(f"  {NM_EV / e0:7.0f}nm{e0:8.3f}{a:11.2e}{abs(w):9.3f}{fo:10.2e}"
              f"{fc:10.2e}{fc / fo:10.2f}   "
              + ", ".join(f"{NM_EV / e:.0f}" for e, _ in under) + " nm")
    print("  The singlet bands test the LENDERS: their computed transition "
          "dipoles are right to\n  within a factor of ~2-3, and too strong "
          "rather than too weak. The borrowed band's\n  shortfall is therefore "
          "not the lenders'; it lies in the coupling to them, or in lenders\n"
          "  missing from the active space.")

    # ---- 2. the visible band in every dataset, fit-free red half-width
    print(f"\n  the visible band, dataset by dataset (red half-width read "
          f"directly, no fit):")
    print(f"  {'dataset':>16}{'peak/nm':>9}{'sigma':>10}{'red HWHM/eV':>13}"
          f"{'sigma(420)/peak':>17}")
    for name, d in sets.items():
        lam_d, sig_d = d[:, 0], d[:, 1]
        m = (lam_d >= 420) & (lam_d <= 520)
        if m.sum() < 3:
            continue
        i = int(np.argmax(np.where(m, sig_d, -1)))
        red = lam_d[i:][sig_d[i:] <= sig_d[i] / 2]
        rh = f"{NM_EV / lam_d[i] - NM_EV / red[0]:.3f}" if red.size else "--"
        flag = "   (no separate band: the maximum is the window edge)" \
            if lam_d[i] <= 420.5 else ""
        print(f"  {name:>16}{lam_d[i]:9.0f}{sig_d[i]:10.2e}{rh:>13}"
              f"{np.interp(420, lam_d, sig_d) / sig_d[i]:17.2f}{flag}")
    b = sets["Barnes(1996)"]
    pb, _ = curve_fit(gauss, NM_EV / b[:, 0], b[:, 1], p0=[9e-21, 2.85, 0.5])
    rb = np.sqrt(np.mean((b[:, 1] - gauss(NM_EV / b[:, 0], *pb)) ** 2))
    print(f"  Barnes (1996) is a single Gaussian to rms {rb:.0e}: a reported "
          f"band fit, {NM_EV / pb[1]:.0f} nm,\n  FWHM {abs(pb[2]):.3f} eV, "
          f"f {f_of(pb[0], pb[2]):.2e}. Ingham's visible band: "
          f"{NM_EV / bands[0][1]:.0f} nm, FWHM {abs(bands[0][2]):.3f} eV, "
          f"f {f_of(bands[0][0], bands[0][2]):.2e}.")

    # ---- 3. against 03's 1D band
    import csv
    rows = [r for r in csv.DictReader(open(DATA / "hobr_band_1d.csv"))
            if r["basis"] == "cc-pvtz-dk"]
    Eb = np.array([float(r["E_eV"]) for r in rows])
    sb = np.array([float(r["sigma_298K"]) for r in rows])
    ip = int(np.argmax(sb))
    lo = Eb[:ip][sb[:ip] <= sb[ip] / 2].max()
    hi = Eb[ip:][sb[ip:] <= sb[ip] / 2].min()
    print(f"\n  03's 1D band (cc-pvtz-dk, 298 K): {NM_EV / Eb[ip]:.0f} nm, FWHM "
          f"{hi - lo:.3f} eV, red HWHM {Eb[ip] - lo:.3f} eV")
    print(f"  -> position and width match Barnes (1996) "
          f"({NM_EV / pb[1]:.0f} nm, {abs(pb[2]):.3f} eV) and NOT Ingham (1998) /"
          f" JPL / IUPAC\n     ({NM_EV / bands[0][1]:.0f} nm, "
          f"{abs(bands[0][2]):.3f} eV). The two measurements carry similar "
          f"integrated intensity\n     (f {f_of(pb[0], pb[2]):.1e} and "
          f"{f_of(bands[0][0], bands[0][2]):.1e}); they disagree on where "
          f"the band sits and how wide it is.")
    f3 = sum(f for e, f in comp if 2.3 <= e < 3.3)
    print(f"\n  borrowed f: computed {f3:.2e} against {f_of(pb[0], pb[2]):.1e}"
          f" (Barnes) - {f_of(bands[0][0], bands[0][2]):.1e} (Ingham): "
          f"{f3 / f_of(bands[0][0], bands[0][2]):.2f}-"
          f"{f3 / f_of(pb[0], pb[2]):.2f}, i.e. mu ~"
          f"{np.sqrt(f_of(pb[0], pb[2]) / f3):.1f}-"
          f"{np.sqrt(f_of(bands[0][0], bands[0][2]) / f3):.1f}x low.")


if __name__ == "__main__":
    main()
