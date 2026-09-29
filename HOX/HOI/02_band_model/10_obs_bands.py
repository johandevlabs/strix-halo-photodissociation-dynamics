#!/usr/bin/env python3
"""
HOI (and HOCl): the measured bands, decomposed -- the numbers
HOBr/01_method/04_soc_vertical.py compares the computed states with.

Each cached spectrum is fitted with Gaussians in energy and each band
integrated, f = 1.1296e12 Int sigma dnu~ (sigma cm2, nu~ cm-1), as
HOBr/02_band_model/10_compare_obs.py does for HOBr. Two bands for HOI (the
spectra stop at 280 nm), three for HOCl (the 240 nm band sets the UV band's
blue wing).

For HOCl and HOBr the reddest band is the triplet (3A") band, borrowing a
little from the singlet above it: visible/UV f ~0.1. HOI's reddest measured
band has visible/UV ~0.6 and f ~20x HOBr's triplet band. This was first read
as a triplet band gone strongly mixed. It is not a triplet band:
01_method/05_eom_vertical.py (and Minaev, JPCA 103, 7294 (1999)) put HOI's
407 nm band on 1A" (+ 3A') and 340 nm on 1A' -- HOBr's 352 and 284 nm bands,
red-shifted. HOI's own triplet band is predicted near 520-600 nm and has not
been measured (05_atmosphere/15_j_hoi_triplet.py).

Pure numpy/scipy on cached files -- runs anywhere.
    python 10_obs_bands.py
"""
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HOX = Path(__file__).resolve().parents[2]
NM_EV = 1239.84193
EV_CM = 8065.544
F_CONST = 1.1296e12

SETS = {
    # label: (file, p0 per band as (peak/1e-20 cm2, E/eV, FWHM/eV), bounds)
    "HOI": [HOX / "HOI/data/obs" / n for n in (
        "HOI_Bauer(1998)_295K_280-500nm.txt",
        "HOI_IUPAC(2007)_295K_280-490nm(rec).txt",
        "HOI_JPL-2010(2011)_295-298K_280-480nm(rec).txt",
        "HOI_Rowley(1999)_298K_280-470nm.txt",
        "HOI_Jenkin(1991)_295K_293-500nm(orig).txt")],
    "HOCl": [HOX / "HOBr/data/obs" / n for n in (
        "HOCl_Barnes(1998)_298K_200-550nm(2nm).txt",
        "HOCl_JPL-2010(2011)_298K_200-420nm(rec).txt")],
}
FIT = {
    "HOI": dict(p0=[30, 3.05, 0.4, 30, 3.65, 0.5],
                lo=[0, 2.7, 0.15, 0, 3.35, 0.15],
                hi=[200, 3.3, 1.0, 200, 4.2, 1.2], weight=False),
    # the weak visible band sits on the UV band's tail: weight the fit
    # towards small sigma or it is absorbed into the tail
    "HOCl": dict(p0=[20, 5.3, 0.8, 6, 4.1, 0.5, 0.4, 3.2, 0.4],
                 lo=[0, 4.8, 0.3, 0, 3.8, 0.2, 0, 2.9, 0.2],
                 hi=[100, 6.5, 2, 30, 4.4, 1.0, 5, 3.5, 0.8], weight=True),
}


def gauss(E, a, e0, w):
    return a * np.exp(-4 * np.log(2) * ((E - e0) / w) ** 2)


def gsum(E, *p):
    return sum(gauss(E, *p[i:i + 3]) for i in range(0, len(p), 3))


def f_of(a, w):
    """f of a Gaussian band in energy, peak a (cm2), FWHM w (eV)."""
    return F_CONST * a * abs(w) * np.sqrt(np.pi / (4 * np.log(2))) * EV_CM


def decompose(path, fit):
    d = np.loadtxt(path)
    d = d[d[:, 1] > 0]
    E, s = NM_EV / d[:, 0], d[:, 1] * 1e20
    sig = 0.02 + 0.05 * s if fit["weight"] else None
    p, _ = curve_fit(gsum, E, s, p0=fit["p0"], bounds=(fit["lo"], fit["hi"]),
                     sigma=sig, maxfev=40000)
    rms = float(np.sqrt(np.mean((gsum(E, *p) - s) ** 2))) * 1e-20
    bands = sorted(((NM_EV / p[i + 1], p[i] * 1e-20, abs(p[i + 2]),
                     f_of(p[i] * 1e-20, p[i + 2]))
                    for i in range(0, len(p), 3)), reverse=True)
    return (d[0, 0], d[-1, 0]), rms, bands


def main():
    for mol, files in SETS.items():
        print(f"\n== {mol}: bands, reddest first (nm, peak cm2, FWHM eV, f)")
        for path in files:
            (a, b), rms, bands = decompose(path, FIT[mol])
            label = path.name.split("_")[1]
            vis, uv = bands[0], bands[1]
            print(f"  {label:<14}{a:4.0f}-{b:3.0f} nm  rms {rms:.1e}")
            for nm, pk, w, f in bands:
                print(f"      {nm:6.1f}  {pk:.2e}  {w:.3f}  {f:.2e}")
            print(f"      visible/UV f = {vis[3] / uv[3]:.2f}")
    print("""
  HOI: Bauer, IUPAC (numerically Bauer's) and JPL agree to 10%; Rowley's
  visible band is 18% weaker; Jenkin (1991) is the outlier (half the UV
  band, fit rms 20x the others). HOBr, from 10_compare_obs.py: visible f
  4.7-6.2e-5 (Barnes, Ingham), 352 nm band 6.5e-4, ratio 0.07-0.10.""")


if __name__ == "__main__":
    main()
