#!/usr/bin/env python3
"""
HOI: what an unmeasured triplet band would do to J(HOI). A sensitivity
estimate, not a result.

The measured HOI spectrum (Bauer et al. 1998, 280-490 nm; the IUPAC and JPL
recommendations) has two bands, at 407 and 340 nm. 05_eom_vertical.py and
Minaev (1999) assign them to 1A" (with 3A') and 1A'. HOI's lowest triplet,
the 3A" band that is HOCl's 368 nm and HOBr's 457 nm band, is computed at
2.40 eV (516 nm) with SOC, f 1.5e-4 (04, cc-pvtz-dk), and is not in any
measured spectrum. For HOCl and HOBr the same method put the triplet
0.06-0.19 eV too high and its f 3-4x too LOW (HOBr).

Bauer's red wing constrains the band: sigma(490 nm) = 8.5e-22 cm2. A band at
530-550 nm with HOBr's computed width would put 1-2e-20 there, so if the band
exists it sits further red or is narrower -- 560-620 nm, FWHM 0.28-0.4 eV
(HOBr's Ingham-band width is 0.28) -- or is weaker than computed. Bauer made
HOI from OH + I2 and calibrated against I2 loss, and I2's visible band peaks
in exactly this region, so the spectrum's red end rests on an I2 correction.

Each case below adds a Gaussian band (in energy) to Bauer's spectrum and
reports J against Bauer's alone, with HOBr's 15_j_hobr.py clear-sky model
(its solar spectrum, ozone, Rayleigh, and the direct / no-Rayleigh-loss
bracket). phi = 1 for the added band.

Pure numpy/scipy on cached files:
    python 15_j_hoi_triplet.py 2>&1 | tee ../logs/j_hoi_triplet.log
"""
import importlib.util
from pathlib import Path

import numpy as np

HOX = Path(__file__).resolve().parents[2]
_s = importlib.util.spec_from_file_location(
    "j15", HOX / "HOBr" / "05_atmosphere" / "15_j_hobr.py")
j15 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(j15)

NM_EV = 1239.841984
BAUER = HOX / "HOI" / "data" / "obs" / "HOI_Bauer(1998)_295K_280-500nm.txt"

# (label, f, centre nm, FWHM eV)
CASES = [
    ("as computed, HOBr-like width", 1.5e-4, 530, 0.40),
    ("as computed, broad", 1.5e-4, 540, 0.55),
    ("f x4 (HOBr's shortfall)", 6.0e-4, 540, 0.40),
    ("--- allowed by Bauer at 490 nm ---", None, None, None),
    ("computed f, Ingham width", 1.5e-4, 563, 0.28),
    ("computed f, further red", 1.5e-4, 600, 0.40),
    ("f x4, narrow", 6.0e-4, 600, 0.28),
    ("f x4, further red", 6.0e-4, 620, 0.40),
]
SZA = (0, 30, 60, 80, 85)


def band(lam, f, nm, w):
    """Gaussian in energy with integrated f; returns sigma(lam), peak."""
    pk = f / (1.1296e12 * w * 8065.544 * np.sqrt(np.pi / (4 * np.log(2))))
    return pk * np.exp(-4 * np.log(2) * ((NM_EV / lam - NM_EV / nm) / w) ** 2), pk


def main():
    lam = np.arange(250.0, 700.5, 1.0)
    F0 = j15.solar(lam)
    tR, sO3 = j15.tau_rayleigh(lam), j15.ozone(lam)
    b = np.loadtxt(BAUER)
    sig = np.interp(lam, b[:, 0], b[:, 1], left=0.0, right=0.0)
    i490 = int(np.argmin(np.abs(lam - 490)))

    def flux(z, mode):
        t = sO3 * 300 * j15.DU * j15.m_ozone(z)
        if mode == "direct":
            t = t + tR * j15.m_rayleigh(z)
        return F0 * np.exp(-t)

    print("=" * 78)
    print("== J(HOI) with a triplet band added to Bauer (1998), clear sky, "
          "O3 300 DU")
    print("=" * 78)
    print(f"  Bauer sigma(490 nm) = {sig[i490]:.1e} cm2: the constraint")
    for mode in ("direct", "noRay"):
        print(f"\n  --- {mode}: " + ("direct beam only (reddest)"
                                    if mode == "direct" else
                                    "no net Rayleigh loss (bluest)")
              + "; J / J(Bauer) at SZA " + ", ".join(map(str, SZA)) + " ---")
        print(f"  {'case':<36}{'f':>8}{'nm':>5}{'FWHM':>6}{'peak':>9}"
              f"{'s(490)':>9}  " + "".join(f"{z:>6}" for z in SZA))
        for label, f, nm, w in CASES:
            if f is None:
                print(f"  {label}")
                continue
            t, pk = band(lam, f, nm, w)
            r = [np.trapezoid((sig + t) * flux(z, mode), lam)
                 / np.trapezoid(sig * flux(z, mode), lam) for z in SZA]
            print(f"  {label:<36}{f:8.1e}{nm:5d}{w:6.2f}{pk:9.1e}{t[i490]:9.1e}"
                  f"  " + "".join(f"{x:6.2f}" for x in r))
    print("""
  Reading it. Even the bands Bauer's red wing allows raise J(HOI) by 20-50%
  with the computed f at high and mid sun, and by 2-3x if f is 4x larger,
  as HOBr's shortfall suggests. At low sun, in the direct beam, much more:
  the visible is all that is left. The direct / noRay bracket is wide there;
  this is a sensitivity, and a case for measuring HOI beyond 490 nm.""")


if __name__ == "__main__":
    main()
