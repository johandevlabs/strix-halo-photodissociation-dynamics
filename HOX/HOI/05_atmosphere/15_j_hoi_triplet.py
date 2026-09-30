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

What constrains it (references/README.md, from the paper itself):
  - NOT Bauer's red wing. The published spectrum is a two-Gaussian fit, and
    the I2-loss subtraction was scaled at 500 nm, which books any HOI
    absorption there as I2 loss. Bauer's sigma(490) = 8.6e-22 is the fit's
    tail. (A first version of this script treated it as a bound.)
  - Bauer's 532 nm photolysis: no OH, so sigma(532) < ~1e-20 cm2 IF OH + I
    is open at 532 nm (Bauer's threshold 582 +- 20 nm). IUPAC's threshold
    is 507 nm, and if that is right there is no dissociative absorption at
    532 nm from any band, and no constraint either. Column s(532) below.

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
BAND = HOX / "HOI" / "data" / "hoi_band_soc_1d.csv"
SHIFT_EV = -0.18        # 11_band_soc_1d.py: obs - calc, -0.173 / -0.192 eV
BOUND_532 = 1.0e-20     # Bauer's 532 nm photolysis null
THRESH_NM = 585.0       # 06_d0.py: HO + I opens at 570-600 nm; phi = 0 beyond

# (label, f, centre nm, FWHM eV)
CASES = [
    ("as computed, HOBr-like width", 1.5e-4, 530, 0.40),
    ("as computed, broad", 1.5e-4, 540, 0.55),
    ("f x4 (HOBr's shortfall)", 6.0e-4, 540, 0.40),
    ("--- further red / narrower ---", None, None, None),
    ("computed f, Ingham width", 1.5e-4, 563, 0.28),
    ("computed f, further red", 1.5e-4, 600, 0.40),
    ("f x4, narrow", 6.0e-4, 600, 0.28),
    ("f x4, further red", 6.0e-4, 620, 0.40),
    ("--- inside Bauer's 532 nm bound ---", None, None, None),
    ("f 4e-5 (s(532) ~1e-20)", 4.0e-5, 540, 0.40),
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
    i532 = int(np.argmin(np.abs(lam - 532)))

    def flux(z, mode):
        t = sO3 * 300 * j15.DU * j15.m_ozone(z)
        if mode == "direct":
            t = t + tR * j15.m_rayleigh(z)
        return F0 * np.exp(-t)

    print("=" * 78)
    print("== J(HOI) with a triplet band added to Bauer (1998), clear sky, "
          "O3 300 DU")
    print("=" * 78)
    print(f"  Bauer sigma(490 nm) = {sig[i490]:.1e} cm2 (the fit's tail, not a"
          f" bound); sigma(532) < ~1e-20 if OH + I is open at 532 nm")
    for mode in ("direct", "noRay"):
        print(f"\n  --- {mode}: " + ("direct beam only (reddest)"
                                    if mode == "direct" else
                                    "no net Rayleigh loss (bluest)")
              + "; J / J(Bauer) at SZA " + ", ".join(map(str, SZA)) + " ---")
        print(f"  {'case':<36}{'f':>8}{'nm':>5}{'FWHM':>6}{'peak':>9}"
              f"{'s(532)':>9}  " + "".join(f"{z:>6}" for z in SZA))
        for label, f, nm, w in CASES:
            if f is None:
                print(f"  {label}")
                continue
            t, pk = band(lam, f, nm, w)
            r = [np.trapezoid((sig + t) * flux(z, mode), lam)
                 / np.trapezoid(sig * flux(z, mode), lam) for z in SZA]
            print(f"  {label:<36}{f:8.1e}{nm:5d}{w:6.2f}{pk:9.1e}{t[i532]:9.1e}"
                  f"  " + "".join(f"{x:6.2f}" for x in r))
    print("""
  Reading it. With the computed f, a triplet band anywhere in 530-620 nm
  raises J(HOI) by 20-50% at high and mid sun; with 4x the f, 2-3x. At low
  sun in the direct beam, much more. Every case puts > 1e-20 cm2 at 532 nm
  unless the band is narrow and far red, so the case stands or falls with
  D0(HO-I): below 2.33 eV Bauer's 532 nm null bounds the band, above it the
  null is silent. A sensitivity, and a case for measuring HOI beyond 500 nm.""")

    # ---- the COMPUTED band (02_band_model/11_band_soc_1d.py), calibrated
    if not BAND.exists():
        print(f"\n  ({BAND.name} not found: run 11_band_soc_1d.py for the "
              f"computed band)")
        return
    d = np.genfromtxt(BAND, delimiter=",", names=True)
    E_ev, st = d["E_eV"], d["sigma_triplet"]
    o = np.argsort(E_ev)
    E_ev, st = E_ev[o], st[o]
    El = NM_EV / lam
    absorb = np.interp(El - SHIFT_EV, E_ev, st, left=0.0, right=0.0)
    base = absorb * (lam <= THRESH_NM)       # photolysis, not absorption
    lost = 1 - np.trapezoid(base * F0, lam) / np.trapezoid(absorb * F0, lam)
    print(f"\n  (phi = 0 red of {THRESH_NM:.0f} nm, the HO + I threshold: "
          f"removes {lost:.0%} of the band's photolysis at the top of the "
          f"atmosphere)")
    nu = np.sort(E_ev) * 8065.544
    f_band = 1.1296e12 * np.trapezoid(st, E_ev * 8065.544)
    print(f"\n== the computed a 3A\" band (1D, SOC states, calibrated "
          f"{SHIFT_EV:+.2f} eV on Bauer's bands), f {f_band:.2e}")
    at_bound = BOUND_532 / base[i532]
    scales = [("as computed", 1.0),
              ("f x 0.42 (407-band f calc/obs)", 1.17e-3 / 2.77e-3),
              ("f x 0.23 (340-band f calc/obs)", 1.85e-3 / 8.0e-3),
              ("at Bauer's 532 nm bound", at_bound),
              ("f x 4 (HOBr's shortfall)", 4.0)]
    for mode in ("direct", "noRay"):
        print(f"\n  --- {mode}; J / J(Bauer) at SZA " + ", ".join(map(str, SZA)))
        print(f"  {'case':<34}{'f':>9}{'s(532)':>10}  "
              + "".join(f"{z:>6}" for z in SZA))
        for label, k in scales:
            t = base * k
            r_ = [np.trapezoid((sig + t) * flux(z, mode), lam)
                  / np.trapezoid(sig * flux(z, mode), lam) for z in SZA]
            print(f"  {label:<34}{f_band * k:9.1e}{t[i532]:10.1e}  "
                  + "".join(f"{x:6.2f}" for x in r_))


if __name__ == "__main__":
    main()
