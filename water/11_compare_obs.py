#!/usr/bin/env python3
"""
Tier 3e: compare the computed cross section against measurement.

Data from the MPI-Mainz UV/VIS Spectral Atlas of Gaseous Molecules of
Atmospheric Interest:
    Keller-Rudek, Moortgat, Sander, Soerensen, Earth Syst. Sci. Data 5,
    365-373 (2013), doi:10.5194/essd-5-365-2013
    https://www.uv-vis-spectral-atlas-mainz.org

Default dataset is Chung et al. (2001), which measured H2O, HDO AND D2O
over 140-193 nm at three temperatures, so the same source covers the
isotopologue comparison later:
    Chung, Chew, Cheng, Bahou, Lee, Nucl. Instr. Meth. Phys. Res. A
    467-468, 1572-1576 (2001), doi:10.1016/S0168-9002(01)00762-8

Reports peak position, FWHM, peak cross section and oscillator strength for
both, and separates the two ways a calculation can be wrong:

    SHAPE  peak position and width, set by the excited-state surface
    SCALE  sigma and f, set by |mu|^2

A deficit that is the same factor in sigma and in f is a pure scale error:
the surface is right and the dipole magnitude is not. That distinction
matters, because a constant scale factor cancels in isotope RATIOS while
it does not in absolute cross sections.

Usage:
    python 11_compare_obs.py --npz cross_section.npz --key sigma_200K
    python 11_compare_obs.py --species HDO --dataset "Chung(2001)_295K_140-192nm(1nm)"
"""
import argparse
import os
import urllib.request

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ("https://www.uv-vis-spectral-atlas-mainz.org/uvvis_data/"
        "cross_sections/Hydrogen+water/")
HC_NM_EV = 1239.841984


def fetch(species, dataset, cache="obs"):
    os.makedirs(cache, exist_ok=True)
    name = f"{species}_{dataset}.txt"
    path = os.path.join(cache, name.replace("/", "_"))
    if not os.path.exists(path):
        url = BASE + urllib.request.quote(name)
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    else:
        print(f"using cached {path}")
    d = np.loadtxt(path)
    return d[:, 0], d[:, 1]


def stats(lam_nm, sig):
    """Peak, FWHM in eV, peak sigma, oscillator strength."""
    i = int(np.argmax(sig))
    m = sig >= sig.max() / 2.0
    lo, hi = lam_nm[m].min(), lam_nm[m].max()
    fwhm = HC_NM_EV / lo - HC_NM_EV / hi
    nu = HC_NM_EV / lam_nm / 1.239841984e-4
    k = np.argsort(nu)
    f = 1.1296e12 * np.trapezoid(sig[k], nu[k])
    return lam_nm[i], sig.max(), abs(fwhm), f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="cross_section.npz")
    p.add_argument("--key", default=None,
                   help="which sigma_*K array to compare; default is the first")
    p.add_argument("--species", default="H2O")
    p.add_argument("--dataset", default="Chung(2001)_295K_140-195nm(1nm)")
    p.add_argument("--png", default="compare_obs.png")
    args = p.parse_args()

    d = np.load(args.npz)
    # sigma_v is the per-state array; the Boltzmann-averaged ones end in K
    keys = [k for k in d.files if k.startswith("sigma_") and k.endswith("K")]
    if not keys:
        raise SystemExit(f"no sigma_*K array in {args.npz}; found {d.files}")
    key = args.key or keys[0]
    print(f"comparing {key}" + (f"  (also available: {keys[1:]})"
                                if len(keys) > 1 else ""))
    lam_c = d["wavelength_nm"]
    sig_c = d[key]
    o = np.argsort(lam_c)
    lam_c, sig_c = lam_c[o], sig_c[o]

    lam_o, sig_o = fetch(args.species, args.dataset)

    pc, sc, wc, fc = stats(lam_c, sig_c)
    po, so, wo, fo = stats(lam_o, sig_o)

    print(f"\n{'':>12}{'peak/nm':>10}{'sigma/cm2':>13}{'FWHM/eV':>10}{'f':>9}")
    print(f"{'computed':>12}{pc:10.1f}{sc:13.3e}{wc:10.3f}{fc:9.4f}")
    print(f"{'measured':>12}{po:10.1f}{so:13.3e}{wo:10.3f}{fo:9.4f}")
    print(f"\nSHAPE  peak {pc - po:+.1f} nm "
          f"({HC_NM_EV/pc - HC_NM_EV/po:+.3f} eV), FWHM {100*(wc/wo - 1):+.1f}%")
    print(f"SCALE  sigma x{sc/so:.3f}, f x{fc/fo:.3f}")
    if abs(sc/so - fc/fo) < 0.05:
        print(f"       consistent -> pure scale error; mu is off by "
              f"x{np.sqrt(so/sc):.3f}.")
        print("       A constant factor cancels in isotope ratios.")
    else:
        print("       sigma and f scale differently -> the band SHAPE is "
              "off too,\n       not just the dipole magnitude.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(lam_o, sig_o, "k-", lw=2, label=f"{args.species} measured")
    ax[0].plot(lam_c, sig_c, label="computed")
    ax[0].plot(lam_c, sig_c * so / sc, "--", label=f"computed x{so/sc:.2f}")
    ax[0].set_ylabel(r"$\sigma$ / cm$^2$")
    ax[0].set_title("absolute")
    ax[1].plot(lam_o, sig_o / so, "k-", lw=2, label="measured")
    ax[1].plot(lam_c, sig_c / sc, label="computed")
    ax[1].set_ylabel("normalised")
    ax[1].set_title("shape only")
    for a in ax:
        a.set_xlim(135, 200)
        a.set_xlabel("wavelength / nm")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"\nwrote {args.png}")


if __name__ == "__main__":
    main()
