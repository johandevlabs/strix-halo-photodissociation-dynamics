#!/usr/bin/env python3
"""
Tier 3i: all isotopologues against Chung et al. (2001), in one table.

All four measured datasets come from the same apparatus and the same paper,
so systematic experimental error largely cancels between them. That makes the
RELATIVE comparison much sharper than any single absolute one, and the
relative quantity is the isotope effect.

The prediction being tested: the band narrows with deuteration because chi_0
narrows, while the peak barely moves because the ZPE drop and the reduced
sampling of the repulsive wall cancel. Both trace back to the ground-state
surface, so the narrowing ratio is a real test of it.

Note the atlas naming is not uniform: HDO uses '140.0-195.0nm' while D2O
uses '140-195nm'.

Usage:
    python 16_compare_all.py
    python 16_compare_all.py --key sigma_298K       # match the 295 K data
"""
import argparse
import importlib.util
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HC = 1239.841984

# label -> (atlas species, dataset string)
DATASETS = {
    "H2O":  ("H2O", "Chung(2001)_295K_140-196nm(0.2nm)"),
    "HDO":  ("HDO", "Chung(2001)_295K_140.0-195.0nm(0.2nm)"),
    "HDOd": ("HDO", "Chung(2001)_295K_140.0-195.0nm(0.2nm)"),
    "D2O":  ("D2O", "Chung(2001)_295K_140-195nm(0.2nm)"),
}


def load_cmp(path="11_compare_obs.py"):
    spec = importlib.util.spec_from_file_location("cmp11", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--key", default=None,
                   help="which sigma_*K array; default is the first found")
    p.add_argument("--png", default="compare_all.png")
    args = p.parse_args()

    C = load_cmp()
    rows = {}
    for lab, (species, dataset) in DATASETS.items():
        npz = os.path.join(args.runs, lab, "cross_section.npz")
        if not os.path.exists(npz):
            print(f"skip {lab}: no {npz}")
            continue
        d = np.load(npz)
        keys = [k for k in d.files
                if k.startswith("sigma_") and k.endswith("K")]
        key = args.key if args.key in keys else keys[0]
        lam_c = d["wavelength_nm"]
        o = np.argsort(lam_c)
        lam_c, sig_c = lam_c[o], d[key][o]
        lam_o, sig_o = C.fetch(species, dataset)
        rows[lab] = (C.stats(lam_c, sig_c), C.stats(lam_o, sig_o),
                     (lam_c, sig_c, lam_o, sig_o), key)

    if not rows:
        raise SystemExit("nothing to compare")

    print(f"\n{'':>6}{'':>10}{'peak/nm':>10}{'sigma/cm2':>13}"
          f"{'FWHM/eV':>10}{'f':>9}")
    for lab, (c, o, _, key) in rows.items():
        print(f"{lab:>6}{'calc':>10}{c[0]:10.1f}{c[1]:13.3e}{c[2]:10.3f}{c[3]:9.4f}")
        print(f"{'':>6}{'meas':>10}{o[0]:10.1f}{o[1]:13.3e}{o[2]:10.3f}{o[3]:9.4f}")

    print(f"\n{'':>6}{'peak err/eV':>13}{'mu scale':>11}"
          f"{'FWHM err':>11}")
    for lab, (c, o, _, _) in rows.items():
        print(f"{lab:>6}{HC/c[0]-HC/o[0]:+13.3f}"
              f"{np.sqrt(o[1]/c[1]):11.3f}{100*(c[2]/o[2]-1):+10.1f}%")

    # the isotope effect: relative narrowing, where experimental systematics
    # largely cancel because all datasets are from one apparatus
    if "H2O" in rows:
        print(f"\nband narrowing relative to H2O (the isotope effect):")
        print(f"{'':>6}{'calc':>10}{'meas':>10}{'diff':>9}")
        c0, o0 = rows["H2O"][0][2], rows["H2O"][1][2]
        for lab, (c, o, _, _) in rows.items():
            print(f"{lab:>6}{c[2]/c0:10.3f}{o[2]/o0:10.3f}"
                  f"{100*((c[2]/c0)/(o[2]/o0)-1):+8.1f}%")

    n = len(rows)
    fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 3.8), squeeze=False)
    for a, (lab, (c, o, curves, _)) in zip(ax[0], rows.items()):
        lam_c, sig_c, lam_o, sig_o = curves
        a.plot(lam_o, sig_o, "k-", lw=2, label="measured")
        a.plot(lam_c, sig_c * o[1] / c[1], "--", label=f"calc x{o[1]/c[1]:.2f}")
        a.set_xlim(135, 200)
        a.set_xlabel("wavelength / nm")
        a.set_title(lab)
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    ax[0][0].set_ylabel(r"$\sigma$ / cm$^2$")
    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"\nwrote {args.png}")


if __name__ == "__main__":
    main()
