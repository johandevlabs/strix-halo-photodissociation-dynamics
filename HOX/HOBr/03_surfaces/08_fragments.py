#!/usr/bin/env python3
"""
HOBr step 8: the OH(X 2Pi) curve and E(Br), the asymptote of the a 3A" surface.

HOCl's 14_fragments.py with Br as the leaving atom. On the a 3A" surface HOBr
breaks the O-Br bond, so the surface must tend to

    V(R -> inf, r_OH, angle)  =  E(Br, 2P) + V_OH(r_OH)

a curve in the spectator coordinate, not a number. Both method families are
computed, UCCSD(T) and CASSCF + SC-NEVPT2, x2c throughout (or the fragments
sit on a different energy scale from the surfaces they anchor), no point
group for the diatomic (PySCF's Abelian subgroup of C-inf-v splits 2Pi).
What made HOCl's splice trustworthy was the NEVPT2 - UCCSD(T) difference
being nearly CONSTANT along the OH curve (36 meV spread over 0.80-1.25 A),
so that is the verdict to read.

One change beyond the atom: E(Br) is written INTO the CSV, in every row, in
both methods. HOCl's splice took E(Cl) as a command-line default copied from
14's log -- a number that silently goes stale if 14 is rerun with a
different basis. 09_splice.py reads it from here.

The OH curve is the same molecule as HOCl's but in cc-pvtz-dk, not def2-tzvp,
so it is recomputed rather than reused: the asymptote has to be on the
raster's basis.

Usage:
    python 08_fragments.py 2>&1 | tee ../logs/fragments.log
"""
import argparse
import os
import csv
import time

import numpy as np

from pyscf import gto, scf, cc, mcscf, mrpt, fci
from pyscf.mcscf import avas
from pathlib import Path

# Data files live in HOBr/data/, one level up from this approach directory,
# so the defaults below work no matter where the script is invoked from.
DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2EV = 27.211386245988
HARTREE2CM = 219474.6313702
AMU2AU = 1822.888486209
M_O, M_H = 15.99491462, 1.00782503

# Measured OH(X 2Pi) radical constants, quoted from memory -- verify before
# these appear anywhere that matters (water/README.md keeps the same caveat).
OBS_RE_ANG, OBS_WE_CM, OBS_DE_EV = 0.9697, 3738.0, 4.62


def oh_mol(r, basis, verbose=0):
    return gto.M(atom=[["O", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, r)]],
                 basis=basis, spin=1, symmetry=False, unit="Angstrom",
                 verbose=verbose, max_memory=8000)


def x_mol(basis, verbose=0):
    return gto.M(atom="Br 0 0 0", basis=basis, spin=1, symmetry=False,
                 verbose=verbose, max_memory=8000)


def uccsd_t(mol):
    mf = scf.ROHF(mol).x2c()
    mf.conv_tol = 1e-11
    mf.kernel()
    mycc = cc.UCCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.kernel()
    return float(mf.e_tot + mycc.e_corr + mycc.ccsd_t()), bool(mycc.converged)


def casscf_nevpt2(mol, aolabels, minao, ss):
    mf = scf.ROHF(mol).x2c()
    mf.conv_tol = 1e-11
    mf.kernel()
    ncas, nelecas, orbs = avas.avas(mf, aolabels, minao=minao, verbose=0)
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    fci.addons.fix_spin_(mc.fcisolver, ss=ss)
    mc.conv_tol, mc.max_cycle_macro, mc.verbose = 1e-10, 200, 0
    mc.kernel(orbs)
    mci = mcscf.CASCI(mf, ncas, nelecas)
    fci.addons.fix_spin_(mci.fcisolver, ss=ss)
    mci.verbose = 0
    mci.kernel(mc.mo_coeff)
    e = float(mci.e_tot + mrpt.NEVPT(mci).kernel())
    return e, ncas, nelecas, bool(mc.converged)


def curve_constants(r_ang, e_ha):
    """re, harmonic omega_e and De from a curve (minimum + outer plateau)."""
    r = np.asarray(r_ang)
    e = np.asarray(e_ha)
    i = int(np.argmin(e))
    if i < 2 or i > len(r) - 3:
        return float("nan"), float("nan"), float("nan"), i
    sel = slice(i - 2, i + 3)
    c = np.polyfit(r[sel] - r[i], e[sel], 2)          # E = c0 x^2 + c1 x + c2
    re = r[i] - c[1] / (2 * c[0])
    bohr_per_ang = 1.0 / 0.529177210903
    k_au = 2.0 * c[0] / bohr_per_ang ** 2             # Ha/A^2 -> Ha/bohr^2
    mu = M_O * M_H / (M_O + M_H) * AMU2AU
    we = np.sqrt(max(k_au, 0.0) / mu) * HARTREE2CM
    de = (e[-1] - e[i]) * HARTREE2EV
    return float(re), float(we), float(de), i


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="cc-pvtz-dk")
    p.add_argument("--rmin", type=float, default=0.70)
    p.add_argument("--rmax", type=float, default=1.40)
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--avas-oh", nargs="+", default=["O 2s", "O 2p", "H 1s"])
    p.add_argument("--avas-x", nargs="+", default=["Br 4p"])
    p.add_argument("--minao", default="ano")
    p.add_argument("--csv", default=str(DATA / "hobr_fragments.csv"))
    args = p.parse_args()

    n = int(round((args.rmax - args.rmin) / args.step)) + 1
    rs = np.round(np.linspace(args.rmin, args.rmin + args.step * (n - 1), n), 4)

    print("=" * 78)
    print("== HOX Phase 2 -- OH(X 2Pi) curve and E(Br): the HOBr a 3A\" asymptote")
    print(f"== {args.basis}, x2c, r(O-H) {rs[0]}-{rs[-1]} A in {len(rs)} points")
    print("=" * 78)

    t0 = time.time()
    e_x_cc, _ = uccsd_t(x_mol(args.basis))
    e_x_nev, ncas_x, nel_x, conv_x = casscf_nevpt2(
        x_mol(args.basis), args.avas_x, args.minao, 0.75)
    print(f"\n  Br(2P):  UCCSD(T) {e_x_cc:.8f} Ha   "
          f"NEVPT2 {e_x_nev:.8f} Ha  CAS({nel_x},{ncas_x})"
          f"{'' if conv_x else '  !! CASSCF unconverged'}")
    print(f"           difference {(e_x_nev - e_x_cc) * HARTREE2EV:+.3f} eV "
          f"(NEVPT2 recovers less correlation; only its CONSTANCY matters)")

    print(f"\n  {'r/A':>7}{'E_uccsdt/Ha':>16}{'E_nevpt2/Ha':>16}"
          f"{'diff/eV':>10}{'t/s':>6}")
    rows = []
    for r in rs:
        t = time.time()
        mol = oh_mol(r, args.basis)
        try:
            e_cc, cc_ok = uccsd_t(mol)
        except Exception as ex:
            e_cc, cc_ok = float("nan"), False
            print(f"  {r:7.3f}  UCCSD(T) failed: {type(ex).__name__}")
        try:
            e_nev, ncas, nel, cas_ok = casscf_nevpt2(
                oh_mol(r, args.basis), args.avas_oh, args.minao, 0.75)
        except Exception as ex:
            e_nev, ncas, nel, cas_ok = float("nan"), 0, 0, False
            print(f"  {r:7.3f}  NEVPT2 failed: {type(ex).__name__}")
        rows.append({"r_A": r, "e_uccsdt_Ha": e_cc, "e_nevpt2_Ha": e_nev,
                     "ncas": ncas, "nelecas": nel,
                     "conv": bool(cc_ok and cas_ok),
                     "e_x_uccsdt_Ha": e_x_cc, "e_x_nevpt2_Ha": e_x_nev})
        print(f"  {r:7.3f}{e_cc:16.8f}{e_nev:16.8f}"
              f"{(e_nev - e_cc) * HARTREE2EV:10.3f}{time.time() - t:6.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    with open(args.csv, "w", newline="") as fh:
        # E(Br) in every row, so the splice reads it rather than taking a
        # number copied from a log (HOCl's 15 did that with E(Cl)).
        w = csv.DictWriter(fh, fieldnames=["r_A", "e_uccsdt_Ha",
                                           "e_nevpt2_Ha", "ncas", "nelecas",
                                           "conv", "e_x_uccsdt_Ha",
                                           "e_x_nevpt2_Ha"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n  wrote {args.csv}")

    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    r = np.array([x["r_A"] for x in rows])
    for name, keyname in (("UCCSD(T)", "e_uccsdt_Ha"), ("NEVPT2", "e_nevpt2_Ha")):
        e = np.array([x[keyname] for x in rows])
        if np.all(~np.isfinite(e)):
            print(f"  {name}: no usable points")
            continue
        re, we, de, i = curve_constants(r, e)
        print(f"  {name:9s} re {re:.4f} A (obs {OBS_RE_ANG})   "
              f"omega_e {we:.0f} cm-1 (obs {OBS_WE_CM:.0f})   "
              f"well depth over this range {de:.3f} eV")
    print(f"  (the range stops at {rs[-1]} A, well inside dissociation, so the "
          f"last column is not De;\n   obs De {OBS_DE_EV} eV is quoted for "
          f"scale only. All three measured values are from memory.)")

    diff = np.array([(x["e_nevpt2_Ha"] - x["e_uccsdt_Ha"]) * HARTREE2EV
                     for x in rows])
    good = np.isfinite(diff)
    print(f"\n  NEVPT2 - UCCSD(T) across the curve: {diff[good].min():.3f} to "
          f"{diff[good].max():.3f} eV, spread "
          f"{diff[good].max() - diff[good].min():.3f} eV")
    print("  A CONSTANT difference is what the splice needs: it means the two "
          "families agree on\n  the shape of the spectator coordinate and "
          "differ only by a fixed offset.")

    i_eq = int(np.argmin(np.abs(r - 0.9646)))
    print(f"\n  asymptote at r(O-H) = {r[i_eq]:.4f} A:")
    print(f"    UCCSD(T)  E(Br) + V_OH = {e_x_cc + rows[i_eq]['e_uccsdt_Ha']:.8f} Ha")
    print(f"    NEVPT2    E(Br) + V_OH = {e_x_nev + rows[i_eq]['e_nevpt2_Ha']:.8f} Ha")
    print(f"  05's fragment check (UCCSD(T), no x2c consistency check) gave "
          f"-536.71407 Ha;\n  the triplet at 4.0 A sat 28 meV from it.")
    print(f"\n  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
