#!/usr/bin/env python3
"""
Phase 1, step 3: the OH(X 2Pi) curve and E(Cl), the asymptote of the a 3A"
surface.

On the a 3A" surface HOCl breaks the O-Cl bond, so the departing atom is Cl
and the surviving fragment is OH -- the same diatomic water needed, by
coincidence. The asymptote is therefore not a number but a curve in the
spectator coordinate:

    V(R -> inf, r_OH, gamma)  =  E(Cl, 2P) + V_OH(r_OH)

Both pieces are computed here in BOTH method families, because the surface is
built from two:

  UCCSD(T)        matches the inner raster (12), which is CCSD(T) + EOM-CCSD
  CASSCF+NEVPT2   matches the outer shell (13)

so each part of the spliced surface can be checked against its own asymptote,
and the constant offset between the two families -- the thing 15 uses to join
them -- is measured here independently of the molecular calculations.

Agreement between the two near re is a check on both, exactly as in
water/05_oh_diatomic.py, where NEVPT2 and UCCSD(T) gave De 4.628 and 4.631 eV
against 4.62 observed. Divergence at long r would be the UCCSD(T) curve
failing as its UHF reference breaks, not the NEVPT2 one -- though OH stays
bound over the range the propagation needs, so that regime is not entered.

x2c scalar relativity everywhere, as in every other HOX surface script. Water
did not need it; here leaving it out would put the fragments on a different
energy scale from the surfaces they are meant to anchor.

NO POINT-GROUP SYMMETRY for the diatomic: PySCF's Abelian subgroup of
C-infinity-v splits the degenerate 2Pi across two irreps, which is water's
note at the same place.

Checks printed: re, the harmonic frequency and De of the OH curve against the
measured radical (re 0.9697 A, omega_e 3738 cm-1, De 4.62 eV -- quoted from
memory, verify), and the two asymptotes with their difference.

Usage:
    python 14_fragments.py 2>&1 | tee logs/fragments.log
"""
import argparse
import csv
import time

import numpy as np

from pyscf import gto, scf, cc, mcscf, mrpt, fci
from pyscf.mcscf import avas

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


def cl_mol(basis, verbose=0):
    return gto.M(atom="Cl 0 0 0", basis=basis, spin=1, symmetry=False,
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
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--rmin", type=float, default=0.70)
    p.add_argument("--rmax", type=float, default=1.40)
    p.add_argument("--step", type=float, default=0.025)
    p.add_argument("--avas-oh", nargs="+", default=["O 2s", "O 2p", "H 1s"])
    p.add_argument("--avas-cl", nargs="+", default=["Cl 3p"])
    p.add_argument("--minao", default="ano")
    p.add_argument("--csv", default="hocl_fragments.csv")
    args = p.parse_args()

    n = int(round((args.rmax - args.rmin) / args.step)) + 1
    rs = np.round(np.linspace(args.rmin, args.rmin + args.step * (n - 1), n), 4)

    print("=" * 78)
    print("== HOX Phase 1 -- OH(X 2Pi) curve and E(Cl): the a 3A\" asymptote")
    print(f"== {args.basis}, x2c, r(O-H) {rs[0]}-{rs[-1]} A in {len(rs)} points")
    print("=" * 78)

    t0 = time.time()
    e_cl_cc, _ = uccsd_t(cl_mol(args.basis))
    e_cl_nev, ncas_cl, nel_cl, conv_cl = casscf_nevpt2(
        cl_mol(args.basis), args.avas_cl, args.minao, 0.75)
    print(f"\n  Cl(2P):  UCCSD(T) {e_cl_cc:.8f} Ha   "
          f"NEVPT2 {e_cl_nev:.8f} Ha  CAS({nel_cl},{ncas_cl})"
          f"{'' if conv_cl else '  !! CASSCF unconverged'}")
    print(f"           difference {(e_cl_nev - e_cl_cc) * HARTREE2EV:+.3f} eV "
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
                     "conv": bool(cc_ok and cas_ok)})
        print(f"  {r:7.3f}{e_cc:16.8f}{e_nev:16.8f}"
              f"{(e_nev - e_cc) * HARTREE2EV:10.3f}{time.time() - t:6.1f}")

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["r_A", "e_uccsdt_Ha",
                                           "e_nevpt2_Ha", "ncas", "nelecas",
                                           "conv"])
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

    i_eq = int(np.argmin(np.abs(r - 0.9644)))
    print(f"\n  asymptote at r(O-H) = {r[i_eq]:.4f} A:")
    print(f"    UCCSD(T)  E(Cl) + V_OH = {e_cl_cc + rows[i_eq]['e_uccsdt_Ha']:.8f} Ha")
    print(f"    NEVPT2    E(Cl) + V_OH = {e_cl_nev + rows[i_eq]['e_nevpt2_Ha']:.8f} Ha")
    print(f"  05's fragment check (UCCSD(T), no x2c consistency check) gave "
          f"-536.71407 Ha;\n  the triplet at 4.0 A sat 28 meV from it.")
    print(f"\n  total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
