#!/usr/bin/env python3
"""
Tier 2c: the 1D OH(X 2Pi) potential and E_H, for the asymptotic splice.

The 3D raster stops around r1 = 3.0-3.5 A, where the A'/A" splitting has
already collapsed below ~0.05 eV. Beyond that the surface is replaced by

    V(r1 -> inf, r2, theta) = E_H + V_OH(r2)

Computing V_OH here as a dedicated 1D curve buys two things. It can be done
at a much larger basis than the 3D grid can afford, and the OH vibrational
levels it supports control the product state distribution the dynamics will
predict, which is hard to get right from a cheap 3D surface.

Two methods are computed at every point because both are cheap in 1D:

  CASSCF+NEVPT2   same method family as the 3D raster, so the splice is
                  internally consistent, and dissociates correctly
  UCCSD(T)        better in the bound region where the vibrational levels
                  live, but degrades past ~2x re as the UHF reference breaks

Agreement between them near re is a check on both. Divergence at large r is
expected and is the UCCSD(T) curve failing, not the NEVPT2 one.

Usage:
    python 05_oh_diatomic.py --basis aug-cc-pVQZ --npts 120
"""
import argparse
import csv
import time

import numpy as np

from pyscf import gto, scf, cc, mcscf, fci, mrpt
from pyscf.mcscf import avas

HARTREE2EV = 27.211386245988


def h_atom(basis):
    """E_H in the same basis, so the splice has no basis-set inconsistency."""
    mol = gto.M(atom="H 0 0 0", basis=basis, spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-11
    return mf.kernel()


def oh_mol(r, basis):
    # No point-group symmetry: PySCF's Abelian subgroup of C_inf_v gives
    # C2v labels for a diatomic, and the doublet Pi ground state is then
    # split across two irreps. Simpler to run without symmetry.
    return gto.M(atom=f"O 0 0 0; H 0 0 {r}", basis=basis, unit="Angstrom",
                 spin=1, verbose=0, max_memory=8000)


def nevpt2_point(mol, avas_aos):
    mf = scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    ncas, nelecas, orbs = avas.avas(mf, avas_aos, canonicalize=False, verbose=0)
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = 1e-9
    mc.verbose = 0
    mc.kernel(orbs)
    e_cas = mc.e_tot

    mci = mcscf.CASCI(mf, ncas, nelecas)
    mci.verbose = 0
    mci.kernel(mc.mo_coeff)
    e_nev = mci.e_tot + mrpt.NEVPT(mci).kernel()
    return e_cas, e_nev, ncas, nelecas


def uccsd_t_point(mol):
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    mycc = cc.UCCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    return mf.e_tot + mycc.e_corr + mycc.ccsd_t()


def make_grid(args):
    """
    Point distribution along r.

    "morse": uniform steps in y = 1 - exp(-a(r-re)), the natural coordinate
    for a diatomic. Dense at re where the vibrational levels live, coarsening
    automatically in the flat tail.

    "quadratic" was the original and is kept only for reproducing earlier
    runs. It anchors the density at RMIN, not re, so it spends points on the
    repulsive wall and the featureless tail while undersampling the well.
    """
    if args.spacing == "linear":
        return np.linspace(args.rmin, args.rmax, args.npts)
    if args.spacing == "quadratic":
        u = np.linspace(0.0, 1.0, args.npts)
        return args.rmin + (args.rmax - args.rmin) * u**2

    # Morse over the structured region, then UNIFORM in the tail.
    # y = 1 - exp(-a(r-re)) saturates as y -> 1, so a pure Morse grid starves
    # the long range however many points it is given: with a=1.6 over
    # 0.75-4.0 A the last three points land near 2.55, 2.92 and 4.00. V_OH is
    # flat there, but r2 in the 3D raster reaches 3.4 A and a spline through
    # three sparse points can ring.
    a, re = args.morse_a, args.r_eq
    r_tail = min(args.r_tail, args.rmax)
    n_tail = min(args.n_tail, args.npts - 2) if args.rmax > r_tail else 0
    n_main = args.npts - n_tail

    y_lo = 1.0 - np.exp(-a * (args.rmin - re))
    y_hi = 1.0 - np.exp(-a * (r_tail - re))
    y = np.linspace(y_lo, y_hi, n_main)
    main_r = re - np.log(1.0 - y) / a

    if n_tail:
        tail = np.linspace(r_tail, args.rmax, n_tail + 1)[1:]
        return np.concatenate([main_r, tail])
    return main_r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", default="aug-cc-pVQZ")
    p.add_argument("--rmin", type=float, default=0.75,
                   help="below ~0.75 A the curve is >2 eV up the repulsive "
                        "wall and chi_0 has no amplitude there")
    p.add_argument("--rmax", type=float, default=4.00,
                   help="the curve is flat to ~0.01 eV beyond 3 A; points "
                        "past that buy nothing")
    p.add_argument("--npts", type=int, default=40)
    p.add_argument("--avas-aos", nargs="+", default=["O 2s", "O 2p", "H 1s"])
    p.add_argument("--no-cc", action="store_true",
                   help="skip UCCSD(T); NEVPT2 alone is enough for the splice")
    p.add_argument("--spacing", choices=["morse", "quadratic", "linear"],
                   default="morse",
                   help="morse: dense at re (default). quadratic: dense at "
                        "rmin, kept only for reproducing earlier runs.")
    p.add_argument("--r-eq", type=float, default=0.9697,
                   help="OH equilibrium bond length, centre of the morse grid")
    p.add_argument("--r-tail", type=float, default=2.5,
                   help="Morse spacing below this, uniform above it")
    p.add_argument("--n-tail", type=int, default=8,
                   help="uniform points in the tail; the Morse variable "
                        "saturates and cannot sample the long range itself")
    p.add_argument("--morse-a", type=float, default=1.6,
                   help="Morse range parameter in 1/A; larger concentrates "
                        "more points near re")
    p.add_argument("--csv", default="oh_curve.csv")
    args = p.parse_args()

    rs = make_grid(args)

    e_h = h_atom(args.basis)
    print(f"basis={args.basis}   E_H = {e_h:.10f} Ha")
    print(f"{args.npts} points, r = {args.rmin}-{args.rmax} A, "
          f"{args.spacing} spacing")
    edges = np.quantile(rs, [0.0, 0.25, 0.5, 0.75, 1.0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rs >= lo) & (rs <= hi)
        step = np.diff(rs[m]).mean() if m.sum() > 1 else float("nan")
        print(f"   {lo:5.2f}-{hi:5.2f} A: {m.sum():3d} pts, "
              f"mean step {step:.4f} A")
    print()
    print(f"{'r/A':>7}{'E_nevpt2/Ha':>15}{'E_casscf/Ha':>15}"
          f"{'E_uccsdt/Ha':>15}{'t/s':>7}")

    rows = []
    for r in rs:
        t0 = time.perf_counter()
        mol = oh_mol(r, args.basis)
        try:
            e_cas, e_nev, ncas, ne = nevpt2_point(mol, args.avas_aos)
        except Exception as ex:
            print(f"{r:7.3f}  NEVPT2 failed: {type(ex).__name__}")
            e_cas = e_nev = np.nan
            ncas = ne = -1

        e_cc = np.nan
        if not args.no_cc:
            try:
                e_cc = uccsd_t_point(mol)
            except Exception:
                pass  # expected to fail at long r; the NEVPT2 curve carries on

        dt = time.perf_counter() - t0
        print(f"{r:7.3f}{e_nev:15.8f}{e_cas:15.8f}{e_cc:15.8f}{dt:7.1f}",
              flush=True)
        rows.append({"r_A": round(float(r), 5),
                     "e_casscf_Ha": e_cas, "e_nevpt2_Ha": e_nev,
                     "e_uccsd_t_Ha": e_cc, "ncas": ncas, "nelecas": str(ne)})

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) + ["e_H_Ha"])
        w.writeheader()
        for row in rows:
            row["e_H_Ha"] = e_h
            w.writerow(row)
    print(f"\nwrote {args.csv}")

    # De from the NEVPT2 curve: asymptote minus minimum
    e = np.array([x["e_nevpt2_Ha"] for x in rows], dtype=float)
    r = np.array([x["r_A"] for x in rows], dtype=float)
    if np.any(np.isfinite(e)):
        imin = int(np.nanargmin(e))
        de = (e[-1] - e[imin]) * HARTREE2EV
        print(f"NEVPT2: re ~ {r[imin]:.4f} A, De ~ {de:.3f} eV "
              f"(experiment: re 0.9697 A, De 4.62 eV)")


if __name__ == "__main__":
    main()
