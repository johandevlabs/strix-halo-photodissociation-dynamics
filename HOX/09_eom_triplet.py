#!/usr/bin/env python3
"""
Phase 0.5, step 10: a third opinion on the triplet slope, from EOM-CCSD.

The band width scales with the slope of the a 3A" surface in the Franck-Condon
window, and two methods disagree about it on an identical grid:

    UCCSD(T), state-specific ROHF reference (05)     -7.00 eV/A at 1.6891 A
    SC-NEVPT2, CAS(10,6) (07, 08)                     -5.95 eV/A
    SC-NEVPT2, CAS(12,7) (08)                         -6.15 eV/A

The NEVPT2 slope steepens as the active space grows, which suggests the
active space is the limitation, but two points are not a trend. UCCSD(T) has
its own weak spot: its ROHF triplet reference wobbles near equilibrium (T1
0.035 -> 0.018 between neighbouring points).

EOM-CCSD reaches the triplet a different way from both. It starts from the
closed-shell CCSD ground state -- T1(S) is only ~0.008 here -- and finds the
triplet as an excitation of it, so there is no triplet reference to wobble and
no active space to choose. It is not fully independent of UCCSD(T), since both
are coupled cluster, but it removes exactly the two suspects: the ROHF
reference on one side and the active space on the other. water/12_vertical.py
used EOM-CCSD for the same job, where it came within 11 meV of NEVPT2.

What it reports, on 07's grid (r_eq + k x 0.05 A, k = -3..4):

  V_T = E_CCSD + omega_T    the triplet surface at the EOM-CCSD level
  V_T = E_CCSD(T) + omega_T  the same excitation on the CCSD(T) ground state,
                             the usual pragmatic combination
  d(omega_T)/dr              the vertical-energy slope itself

compared with UCCSD(T) and NEVPT2 read from the earlier CSVs where present.

ASSIGNMENT. Without symmetry the lowest EOM triplet is taken as a 3A". That
holds if it stays well separated from the next triplet (03 put the next one
~1 eV higher at equilibrium); the gap is checked at every point and printed.

Usage:
    python 09_eom_triplet.py 2>&1 | tee logs/eom_triplet.log
"""
import argparse
import csv
import os
import time
import traceback

import numpy as np

from pyscf import gto, scf, cc

HARTREE2EV = 27.211386245988
R_OH_ANG = 0.9644
R_OCL_EQ_ANG = 1.6891
ANGLE_HOCL_DEG = 102.96


def geometry(r_ocl):
    th = np.deg2rad(ANGLE_HOCL_DEG)
    return [["O", (0.0, 0.0, 0.0)],
            ["Cl", (r_ocl, 0.0, 0.0)],
            ["H", (R_OH_ANG * np.cos(th), R_OH_ANG * np.sin(th), 0.0)]]


def eom_triplets(mycc, nroots):
    """Lowest EOM-EE-CCSD triplet excitation energies (Ha) and convergence."""
    try:
        from pyscf.cc import eom_rccsd
        eom = eom_rccsd.EOMEETriplet(mycc)
        e, _ = eom.kernel(nroots=nroots)
        conv = getattr(eom, "converged", True)
    except (ImportError, AttributeError):
        e, _ = mycc.eomee_ccsd_triplet(nroots=nroots)
        conv = True
    e = np.sort(np.atleast_1d(np.asarray(e, dtype=float)))
    return e, bool(np.all(np.atleast_1d(conv)))


def run_point(r, args):
    t0, c0 = time.time(), time.process_time()
    mol = gto.M(atom=geometry(r), basis=args.basis, spin=0, charge=0,
                symmetry=False, unit="Angstrom", verbose=args.verbose,
                max_memory=16000)
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-10
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.kernel()
    t1 = float(np.linalg.norm(mycc.t1) / np.sqrt(2 * mycc.t1.shape[0]))
    e_t = float(mycc.ccsd_t())
    w, conv_eom = eom_triplets(mycc, args.nroots)
    row = {"r": r, "e_ccsd": float(mycc.e_tot),
           "e_ccsdt": float(mycc.e_tot) + e_t, "t1_s": t1,
           "conv": bool(mf.converged and mycc.converged and conv_eom)}
    for i, wi in enumerate(w):
        row[f"w_{i}"] = float(wi)
    row["wall"], row["cpu"] = time.time() - t0, time.process_time() - c0
    return row


def deriv_at(r, e_ev, at, deg=4):
    r = np.asarray(r, dtype=float)
    e = np.asarray(e_ev, dtype=float)
    deg = min(deg, len(r) - 2) if len(r) > 2 else 1
    c = np.polyfit(r - at, e - e.mean(), deg)
    return float(np.polyval(np.polyder(c), 0.0))


def load_curve(path, rkey, ekey, filt=None):
    if not os.path.exists(path):
        return None
    rs, es = [], []
    with open(path) as fh:
        for rec in csv.DictReader(fh):
            if filt and not filt(rec):
                continue
            try:
                rs.append(float(rec[rkey]))
                es.append(float(rec[ekey]) * HARTREE2EV)
            except (KeyError, ValueError):
                continue
    return (np.array(rs), np.array(es)) if rs else None


def analyse(rows, args):
    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    bad = [x["r"] for x in rows if not x["conv"]]
    if bad:
        print(f"  !! unconverged SCF/CCSD/EOM at r = "
              f"{', '.join(f'{v:.4f}' for v in bad)} A -- EXCLUDED")
    rows = [x for x in rows if x["conv"]]
    if len(rows) < 3:
        print("  too few converged points for a slope")
        return

    r = np.array([x["r"] for x in rows])
    w0 = np.array([x["w_0"] for x in rows]) * HARTREE2EV
    gaps = [(x["w_1"] - x["w_0"]) * HARTREE2EV for x in rows if "w_1" in x]
    if gaps:
        print(f"  lowest-triplet assignment: gap to the next EOM triplet "
              f"{min(gaps):.2f}-{max(gaps):.2f} eV"
              + ("" if min(gaps) > 0.3 else
                 "  !! under 0.3 eV somewhere: the lowest root may not be "
                 "the same state throughout"))

    at = args.at
    vt_ccsd = np.array([x["e_ccsd"] for x in rows]) * HARTREE2EV + w0
    vt_ccsdt = np.array([x["e_ccsdt"] for x in rows]) * HARTREE2EV + w0
    s_eom = deriv_at(r, vt_ccsd, at)
    s_eomt = deriv_at(r, vt_ccsdt, at)
    s_w = deriv_at(r, w0, at)
    i_eq = int(np.argmin(abs(r - R_OCL_EQ_ANG)))
    print(f"  EOM-CCSD vertical triplet energy at {r[i_eq]:.4f} A: "
          f"{w0[i_eq]:.4f} eV")

    table = [("EOM-CCSD:  E_CCSD + omega", s_eom),
             ("EOM-CCSD:  E_CCSD(T) + omega", s_eomt)]
    cc_t = load_curve(args.uccsdt_csv, "r", "e_t")
    if cc_t is not None:
        table.append(("UCCSD(T), 05 fine grid", deriv_at(*cc_t, at)))
    nev = load_curve(args.nevpt2_csv, "r", "e_nev_t",
                     filt=lambda rec: rec.get("space") == "(10,6)")
    if nev is not None:
        table.append(("SC-NEVPT2 CAS(10,6), 07", deriv_at(*nev, at)))
    n127 = load_curve(args.cas127_csv, "r", "e_sc_t")
    if n127 is not None:
        table.append(("SC-NEVPT2 CAS(12,7), 08", deriv_at(*n127, at, deg=1)))

    print(f"\n  slope of the triplet energy at r = {at:.4f} A (eV/A):")
    for name, s in table:
        print(f"    {name:32s} {s:+.3f}")
    print(f"  vertical-energy slope d(omega)/dr, EOM-CCSD: {s_w:+.3f} eV/A")

    ref = dict(table)
    uc = ref.get("UCCSD(T), 05 fine grid")
    nv = ref.get("SC-NEVPT2 CAS(10,6), 07")
    if uc is not None and nv is not None:
        e = s_eomt
        closer = "UCCSD(T)" if abs(e - uc) < abs(e - nv) else "NEVPT2"
        print(f"\n  EOM-CCSD(T)-ground slope {e:+.3f} is {abs(e - uc) / abs(uc):.1%} "
              f"from UCCSD(T) and {abs(e - nv) / abs(nv):.1%} from NEVPT2 "
              f"CAS(10,6): closer to {closer}.")
        if closer == "UCCSD(T)" and abs(e - uc) / abs(uc) < 0.05:
            print("  -> EOM-CCSD sides with UCCSD(T). The ROHF wobble did not "
                  "set the UCCSD(T)\n     slope, and the NEVPT2 slope is the "
                  "outlier -- consistent with its\n     active-space trend. "
                  "Trust coupled cluster in the Franck-Condon window.")
        elif closer == "NEVPT2" and abs(e - nv) / abs(nv) < 0.05:
            print("  -> EOM-CCSD sides with NEVPT2. The steep UCCSD(T) slope "
                  "is suspect,\n     plausibly its triplet ROHF reference. "
                  "Do not splice UCCSD(T) into the\n     Franck-Condon window "
                  "without resolving this.")
        else:
            print("  -> EOM-CCSD lands between the two, within 5% of neither. "
                  "Three methods,\n     three slopes: the band width is "
                  "uncertain at this level, and the\n     measured band is "
                  "the referee that is left.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--nroots", type=int, default=3)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--kmin", type=int, default=-3)
    p.add_argument("--kmax", type=int, default=4)
    p.add_argument("--at", type=float, default=R_OCL_EQ_ANG,
                   help="where to compare slopes (default the 1.6891 A used "
                        "throughout; 05's CCSD(T) minimum is ~1.698 A)")
    p.add_argument("--uccsdt-csv", default="hocl_scan_fc.csv")
    p.add_argument("--nevpt2-csv", default="hocl_fc_active_space.csv")
    p.add_argument("--cas127-csv", default="hocl_nevpt2_cas127.csv")
    p.add_argument("--csv", default="hocl_eom_triplet.csv")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    rs = [R_OCL_EQ_ANG + k * args.step
          for k in range(args.kmin, args.kmax + 1)]
    print("=" * 78)
    print("== HOX Phase 0.5 -- EOM-CCSD triplet slope, a third opinion")
    print(f"== {args.basis}, {args.nroots} EOM triplet roots, "
          f"r = {rs[0]:.4f} to {rs[-1]:.4f} A in {len(rs)} points")
    print("=" * 78)
    print(f"\n  {'r/A':>7}{'E_CCSD(T)/Ha':>18}{'T1(S)':>8}"
          f"{'omega0/eV':>11}{'omega1/eV':>11}{'conv':>6}{'t/s':>6}")
    rows = []
    for r in rs:
        try:
            row = run_point(r, args)
        except Exception as exc:
            print(f"  {r:7.4f}  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        rows.append(row)
        w1 = row.get("w_1", float("nan")) * HARTREE2EV
        print(f"  {r:7.4f}{row['e_ccsdt']:18.8f}{row['t1_s']:8.4f}"
              f"{row['w_0'] * HARTREE2EV:11.4f}{w1:11.4f}"
              f"{'yes' if row['conv'] else 'NO':>6}{row['wall']:6.0f}")
    if not rows:
        return
    cols = (["r", "e_ccsd", "e_ccsdt", "t1_s", "conv"]
            + [f"w_{i}" for i in range(args.nroots)] + ["wall", "cpu"])
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(f"\n  wrote {args.csv}")
    analyse(rows, args)


if __name__ == "__main__":
    main()
