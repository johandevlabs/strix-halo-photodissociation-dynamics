#!/usr/bin/env python3
"""
Tier 1: CCSD(T)/cc-pVXZ ladder on H2O.

Walks the correlation-consistent series on a fixed geometry and times each
stage separately (SCF, CCSD iterations, perturbative triples). The point is
to recover the formal scaling on *this* machine and get a wall-clock number
per basis size that can be set against a 2012 cluster allocation.

Nbf for H2O: DZ 24/aDZ 41, TZ 58/aTZ 92, QZ 115/aQZ 172, 5Z 201/a5Z 287, 6Z 322.
CCSD is O(N^6), (T) is O(N^7), so expect the triples to dominate from QZ up.

Verified reference values (frozen-core, this geometry, PySCF 2.14):
    cc-pVDZ  E[CCSD(T)] = -76.2410412160
    cc-pVTZ  E[CCSD(T)] = -76.3321941991
Use these as a correctness check before trusting any timing.

Note: the fitted exponents are overhead-dominated below ~100 basis functions
(DZ->TZ gives k ~ 2.8 for CCSD). They only approach the formal 6 and 7 from
QZ upward, which is the interesting part of the curve.

Usage:
    OMP_NUM_THREADS=16 python3 02_h2o_ladder.py --threads 16 \
        --basis cc-pVDZ cc-pVTZ cc-pVQZ cc-pV5Z --incore --mem 80000
"""
import argparse
import csv
import os
import resource
import time

from pyscf import cc, gto, scf

# Experimental equilibrium geometry, r(OH) = 0.9578 A, angle = 104.48 deg
GEOM = """
O   0.0000000   0.0000000   0.1173000
H   0.0000000   0.7572000  -0.4692000
H   0.0000000  -0.7572000  -0.4692000
"""


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def run_one(basis: str, args) -> dict:
    mol = gto.M(atom=GEOM, basis=basis, unit="Angstrom",
                verbose=0, max_memory=args.mem)
    nbf = mol.nao_nr()
    print(f"\n=== {basis}  ({nbf} basis functions) ===", flush=True)

    t0 = time.perf_counter()
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    e_hf = mf.kernel()
    t_scf = time.perf_counter() - t0
    print(f"  SCF    {t_scf:9.2f} s   E(HF)   = {e_hf:.10f}", flush=True)

    mycc = cc.CCSD(mf)
    mycc.max_memory = args.mem
    mycc.conv_tol = 1e-8
    if args.frozen_core:
        mycc.frozen = 1  # O 1s
    if args.incore:
        mycc.incore_complete = True

    t0 = time.perf_counter()
    mycc.kernel()
    t_ccsd = time.perf_counter() - t0
    print(f"  CCSD   {t_ccsd:9.2f} s   E_corr  = {mycc.e_corr:.10f}  "
          f"({mycc.cycles} iters)", flush=True)

    t0 = time.perf_counter()
    e_t = mycc.ccsd_t()
    t_triples = time.perf_counter() - t0
    print(f"  (T)    {t_triples:9.2f} s   E(T)    = {e_t:.10f}", flush=True)

    total = e_hf + mycc.e_corr + e_t
    print(f"  TOTAL  {t_scf + t_ccsd + t_triples:9.2f} s   "
          f"E[CCSD(T)] = {total:.10f}   peak RSS {peak_rss_gb():.1f} GB",
          flush=True)

    return {
        "basis": basis,
        "nbf": nbf,
        "threads": args.threads,
        "incore": args.incore,
        "frozen_core": args.frozen_core,
        "t_scf_s": round(t_scf, 3),
        "t_ccsd_s": round(t_ccsd, 3),
        "t_triples_s": round(t_triples, 3),
        "t_total_s": round(t_scf + t_ccsd + t_triples, 3),
        "ccsd_iters": mycc.cycles,
        "e_hf": e_hf,
        "e_corr_ccsd": mycc.e_corr,
        "e_triples": e_t,
        "e_ccsd_t": total,
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--basis", nargs="+",
                   default=["cc-pVDZ", "cc-pVTZ", "cc-pVQZ"],
                   help="basis sets to walk, smallest first")
    p.add_argument("--threads", type=int, default=int(
        os.environ.get("OMP_NUM_THREADS", 0)),
        help="thread count this run was launched with (label only)")
    p.add_argument("--mem", type=int, default=8000,
                   help="max_memory in MB handed to PySCF")
    p.add_argument("--incore", action="store_true",
                   help="force fully in-core CCSD (the point of 96 GB)")
    p.add_argument("--frozen-core", action="store_true", default=True)
    p.add_argument("--all-electron", dest="frozen_core", action="store_false")
    p.add_argument("--csv", default="h2o_ladder.csv")
    args = p.parse_args()

    print(f"threads={args.threads}  max_memory={args.mem} MB  "
          f"incore={args.incore}  frozen_core={args.frozen_core}")

    rows = []
    for basis in args.basis:
        try:
            rows.append(run_one(basis, args))
        except MemoryError:
            print(f"  !! {basis} ran out of memory, stopping ladder")
            break

    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended {len(rows)} rows to {args.csv}")

    if len(rows) >= 2:
        print("\nobserved scaling exponent (t ~ nbf^k):")
        for a, b in zip(rows, rows[1:]):
            import math
            r = math.log(b["nbf"] / a["nbf"])
            for stage, label in [("t_ccsd_s", "CCSD"), ("t_triples_s", "(T) ")]:
                if a[stage] > 0.05:
                    k = math.log(b[stage] / a[stage]) / r
                    print(f"  {label} {a['basis']:>10s} -> "
                          f"{b['basis']:<10s} k = {k:.2f}")


if __name__ == "__main__":
    main()
