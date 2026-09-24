#!/usr/bin/env python3
"""
HOBr step 7: the multireference outer shell of the a 3A" surface.

HOCl's 13_outer_shell.py with HOBr's numbers in it. 05's raster stops at
2.45 A because EOM-CCSD does (02: single-excitation weight below 0.90 from
2.45, 3A' lowest from 2.55), but the propagation grid runs to the asymptote.
This fills 2.25-3.85 A with the symmetry-forced 3A" SA-CASSCF(10,6) +
SC-NEVPT2 that carried HOCl's surface out -- sound where EOM is not, and only
asked for a smooth path into the absorber and onto E(Br) + V_OH, since 03
showed the band cannot see past ~2.0 A.

What carries over unchanged: 4 averaged 3A" roots (single-root CASSCF did not
converge for HOCl), triplet SCF seeded from the closed-shell density of the
SAME geometry with a second-order fallback (the fix for HOCl's 41 fail:rohf),
the active space rebuilt per point, and --retry-failed.

What changed, and why:
  labels     AVAS projector on ["Br 4p", "O 2p"], 5 occupied + 1 virtual:
             CAS(10,6), as for HOCl. The active-space machinery is imported
             from HOCl/01_method/07_fc_active_space.py, the code 07 validated,
             rather than copied into a second place to drift.
  extent     2.25-3.85 A. Starts INSIDE the raster -- 2.25, 2.35, 2.45 are
             raster radii -- because the overlap is what fixes the offset
             between the two methods. Ends r_eq + ~2.0 A, as HOCl's did.
  basis      cc-pvtz-dk, the raster's.
  pool       capped at the task count; data directory created if missing.

Usage:
    python 07_outer_shell.py --dry-run
    python 07_outer_shell.py --pilot 30 --nproc 30
    python 07_outer_shell.py --nproc 30
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["OMP_PROC_BIND"] = "false"
os.environ.pop("OMP_PLACES", None)

import argparse  # noqa: E402
import csv  # noqa: E402
import importlib.util  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

# Data files live in HOBr/data/, one level up from this approach directory,
# so the defaults below work no matter where the script is invoked from.
DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2EV = 27.211386245988
DEG = np.pi / 180.0
R_OX_EQ, R_OH_EQ, ANGLE_EQ = 1.8357, 0.9646, 102.02   # 01_geometry.py

FIELDS = ["r_obr_A", "r_oh_A", "theta_deg", "ncas", "nelecas", "conv_cas",
          "e_casscf_Ha", "e_nevpt2_Ha", "s2", "gap_app_eV", "status", "wall_s"]

_HERE = os.path.dirname(os.path.abspath(__file__))
# 07 lives in the sibling approach directory, where the method was chosen.
# The active-space machinery lives with HOCl, where 07 validated it; it is
# imported from there rather than copied, so there is one version.
_FC07_PY = os.path.join(_HERE, os.pardir, os.pardir, "HOCl", "01_method",
                        "07_fc_active_space.py")
_fc07 = None


def fc07():
    """07's active-space machinery, loaded once (workers inherit by fork)."""
    global _fc07
    if _fc07 is None:
        spec = importlib.util.spec_from_file_location("fc07", _FC07_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _fc07 = mod
    return _fc07


def geometry(r_ox, r_oh, angle):
    th = angle * DEG
    return [["O", (0.0, 0.0, 0.0)],
            ["Br", (r_ox, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def compute_point(task):
    """Lowest 3A" by SA-CASSCF + SC-NEVPT2 at one geometry."""
    r_ox, r_oh, theta, basis, nroots, n_occ, n_vir, mem, max_cycle = task
    t0 = time.perf_counter()
    row = {"r_obr_A": round(r_ox, 4), "r_oh_A": round(r_oh, 4),
           "theta_deg": round(theta, 3), "status": "ok"}
    try:
        from pyscf import gto, scf, mcscf, mrpt, fci
        from pyscf.fci import spin_op
        f = fc07()

        atom = geometry(r_ox, r_oh, theta)
        mol_s = gto.M(atom=atom, basis=basis, spin=0, charge=0, symmetry="Cs",
                      unit="Angstrom", verbose=0, max_memory=mem)
        mf_s = scf.RHF(mol_s).x2c()
        mf_s.conv_tol = 1e-10
        mf_s.kernel()
        if not mf_s.converged:
            row["status"] = "fail:rhf"
            return _stamp(row, t0)

        proj = f.Projector(mol_s, ["Br 4p", "O 2p"], "ano")
        mo, ncore, ncas, nelecas = f.select_fixed(mf_s, proj, n_occ, n_vir)
        mo = f.canonicalize_like_avas(mol_s, mf_s, mo, ncore, ncas)
        mo = f.symmetrize_blocks(mol_s, mo, mf_s.get_ovlp(), ncore, ncas)
        row["ncas"], row["nelecas"] = ncas, nelecas

        mol_t = gto.M(atom=atom, basis=basis, spin=2, charge=0, symmetry="Cs",
                      unit="Angstrom", verbose=0, max_memory=mem)
        # Seed the triplet SCF from the converged closed-shell density of the
        # same geometry, and fall back to a second-order solver. Plain ROHF
        # from the default guess failed at 41 of 1050 points, all at
        # r(O-Cl) >= 2.4 A where the fragments become two open-shell radicals
        # and the default guess is poor. This is a local guess, not orbitals
        # chained from a neighbouring geometry -- water/README.md's warning is
        # about propagating an ACTIVE SPACE along a scan, which AVAS still
        # rebuilds per point here.
        mf_t = scf.ROHF(mol_t).x2c()
        mf_t.conv_tol = 1e-10
        mf_t.max_cycle = 100
        mf_t.kernel(dm0=mf_s.make_rdm1())
        if not mf_t.converged:
            mf_t = mf_t.newton()
            mf_t.kernel(mf_t.mo_coeff, mf_t.mo_occ)
            if mf_t.converged:
                row["status"] = "ok:newton"
        if not mf_t.converged:
            row["status"] = "fail:rohf"
            return _stamp(row, t0)

        mc = mcscf.CASSCF(mf_t, ncas, nelecas)
        mc.fcisolver = f.triplet_solver(mol_t, nroots)
        if nroots > 1:
            mc = mc.state_average_(np.ones(nroots) / nroots)
        mc.conv_tol, mc.max_cycle_macro, mc.verbose = 1e-8, max_cycle, 0
        mc.kernel(mo)
        row["conv_cas"] = bool(mc.converged)

        mci = mcscf.CASCI(mf_t, ncas, nelecas)
        mci.fcisolver, mci.verbose = f.triplet_solver(mol_t, nroots), 0
        mci.kernel(mc.mo_coeff)
        e_cas = np.atleast_1d(np.asarray(mci.e_tot, dtype=float))
        cis = mci.ci if isinstance(mci.ci, (list, tuple)) else [mci.ci]
        row["e_casscf_Ha"] = float(e_cas[0])
        row["s2"] = float(spin_op.spin_square0(cis[0], ncas, mci.nelecas)[0])
        row["gap_app_eV"] = (float((e_cas[1] - e_cas[0]) * HARTREE2EV)
                             if e_cas.size > 1 else float("nan"))
        row["e_nevpt2_Ha"] = float(e_cas[0]
                                   + mrpt.NEVPT(mci, root=0).kernel())

        flags = []
        if not row["conv_cas"]:
            flags.append("cas")
        if abs(row["s2"] - 2.0) > 0.02:
            flags.append("s2")
        if flags:
            row["status"] = "warn:" + "+".join(flags)
    except Exception as ex:
        row["status"] = f"fail:{type(ex).__name__}"
    return _stamp(row, t0)


def _stamp(row, t0):
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row


def _init_worker(counter):
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    try:
        os.sched_setaffinity(0, {idx % (os.cpu_count() or 1)})
    except (AttributeError, OSError):
        pass


def key(a, b, c):
    return (round(float(a), 4), round(float(b), 4), round(float(c), 3))


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add(key(row["r_obr_A"], row["r_oh_A"], row["theta_deg"]))
            except (KeyError, ValueError):
                continue
    return done


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="cc-pvtz-dk")
    p.add_argument("--rox-min", type=float, default=2.25,
                   help="starts INSIDE the raster: the overlap is what fixes "
                        "the offset between the two methods")
    p.add_argument("--rox-max", type=float, default=3.85)
    p.add_argument("--rox-step", type=float, default=0.10)
    p.add_argument("--roh-min", type=float, default=0.80)
    p.add_argument("--roh-max", type=float, default=1.25)
    p.add_argument("--roh-step", type=float, default=0.05)
    p.add_argument("--theta-min", type=float, default=75.0)
    p.add_argument("--theta-max", type=float, default=135.0)
    p.add_argument("--theta-step", type=float, default=10.0)
    p.add_argument("--nroots", type=int, default=4,
                   help="3A\" roots to average; 4 is what 06 and 08 used, and "
                        "single-root CASSCF did not converge in 04/08")
    p.add_argument("--n-occ", type=int, default=5)
    p.add_argument("--n-vir", type=int, default=1)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--nproc", type=int, default=30)
    p.add_argument("--mem", type=int, default=3000)
    p.add_argument("--pilot", type=int, default=0)
    p.add_argument("--sec-per-point", type=float, default=20.0,
                   help="single-threaded CPU-s per point; HOCl measured 4, "
                        "and Br adds 18 correlated core electrons. The "
                        "pilot measures it")
    p.add_argument("--retry-failed", action="store_true",
                   help="drop rows whose status is fail (keeping a .bak) "
                        "so this run recomputes exactly those points")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--csv", default=str(DATA / "hobr_outer_shell.csv"))
    args = p.parse_args()

    def grid(lo, hi, step):
        n = int(round((hi - lo) / step)) + 1
        return np.round(np.linspace(lo, lo + step * (n - 1), n), 4)

    rox = grid(args.rox_min, args.rox_max, args.rox_step)
    roh = grid(args.roh_min, args.roh_max, args.roh_step)
    theta = grid(args.theta_min, args.theta_max, args.theta_step)
    points = [(a, b, c) for a in rox for b in roh for c in theta]

    print("=" * 78)
    print('== HOX Phase 2 -- HOBr multireference outer shell of the a 3A" surface')
    print("=" * 78)
    print(f"  r(O-Br) {rox[0]}-{rox[-1]} A, {len(rox)} points "
          f"(overlaps the raster below 2.40)")
    print(f"  r(O-H)  {roh[0]}-{roh[-1]} A, {len(roh)} points")
    print(f"  angle   {theta[0]}-{theta[-1]} deg, {len(theta)} points")
    print(f"  total   {len(points)} geometries, CAS({2 * args.n_occ},"
          f"{args.n_occ + args.n_vir}) SA-{args.nroots} + SC-NEVPT2, "
          f"{args.basis}")
    cpu_h = len(points) * args.sec_per_point / 3600.0
    print(f"  estimate {cpu_h:.0f} CPU-hours = {cpu_h / args.nproc:.1f} h on "
          f"{args.nproc} workers at {args.sec_per_point:.0f} s/point")

    if args.pilot:
        idx = np.unique(np.linspace(0, len(points) - 1,
                                    args.pilot).astype(int))
        points = [points[i] for i in idx]
        print(f"  PILOT: {len(points)} points spread over the shell; same CSV")

    if args.retry_failed and os.path.exists(args.csv):
        keep, dropped = [], 0
        with open(args.csv, newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status", "").startswith("fail"):
                    dropped += 1
                else:
                    keep.append(row)
        if dropped:
            os.replace(args.csv, args.csv + ".bak")
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                for row in keep:
                    w.writerow({k: row.get(k, "") for k in FIELDS})
            print(f"  --retry-failed: dropped {dropped} failed rows "
                  f"(old file kept as {args.csv}.bak)")

    done = load_done(args.csv)
    todo = [t for t in points if key(*t) not in done]
    print(f"  {len(done)} already in {args.csv}, {len(todo)} to run\n")
    if args.dry_run or not todo:
        print("dry run: stopping here" if todo else "nothing to do")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    new_file = not os.path.exists(args.csv)
    tasks = [(a, b, c, args.basis, args.nroots, args.n_occ, args.n_vir,
              args.mem, args.max_cycle) for a, b, c in todo]
    t_start = time.perf_counter()
    n_ok = n_warn = n_bad = 0
    with open(args.csv, "a", newline="", buffering=1) as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        ctx = (mp.get_context("fork") if "fork" in mp.get_all_start_methods()
               else mp.get_context())
        counter = ctx.Value("i", 0)
        with ctx.Pool(min(args.nproc, len(tasks)), initializer=_init_worker,
                      initargs=(counter,)) as pool:
            for i, row in enumerate(
                    pool.imap_unordered(compute_point, tasks, chunksize=1), 1):
                w.writerow({k: row.get(k, "") for k in FIELDS})
                st = row["status"]
                n_ok += st.startswith("ok")
                n_warn += st.startswith("warn")
                n_bad += st.startswith("fail")
                if i % 10 == 0 or i == len(tasks):
                    el = time.perf_counter() - t_start
                    rate = i / el
                    print(f"  {i}/{len(tasks)}  ok={n_ok} warn={n_warn} "
                          f"bad={n_bad}  {rate * 3600:.0f} pts/h  "
                          f"eta {(len(tasks) - i) / rate / 60:.1f} min",
                          flush=True)

    el = time.perf_counter() - t_start
    per_point = el * min(args.nproc, len(tasks)) / max(len(tasks), 1)
    print(f"\ndone: {n_ok} ok, {n_warn} warned, {n_bad} failed, {el / 60:.1f} min")
    print(f"measured {per_point:.0f} CPU-s per point "
          f"(was assuming {args.sec_per_point:.0f})")
    full = len(rox) * len(roh) * len(theta)
    print(f"-> the full {full}-point shell is {full * per_point / 3600:.0f} "
          f"CPU-hours = {full * per_point / 3600 / args.nproc:.1f} h on "
          f"{args.nproc} workers")


if __name__ == "__main__":
    main()
