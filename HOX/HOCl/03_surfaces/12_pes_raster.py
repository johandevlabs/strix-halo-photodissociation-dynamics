#!/usr/bin/env python3
"""
Phase 1, step 1: the 3D CCSD(T) + EOM-CCSD raster for the HOCl a 3A" band.

One calculation per geometry gives BOTH surfaces the band needs: the CCSD(T)
ground state, and the a 3A" triplet as E_CCSD(T) + omega(EOM-CCSD). Phase 0.5
established why this method and this extent:

  09  EOM-CCSD matches UCCSD(T)'s slope to 0.5% and is far smoother (0.1 vs
      4.2 meV), with no ROHF triplet reference to wobble and no active space
      to choose. Trustworthy along O-Cl to 2.29 A, where a second triplet,
      probably 3A', crosses.
  10  the 1D band depends on the surface only out to ~2 A: flattening V_T
      beyond 2.10 A changes it 0.1%, and nothing beyond 2.3 A matters at all.
      So the raster can stop inside the crossing.
  11  across the O-H stretch, the bend and the corners of the Franck-Condon
      region, the lowest triplet is 3A" everywhere, T1(S) <= 0.010, and both
      surfaces are smooth.

COORDINATES are valence (r(O-Cl), r(O-H), H-O-Cl angle), transformed to
Jacobi later, as water/07_jacobi.py does. Cs is FORCED, not detected
(water/README.md), which also lets every EOM root be labelled by the irrep of
its dominant single excitation, so what is stored is the lowest 3A" rather
than merely the lowest triplet -- the check 11 added, kept here because a
raster reaches geometries no cut visited.

PARALLELISM is water's job-array pattern: one single-threaded process per
geometry, many processes. The environment variables at the top of this file
must precede numpy and pyscf -- BLAS reads them at load time -- and OpenMP
pinning is cleared explicitly, because OMP_PLACES=cores with
OMP_PROC_BIND=close makes every worker bind to place 0 and pile onto one
physical core (water/README.md's first gotcha).

RESUMABLE: rows append to the CSV as they finish and a rerun skips what is
already there, so interrupting is safe. --pilot writes into the SAME csv, so
its points count towards the full run rather than being thrown away.

Usage:
    python 12_pes_raster.py --dry-run                 # grid size and estimate
    python 12_pes_raster.py --pilot 60                # ~20 min, measures cost
    python 12_pes_raster.py --nproc 30                # the full raster
"""
import os

# MUST precede numpy/pyscf: BLAS reads these at library load time, and the
# design here is one thread per process, many processes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
# Clear OpenMP thread PINNING, or every independent worker binds to place 0
# and they all contend for one physical core while the load average looks fine.
os.environ["OMP_PROC_BIND"] = "false"
os.environ.pop("OMP_PLACES", None)

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from pathlib import Path  # noqa: E402

# Data files live in HOCl/data/, one level up from this approach directory,
# so the defaults below work no matter where the script is invoked from.
DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2EV = 27.211386245988
DEG = np.pi / 180.0

R_OCL_EQ, R_OH_EQ, ANGLE_EQ = 1.6891, 0.9644, 102.96
T1_MAX = 0.02

FIELDS = ["r_ocl_A", "r_oh_A", "theta_deg", "nbf", "t1_s",
          "e_hf_Ha", "e_ccsd_Ha", "e_ccsdt_Ha", "omega_app_eV",
          "root0_sym", "gap_any_eV", "gap_app_eV", "w1", "status", "wall_s"]


def geometry(r_ocl, r_oh, angle):
    th = angle * DEG
    return [["O", (0.0, 0.0, 0.0)],
            ["Cl", (r_ocl, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def compute_point(task):
    """One geometry: CCSD(T) ground state and the lowest 3A" by EOM-CCSD."""
    r_ocl, r_oh, theta, basis, nroots, mem = task
    t0 = time.perf_counter()
    row = {"r_ocl_A": round(r_ocl, 4), "r_oh_A": round(r_oh, 4),
           "theta_deg": round(theta, 3), "status": "ok",
           "root0_sym": "", "w1": ""}
    try:
        from pyscf import gto, scf, cc, symm
        from pyscf.cc import eom_rccsd

        mol = gto.M(atom=geometry(r_ocl, r_oh, theta), basis=basis, spin=0,
                    charge=0, symmetry="Cs", unit="Angstrom", verbose=0,
                    max_memory=mem)
        row["nbf"] = mol.nao_nr()
        mf = scf.RHF(mol).x2c()
        mf.conv_tol = 1e-10
        mf.kernel()
        if not mf.converged:
            row["status"] = "fail:scf"
            return _stamp(row, t0)
        row["e_hf_Ha"] = float(mf.e_tot)

        mycc = cc.RCCSD(mf)
        mycc.conv_tol = 1e-9
        mycc.kernel()
        if not mycc.converged:
            row["status"] = "fail:ccsd"
            return _stamp(row, t0)
        nocc = mycc.t1.shape[0]
        row["t1_s"] = float(np.linalg.norm(mycc.t1) / np.sqrt(2 * nocc))
        row["e_ccsd_Ha"] = float(mycc.e_tot)
        row["e_ccsdt_Ha"] = float(mycc.e_tot) + float(mycc.ccsd_t())

        eom = eom_rccsd.EOMEETriplet(mycc)
        e, vecs = eom.kernel(nroots=nroots)
        e = np.atleast_1d(np.asarray(e, dtype=float))
        vecs = vecs if isinstance(vecs, (list, tuple)) else list(vecs)
        order = np.argsort(e)
        e, vecs = e[order], [vecs[k] for k in order]
        if not np.all(np.atleast_1d(getattr(eom, "converged", True))):
            row["status"] = "warn:eom_unconverged"

        # Label each root A' or A" by its dominant single excitation, as 11
        # does. Duplicated rather than imported: a worker process should not
        # depend on another script's import side effects.
        ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb,
                                  mf.mo_coeff, s=mf.get_ovlp())
        names = dict(zip(mol.irrep_id, mol.irrep_name))
        labels = []
        for v in vecs:
            r1 = np.asarray(eom.vector_to_amplitudes(v)[0])
            i, a = np.unravel_index(int(np.argmax(np.abs(r1))), r1.shape)
            labels.append((names.get(int(ids[i]) ^ int(ids[nocc + a]), "?"),
                           float(np.sum(np.abs(r1) ** 2)
                                 / np.sum(np.abs(v) ** 2))))
        row["root0_sym"] = labels[0][0]
        app = [k for k, (nm, _) in enumerate(labels) if '"' in nm]
        if not app:
            row["status"] = f"fail:no_App_root_in_{len(e)}"
            return _stamp(row, t0)
        k = app[0]
        row["omega_app_eV"] = float(e[k] * HARTREE2EV)
        row["w1"] = round(labels[k][1], 4)
        others = [e[j] for j in range(len(e)) if j != k]
        row["gap_any_eV"] = (float((min(others, key=lambda x: abs(x - e[k]))
                                    - e[k]) * HARTREE2EV)
                             if others else float("nan"))
        row["gap_app_eV"] = (float((e[app[1]] - e[k]) * HARTREE2EV)
                             if len(app) > 1 else float("nan"))
        # Both are informational: the stored value is still the lowest 3A".
        # A 3A' below it is the crossing arriving (seen from r(O-Cl) 2.30 A at
        # wide angles), and it is exactly what the labelling is for -- without
        # it a 3A' energy would go into the surface unnoticed.
        flags = []
        if row["t1_s"] >= T1_MAX:
            flags.append("t1")
        if row["root0_sym"] != 'A"':
            flags.append("Ap_below")
        if row["status"] == "ok" and flags:
            row["status"] = "warn:" + "+".join(flags)
    except Exception as ex:                       # keep the raster running
        row["status"] = f"fail:{type(ex).__name__}"
    return _stamp(row, t0)


def _stamp(row, t0):
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row


_COUNTER = None


def _init_worker(counter):
    """Give each worker its own logical CPU; see water/04_pes_grid.py."""
    global _COUNTER
    _COUNTER = counter
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    try:
        os.sched_setaffinity(0, {idx % (os.cpu_count() or 1)})
    except (AttributeError, OSError):
        pass


def key(r_ocl, r_oh, theta):
    return (round(float(r_ocl), 4), round(float(r_oh), 4),
            round(float(theta), 3))


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add(key(row["r_ocl_A"], row["r_oh_A"], row["theta_deg"]))
            except (KeyError, ValueError):
                continue
    return done


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--rocl-min", type=float, default=1.40)
    p.add_argument("--rocl-max", type=float, default=2.40,
                   help="10 showed the band needs the surface only to ~2.0-2.3 A "
                        "and 09 trusts EOM-CCSD to 2.29 A; 2.40 leaves margin "
                        "for the Jacobi transform, 2.30 trims ~15%% of points")
    p.add_argument("--rocl-step", type=float, default=0.05)
    p.add_argument("--roh-min", type=float, default=0.80)
    p.add_argument("--roh-max", type=float, default=1.25)
    p.add_argument("--roh-step", type=float, default=0.05)
    p.add_argument("--theta-min", type=float, default=75.0)
    p.add_argument("--theta-max", type=float, default=135.0)
    p.add_argument("--theta-step", type=float, default=5.0)
    p.add_argument("--nroots", type=int, default=4)
    p.add_argument("--nproc", type=int, default=30)
    p.add_argument("--mem", type=int, default=3000, help="MB per worker")
    p.add_argument("--pilot", type=int, default=0,
                   help="run only N points spread evenly over the grid, plus "
                        "equilibrium. Written to the same CSV, so they count "
                        "towards the full raster afterwards.")
    p.add_argument("--sec-per-point", type=float, default=111.0,
                   help="single-threaded CPU-seconds per point, for the "
                        "estimate; 111 measured over the 60-point pilot. The "
                        "570 assumed before came from 09's wall time times "
                        "its parallel factor, which counts threading "
                        "overhead as work")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--csv", default=str(DATA / "hocl_pes_raster.csv"))
    args = p.parse_args()

    def grid(lo, hi, step):
        n = int(round((hi - lo) / step)) + 1
        return np.round(np.linspace(lo, lo + step * (n - 1), n), 4)

    rocl = grid(args.rocl_min, args.rocl_max, args.rocl_step)
    roh = grid(args.roh_min, args.roh_max, args.roh_step)
    theta = grid(args.theta_min, args.theta_max, args.theta_step)
    points = [(a, b, c) for a in rocl for b in roh for c in theta]

    print("=" * 78)
    print("== HOX Phase 1 -- 3D CCSD(T) + EOM-CCSD raster for HOCl")
    print("=" * 78)
    print(f"  r(O-Cl) {rocl[0]}-{rocl[-1]} A, {len(rocl)} points")
    print(f"  r(O-H)  {roh[0]}-{roh[-1]} A, {len(roh)} points")
    print(f"  angle   {theta[0]}-{theta[-1]} deg, {len(theta)} points")
    print(f"  total   {len(points)} geometries, {args.basis}, "
          f"{args.nroots} EOM roots")
    cpu_h = len(points) * args.sec_per_point / 3600.0
    print(f"  estimate {cpu_h:.0f} CPU-hours = {cpu_h / args.nproc:.1f} h "
          f"on {args.nproc} workers at {args.sec_per_point:.0f} s/point")

    if args.pilot:
        eq = min(points, key=lambda t: abs(t[0] - R_OCL_EQ)
                 + abs(t[1] - R_OH_EQ) + abs(t[2] - ANGLE_EQ) / 100.0)
        idx = np.unique(np.linspace(0, len(points) - 1,
                                    max(args.pilot - 1, 1)).astype(int))
        chosen = [points[i] for i in idx]
        if eq not in chosen:
            chosen.append(eq)
        points = chosen
        print(f"  PILOT: {len(points)} points spread over the grid, plus "
              f"equilibrium; same CSV, so they count later")

    done = load_done(args.csv)
    todo = [t for t in points if key(*t) not in done]
    print(f"  {len(done)} already in {args.csv}, {len(todo)} to run\n")
    if args.dry_run or not todo:
        print("nothing to do" if not todo else "dry run: stopping here")
        return

    new_file = not os.path.exists(args.csv)
    tasks = [(a, b, c, args.basis, args.nroots, args.mem) for a, b, c in todo]
    t_start = time.perf_counter()
    n_ok = n_warn = n_bad = 0
    with open(args.csv, "a", newline="", buffering=1) as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        # Ask for fork explicitly. It is the default on the EVO's Python
        # 3.12, but 3.14 defaults to forkserver, where each worker re-imports
        # this module and re-runs the environment setup at the top -- slower,
        # and it loses the single-thread settings this design depends on.
        ctx = (mp.get_context("fork") if "fork" in mp.get_all_start_methods()
               else mp.get_context())
        counter = ctx.Value("i", 0)
        with ctx.Pool(args.nproc, initializer=_init_worker,
                      initargs=(counter,)) as pool:
            for i, row in enumerate(
                    pool.imap_unordered(compute_point, tasks, chunksize=1), 1):
                w.writerow({k: row.get(k, "") for k in FIELDS})
                st = row["status"]
                n_ok += st == "ok"
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
    per_point_cpu = el * args.nproc / max(len(tasks), 1)
    print(f"\ndone: {n_ok} ok, {n_warn} warned, {n_bad} failed, "
          f"{el / 60:.1f} min")
    print(f"measured {per_point_cpu:.0f} CPU-s per point "
          f"(was assuming {args.sec_per_point:.0f})")
    full = len(rocl) * len(roh) * len(theta)
    print(f"-> the full {full}-point raster is "
          f"{full * per_point_cpu / 3600:.0f} CPU-hours = "
          f"{full * per_point_cpu / 3600 / args.nproc:.1f} h on "
          f"{args.nproc} workers")
    if n_bad or n_warn:
        print(f"\ninspect: awk -F, 'NR==1||$14!=\"ok\"' {args.csv} | head -30")


if __name__ == "__main__":
    main()
