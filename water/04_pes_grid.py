#!/usr/bin/env python3
"""
Tier 2b: the 3D ab initio raster for H2O A-state photodissociation.

Coordinates are (r1, r2, theta) with the two OH bonds independent. Everything
runs in Cs, forced rather than detected, so r1 == r2 does not silently become
C2v mid-raster and relabel the irreps.

Three things keep the cost down:

  1. V(r1, r2, theta) == V(r2, r1, theta), so only the r1 >= r2 triangle is
     computed. Mirroring afterwards makes the symmetry exact by construction
     instead of approximately recovered from independently converged points.
  2. The raster stops around 3.0-3.5 A, where the A'/A" splitting has already
     collapsed below 0.05 eV. Beyond that the surface is E_H + V_OH(r2) from
     05_oh_diatomic.py.
  3. Each geometry is an independent single-threaded job. Parallelising over
     GEOMETRIES with 16 processes beats threading one CASSCF across 16 cores
     by a wide margin at this problem size. This is the job-array pattern.

NEVPT2 costs ~5% on top of the CASSCF that has to run anyway, so both
surfaces are stored at every point and you can propagate on either.

Resumable: results append to the CSV as they complete, and a rerun skips
geometries already present. Safe to interrupt.

Usage:
    python 04_pes_grid.py --basis aug-cc-pVTZ --nproc 16
    python 04_pes_grid.py --dry-run          # report grid size and cost only
"""
import os

# MUST precede numpy/pyscf: BLAS reads these at library load time and the
# whole design here is one thread per process, many processes.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"

# Clear OpenMP thread PINNING. OMP_PLACES=cores with OMP_PROC_BIND=close is
# correct for ONE process threading over many cores, and catastrophic here:
# every independent process sees the whole machine, binds its single thread
# to place 0, and all N workers pile onto the same physical core. Observed
# symptom: two logical CPUs (an SMT sibling pair) at 100%, every other core
# idle, each worker at 1/N of a core, while the load average still reads N
# because N processes are runnable. Affinity is assigned per worker below.
os.environ["OMP_PROC_BIND"] = "false"
os.environ.pop("OMP_PLACES", None)

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DEG = np.pi / 180.0
HARTREE2EV = 27.211386245988

FIELDS = ["r1_A", "r2_A", "theta_deg", "nbf", "ncas", "nelecas",
          "e_gs_casscf_Ha", "e_ex_casscf_Ha",
          "e_gs_nevpt2_Ha", "e_ex_nevpt2_Ha",
          "mu_x", "mu_y", "mu_z", "mu_abs", "status", "wall_s"]


def make_r_grid(args):
    """
    Shared grid for r1 and r2 (the r1 >= r2 triangle requires one grid).

    Morse spacing, y = 1 - exp(-a(r-re)), dense where the OH well and the FC
    region are, then a uniform tail because y saturates as it approaches 1
    and cannot sample the long range on its own. The earlier quadratic
    mapping anchored density at RMIN, spending points on the repulsive wall
    and starving the region that matters.
    """
    if args.spacing == "linear":
        return np.linspace(args.rmin, args.rmax, args.nr)
    if args.spacing == "quadratic":
        u = np.linspace(0.0, 1.0, args.nr)
        return args.rmin + (args.rmax - args.rmin) * u**2

    a, re = args.morse_a, args.r_eq
    r_tail = min(args.r_tail, args.rmax)
    n_tail = min(args.n_tail, args.nr - 2) if args.rmax > r_tail else 0
    n_main = args.nr - n_tail
    y = np.linspace(1.0 - np.exp(-a * (args.rmin - re)),
                    1.0 - np.exp(-a * (r_tail - re)), n_main)
    main_r = re - np.log(1.0 - y) / a
    if n_tail:
        return np.concatenate(
            [main_r, np.linspace(r_tail, args.rmax, n_tail + 1)[1:]])
    return main_r


def build_mol(r1, r2, theta_deg, basis, verbose=0):
    from pyscf import gto
    a = theta_deg * DEG
    atom = [["O", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, r1)],
            ["H", (0.0, r2 * np.sin(a), r2 * np.cos(a))]]
    return gto.M(atom=atom, basis=basis, unit="Angstrom",
                 symmetry="Cs", verbose=verbose, max_memory=4000)


def compute_point(task):
    """
    One geometry. Returns a dict matching FIELDS. Never raises: a failed
    point is recorded with status set and the raster continues, so a single
    pathological geometry cannot kill a multi-hour run.
    """
    r1, r2, theta, basis, avas_aos, gs_sym, ex_sym = task
    t0 = time.perf_counter()
    row = dict.fromkeys(FIELDS, np.nan)
    row.update({"r1_A": r1, "r2_A": r2, "theta_deg": theta, "status": "ok"})

    try:
        from pyscf import scf, mcscf, fci, mrpt
        from pyscf.mcscf import avas

        mol = build_mol(r1, r2, theta, basis)
        row["nbf"] = mol.nao_nr()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-10
        mf.max_cycle = 200
        mf.kernel()
        if not mf.converged:
            # Second-order (Newton) SCF. Slower per iteration but far more
            # robust at the awkward geometries: near-linear at short bond
            # length, and long r1 where open-shell character develops.
            mf = mf.newton()
            mf.max_cycle = 100
            mf.kernel()
            row["status"] = "ok" if mf.converged else "scf_unconverged"

        # active space rebuilt from atomic character at every geometry.
        # Do NOT seed from a neighbouring point: in a diffuse basis the aug
        # functions lie near sigma*, CASSCF rotates them in, and the damage
        # compounds along the scan.
        ncas, nelecas, orbs = avas.avas(mf, avas_aos,
                                        canonicalize=False, verbose=0)
        row["ncas"], row["nelecas"] = ncas, str(nelecas)

        mc = mcscf.CASSCF(mf, ncas, nelecas)
        s_gs = fci.direct_spin0_symm.FCI(mol)
        s_gs.wfnsym, s_gs.nroots = gs_sym, 1
        s_ex = fci.direct_spin0_symm.FCI(mol)
        s_ex.wfnsym, s_ex.nroots = ex_sym, 1
        mcscf.state_average_mix_(mc, [s_gs, s_ex], [0.5, 0.5])
        mc.conv_tol = 1e-8
        mc.max_cycle_macro = 200
        mc.verbose = 0
        mc.kernel(orbs)
        if not mc.converged:
            row["status"] = "casscf_unconverged"

        # NEVPT2 refuses state-averaged solvers: re-diagonalise each irrep
        # singly on the optimised SA orbitals.
        energies, civecs, casci = [], [], []
        for sym in (gs_sym, ex_sym):
            mci = mcscf.CASCI(mf, ncas, nelecas)
            mci.fcisolver = fci.direct_spin0_symm.FCI(mol)
            mci.fcisolver.wfnsym = sym
            mci.verbose = 0
            mci.kernel(mc.mo_coeff)
            energies.append(mci.e_tot)
            civecs.append(mci.ci)
            casci.append(mci)

        row["e_gs_casscf_Ha"], row["e_ex_casscf_Ha"] = energies
        row["e_gs_nevpt2_Ha"] = energies[0] + mrpt.NEVPT(casci[0]).kernel()
        row["e_ex_nevpt2_Ha"] = energies[1] + mrpt.NEVPT(casci[1]).kernel()

        ncore = casci[0].ncore
        mo_cas = mc.mo_coeff[:, ncore:ncore + ncas]
        t_dm1 = fci.direct_spin1.trans_rdm1(
            civecs[0], civecs[1], ncas, casci[0].nelecas)
        with mol.with_common_orig(mol.atom_coords().mean(axis=0)):
            dip_ao = mol.intor("int1e_r", comp=3)
        mu = -np.einsum("xij,ji->x", dip_ao, mo_cas @ t_dm1 @ mo_cas.T)
        row["mu_x"], row["mu_y"], row["mu_z"] = mu
        row["mu_abs"] = float(np.linalg.norm(mu))

    except Exception as ex:
        row["status"] = f"fail:{type(ex).__name__}"

    row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row


_CPU_COUNTER = None


def _init_worker(counter):
    """
    Give each worker its own physical core.

    Relying on the scheduler alone usually works, but explicit affinity keeps
    workers off each other's SMT siblings, which matters because two threads
    on one core contend for the same FP pipes. Logical CPUs 0..n_phys-1 are
    the first sibling of each physical core on this layout.
    """
    global _CPU_COUNTER
    _CPU_COUNTER = counter
    with counter.get_lock():
        idx = counter.value
        counter.value += 1
    try:
        ncpu = os.cpu_count() or 1
        os.sched_setaffinity(0, {idx % ncpu})
    except (AttributeError, OSError):
        pass  # not fatal: without pinning the scheduler still spreads them


def key(r1, r2, th):
    return (round(float(r1), 4), round(float(r2), 4), round(float(th), 3))


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add(key(row["r1_A"], row["r2_A"], row["theta_deg"]))
            except (KeyError, ValueError):
                continue
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", default="aug-cc-pVTZ")
    p.add_argument("--rmin", type=float, default=0.70)
    p.add_argument("--rmax", type=float, default=3.40,
                   help="raster stops here; beyond it the surface is the "
                        "analytic E_H + V_OH(r2) splice")
    p.add_argument("--nr", type=int, default=20)
    p.add_argument("--theta-min", type=float, default=55.0)
    p.add_argument("--theta-max", type=float, default=175.0,
                   help="stay off 180: at exactly linear the point group is "
                        "not Cs and the forced subgroup will fail")
    p.add_argument("--ntheta", type=int, default=13)
    p.add_argument("--spacing", choices=["morse", "quadratic", "linear"],
                   default="morse")
    p.add_argument("--r-eq", type=float, default=0.9697)
    p.add_argument("--morse-a", type=float, default=1.6)
    p.add_argument("--r-tail", type=float, default=2.5,
                   help="Morse spacing below this, uniform above")
    p.add_argument("--n-tail", type=int, default=5)
    p.add_argument("--avas-aos", nargs="+", default=["O 2s", "O 2p", "H 1s"])
    p.add_argument("--gs-sym", default="A'")
    p.add_argument("--ex-sym", default='A"')
    p.add_argument("--nproc", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--csv", default="pes_grid.csv")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sec-per-point", type=float, default=9.0,
                   help="for the dry-run estimate only")
    args = p.parse_args()

    rs = make_r_grid(args)
    thetas = np.linspace(args.theta_min, args.theta_max, args.ntheta)

    tasks = []
    for th in thetas:
        for i, r1 in enumerate(rs):
            for r2 in rs[:i + 1]:          # r1 >= r2 triangle only
                tasks.append((float(r1), float(r2), float(th),
                              args.basis, args.avas_aos,
                              args.gs_sym, args.ex_sym))

    full = len(rs) ** 2 * len(thetas)
    print(f"basis    {args.basis}")
    print(f"r grid   {args.nr} points, {args.rmin}-{args.rmax} A "
          f"({args.spacing})")
    _e = np.quantile(rs, [0.0, 0.25, 0.5, 0.75, 1.0])
    for _lo, _hi in zip(_e[:-1], _e[1:]):
        _m = (rs >= _lo) & (rs <= _hi)
        _s = np.diff(rs[_m]).mean() if _m.sum() > 1 else float("nan")
        print(f"         {_lo:5.2f}-{_hi:5.2f} A: {_m.sum():3d} pts, "
              f"step {_s:.4f} A")
    print(f"theta    {args.ntheta} points, {args.theta_min}-{args.theta_max} deg")
    print(f"points   {len(tasks)} computed / {full} full grid "
          f"({100 * len(tasks) / full:.0f}%, r1>=r2 symmetry)")

    done = load_done(args.csv)
    todo = [t for t in tasks if key(t[0], t[1], t[2]) not in done]
    if done:
        print(f"resume   {len(done)} already in {args.csv}, "
              f"{len(todo)} remaining")

    core_h = len(todo) * args.sec_per_point / 3600.0
    print(f"estimate {core_h:.1f} core-hours "
          f"=> {core_h / max(args.nproc, 1):.2f} h on {args.nproc} procs "
          f"at {args.sec_per_point:.0f} s/point\n")

    if args.dry_run:
        return
    if not todo:
        print("nothing to do")
        return

    new_file = not os.path.exists(args.csv)
    t_start = time.perf_counter()
    n_ok = n_bad = 0

    with open(args.csv, "a", newline="", buffering=1) as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        counter = mp.Value("i", 0)
        with mp.Pool(args.nproc, initializer=_init_worker,
                     initargs=(counter,)) as pool:
            for i, row in enumerate(
                    pool.imap_unordered(compute_point, todo, chunksize=1), 1):
                w.writerow(row)          # flushed per line: safe to interrupt
                if row["status"] == "ok":
                    n_ok += 1
                else:
                    n_bad += 1
                if i % 25 == 0 or i == len(todo):
                    el = time.perf_counter() - t_start
                    rate = i / el
                    print(f"  {i}/{len(todo)}  ok={n_ok} bad={n_bad}  "
                          f"{rate * 3600:.0f} pts/h  "
                          f"eta {(len(todo) - i) / rate / 60:.1f} min",
                          flush=True)

    print(f"\ndone: {n_ok} ok, {n_bad} problematic, "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")
    if n_bad:
        print(f"inspect with: awk -F, '$15!=\"ok\"' {args.csv} | head")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
