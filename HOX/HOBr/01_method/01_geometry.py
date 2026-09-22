#!/usr/bin/env python3
"""
HOBr step 1: the equilibrium geometry and harmonic force field, from CCSD(T).

Two reasons this comes first, before any surface.

1. HOCl's geometry was taken from the literature. For HOBr the numbers I have
   are recalled rather than looked up, and a wrong r(O-Br) puts the whole
   Franck-Condon window in the wrong place. Computing it costs minutes.

2. **The temperature dependence depends on the force field exponentially.**
   sigma(lambda, T) is a Boltzmann average over initial vibrational states,
   and at 298 K the O-Br stretch (620 cm-1 observed) carries ~5% of the
   population while the bend (1163) carries ~0.4%. A frequency 10% off moves
   that population by ~15% of itself. So the force field is not decoration
   here, it is an input to the deliverable.

METHOD: CCSD(T) with x2c scalar relativity on a 3x3x3 grid in the valence
coordinates, then a full quadratic fit (10 coefficients, 27 points). The fit
gives the stationary point and the internal-coordinate Hessian in one step.
Frequencies follow from H_cart = B^T H_int B, with B = dq/dx by finite
differences -- so no Wilson G-matrix formula is taken on trust, only the
geometric definition of the coordinates.

VALIDATION: run --molecule HOCl first. Its fundamentals are known (724, 1239,
3609 cm-1) and it is cheap, so it tests this entire chain against measurement
before HOBr is asked to produce a number nobody can check. Harmonic
frequencies should sit a few percent ABOVE the observed fundamentals --
anharmonicity is negative -- and if they come out below, something is wrong.

Parallelism is water's job-array pattern, as 12_pes_raster.py; the
environment block must precede numpy and pyscf.

Usage:
    python 01_geometry.py --molecule HOCl              # validate the chain
    python 01_geometry.py --molecule HOBr              # the real run
    python 01_geometry.py --molecule HOBr --refine     # re-centre and redo
    python 01_geometry.py --molecule HOBr --report-only
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["OMP_PROC_BIND"] = "false"
os.environ.pop("OMP_PLACES", None)

import argparse  # noqa: E402
import csv  # noqa: E402
import itertools  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2CM = 219474.6313702
DEG = np.pi / 180.0
AMU2ME = 1822.888486209          # atomic mass unit -> electron mass
ANG2BOHR = 1.8897261254578281

# Most-abundant-isotope masses, amu. The observed fundamentals below are for
# these isotopologues, so averaged atomic weights would be comparing different
# molecules: 79Br is 78.918 against an average of 79.904, a 1.2% mass
# difference and ~0.6% on the O-Br stretch.
MASS = {"H": 1.0078250319, "O": 15.9949146221,
        "Cl": 34.968852682, "Br": 78.9183376}

# Starting geometries and observed fundamentals.
#
# HOCl's geometry is the one used throughout the HOCl work. HOBr's is RECALLED
# and is a starting guess only -- replacing it with a computed value is this
# script's job. The fundamentals are quoted from the NIST WebBook via
# TASKS.md and carry that file's standing caveat: CHECK THEM against the
# source before they appear anywhere that matters.
MOLECULES = {
    "HOCl": dict(halogen="Cl",
                 start=(1.6891, 0.9644, 102.96),
                 obs_fundamental=(724.36, 1238.62, 3609.48),
                 obs_geom=(1.6891, 0.9644, 102.96)),
    "HOBr": dict(halogen="Br",
                 start=(1.834, 0.961, 102.3),
                 obs_fundamental=(620.23, 1162.57, 3614.90),
                 obs_geom=None),
}

FIELDS = ["r_ox_A", "r_oh_A", "theta_deg", "nbf", "e_hf_Ha", "e_ccsd_Ha",
          "e_ccsdt_Ha", "t1", "status", "wall_s"]


def geometry(halogen, r_ox, r_oh, theta):
    """Cartesians from the three valence coordinates, O at the origin."""
    th = theta * DEG
    return [["O", (0.0, 0.0, 0.0)],
            [halogen, (r_ox, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def internals(x):
    """(r_OX, r_OH, angle) from a flat Cartesian vector laid out (O, X, H).

    Unit-agnostic in length -- it returns whatever the input is in -- and the
    angle in RADIANS, so feeding it bohr gives atomic units directly. This is
    the only definition of the coordinates in the script; B is differenced
    from it, so the two cannot disagree.
    """
    o, xx, h = x[0:3], x[3:6], x[6:9]
    a, b = xx - o, h - o
    r_ox = np.linalg.norm(a)
    r_oh = np.linalg.norm(b)
    cos = np.dot(a, b) / (r_ox * r_oh)
    return np.array([r_ox, r_oh, np.arccos(np.clip(cos, -1.0, 1.0))])


def compute_point(task):
    """CCSD(T) with x2c at one geometry."""
    halogen, r_ox, r_oh, theta, basis, mem = task
    t0 = time.perf_counter()
    row = {"r_ox_A": round(r_ox, 5), "r_oh_A": round(r_oh, 5),
           "theta_deg": round(theta, 4), "status": "ok"}
    try:
        from pyscf import gto, scf, cc

        mol = gto.M(atom=geometry(halogen, r_ox, r_oh, theta), basis=basis,
                    spin=0, charge=0, symmetry="Cs", unit="Angstrom",
                    verbose=0, max_memory=mem)
        row["nbf"] = mol.nao_nr()
        mf = scf.RHF(mol).x2c()
        mf.conv_tol = 1e-11
        mf.kernel()
        if not mf.converged:
            row["status"] = "fail:rhf"
            return _stamp(row, t0)
        row["e_hf_Ha"] = float(mf.e_tot)

        mcc = cc.CCSD(mf)
        mcc.conv_tol = 1e-9
        mcc.max_cycle = 100
        mcc.kernel()
        if not mcc.converged:
            row["status"] = "fail:ccsd"
            return _stamp(row, t0)
        row["e_ccsd_Ha"] = float(mcc.e_tot)
        nocc = mcc.t1.shape[0]          # Lee-Taylor, as 12_pes_raster.py
        row["t1"] = float(np.linalg.norm(mcc.t1) / np.sqrt(2 * nocc))
        row["e_ccsdt_Ha"] = float(mcc.e_tot + mcc.ccsd_t())
        if row["t1"] > 0.02:
            row["status"] = "warn:t1"
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
        os.sched_setaffinity(0, {idx % os.cpu_count()})
    except (AttributeError, OSError):
        pass


def key(r_ox, r_oh, theta):
    """The identity of a grid point, used by BOTH the grid and the CSV.

    Two different roundings here is a silent way to recompute every point on
    every rerun while the resume logic looks like it works.
    """
    return (round(float(r_ox), 5), round(float(r_oh), 5),
            round(float(theta), 4))


def load_done(path):
    """Rows already in the CSV, so a rerun costs nothing."""
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                k = key(row["r_ox_A"], row["r_oh_A"], row["theta_deg"])
            except (KeyError, ValueError):
                continue
            done[k] = row
    return done


def build_grid(center, steps):
    """3x3x3 around the centre: 27 points for 10 quadratic coefficients."""
    axes = [[c - s, c, c + s] for c, s in zip(center, steps)]
    return [key(*p) for p in itertools.product(*axes)]


def quadratic_fit(pts, energies):
    """Least-squares full quadratic in 3 variables, centred on the mean.

    Returns (minimum_coords, gradient_at_centre, hessian, rms_residual_Ha),
    all in the grid's own units of (Angstrom, Angstrom, degree).

    Fitting in DISPLACEMENTS from the grid centre, not in raw values: r ~ 1.8
    and theta ~ 102 differ by two orders of magnitude, and a design matrix
    built from raw values loses precision in exactly the second-derivative
    coefficients this exists to extract.
    """
    pts = np.asarray(pts, dtype=float)
    c = pts.mean(axis=0)
    d = pts - c
    A = np.column_stack([np.ones(len(d)), d[:, 0], d[:, 1], d[:, 2],
                         d[:, 0] ** 2, d[:, 1] ** 2, d[:, 2] ** 2,
                         d[:, 0] * d[:, 1], d[:, 0] * d[:, 2],
                         d[:, 1] * d[:, 2]])
    coef, *_ = np.linalg.lstsq(A, np.asarray(energies, dtype=float), rcond=None)
    resid = float(np.sqrt(np.mean((A @ coef - np.asarray(energies)) ** 2)))

    g = coef[1:4].copy()
    H = np.array([[2 * coef[4], coef[7], coef[8]],
                  [coef[7], 2 * coef[5], coef[9]],
                  [coef[8], coef[9], 2 * coef[6]]])
    dx = np.linalg.solve(H, -g)          # stationary point: H dx = -g
    return c + dx, g, H, resid


def cartesians_bohr(halogen, coords):
    """Flat 9-vector of Cartesians in BOHR from (r_OX/A, r_OH/A, theta/deg)."""
    return np.array([c for _, xyz in geometry(halogen, *coords)
                     for c in xyz], dtype=float) * ANG2BOHR


def b_matrix(halogen, coords, h=1e-5):
    """B[k, i] = dq_k/dx_i in atomic units, by central differences.

    Finite-differenced rather than written out. The analytic B (equivalently
    the Wilson G matrix) for a bent triatomic is exactly the sort of recalled
    formula this repo keeps getting caught by; internals() is the definition
    and B follows from it mechanically.
    """
    x0 = cartesians_bohr(halogen, coords)
    B = np.zeros((3, 9))
    for i in range(9):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        B[:, i] = (internals(xp) - internals(xm)) / (2 * h)
    return B


# q_au = q_fit * S:  Angstrom -> bohr for the two stretches, degree -> radian
# for the bend. The Hessian from the fit is in Hartree per (A, A, deg), and
# every term has to be converted before it meets a mass in atomic units.
S_FIT2AU = np.array([ANG2BOHR, ANG2BOHR, DEG])


def frequencies(halogen, coords, H_fit):
    """Harmonic frequencies in cm-1 from the fitted internal Hessian.

    H_cart = B^T H_int B is exact AT A STATIONARY POINT: the neglected term is
    sum_k (dE/dq_k) d2q_k/dx2, and dE/dq = 0 there. So this must be evaluated
    at the fitted minimum, never at the grid centre.

    Six of the nine eigenvalues are zero BY CONSTRUCTION -- B^T H B has rank
    at most 3 -- so their vanishing is not a check on anything, and the three
    largest are taken. The check that does mean something is the ratio to the
    observed fundamentals, and --molecule HOCl against a known spectrum.
    """
    H_au = H_fit / np.outer(S_FIT2AU, S_FIT2AU)
    B = b_matrix(halogen, coords)
    H_cart = B.T @ H_au @ B

    m = np.repeat([MASS["O"], MASS[halogen], MASS["H"]], 3) * AMU2ME
    w = np.linalg.eigvalsh(H_cart / np.sqrt(np.outer(m, m)))
    return np.array([np.sign(v) * np.sqrt(abs(v)) * HARTREE2CM
                     for v in np.sort(w)[-3:]])


def report(molecule, rows, args):
    """Fit, frequencies, and the comparisons that can fail."""
    spec = MOLECULES[molecule]
    good = [r for r in rows if r["status"].startswith(("ok", "warn"))]
    print(f"\n  {len(good)} of {len(rows)} points usable")
    if len(good) < 10:
        print("  !! a full quadratic in 3 variables needs 10; cannot fit.")
        return
    bad = [r for r in rows if not r["status"].startswith(("ok", "warn"))]
    for r in bad:
        print(f"     dropped {r['r_ox_A']}, {r['r_oh_A']}, {r['theta_deg']}: "
              f"{r['status']}")
    t1s = [float(r["t1"]) for r in good if r.get("t1") not in (None, "")]
    if t1s:
        t1max = max(t1s)
        print(f"  worst T1 diagnostic {t1max:.4f}" + (
            "  -- above 0.02, the single reference is strained"
            if t1max > 0.02 else ""))

    pts = [(float(r["r_ox_A"]), float(r["r_oh_A"]), float(r["theta_deg"]))
           for r in good]
    e = [float(r["e_ccsdt_Ha"]) for r in good]
    coords, g, H, resid = quadratic_fit(pts, e)

    center = np.array(args.center if args.center else spec["start"], float)
    steps = np.array(args.steps, float)
    shift = coords - center
    print(f"\n  quadratic fit rms residual {resid * 27211.386:.3f} meV "
          f"(how far the surface departs from a quadratic over this grid)")
    print(f"  stationary point moved from the grid centre by "
          f"{shift[0]:+.4f} A, {shift[1]:+.4f} A, {shift[2]:+.3f} deg")
    if np.any(np.abs(shift) > steps):
        print("  !! that is more than one grid step: the quadratic is being "
              "extrapolated. Rerun with --center at the new point "
              "(or --refine).")

    evals = np.linalg.eigvalsh(H)
    if np.any(evals <= 0):
        print(f"  !! Hessian is not positive definite (eigenvalues {evals}). "
              f"This is not a minimum.")
    print(f"\n  equilibrium geometry (CCSD(T)/{args.basis}, x2c):")
    print(f"      r(O-{spec['halogen']}) = {coords[0]:.4f} A")
    print(f"      r(O-H)  = {coords[1]:.4f} A")
    print(f"      angle   = {coords[2]:.2f} deg")
    if spec["obs_geom"]:
        o = spec["obs_geom"]
        print(f"      against the literature values used so far: "
              f"{o[0]:.4f}, {o[1]:.4f}, {o[2]:.2f}  "
              f"(diff {coords[0]-o[0]:+.4f}, {coords[1]-o[1]:+.4f}, "
              f"{coords[2]-o[2]:+.2f})")

    freq = frequencies(spec["halogen"], coords, H)
    obs = np.array(spec["obs_fundamental"], float)
    print(f"\n  harmonic frequencies, cm-1 (ascending):")
    print(f"      {'calc':>10}{'obs fund.':>12}{'calc/obs':>10}   assignment")
    names = [f"nu3  O-{spec['halogen']} stretch", "nu2  bend",
             "nu1  O-H stretch"]
    for w, o, n in zip(freq, obs, names):
        print(f"      {w:10.1f}{o:12.2f}{w / o:10.3f}   {n}")
    print("""
  Harmonic frequencies should land a few percent ABOVE the observed
  fundamentals: anharmonicity is negative, and for X-H stretches it is worth
  4-6%. A ratio below 1.00, or above ~1.10, means the fit or the force field
  is wrong rather than merely approximate.""")

    kT = 0.695034800                                    # cm-1 per K
    print(f"\n  v=1 populations from the CALCULATED frequencies:")
    print(f"      {'mode':>22}{'220 K':>10}{'298 K':>10}")
    for w, n in zip(freq, names):
        print(f"      {n:>22}{100 * np.exp(-w / (kT * 220)):9.2f}%"
              f"{100 * np.exp(-w / (kT * 298)):9.2f}%")
    print("      (this is what sigma(lambda, T) is a Boltzmann average over;\n"
          "       the O-X stretch dominates and it IS the dissociation\n"
          "       coordinate, so the 1D model already carries it)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--molecule", default="HOBr", choices=sorted(MOLECULES))
    p.add_argument("--basis", default="def2-tzvp",
                   help="all-electron; the Br choice is settled by "
                        "toolchain/02_soc_atoms.py --sweep")
    p.add_argument("--center", nargs=3, type=float, default=None,
                   metavar=("R_OX", "R_OH", "THETA"),
                   help="grid centre; default is the molecule's start value")
    p.add_argument("--steps", nargs=3, type=float,
                   default=[0.03, 0.03, 3.0],
                   metavar=("D_OX", "D_OH", "D_THETA"),
                   help="half-widths in A, A, deg (default 0.03 0.03 3)")
    p.add_argument("--refine", action="store_true",
                   help="after fitting, re-centre on the minimum and run a "
                        "second grid into the same CSV")
    p.add_argument("--csv", default=None)
    p.add_argument("--nproc", type=int, default=min(27, os.cpu_count() or 1))
    p.add_argument("--memory", type=int, default=4000)
    p.add_argument("--report-only", action="store_true",
                   help="fit and report from the existing CSV, no new points")
    args = p.parse_args()

    spec = MOLECULES[args.molecule]
    csv_path = args.csv or str(DATA / f"{args.molecule.lower()}_geometry.csv")
    # git does not track empty directories, so HOBr/data/ does not exist on a
    # fresh clone and the first append dies at the very end of the first round
    # -- after every point has been computed. Make it rather than assume it.
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    center = list(args.center) if args.center else list(spec["start"])

    print("=" * 72)
    print(f"== {args.molecule} equilibrium geometry and harmonic force field")
    print(f"== CCSD(T)/{args.basis} with x2c, 3x3x3 grid, quadratic fit")
    print("=" * 72)

    rounds = [center] + ([None] if args.refine else [])
    for n, c in enumerate(rounds):
        if c is None:                       # --refine: re-centre and repeat
            rows = list(load_done(csv_path).values())
            good = [r for r in rows if r["status"].startswith(("ok", "warn"))]
            pts = [(float(r["r_ox_A"]), float(r["r_oh_A"]),
                    float(r["theta_deg"])) for r in good]
            c = list(quadratic_fit(pts, [float(r["e_ccsdt_Ha"])
                                         for r in good])[0])
            print(f"\n--- refining: new centre "
                  f"{c[0]:.4f}, {c[1]:.4f}, {c[2]:.2f}")
            args.center = c

        grid = build_grid(c, args.steps)
        done = load_done(csv_path)
        todo = [g for g in grid if g not in done]
        print(f"\n  round {n + 1}: {len(grid)} points, {len(todo)} to compute")

        if todo and not args.report_only:
            tasks = [(spec["halogen"], a, b, t, args.basis, args.memory)
                     for a, b, t in todo]
            new = os.path.exists(csv_path)
            t0 = time.time()
            ctx = mp.get_context("fork")     # 3.14 defaults to forkserver,
            counter = ctx.Value("i", 0)      # which re-imports and loses the
            with open(csv_path, "a", newline="") as fh:   # env block above
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if not new:
                    w.writeheader()
                with ctx.Pool(args.nproc, initializer=_init_worker,
                              initargs=(counter,)) as pool:
                    for i, row in enumerate(
                            pool.imap_unordered(compute_point, tasks), 1):
                        w.writerow(row)
                        fh.flush()
                        print(f"    [{i}/{len(tasks)}] "
                              f"{row['r_ox_A']}, {row['r_oh_A']}, "
                              f"{row['theta_deg']}: {row['status']} "
                              f"({row['wall_s']} s)")
            print(f"  round done in {(time.time() - t0) / 60:.1f} min")

        have = load_done(csv_path)
        rows = [have[g] for g in grid if g in have]
        report(args.molecule, rows, args)

    print(f"\n  csv: {csv_path}")


if __name__ == "__main__":
    main()
