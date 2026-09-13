#!/usr/bin/env python3
"""
Tier 3g: a small, accurate CCSD(T) ground-state surface for the vibrational
states.

The production V_gs comes from SA-CASSCF/NEVPT2 with a full-valence CAS(8,6),
chosen because the EXCITED state needs a multireference treatment out to
dissociation. Near equilibrium the ground state needs no such thing: it is a
closed shell where CCSD(T) is far more accurate. The symptoms of the
compromise are a ZPE 174 cm-1 too high and De underbound by 0.23 eV.

The vibrational wavefunctions for v = 0..4 live inside roughly
r(OH) = 0.8-1.3 A, so the region that has to be accurate is small and the
whole raster costs minutes rather than hours.

Writes both a valence CSV and a Jacobi .npz that 08_relax.py reads directly,
so the bound-state problem is self-contained and does not disturb the
dissociative surface used for the propagation.

CAVEAT: CCSD(T) on an RHF reference degrades as the bond stretches. The T1
diagnostic is reported per point and flagged above 0.02; that only matters
near the box edge, where the bound wavefunctions have no amplitude.

Usage:
    python 13_gs_well.py --basis aug-cc-pVTZ --nproc 16
    python 13_gs_well.py --basis aug-cc-pVQZ --nr 14 --ntheta 11   # slower
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ["OMP_PROC_BIND"] = "false"
os.environ.pop("OMP_PLACES", None)

import argparse  # noqa: E402
import csv  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

DEG = np.pi / 180.0
BOHR = 0.529177210903
HARTREE2EV = 27.211386245988
AMU = 1822.888486209
M_H, M_D, M_O = 1.00782503207, 2.01410177812, 15.9949146196

FIELDS = ["r1_A", "r2_A", "theta_deg", "nbf", "e_hf_Ha", "e_ccsd_t_Ha",
          "t1_diag", "status", "wall_s"]


def compute_point(task):
    r1, r2, theta, basis, frozen = task
    t0 = time.perf_counter()
    row = dict.fromkeys(FIELDS, np.nan)
    row.update({"r1_A": r1, "r2_A": r2, "theta_deg": theta, "status": "ok"})
    try:
        from pyscf import gto, scf, cc
        a = theta * DEG
        mol = gto.M(atom=[["O", (0.0, 0.0, 0.0)],
                          ["H", (0.0, 0.0, r1)],
                          ["H", (0.0, r2 * np.sin(a), r2 * np.cos(a))]],
                    basis=basis, unit="Angstrom", verbose=0, max_memory=4000)
        row["nbf"] = mol.nao_nr()
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-11
        mf.max_cycle = 200
        mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            mf.kernel()
        row["e_hf_Ha"] = mf.e_tot
        mycc = cc.CCSD(mf)
        mycc.verbose = 0
        mycc.conv_tol = 1e-9
        if frozen:
            mycc.frozen = 1
        mycc.kernel()
        row["t1_diag"] = float(mycc.get_t1_diagnostic())
        row["e_ccsd_t_Ha"] = mf.e_tot + mycc.e_corr + mycc.ccsd_t()
        if not mycc.converged:
            row["status"] = "ccsd_unconverged"
        elif row["t1_diag"] > 0.02:
            row["status"] = "high_t1"
    except Exception as ex:
        row["status"] = f"fail:{type(ex).__name__}"
    row["wall_s"] = round(time.perf_counter() - t0, 2)
    return row


def key(a, b, c):
    return (round(float(a), 4), round(float(b), 4), round(float(c), 3))


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    done.add(key(row["r1_A"], row["r2_A"], row["theta_deg"]))
                except (KeyError, ValueError):
                    pass
    return done


def to_jacobi(rs, thetas, V, args):
    """Same transform as 07_jacobi.py, on a box sized for the bound states."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((rs, rs, thetas), V,
                                bounds_error=False, fill_value=None)
    R = np.linspace(args.R_min, args.R_max, args.nR)
    r = np.linspace(args.r_min, args.r_max, args.nr_j)
    x_gl, w_gl = np.polynomial.legendre.leggauss(args.ngamma)
    gamma = np.arccos(x_gl)
    RR, rr, GG = np.meshgrid(R, r, gamma, indexing="ij")

    mB, mA = args.m_spectator, args.m_departing
    d = (mB / (mB + M_O)) * rr
    r1 = np.sqrt(RR**2 + d**2 + 2.0 * RR * d * np.cos(GG))
    cos_th = np.clip((RR * np.cos(GG) + d) / np.maximum(r1, 1e-12), -1, 1)
    theta = np.degrees(np.arccos(cos_th))
    r1a, r2a = r1 * BOHR, rr * BOHR

    outside = ((theta < thetas[0]) | (theta > thetas[-1])
               | (r1a > rs[-1]) | (r2a > rs[-1])
               | (r1a < rs[0]) | (r2a < rs[0]))
    pts = np.stack([np.clip(r1a, rs[0], rs[-1]),
                    np.clip(r2a, rs[0], rs[-1]),
                    np.clip(theta, thetas[0], thetas[-1])], axis=-1)
    Vj = f(pts)
    Vj[outside] = args.wall
    print(f"jacobi   {100*outside.mean():.1f}% of the bound box lies outside "
          f"the raster -> walled at {args.wall:.1f} Ha")
    print(f"         (bound states must have no amplitude there; check the "
          f"participation ratio 08_relax reports)")

    mu_r = mB * M_O / (mB + M_O) * AMU
    mu_R = mA * (mB + M_O) / (mA + mB + M_O) * AMU
    return dict(R=R, r=r, gamma=gamma, x_gl=x_gl, w_gl=w_gl,
                V_gs=Vj, mu_R=mu_R, mu_r=mu_r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", default="aug-cc-pVTZ")
    p.add_argument("--rmin", type=float, default=0.65)
    p.add_argument("--rmax", type=float, default=1.80,
                   help="keep modest: CCSD(T) on an RHF reference degrades "
                        "as the bond stretches, and the bound states do not "
                        "reach past ~1.3 A anyway")
    p.add_argument("--nr", type=int, default=16)
    p.add_argument("--theta-min", type=float, default=55.0)
    p.add_argument("--theta-max", type=float, default=175.0)
    p.add_argument("--ntheta", type=int, default=13)
    p.add_argument("--r-eq", type=float, default=0.9578)
    p.add_argument("--morse-a", type=float, default=2.0)
    p.add_argument("--all-electron", dest="frozen", action="store_false",
                   default=True)
    p.add_argument("--nproc", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--csv", default="gs_well.csv")
    # Jacobi box for the bound states
    p.add_argument("--R-min", type=float, default=1.2)
    p.add_argument("--R-max", type=float, default=3.6)
    p.add_argument("--nR", type=int, default=96)
    p.add_argument("--r-min", type=float, default=1.2)
    p.add_argument("--r-max", type=float, default=3.0)
    p.add_argument("--nr-j", type=int, default=77)
    p.add_argument("--ngamma", type=int, default=36)
    p.add_argument("--m-spectator", type=float, default=M_H)
    p.add_argument("--m-departing", type=float, default=M_H)
    p.add_argument("--wall", type=float, default=1.0)
    p.add_argument("--npz", default="jacobi_bound_ccsdt.npz")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    # Morse spacing, dense at re where the curvature matters most
    a, re = args.morse_a, args.r_eq
    y = np.linspace(1 - np.exp(-a * (args.rmin - re)),
                    1 - np.exp(-a * (args.rmax - re)), args.nr)
    rs = re - np.log(1 - y) / a
    thetas = np.linspace(args.theta_min, args.theta_max, args.ntheta)

    tasks = [(float(r1), float(r2), float(th), args.basis, args.frozen)
             for th in thetas
             for i, r1 in enumerate(rs) for r2 in rs[:i + 1]]
    print(f"basis    {args.basis}, frozen core = {args.frozen}")
    print(f"r grid   {args.nr} points, {args.rmin}-{args.rmax} A (morse)")
    print(f"theta    {args.ntheta} points")
    print(f"points   {len(tasks)} (r1>=r2 triangle)")

    done = load_done(args.csv)
    todo = [t for t in tasks if key(t[0], t[1], t[2]) not in done]
    if done:
        print(f"resume   {len(done)} done, {len(todo)} remaining")
    if args.dry_run:
        return

    new = not os.path.exists(args.csv)
    t0 = time.perf_counter()
    rows = []
    if todo:
        with open(args.csv, "a", newline="", buffering=1) as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            with mp.Pool(args.nproc) as pool:
                for i, row in enumerate(
                        pool.imap_unordered(compute_point, todo, chunksize=1), 1):
                    w.writerow(row)
                    rows.append(row)
                    if i % 50 == 0 or i == len(todo):
                        el = time.perf_counter() - t0
                        print(f"  {i}/{len(todo)}  {i/el*3600:.0f} pts/h  "
                              f"eta {(len(todo)-i)/(i/el)/60:.1f} min",
                              flush=True)
        print(f"\nraster done in {(time.perf_counter()-t0)/60:.1f} min")

    # assemble, mirror, fill
    allrows = []
    with open(args.csv, newline="") as fh:
        allrows = [r for r in csv.DictReader(fh)]
    ri = {round(float(v), 4): i for i, v in enumerate(np.round(rs, 4))}
    ti = {round(float(v), 3): i for i, v in enumerate(np.round(thetas, 3))}
    V = np.full((len(rs), len(rs), len(thetas)), np.nan)
    t1max, nbad = 0.0, 0
    for row in allrows:
        if row["status"] not in ("ok", "high_t1"):
            nbad += 1
            continue
        try:
            i, j = ri[round(float(row["r1_A"]), 4)], ri[round(float(row["r2_A"]), 4)]
            k = ti[round(float(row["theta_deg"]), 3)]
            e = float(row["e_ccsd_t_Ha"])
            t1max = max(t1max, float(row["t1_diag"]))
        except (KeyError, ValueError):
            nbad += 1
            continue
        V[i, j, k] = V[j, i, k] = e
    print(f"assembled: {np.isfinite(V).sum()} filled, {np.isnan(V).sum()} "
          f"holes, {nbad} skipped, max T1 diagnostic {t1max:.4f}")

    for j in range(V.shape[1]):
        for k in range(V.shape[2]):
            col = V[:, j, k]
            bad = np.isnan(col)
            if bad.any() and not bad.all():
                col[bad] = np.interp(rs[bad], rs[~bad], col[~bad])
    V = 0.5 * (V + V.transpose(1, 0, 2))

    emin = np.nanmin(V)
    idx = np.unravel_index(np.nanargmin(V), V.shape)
    print(f"minimum  {emin:.10f} Ha at r1=r2={rs[idx[0]]:.4f} A, "
          f"theta={thetas[idx[2]]:.2f} deg")
    V = V - emin

    out = to_jacobi(rs, thetas, V, args)
    out["e_min_Ha"] = emin
    np.savez_compressed(args.npz, **out)
    print(f"\nwrote {args.npz}")
    print(f"next: python 08_relax.py --jacobi {args.npz} --nstates 5 "
          f"--tol 1e-8")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
