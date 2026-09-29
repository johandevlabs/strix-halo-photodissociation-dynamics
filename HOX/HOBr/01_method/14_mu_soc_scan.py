#!/usr/bin/env python3
"""
HOBr step 14: the SOC-borrowed transition dipole across the Franck-Condon
window -- the one approximation left that reaches sigma(298)/sigma(220).

13 propagated with a CONSTANT mu (Condon). That cancels exactly in every
temperature ratio only if mu is really constant. It is not guaranteed: the
borrowed dipole goes as <T|H_SO|S_k> mu(S_k) / (E_T - E_Sk), and the energy
gaps change along O-Br. A mu that varies weights v=1's two lobes -- one inner,
one outer -- differently from v=0's single one, and that reaches exactly the
wing ratios where the temperature effect lives (13: +12% to +43% at 520-550
nm). water/10_mu_sensitivity.py asked the same question; there, freezing mu
moved the peak 4 nm and narrowed the band 15%.

What is computed, at each geometry, with 04's method (Ms = 0 SA-CASSCF, 6
roots, DKH1-QD-NEVPT2 + SOC, cc-pvtz-dk):

    f_sum   the summed f of the three a 3A" components (lowest three
            excited SOC states)
    mu2     = 3 f_sum / (2 dE), |mu|^2 in atomic units, what 13 needs

THE ACTIVE SPACE IS FIXED, not left to AVAS's threshold. In HOCl, AVAS
switched between CAS(12,7) and CAS(10,6) right inside this window (07), which
would put a step into mu that is pure artefact. So every point uses 07's
fixed-count selection -- the top 6 occupied and 1 virtual by projection onto
Br 4p / O 2p -- imported from HOCl/01_method/07, the code 07 validated. The
projection eigenvalues of the last orbital in and the first one out are
recorded, so a selection that is only barely stable shows up in the CSV.

GEOMETRIES. The O-Br stretch is what carries the temperature dependence
(4.8% in v=1 at 298 K), and chi_0 spans r_eq +/- 0.09 A at two standard
deviations, v=1 somewhat more; so r(O-Br) 1.66-2.01 A in 0.05 A steps at
01's spectators, plus r_eq itself. Plus the bend at 90 and 115 deg (its hot
band holds 0.3%, and its zero-point spread is every state's), at r_eq.
Eleven points, run one at
a time since each uses the whole machine; ~4 min each on the EVO.

One more thing is assumed and checked, not hidden: 13 propagates a single
mu(q) chi. The three components each have their own dipole; using
sqrt(sum |mu_k|^2) is right if they keep their proportions across the window,
and the CSV records each component's f so that can be seen.

Usage:
    python 14_mu_soc_scan.py 2>&1 | tee ../logs/mu_soc_scan.log
    python 14_mu_soc_scan.py --report-only
"""
import argparse
import csv
import importlib.util
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parents[1] / "data"
HARTREE2EV = 27.211386245988
R_EQ, ROH_EQ, TH_EQ = 1.8357, 0.9646, 102.02      # 01_geometry.py

FIELDS = ["r_ox_A", "r_oh_A", "theta_deg", "ncas", "nelecas", "conv_cas",
          "proj_last_in", "proj_first_out", "e_T_eV", "spread_cm",
          "gap_next_eV", "f1", "f2", "f3", "f_sum", "mu2_au", "status",
          "wall_s"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def default_points():
    pts = [(round(float(r), 4), ROH_EQ, TH_EQ)
           for r in np.arange(1.66, 2.0101, 0.05)]
    # r_eq itself: the reference every ratio is taken against, and the
    # centre of the bend points
    pts += [(R_EQ, ROH_EQ, TH_EQ), (R_EQ, ROH_EQ, 90.0), (R_EQ, ROH_EQ, 115.0)]
    return pts


def key(r, h, t):
    return (round(float(r), 4), round(float(h), 4), round(float(t), 3))


def compute_point(geom, args, s04, f07):
    from pyscf import gto, scf, mcscf, fci
    t0 = time.time()
    row = dict(zip(("r_ox_A", "r_oh_A", "theta_deg"), key(*geom)),
               status="ok")
    mol = gto.M(atom=s04.geometry("Br", *geom), basis=args.basis, spin=0,
                charge=0, symmetry=False, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        row["status"] = "fail:scf"
        return row, t0

    proj = f07.Projector(mol, ["Br 4p", "O 2p"], "ano")
    occ = mf.mo_occ > 0
    w_o, _ = proj.eig(mf.mo_coeff[:, occ])
    w_o = np.sort(w_o)[::-1]
    row["proj_last_in"] = round(float(w_o[args.n_occ - 1]), 4)
    row["proj_first_out"] = round(float(w_o[args.n_occ]), 4)
    mo, ncore, ncas, nelecas = f07.select_fixed(mf, proj, args.n_occ, args.n_vir)
    mo = f07.canonicalize_like_avas(mol, mf, mo, ncore, ncas)
    row["ncas"], row["nelecas"] = ncas, nelecas

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol, mc.conv_tol_grad = 1e-11, 1e-6
    mc.max_cycle_macro = args.max_cycle
    mc.fcisolver = fci.direct_spin1.FCI(mol)          # Ms = 0, S free (04)
    mc = mc.state_average_(np.ones(args.nroots) / args.nroots)
    mc.verbose = 0
    mc.kernel(mo)
    row["conv_cas"] = bool(mc.converged)

    e_soc, osc = s04.qd_nevpt2(mf, mc, soc=args.soc)
    f = np.atleast_1d(np.asarray(osc, dtype=float)).ravel()
    rel = (e_soc - e_soc[0]) * HARTREE2EV
    comp = [1, 2, 3]
    fk = [float(f[i - 1]) for i in comp]
    e_t = float(np.mean(rel[comp]))
    row.update(e_T_eV=round(e_t, 5),
               spread_cm=round(float(np.ptp(rel[comp]) / HARTREE2EV * 219474.63), 1),
               gap_next_eV=round(float(rel[4] - rel[3]), 4),
               f1=fk[0], f2=fk[1], f3=fk[2], f_sum=sum(fk),
               mu2_au=3.0 * sum(fk) / (2.0 * e_t / HARTREE2EV))
    if row["gap_next_eV"] < 0.2:
        row["status"] = "warn:gap"
    return row, t0


def load(path):
    if not os.path.exists(path):
        return {}
    return {key(r["r_ox_A"], r["r_oh_A"], r["theta_deg"]): r
            for r in csv.DictReader(open(path))}


def report(rows):
    ok = [r for r in rows if not r["status"].startswith("fail")]
    if not ok:
        print("  nothing usable")
        return
    ref = min(ok, key=lambda r: abs(float(r["r_ox_A"]) - R_EQ)
              + abs(float(r["theta_deg"]) - TH_EQ) / 100)
    mu2_eq = float(ref["mu2_au"])
    print(f"\n  {'r_OBr':>7}{'theta':>7}{'E_T/eV':>8}{'f_sum':>11}{'mu2/mu2_eq':>12}"
          f"{'f1:f2:f3 (fractions)':>26}{'proj in/out':>14}  status")
    for r in sorted(ok, key=lambda r: (float(r["theta_deg"]), float(r["r_ox_A"]))):
        fs = float(r["f_sum"])
        fr = [float(r[k]) / fs for k in ("f1", "f2", "f3")] if fs > 0 else [0, 0, 0]
        print(f"  {float(r['r_ox_A']):7.3f}{float(r['theta_deg']):7.1f}"
              f"{float(r['e_T_eV']):8.3f}{fs:11.3e}{float(r['mu2_au']) / mu2_eq:12.3f}"
              f"{'  ' + ':'.join(f'{x:.2f}' for x in fr):>26}"
              f"{float(r['proj_last_in']):7.3f}/{float(r['proj_first_out']):.3f}"
              f"  {r['status']}{'' if r.get('conv_cas') in ('True', True) else ' (CAS unconv.)'}")
    rs = [r for r in ok if abs(float(r["theta_deg"]) - TH_EQ) < 0.5]
    if len(rs) >= 3:
        x = np.array([float(r["r_ox_A"]) for r in rs])
        y = np.log(np.array([float(r["mu2_au"]) for r in rs]) / mu2_eq)
        c = np.polyfit(x - R_EQ, y, 2)
        print(f"\n  along O-Br: d ln|mu|^2 / dr = {c[1]:+.2f} /A at r_eq, "
              f"curvature {2 * c[0]:+.1f} /A^2")
        s = 0.0435                                   # sd of |chi_0|^2, A
        print(f"  across chi_0's +/- 1 sd ({s} A) |mu|^2 changes by "
              f"{100 * (np.exp(c[1] * s) - 1):+.0f}% / "
              f"{100 * (np.exp(-c[1] * s) - 1):+.0f}%")
        if np.min(np.exp(y)) < 0.1:
            print("  !! |mu|^2 falls below 10% of its equilibrium value inside the"
                  " scan: mu may change sign,\n     and sqrt(|mu|^2) would put a "
                  "cusp where the true dipole passes smoothly through zero.")
    print("\n  If the component fractions stay put, a single mu(q) = sqrt(sum "
          "|mu_k|^2) is exact for\n  the band; if they shift, it is an "
          "approximation, and the table says by how much.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="cc-pvtz-dk")
    p.add_argument("--soc", default="DKH1")
    p.add_argument("--nroots", type=int, default=6)
    p.add_argument("--n-occ", type=int, default=6)
    p.add_argument("--n-vir", type=int, default=1)
    p.add_argument("--max-cycle", type=int, default=100,
                   help="100 gives f to 0.1% of 300 cycles (04 runs, 2026-09-24)")
    p.add_argument("--csv", default=str(DATA / "hobr_mu_soc_scan.csv"))
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    print("=" * 76)
    print("== HOBr: SOC-borrowed |mu|^2 across the Franck-Condon window")
    print(f"== {args.basis}, {args.soc}-QD-NEVPT2, fixed CAS({2 * args.n_occ},"
          f"{args.n_occ + args.n_vir}), {args.nroots} Ms=0 roots")
    print("=" * 76)
    done = load(args.csv)
    pts = default_points()
    todo = [g for g in pts if key(*g) not in done]
    print(f"  {len(pts)} geometries, {len(todo)} to compute")

    if todo and not args.report_only:
        s04 = _load("s04", HERE.parent / "04_soc_vertical.py")
        f07 = _load("f07", HERE.parents[2] / "HOCl" / "01_method" / "07_fc_active_space.py")
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            for g in todo:
                try:
                    row, t0 = compute_point(g, args, s04, f07)
                except Exception as exc:
                    row, t0 = dict(zip(("r_ox_A", "r_oh_A", "theta_deg"), key(*g)),
                                   status=f"fail:{type(exc).__name__}"), time.time()
                row["wall_s"] = round(time.time() - t0, 1)
                w.writerow(row)
                fh.flush()
                print(f"    {g}: {row['status']}, f_sum {row.get('f_sum', '--')},"
                      f" {row['wall_s']} s", flush=True)
    report(list(load(args.csv).values()))


if __name__ == "__main__":
    main()
