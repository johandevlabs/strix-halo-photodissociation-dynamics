#!/usr/bin/env python3
"""
HOI step 7: the spin-orbit states along O-I -- what the band model needs
where one spin-free triplet surface is no longer enough.

For HOBr the band ran on a single spin-free a 3A" surface with a borrowed
dipole (c^2 ~2%). For HOI 04 gives c^2 ~20%: the three triplet components
split by 450 cm-1, the one carrying the intensity (f 1.5e-4) is 0.05 eV
above the other two, and the singlet 1A" is only 0.4 eV higher. So the band
model propagates on the SPIN-ORBIT states themselves, each with its own
curve and its own dipole, and this scan supplies them.

At each r(O-I), spectators at 01's minimum, 04's method: Ms = 0 SA-CASSCF,
10 roots, then QD-NEVPT2 without SOC and with DKH1 SOC, cc-pvtz-dk. Stored:

    e_soc0_Ha, e_sf0_Ha   the ground state with and without SOC (absolute)
    dE_k, f_k             the lowest NK excited SOC states: excitation
                          energy (eV) and oscillator strength
    sf_k                  the spin-free excitation energies (eV), for the
                          SOC shift and a check against the EOM cut (02)

NK = 8 reaches through the 1A"/3A' pair and the 368 nm mixed state to 1A'
at r_eq, i.e. every state under Bauer's two measured bands, so the model can
be tested on the bands that WERE measured as well as the one that was not.

THE ACTIVE SPACE IS FIXED at CAS(12,7) by projection onto I 5p / O 2p, as
HOBr's 14 does with HOCl 07's code: an AVAS threshold that switched spaces
inside the scan would put a step into every curve. The projection
eigenvalues of the last orbital in and the first out are recorded.

Grid: 1.75-2.70 A in 0.05 A steps, plus r_eq; ~3.5 min a point on the EVO,
one at a time (each uses the machine). ~75 min.

Usage (from this directory):
    python 07_soc_scan.py 2>&1 | tee ../logs/soc_scan.log
    python 07_soc_scan.py --report-only
"""
import argparse
import csv
import importlib.util
import os
import time
from pathlib import Path

import numpy as np

HOX = Path(__file__).resolve().parents[2]
CSV = HOX / "HOI" / "data" / "hoi_soc_scan.csv"
HARTREE2EV = 27.211386245988
NM_PER_EV = 1239.841984
R_EQ, ROH_EQ, TH_EQ = 1.9907, 0.9694, 104.65      # 01_geometry.py
NK, NSF = 8, 6
FIELDS = (["r_ox_A", "theta_deg", "ncas", "nelecas", "conv_cas", "proj_last_in",
           "proj_first_out", "e_soc0_Ha", "e_sf0_Ha"]
          + [f"dE_{k}" for k in range(1, NK + 1)]
          + [f"f_{k}" for k in range(1, NK + 1)]
          + [f"sf_{k}" for k in range(1, NSF)]
          + ["status", "wall_s"])


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def radii():
    rs = [round(float(r), 4) for r in np.arange(1.75, 2.7001, 0.05)]
    return sorted(set(rs + [R_EQ]))


def points(grid2d):
    """(r, theta) to compute. 1D: the cut at the equilibrium angle. 2D: the
    coarse grid the 2D band model interpolates the SOC offsets and dipoles
    on (they vary slowly; the fine shape comes from the EOM raster)."""
    if not grid2d:
        return [(r, TH_EQ) for r in radii()]
    return [(round(float(r), 4), float(t))
            for t in np.arange(85.0, 125.01, 10.0)
            for r in np.arange(1.80, 2.4001, 0.10)]


def pkey(r, t):
    return (round(float(r), 4), round(float(t), 3))


def compute_point(r, theta, args, s04, f07):
    from pyscf import gto, scf, mcscf, fci
    t0 = time.time()
    row = dict(r_ox_A=r, theta_deg=theta, status="ok")
    mol = gto.M(atom=s04.geometry("I", r, ROH_EQ, theta), basis=args.basis,
                spin=0, charge=0, symmetry=False, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        row["status"] = "fail:scf"
        return row, t0

    proj = f07.Projector(mol, ["I 5p", "O 2p"], "ano")
    w_o, _ = proj.eig(mf.mo_coeff[:, mf.mo_occ > 0])
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

    e_sf, _ = s04.qd_nevpt2(mf, mc, soc=None, verbose=0)
    e_soc, osc = s04.qd_nevpt2(mf, mc, soc=args.soc, verbose=0)
    f = np.atleast_1d(np.asarray(osc, dtype=float)).ravel()
    row["e_soc0_Ha"], row["e_sf0_Ha"] = float(e_soc[0]), float(e_sf[0])
    for k in range(1, NK + 1):
        row[f"dE_{k}"] = round(float((e_soc[k] - e_soc[0]) * HARTREE2EV), 6)
        row[f"f_{k}"] = float(f[k - 1])
    for k in range(1, NSF):
        row[f"sf_{k}"] = round(float((e_sf[k] - e_sf[0]) * HARTREE2EV), 6)
    return row, t0


def load(path):
    """Keyed (r, theta); the 1D CSV predates the theta column, so a missing
    or empty one means the equilibrium angle."""
    if not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        t = r.get("theta_deg") or TH_EQ
        r["theta_deg"] = t
        out[pkey(r["r_ox_A"], t)] = r
    return out


def report_2d(rows):
    ok = [r for r in rows if not r["status"].startswith("fail")]
    th = sorted({float(r["theta_deg"]) for r in ok})
    rs = sorted({float(r["r_ox_A"]) for r in ok})
    grid = {pkey(r["r_ox_A"], r["theta_deg"]): r for r in ok}
    for title, fn in (("E3 (the bright component) / eV",
                       lambda x: f"{float(x['dE_3']):8.3f}"),
                      ("f1+f2+f3",
                       lambda x: f"{sum(float(x[f'f_{k}']) for k in (1, 2, 3)):8.1e}"),
                      ("E3 - spin-free T1 / meV",
                       lambda x: f"{(float(x['dE_3']) - float(x['sf_1'])) * 1000:8.0f}")):
        print(f"\n  {title}, rows r/A, columns angle/deg")
        print("  " + " " * 6 + "".join(f"{t:8.0f}" for t in th))
        for r_ in rs:
            print(f"  {r_:6.2f}" + "".join(
                fn(grid[pkey(r_, t)]) if pkey(r_, t) in grid else f"{'--':>8}"
                for t in th))


def report(rows):
    rows = [r for r in rows if abs(float(r["theta_deg"]) - TH_EQ) < 1e-3]
    ok = sorted((r for r in rows if not r["status"].startswith("fail")),
                key=lambda r: float(r["r_ox_A"]))
    if not ok:
        print("  nothing usable")
        return
    print(f"\n  SOC excitation energies (eV) and f, lowest {NK} states; "
          f"spin-free T1/S1 (eV)")
    print(f"  {'r/A':>6}" + "".join(f"{'E' + str(k):>7}" for k in range(1, NK + 1))
          + f"{'f1+f2+f3':>10}{'f4..f8':>10}{'sf T1':>7}{'sf S1':>7}"
          f"{'SO gs meV':>10}  CAS")
    for r in ok:
        dE = [float(r[f"dE_{k}"]) for k in range(1, NK + 1)]
        f3 = sum(float(r[f"f_{k}"]) for k in (1, 2, 3))
        f8 = sum(float(r[f"f_{k}"]) for k in range(4, NK + 1))
        so = (float(r["e_soc0_Ha"]) - float(r["e_sf0_Ha"])) * HARTREE2EV * 1000
        flag = "" if r.get("conv_cas") in ("True", True) else " unconv"
        print(f"  {float(r['r_ox_A']):6.3f}" + "".join(f"{e:7.3f}" for e in dE)
              + f"{f3:10.2e}{f8:10.2e}{float(r['sf_1']):7.3f}"
              f"{float(r['sf_2']):7.3f}{so:10.1f}  "
              f"{r['proj_last_in']}/{r['proj_first_out']}{flag}")
    print("\n  Adiabatic SOC states, ordered by energy at every r: where two"
          " approach, the\n  character (and f) passes from one to the other, "
          "which the f columns show.\n  proj in/out close together = the "
          "fixed CAS is near a swap.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="cc-pvtz-dk")
    p.add_argument("--soc", default="DKH1")
    p.add_argument("--nroots", type=int, default=10)
    p.add_argument("--n-occ", type=int, default=6)
    p.add_argument("--n-vir", type=int, default=1)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--csv", default=None)
    p.add_argument("--grid2d", action="store_true",
                   help="r 1.80-2.40 A x angle 85-125 deg (35 points), for the "
                        "2D band model; own CSV hoi_soc_2d.csv")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    args.csv = args.csv or str(CSV if not args.grid2d
                               else CSV.with_name("hoi_soc_2d.csv"))

    print("=" * 76)
    print("== HOI: spin-orbit states along O-I"
          + (" and the angle (2D grid)" if args.grid2d else ""))
    print(f"== {args.basis}, {args.soc}-QD-NEVPT2, fixed CAS({2 * args.n_occ},"
          f"{args.n_occ + args.n_vir}), {args.nroots} Ms=0 roots")
    print("=" * 76)
    done = load(args.csv)
    pts = points(args.grid2d)
    todo = [(r, t) for r, t in pts if pkey(r, t) not in done
            or done[pkey(r, t)]["status"].startswith("fail")]
    print(f"  {len(pts)} points, {len(todo)} to compute")

    if todo and not args.report_only:
        s04 = _load("s04", HOX / "HOBr" / "01_method" / "04_soc_vertical.py")
        f07 = _load("f07", HOX / "HOCl" / "01_method" / "07_fc_active_space.py")
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        new = not (os.path.exists(args.csv) and os.path.getsize(args.csv) > 0)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            for r, t in todo:
                try:
                    row, t0 = compute_point(r, t, args, s04, f07)
                except Exception as exc:
                    row, t0 = dict(r_ox_A=r, theta_deg=t,
                                   status=f"fail:{type(exc).__name__}"
                                   f":{exc}"[:100]), time.time()
                row["wall_s"] = round(time.time() - t0, 1)
                w.writerow(row)
                fh.flush()
                print(f"    r = {r:.4f}, angle {t:.1f}: {row['status']}, E1-3 "
                      + ", ".join(str(row.get(f'dE_{k}', '--')) for k in (1, 2, 3))
                      + f", {row['wall_s']} s", flush=True)
    rows = list(load(args.csv).values())
    report_2d(rows) if args.grid2d else report(rows)


if __name__ == "__main__":
    main()
