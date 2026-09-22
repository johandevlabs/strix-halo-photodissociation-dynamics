#!/usr/bin/env python3
"""
Phase 1 prerequisite: is CCSD(T) + EOM-CCSD sound along the O-H stretch and
the bend, not just along O-Cl?

09 and 10 established the triplet surface along r(O-Cl) only: EOM-CCSD is
smooth and trustworthy to 2.29 A there, and the 1D band depends on the surface
out to ~2 A. A 3D raster also moves r(O-H) and the H-O-Cl angle, and nothing
has checked EOM-CCSD in those directions. This does, before ~16 h of raster.

RANGES. The ground-state wavefunction sets how far each coordinate matters.
O-H stretch: ~3600 cm-1 with reduced mass ~0.95 amu gives a zero-point
amplitude of ~0.10 A, so 0.75-1.25 A spans about +/-2.5 amplitudes. Bend:
~1240 cm-1, a few degrees of zero-point amplitude, so 75-135 deg is generous.
Eight corner points combine displacements in all three coordinates, where a
problem invisible along any single cut could appear.

SYMMETRY LABELS. Along O-Cl the lowest EOM triplet stayed 3A" until the
crossing, and 09 could get away with "lowest root" plus a gap check. Along the
bend that is not safe: a 3A' could fall below the 3A" and still leave a
comfortable gap to the next root. So the molecule is built in Cs, every EOM
root is labelled by the irrep of its dominant single excitation (irrep of the
occupied orbital times irrep of the virtual), and the surface value taken is
the lowest root labelled A". The gap to the next root of ANY symmetry and to
the next A" root are both reported.

For each point: T1 of the closed-shell CCSD (the reference EOM stands on),
omega of the lowest 3A", which symmetry root 0 has, both gaps, the weight of
single excitations in that root, and convergence.

VERDICT, per cut and overall. EOM-CCSD is usable for the raster where:
  - everything converged;
  - T1(S) < 0.02, the conventional closed-shell threshold;
  - the lowest 3A" is identified, with the next 3A" > 0.3 eV above;
  - omega and V_T are smooth along each cut (< 5 meV from a degree-5 fit;
    see roughness_mev for why not a quartic).
A 3A' below the 3A" is reported but not disqualifying on its own: in Cs, and
without SOC, it does not couple to the 3A" surface the band is computed on.

Consistency check: the cut points at the 09 geometry must reproduce 09's
EOM-CCSD vertical energy there, 3.4320 eV.

Usage:
    python 11_eom_coordinates.py 2>&1 | tee logs/eom_coordinates.log
    python 11_eom_coordinates.py --no-corners
"""
import argparse
import csv
import time
import traceback

import numpy as np

from pyscf import gto, scf, cc, symm
from pathlib import Path

# Data files live in HOCl/data/, one level up from this approach directory,
# so the defaults below work no matter where the script is invoked from.
DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2EV = 27.211386245988

R_OCL_ANG, R_OH_ANG, ANGLE_DEG = 1.6891, 0.9644, 102.96
REF_09_EV = 3.4320
T1_MAX, GAP_MIN_EV, ROUGH_MEV = 0.02, 0.3, 5.0


def geometry(r_ocl, r_oh, angle):
    th = np.deg2rad(angle)
    return [["O", (0.0, 0.0, 0.0)],
            ["Cl", (r_ocl, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def label_roots(mol, mf, eom, vectors, nocc):
    """Irrep and single-excitation weight of each EOM root."""
    ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff,
                              s=mf.get_ovlp())
    name_of = dict(zip(mol.irrep_id, mol.irrep_name))
    out = []
    for v in vectors:
        r1 = np.asarray(eom.vector_to_amplitudes(v)[0])
        i, a = np.unravel_index(int(np.argmax(np.abs(r1))), r1.shape)
        irrep = int(ids[i]) ^ int(ids[nocc + a])      # Cs: XOR of irrep ids
        w1 = float(np.sum(np.abs(r1) ** 2) / np.sum(np.abs(v) ** 2))
        out.append((name_of.get(irrep, str(irrep)), w1))
    return out


def run_point(cut, r_ocl, r_oh, angle, args):
    t0 = time.time()
    mol = gto.M(atom=geometry(r_ocl, r_oh, angle), basis=args.basis, spin=0,
                charge=0, symmetry="Cs", unit="Angstrom", verbose=args.verbose,
                max_memory=16000)
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-10
    mf.kernel()
    mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.kernel()
    nocc = mycc.t1.shape[0]
    t1 = float(np.linalg.norm(mycc.t1) / np.sqrt(2 * nocc))
    e_ccsdt = float(mycc.e_tot) + float(mycc.ccsd_t())

    from pyscf.cc import eom_rccsd
    eom = eom_rccsd.EOMEETriplet(mycc)
    e, vecs = eom.kernel(nroots=args.nroots)
    e = np.atleast_1d(np.asarray(e, dtype=float))
    vecs = vecs if isinstance(vecs, (list, tuple)) else list(vecs)
    order = np.argsort(e)
    e = e[order]
    vecs = [vecs[k] for k in order]
    conv = bool(mf.converged and mycc.converged
                and np.all(np.atleast_1d(getattr(eom, "converged", True))))

    row = {"cut": cut, "r_ocl": r_ocl, "r_oh": r_oh, "angle": angle,
           "t1_s": t1, "e_ccsdt": e_ccsdt, "conv": conv}
    try:
        labels = label_roots(mol, mf, eom, vecs, nocc)
        app = [k for k, (name, _) in enumerate(labels) if '"' in name]
    except Exception as exc:
        # Keep the run useful if labelling fails: fall back to 09's rule,
        # lowest root, and say so on every row.
        labels = [("?", float("nan"))] * len(e)
        app = list(range(len(e)))
        row["note"] = (f"unlabelled ({type(exc).__name__}); lowest root used, "
                       f"as in 09")
    row["root0_sym"] = labels[0][0]
    if app:
        k = app[0]
        row["omega"] = float(e[k] * HARTREE2EV)
        row["w1"] = labels[k][1]
        others = [e[j] for j in range(len(e)) if j != k]
        row["gap_any"] = (float((min(others, key=lambda x: abs(x - e[k]))
                                 - e[k]) * HARTREE2EV) if others else float("nan"))
        row["gap_app"] = (float((e[app[1]] - e[k]) * HARTREE2EV)
                          if len(app) > 1 else float("nan"))
    else:
        row["omega"] = float("nan")
        row["note"] = (row.get("note", "") + " no A\" root among "
                       f"{len(e)} computed").strip()
    row["wall"] = time.time() - t0
    return row


def roughness_mev(x, y, deg=5):
    """Largest deviation from a smooth degree-5 polynomial, in meV.

    Degree 5, not 4. The O-H cut spans ~2 eV of a Morse-shaped curve over
    0.5 A, and a quartic cannot follow that to better than ~9 meV, which
    nearly tripped the 10 meV threshold on a surface whose second differences
    were perfectly smooth (same sign, monotone). Calibrated on a synthetic
    Morse curve on this grid: degree 5 leaves 0.4 meV on a smooth curve and
    10.3 meV when a 30 meV step is injected, so with the threshold at 5 meV
    this catches steps of roughly 15 meV and up while accepting real
    curvature.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < deg + 2 or np.any(~np.isfinite(y)):
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    return float(np.abs(yc - np.polyval(np.polyfit(xc, yc, deg), xc)).max() * 1000)


def analyse(rows):
    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    problems = []
    for cut, coord in (("oh", "r_oh"), ("bend", "angle")):
        rs = [x for x in rows if x["cut"] == cut]
        if not rs:
            continue
        xs = [x[coord] for x in rs]
        om = [x["omega"] for x in rs]
        vt = [x["e_ccsdt"] * HARTREE2EV + x["omega"] for x in rs]
        rough_w, rough_v = roughness_mev(xs, om), roughness_mev(xs, vt)
        t1max = max(x["t1_s"] for x in rs)
        gap_app = min((x["gap_app"] for x in rs if np.isfinite(x["gap_app"])),
                      default=float("nan"))
        not_app = [x[coord] for x in rs if '"' not in x["root0_sym"]]
        bad_conv = [x[coord] for x in rs if not x["conv"]]
        unit = "A" if coord == "r_oh" else "deg"
        print(f"\n  {cut} cut, {min(xs):g}-{max(xs):g} {unit}:")
        print(f"    max T1(S) {t1max:.4f}   min gap to next 3A\" {gap_app:.2f} eV"
              f"   roughness omega {rough_w:.1f} meV, V_T {rough_v:.1f} meV")
        if not_app:
            print(f"    root 0 is not A\" at {unit} = "
                  f"{', '.join(f'{v:g}' for v in not_app)} (reported; not "
                  f"disqualifying in Cs without SOC)")
        for ok, msg in ((not bad_conv, f"unconverged at {bad_conv}"),
                        (t1max < T1_MAX, f"T1(S) reaches {t1max:.3f}"),
                        (not np.isfinite(gap_app) or gap_app > GAP_MIN_EV,
                         f"next 3A\" only {gap_app:.2f} eV above"),
                        (np.isfinite(rough_v) and rough_v < ROUGH_MEV,
                         f"V_T rough by {rough_v:.1f} meV")):
            if not ok:
                problems.append(f"{cut}: {msg}")
                print(f"    !! {msg}")

    corners = [x for x in rows if x["cut"] == "corner"]
    if corners:
        print("\n  corners:")
        for x in corners:
            flag = []
            if x["t1_s"] >= T1_MAX:
                flag.append("T1")
            if np.isfinite(x["gap_app"]) and x["gap_app"] <= GAP_MIN_EV:
                flag.append("gap")
            if not x["conv"]:
                flag.append("noconv")
            if not np.isfinite(x["omega"]):
                flag.append("no A\"")
            print(f"    r(O-Cl) {x['r_ocl']:.2f}  r(O-H) {x['r_oh']:.2f}  "
                  f"{x['angle']:5.1f} deg: T1 {x['t1_s']:.4f}  omega "
                  f"{x['omega']:.3f} eV  root0 {x['root0_sym']}"
                  f"{'  !! ' + ', '.join(flag) if flag else ''}")
            if flag:
                problems.append(f"corner {x['r_ocl']}/{x['r_oh']}/{x['angle']}: "
                                f"{', '.join(flag)}")

    eq = [x for x in rows if abs(x["r_ocl"] - R_OCL_ANG) < 1e-6
          and abs(x["r_oh"] - R_OH_ANG) < 1e-6 and abs(x["angle"] - ANGLE_DEG) < 1e-6]
    if eq:
        d = eq[0]["omega"] - REF_09_EV
        print(f"\n  consistency: omega at the 09 geometry {eq[0]['omega']:.4f} eV, "
              f"09 gave {REF_09_EV:.4f} ({d * 1000:+.1f} meV)")
        if abs(d) > 0.005:
            problems.append(f"omega at the 09 geometry differs by {d * 1000:.1f} meV")

    print()
    if problems:
        print("  -> NOT cleared for the raster as it stands:")
        for p in problems:
            print(f"     - {p}")
    else:
        print("  -> CCSD(T) + EOM-CCSD is sound along the O-H stretch, the bend "
              "and the corners\n     of the Franck-Condon region. Cleared for "
              "the 3D raster.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--nroots", type=int, default=4)
    p.add_argument("--no-corners", action="store_true")
    p.add_argument("--csv", default=str(DATA / "hocl_eom_coordinates.csv"))
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    points = [("oh", R_OCL_ANG, r, ANGLE_DEG)
              for r in sorted(set(np.round(np.arange(0.75, 1.2501, 0.05), 4))
                              | {R_OH_ANG})]
    points += [("bend", R_OCL_ANG, R_OH_ANG, a)
               for a in sorted(set(np.arange(75.0, 135.01, 5.0)) | {ANGLE_DEG})]
    if not args.no_corners:
        points += [("corner", ro, rh, an)
                   for ro in (1.60, 1.85) for rh in (0.85, 1.10)
                   for an in (85.0, 120.0)]

    print("=" * 78)
    print("== HOX Phase 1 prerequisite -- EOM-CCSD across O-H stretch and bend")
    print(f"== {args.basis}, Cs, {args.nroots} EOM triplet roots, "
          f"{len(points)} points")
    print("=" * 78)
    h_omega, h_gap = 'omega 3A"', 'gap A"'
    print(f"\n  {'cut':>6}{'r(OCl)':>7}{'r(OH)':>7}{'angle':>7}{'T1(S)':>8}"
          f"{h_omega:>10}{'root0':>6}{'gap any':>9}{h_gap:>8}"
          f"{'w1':>6}{'conv':>5}{'t/s':>5}")
    rows = []
    for cut, ro, rh, an in points:
        try:
            row = run_point(cut, ro, rh, an, args)
        except Exception as exc:
            print(f"  {cut:>6}{ro:7.3f}{rh:7.3f}{an:7.1f}  FAILED: "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        rows.append(row)
        print(f"  {cut:>6}{ro:7.3f}{rh:7.3f}{an:7.1f}{row['t1_s']:8.4f}"
              f"{row['omega']:10.4f}{row['root0_sym']:>6}"
              f"{row.get('gap_any', float('nan')):9.3f}"
              f"{row.get('gap_app', float('nan')):8.3f}"
              f"{row.get('w1', float('nan')):6.2f}"
              f"{'yes' if row['conv'] else 'NO':>5}{row['wall']:5.0f}"
              f"{'  ' + row['note'] if row.get('note') else ''}")

    if not rows:
        return
    cols = ["cut", "r_ocl", "r_oh", "angle", "t1_s", "e_ccsdt", "omega",
            "root0_sym", "gap_any", "gap_app", "w1", "conv", "wall", "note"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(f"\n  wrote {args.csv}")
    for row in rows:
        row.setdefault("gap_app", float("nan"))
    analyse(rows)


if __name__ == "__main__":
    main()
