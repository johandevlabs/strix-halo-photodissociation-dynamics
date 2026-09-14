#!/usr/bin/env python3
"""
Phase 0.5, step 6: does the state-specific route survive bond dissociation?

04 showed dCCSD(T) reproduces the multi-state vertical energy to 0.034 eV at
EQUILIBRIUM. That is the easy case. A raster spends most of its points away
from equilibrium, and single-reference coupled cluster fails where
multireference character grows. This scans r(O-Cl) and measures that failure
directly, rather than assuming it does not happen.

THE TWO SURFACES HAVE OPPOSITE DIFFICULTY, which is the useful realisation:

  a 3A" over the FULL range is the EASY one. It dissociates to OH(2Pi) +
  Cl(2P) coupled high-spin, and a high-spin open-shell pair is exactly what a
  spin-unrestricted or ROHF reference describes well. This is the surface the
  wavepacket actually moves on, and it is single-reference friendly all the
  way out.

  X 1A' at large r is the HARD one. It becomes a singlet diradical, OH. + Cl.,
  which RHF cannot represent at all and RCCSD(T) inherits that failure.

But the ground state is only ever needed over the BOUND region, to make chi_0
-- water/13_gs_well.py computes V_gs "over the bound region only" for exactly
this reason. So the hard case is one we never have to enter. The scan verifies
that claim instead of asserting it: it reports T1 diagnostics for both spins at
every point, so the breakdown is visible where it happens.

Two checks beyond the diagnostics:

  SIZE CONSISTENCY. As r -> infinity the triplet energy must approach
  E(OH) + E(Cl) computed as separate fragments. This is the same test water
  applied to its splice, where the offset was constant to 13 meV and the
  A'/A" splitting went to 0.0000 eV at 10 A. CCSD(T) is size consistent, so
  a drift here means the reference is failing, not the theory.

  T-S GAP -> 0. At infinite separation the singlet and triplet are the same
  two radicals with no interaction, so the gap must vanish. It will not, in
  this scan, because the singlet reference collapses first -- and watching
  WHERE it stops vanishing is a clean measure of where RHF gives up.

T1 DIAGNOSTIC thresholds, the conventional ones: above 0.02 for a closed
shell, or 0.04 for an open shell, single-reference CC is suspect. These are
rules of thumb, not physics, but they are the standard rules of thumb.

Usage:
    python 05_triplet_scan.py 2>&1 | tee logs/triplet_scan.log
    python 05_triplet_scan.py --rmin 1.4 --rmax 5.0 --npoints 19
    python 05_triplet_scan.py --triplet-only      # skip the singlet entirely
    python 05_triplet_scan.py --symmetry Cs
"""
import argparse
import os
import time
import traceback

import numpy as np

from pyscf import gto, scf, cc

HARTREE2EV = 27.211386245988

R_OH_ANG = 0.9644
R_OCL_EQ_ANG = 1.6891
ANGLE_HOCL_DEG = 102.96

T1_THRESHOLD_CLOSED = 0.02
T1_THRESHOLD_OPEN = 0.04


def geometry(r_ocl):
    th = np.deg2rad(ANGLE_HOCL_DEG)
    return [["O", (0.0, 0.0, 0.0)],
            ["Cl", (r_ocl, 0.0, 0.0)],
            ["H", (R_OH_ANG * np.cos(th), R_OH_ANG * np.sin(th), 0.0)]]


def t1_diagnostic(mycc):
    """||t1|| / sqrt(n correlated electrons), the Lee-Taylor diagnostic."""
    t1 = mycc.t1
    if isinstance(t1, (list, tuple)):                    # UCCSD: (t1a, t1b)
        norm = np.sqrt(sum(float(np.linalg.norm(t)) ** 2 for t in t1))
        nelec = sum(t.shape[0] for t in t1)
    else:                                                # RCCSD
        norm = float(np.linalg.norm(t1))
        nelec = 2 * t1.shape[0]
    return norm / np.sqrt(nelec) if nelec else float("nan")


def irrep_occupation(mf):
    """Electrons per irrep as {irrep: (alpha, beta)}; None without symmetry."""
    if not mf.mol.symmetry:
        return None
    try:
        raw = mf.get_irrep_nelec()
    except Exception:
        return None
    occ = {}
    for k, v in raw.items():
        if isinstance(v, (tuple, list, np.ndarray)):
            occ[k] = (int(v[0]), int(v[1]))
        else:                                   # RHF reports the total
            occ[k] = (int(v) // 2, int(v) // 2)
    return occ


def state_label(occ, spin):
    """Multiplicity plus overall Cs symmetry, read off the open shells.

    In Cs, A' x A' = A" x A" = A' and A' x A" = A", so the state is A" exactly
    when an odd number of open-shell electrons sit in a" orbitals.
    """
    if occ is None:
        return "--"
    n_app = sum(a - b for k, (a, b) in occ.items() if '"' in k)
    sym = 'A"' if n_app % 2 else "A'"
    return f"{spin + 1}{sym}"


def run_point(atom, spin, basis, symmetry, verbose=0, pin=None, cc_on=True):
    """CCSD(T) on one state of this spin. Returns a dict of results.

    pin: {irrep: (alpha, beta)} to impose via irrep_nelec. Without it, an SCF
    fills orbitals by aufbau and can land on a different state at a different
    geometry. With symmetry on but nothing pinned, the orbitals are merely
    symmetry-PURE -- which for a planar molecule they already were, and why a
    bare --symmetry Cs reproduced the unsymmetric scan to 3e-7 Ha.
    """
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=0,
                symmetry=symmetry, unit="Angstrom", verbose=verbose)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).x2c()
    mf.conv_tol = 1e-11
    if pin is not None:
        # RHF wants a total per irrep, ROHF an (alpha, beta) pair.
        mf.irrep_nelec = {k: (a + b if spin == 0 else (a, b))
                          for k, (a, b) in pin.items()}
    mf.kernel()

    occ = irrep_occupation(mf)
    out = {"e_scf": mf.e_tot, "scf_conv": bool(mf.converged),
           "s2": float("nan"), "occ": occ, "label": state_label(occ, spin)}
    if spin != 0:
        out["s2"] = float(mf.spin_square()[0])
    if not cc_on:
        return out

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.kernel()
    out["cc_conv"] = bool(mycc.converged)
    out["t1"] = t1_diagnostic(mycc)
    out["e_tot"] = mycc.e_tot + mycc.ccsd_t()
    return out


def fragment_asymptote(basis, verbose=0):
    """E(OH) + E(Cl) as separated fragments -- water's 05_oh_diatomic move.

    Always without symmetry: OH is C-infinity-v and Cl is an atom, so asking
    for Cs raises PointGroupSymmetryError, and the energies do not need it.
    """
    oh = run_point([["O", (0.0, 0.0, 0.0)], ["H", (0.0, 0.0, R_OH_ANG)]],
                   1, basis, False, verbose)
    cl = run_point([["Cl", (0.0, 0.0, 0.0)]], 1, basis, False, verbose)
    return oh, cl


def analyse_smoothness(rows, key="e_t"):
    """Find kinks in a curve that physics says should be smooth.

    Past its minimum a dissociating state must rise monotonically towards the
    asymptote. A decrease after that point is not a feature of the potential,
    it is the reference changing character -- and unlike a T1 spike it shows up
    directly as a defect in the surface the propagation would use.

    Returns (monotonicity violations, worst second difference), both as lists
    of (r, value).
    """
    pts = [(r["r"], r[key]) for r in rows if key in r]
    if len(pts) < 3:
        return [], []
    rs = [p[0] for p in pts]
    es = [p[1] for p in pts]

    imin = int(np.argmin(es))
    violations = []
    for i in range(imin + 1, len(es)):
        if es[i] < es[i - 1]:
            violations.append((rs[i], (es[i] - es[i - 1]) * HARTREE2EV))

    # Second differences, but ONLY past the minimum. Applied to the whole
    # curve this test flags the repulsive wall at short r, where curvature is
    # large because the potential really is strongly curved -- a false
    # positive that also drowns out the genuine kink further out. Past the
    # minimum the curve should decay smoothly to the asymptote, so any sharp
    # curvature there is suspect.
    d2 = []
    for i in range(max(imin + 1, 1), len(es) - 1):
        val = (es[i + 1] - 2 * es[i] + es[i - 1]) * HARTREE2EV
        d2.append((rs[i], val))
    if d2:
        med = float(np.median([abs(v) for _, v in d2]))
        d2 = [(r, v) for r, v in d2 if med > 0 and abs(v) > 8 * med]
    return violations, d2


def make_plot(rows, path, e_inf=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rs = [r["r"] for r in rows if "e_t" in r]
    et = [r["e_t"] for r in rows if "e_t" in r]
    rs_s = [r["r"] for r in rows if "e_s" in r]
    es = [r["e_s"] for r in rows if "e_s" in r]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    if es:
        ax1.plot(rs_s, es, "o-", label="X 1A' (CCSD(T))", color="#1f77b4")
    ax1.plot(rs, et, "s-", label='a 3A" (UCCSD(T))', color="#d62728")
    if e_inf is not None:
        ax1.axhline(e_inf, ls=":", color="grey", label="OH + Cl fragments")
    ax1.axvline(R_OCL_EQ_ANG, ls="--", color="k", lw=0.8, label="equilibrium")
    ax1.set_ylabel("E / Ha")
    ax1.legend(fontsize=8)
    ax1.set_title("HOCl, O-Cl cut at fixed r(OH) and angle")

    t1t = [r["t1_t"] for r in rows if "t1_t" in r]
    ax2.plot(rs, t1t, "s-", color="#d62728", label='T1, a 3A"')
    if any("t1_s" in r for r in rows):
        ax2.plot([r["r"] for r in rows if "t1_s" in r],
                 [r["t1_s"] for r in rows if "t1_s" in r],
                 "o-", color="#1f77b4", label="T1, X 1A'")
    ax2.axhline(T1_THRESHOLD_OPEN, ls=":", color="#d62728",
                label=f"open-shell {T1_THRESHOLD_OPEN}")
    ax2.axhline(T1_THRESHOLD_CLOSED, ls=":", color="#1f77b4",
                label=f"closed-shell {T1_THRESHOLD_CLOSED}")
    ax2.axvline(R_OCL_EQ_ANG, ls="--", color="k", lw=0.8)
    ax2.set_xlabel("r(O-Cl) / A")
    ax2.set_ylabel("T1 diagnostic")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--symmetry", default=None,
                   help="point group to FORCE, e.g. Cs. For a real raster "
                        "this is not optional -- see TASKS.md.")
    p.add_argument("--rmin", type=float, default=1.4)
    p.add_argument("--rmax", type=float, default=4.0)
    p.add_argument("--npoints", type=int, default=14)
    p.add_argument("--pin-irrep", action="store_true",
                   help="hold each spin's EQUILIBRIUM irrep occupation fixed "
                        "along the scan via irrep_nelec (implies Cs). Also "
                        "runs an unconstrained SCF at every point and reports "
                        "whether it would have landed on a different state.")
    p.add_argument("--triplet-only", action="store_true",
                   help="skip the singlet; it is only needed in the bound "
                        "region anyway")
    p.add_argument("--no-asymptote", action="store_true")
    p.add_argument("--csv", default="hocl_scan.csv")
    p.add_argument("--png", default="hocl_scan.png")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    sym = args.symmetry if args.symmetry else False
    if args.pin_irrep and not sym:
        sym = "Cs"
        print("  --pin-irrep needs a point group; using Cs")
    rs = np.linspace(args.rmin, args.rmax, args.npoints)

    pin_s = pin_t = None
    if args.pin_irrep:
        # Take the occupation from EQUILIBRIUM, where 04 showed this reference
        # reproduces the multi-state a 3A" to 0.034 eV. That is the state we
        # want to follow; the question is whether aufbau abandons it.
        eq = geometry(R_OCL_EQ_ANG)
        t_eq = run_point(eq, 2, args.basis, sym, args.verbose, cc_on=False)
        pin_t = t_eq["occ"]
        print(f"  pinned triplet occupation from r = {R_OCL_EQ_ANG} A: "
              f"{pin_t}  -> {t_eq['label']}")
        if t_eq["label"] != '3A"':
            print("  !! the equilibrium triplet reference is NOT 3A\". Pinning"
                  " it would follow the wrong state; check the geometry.")
        if not args.triplet_only:
            s_eq = run_point(eq, 0, args.basis, sym, args.verbose, cc_on=False)
            pin_s = s_eq["occ"]
            print(f"  pinned singlet occupation: {pin_s}  -> {s_eq['label']}")

    print("=" * 78)
    print("== HOX Phase 0.5 -- dCCSD(T) along the O-Cl dissociation coordinate")
    print(f"== basis {args.basis}, symmetry {sym}, "
          f"r = {args.rmin} to {args.rmax} A in {args.npoints} points")
    print(f"== equilibrium is r(O-Cl) = {R_OCL_EQ_ANG} A")
    print("=" * 78)

    t0, c0 = time.time(), time.process_time()
    rows = []
    print(f"\n  {'r/A':>7}{'E(S)/Ha':>18}{'T1(S)':>8}"
          f"{'E(T)/Ha':>18}{'T1(T)':>8}{'<S^2>':>8}{'state':>7}"
          f"{'dE/eV':>9}  flags")
    for r in rs:
        atom = geometry(r)
        row = {"r": r}
        flags = []
        try:
            if not args.triplet_only:
                s = run_point(atom, 0, args.basis, sym, args.verbose,
                              pin=pin_s)
                row["e_s"], row["t1_s"] = s["e_tot"], s["t1"]
                if s["t1"] > T1_THRESHOLD_CLOSED:
                    flags.append("S:T1")
                if not (s["scf_conv"] and s["cc_conv"]):
                    flags.append("S:noconv")

            if pin_t is not None:
                # Where would an unconstrained SCF have gone? SCF alone is
                # seconds, so asking costs almost nothing next to CCSD(T).
                free = run_point(atom, 2, args.basis, sym, args.verbose,
                                 cc_on=False)
                row["label_free"] = free["label"]
                if free["occ"] != pin_t:
                    flags.append(f"T:aufbau->{free['label']}")
                    row["switch"] = True

            t = run_point(atom, 2, args.basis, sym, args.verbose, pin=pin_t)
            row["e_t"], row["t1_t"], row["s2"] = t["e_tot"], t["t1"], t["s2"]
            row["label_t"] = t["label"]
            if pin_t is not None and row.get("switch"):
                gap = (free["e_scf"] - t["e_scf"]) * HARTREE2EV
                row["free_minus_pinned_scf_ev"] = gap
            if t["t1"] > T1_THRESHOLD_OPEN:
                flags.append("T:T1")
            if not (t["scf_conv"] and t["cc_conv"]):
                flags.append("T:noconv")
            if abs(t["s2"] - 2.0) > 0.02:
                flags.append("T:spin")

            de = ((row["e_t"] - row["e_s"]) * HARTREE2EV
                  if "e_s" in row else float("nan"))
            row["de"] = de
            es = f"{row.get('e_s', float('nan')):18.8f}"
            t1s = f"{row.get('t1_s', float('nan')):8.4f}"
            print(f"  {r:7.3f}{es}{t1s}{row['e_t']:18.8f}{row['t1_t']:8.4f}"
                  f"{row['s2']:8.4f}{row['label_t']:>7}{de:9.4f}  "
                  f"{' '.join(flags)}")
        except Exception as exc:
            print(f"  {r:7.3f}  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        rows.append(row)

    # ------------------------------------------------------------ asymptote
    e_inf = None
    if not args.no_asymptote:
        print("\n  --- separated fragments (size-consistency check)")
        try:
            oh, cl = fragment_asymptote(args.basis, args.verbose)
            e_inf = oh["e_tot"] + cl["e_tot"]
            print(f"      OH(2Pi)   {oh['e_tot']:18.8f} Ha  "
                  f"T1 {oh['t1']:.4f}")
            print(f"      Cl(2P)    {cl['e_tot']:18.8f} Ha  "
                  f"T1 {cl['t1']:.4f}")
            print(f"      sum       {e_inf:18.8f} Ha")
            last = [r for r in rows if "e_t" in r]
            if last:
                far = last[-1]
                diff = (far["e_t"] - e_inf) * HARTREE2EV
                print(f"      triplet at r = {far['r']:.3f} A is "
                      f"{diff:+.4f} eV from the fragment sum")
                if abs(diff) < 0.05:
                    print("      -> size consistent to better than 50 meV. The "
                          "triplet surface can be\n         spliced onto the "
                          "fragment asymptote as water/06_splice.py does.")
                else:
                    print("      -> the gap is large. Either r_max is not yet "
                          "asymptotic (check whether\n         dE/eV is still "
                          "falling above), or the reference is failing out "
                          "there.")
        except Exception as exc:
            print(f"      asymptote FAILED: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ csv
    if rows:
        import csv
        keys = ["r", "e_s", "t1_s", "e_t", "t1_t", "s2", "de",
                "label_t", "label_free", "free_minus_pinned_scf_ev"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in keys})
        print(f"\n  wrote {args.csv}")
        try:
            make_plot(rows, args.png, e_inf)
        except Exception as exc:
            print(f"  plot FAILED: {type(exc).__name__}: {exc}")

    # -------------------------------------------------------------- verdict
    wall, cpu = time.time() - t0, time.process_time() - c0
    ncpu = os.cpu_count() or 1
    phys = ncpu // 2 if ncpu > 1 else 1
    per_point = cpu / max(len(rs), 1)

    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)

    bad_t = [r["r"] for r in rows
             if "t1_t" in r and r["t1_t"] > T1_THRESHOLD_OPEN]
    bad_s = [r["r"] for r in rows
             if "t1_s" in r and r["t1_s"] > T1_THRESHOLD_CLOSED]

    # A kink in the curve is more damning than a T1 value: T1 warns that the
    # reference is strained, a non-monotonic tail proves the surface is wrong.
    viol, kinks = analyse_smoothness(rows, "e_t")
    if viol or kinks:
        print("  TRIPLET SURFACE IS NOT SMOOTH.")
        for r, dv in viol:
            print(f"    r = {r:.2f} A: energy DROPS by {abs(dv):.4f} eV past "
                  f"the minimum -- unphysical for a dissociating state")
        for r, dv in kinks:
            print(f"    r = {r:.2f} A: second difference {dv:+.4f} eV, far "
                  f"above the curve's typical curvature")
        print("  This is a defect in the potential itself, not a warning about")
        print("  the reference. A propagation on this surface would scatter the")
        print("  wavepacket off an artefact.")
        print()

    # Did the reference change STATE along the scan? Decisive for the repair:
    # an occupation flip is fixed by pinning; a same-symmetry problem is not.
    switches = [r for r in rows if r.get("switch")]
    if args.pin_irrep:
        if switches:
            where = ", ".join(f"{r['r']:.2f} ({r['label_free']}, "
                              f"{r.get('free_minus_pinned_scf_ev', 0):+.3f} eV)"
                              for r in switches)
            print(f"  STATE SWITCH: unconstrained aufbau left the equilibrium "
                  f"occupation at r = {where}")
            print("  (the eV figure is SCF(free) - SCF(pinned); negative means")
            print("  aufbau found a LOWER determinant of different occupation.)")
            if viol or kinks:
                print("  The pinned curve is STILL kinked, so the occupation flip")
                print("  is not the whole story -- see the T1 values below.")
            else:
                print("  The pinned curve is smooth. The kink was the SCF")
                print("  changing state, and irrep_nelec repairs it cheaply.")
        else:
            print("  No state switch: unconstrained aufbau kept the equilibrium")
            print("  occupation at every point.")
            if viol or kinks:
                print("  So the kink is NOT an A'/A\" occupation flip. It is a")
                print("  same-symmetry problem -- a 3A\" avoided crossing or the")
                print("  SCF landing in a different local solution of the same")
                print("  occupation -- which pinning cannot reach. That region")
                print("  needs a multireference treatment.")
        print()
    elif sym:
        labels = sorted({r["label_t"] for r in rows
                         if r.get("label_t") not in (None, "--")})
        print(f"  triplet state labels along the scan: {', '.join(labels)}")
        print()

    if bad_t:
        print(f"  TRIPLET T1 above {T1_THRESHOLD_OPEN} at r = "
              f"{', '.join(f'{r:.2f}' for r in bad_t)} A")
        lo, hi = min(bad_t), max(bad_t)
        fc_lo, fc_hi = R_OCL_EQ_ANG - 0.2, R_OCL_EQ_ANG + 0.2
        if hi < fc_lo or lo > fc_hi:
            print(f"  The flagged region ({lo:.2f}-{hi:.2f} A) lies OUTSIDE the")
            print(f"  Franck-Condon window (~{fc_lo:.2f}-{fc_hi:.2f} A), where")
            print("  the band position and width are set by the reflection")
            print("  principle. That is the good case: the absorption profile")
            print("  is governed by a region the reference handles, and the")
            print("  defect sits where it mainly affects fine structure. It")
            print("  still has to be repaired before propagating -- the packet")
            print("  travels through it -- but it does not invalidate the")
            print("  state-specific approach near equilibrium.")
        else:
            print("  The flagged region OVERLAPS the Franck-Condon window.")
            print("  That is the bad case: the band position, width and")
            print("  intensity all derive from precisely this region, so the")
            print("  state-specific route cannot be trusted for the spectrum.")
    else:
        print(f"  Triplet T1 stayed below {T1_THRESHOLD_OPEN} across the whole")
        print("  scan. The state-specific triplet surface is sound along the")
        print("  dissociation coordinate, which is what Phase 1 depends on.")

    if bad_s:
        print(f"\n  Singlet T1 above {T1_THRESHOLD_CLOSED} from r = "
              f"{min(bad_s):.2f} A outward. EXPECTED: the ground state becomes")
        print("  a diradical that RHF cannot represent. Harmless, provided the")
        print("  ground-state surface is built only over the bound region, as")
        print("  water/13_gs_well.py does. Treat min(flagged r) as the outer")
        print("  limit of where V_gs may be trusted.")

    print(f"\n  cost: {cpu:.0f} CPU-s for {len(rs)} points = "
          f"{per_point:.0f} CPU-s/point ({wall:.0f} s wall)")
    print(f"  a 3289-point raster at this rate: "
          f"{per_point * 3289 / 3600:.0f} CPU-hours = "
          f"{per_point * 3289 / 3600 / phys:.0f} h on {phys} cores")


if __name__ == "__main__":
    main()
