#!/usr/bin/env python3
"""
HOBr step 2: the a 3A" triplet along O-Br, in two bases.

Descends from HOCl's 09 (EOM slopes), 11 (EOM validity and root labelling)
and 06 (does a second triplet cross?). One cut answers all three for HOBr,
and a second basis answers a fourth question at almost no extra cost.

FOUR THINGS COME OUT OF ONE RUN:

1. **Is EOM-CCSD valid for HOBr, and out to what radius?** For HOCl it was
   trustworthy to 2.29 A, breaking where a second triplet -- probably 3A' --
   crossed, and T1(S) passed 0.02 by 2.6 A. Br is heavier and its SOC is 4x
   larger, so none of that transfers. The raster's outer edge depends on the
   answer.

2. **Where does 3A' cross 3A"?** The surface must store the lowest 3A", not
   merely the lowest triplet. HOCl's raster found 3A' below at 112 of 2730
   points, all at r >= 2.25 A and wide angles, and the symmetry labelling is
   the only reason a 3A' energy did not go into the surface unnoticed.

3. **The band's position and width, to first order.** Vertical excitation and
   slope at the Franck-Condon geometry, against Ingham's measured 457 nm.

4. **Does aug- matter in the bond-breaking region?** The atomic sweep cannot
   say: SOC is a near-nuclear property, so aug-cc-pvtz-dk matching cc-pvtz-dk
   to 0.1 points on the 2P splitting is no evidence at all about 2-3 A. The
   leading expectation is that it does little -- a 3A" n -> sigma* going to
   OH(2Pi) + Br(2P) is valence throughout, with no Rydberg, anion or
   charge-transfer character. But two things could differ:

     - BSSE, which in a non-augmented basis is an attractive error peaking
       exactly where the fragments are close but separating. Its signature is
       a cc-minus-aug difference that GROWS with r, so the script tests for
       the trend, not the offset.
     - Spurious diffuse roots. This is the risk that aug- makes things worse:
       the raster identifies its state by labelling roots, and diffuse
       functions add low-lying ones to sort through, near dissociation where
       the level density is already high. The per-root symmetries are stored
       so this is visible rather than assumed.

Every root is labelled A' or A" by the irrep of its dominant single
excitation, as in 11 and 12, and ALL roots are stored -- not just the lowest
3A" -- because the crossing is the point.

Usage:
    python 02_obr_cut.py --dry-run
    python 02_obr_cut.py --bases cc-pvtz-dk aug-cc-pvtz-dk
    python 02_obr_cut.py --molecule HOCl        # reproduce 09 as a control
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
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
HARTREE2EV = 27.211386245988
DEG = np.pi / 180.0
NM_PER_EV = 1239.841984

# Spectator coordinates, held fixed along the cut. Update from 01_geometry.py
# once it has run: these are the recalled starting values, good enough to
# compare two bases against each other but not to quote.
MOLECULES = {
    "HOBr": dict(halogen="Br", r_eq=1.834, r_oh=0.961, theta=102.3,
                 obs_nm=457.0),
    "HOCl": dict(halogen="Cl", r_eq=1.6891, r_oh=0.9644, theta=102.96,
                 obs_nm=380.0),
}

NROOTS = 6
T1_MAX = 0.02


def fields(nroots):
    base = ["basis", "r_ox_A", "nbf", "t1_s", "e_hf_Ha", "e_ccsd_Ha",
            "e_ccsdt_Ha", "omega_app_eV", "w1_app", "root0_sym",
            "gap_any_eV", "gap_app_eV", "status", "wall_s"]
    for k in range(1, nroots + 1):
        base += [f"omega{k}_eV", f"sym{k}", f"w1_{k}"]
    return base


def key(basis, r_ox):
    """Basis is part of a point's identity -- see 01_geometry.py's note."""
    return (str(basis), round(float(r_ox), 5))


def geometry(halogen, r_ox, r_oh, theta):
    th = theta * DEG
    return [["O", (0.0, 0.0, 0.0)],
            [halogen, (r_ox, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def compute_point(task):
    """CCSD(T) ground state and EOM-CCSD triplet roots at one radius."""
    halogen, r_ox, r_oh, theta, basis, nroots, mem = task
    t0 = time.perf_counter()
    row = {"basis": basis, "r_ox_A": round(r_ox, 5), "status": "ok"}
    try:
        from pyscf import gto, scf, cc, symm
        from pyscf.cc import eom_rccsd

        mol = gto.M(atom=geometry(halogen, r_ox, r_oh, theta), basis=basis,
                    spin=0, charge=0, symmetry="Cs", unit="Angstrom",
                    verbose=0, max_memory=mem)
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
        mycc.max_cycle = 100
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

        # Irrep of the dominant single excitation, as 11 and 12 do. Duplicated
        # rather than imported: a worker should not depend on another script's
        # import side effects.
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

        # EVERY root is stored: the crossing is the thing being looked for,
        # and a script that keeps only its target cannot show one.
        for k, (om, (nm, w1)) in enumerate(zip(e, labels), 1):
            row[f"omega{k}_eV"] = float(om * HARTREE2EV)
            row[f"sym{k}"] = nm
            row[f"w1_{k}"] = round(w1, 4)

        row["root0_sym"] = labels[0][0]
        app = [k for k, (nm, _) in enumerate(labels) if '"' in nm]
        if not app:
            row["status"] = f"fail:no_App_in_{len(e)}_roots"
            return _stamp(row, t0)
        k = app[0]
        row["omega_app_eV"] = float(e[k] * HARTREE2EV)
        row["w1_app"] = round(labels[k][1], 4)
        others = [e[j] for j in range(len(e)) if j != k]
        row["gap_any_eV"] = (float((min(others, key=lambda x: abs(x - e[k]))
                                    - e[k]) * HARTREE2EV)
                             if others else float("nan"))
        row["gap_app_eV"] = (float((e[app[1]] - e[k]) * HARTREE2EV)
                             if len(app) > 1 else float("nan"))

        flags = []
        if row["t1_s"] >= T1_MAX:
            flags.append("t1")
        if k != 0:
            flags.append("Ap_below")
        if row["w1_app"] < 0.90:
            flags.append("w1")
        if flags and row["status"] == "ok":
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
        os.sched_setaffinity(0, {idx % os.cpu_count()})
    except (AttributeError, OSError):
        pass


def load_done(path, nroots):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                done[key(row["basis"], row["r_ox_A"])] = row
            except (KeyError, ValueError):
                continue
    return done


def build_radii(r_eq, args):
    """Fine where the band lives, coarse where only the shape matters."""
    fine = np.arange(args.rmin, args.rfine + 1e-9, args.dfine)
    coarse = np.arange(args.rfine + args.dcoarse, args.rmax + 1e-9,
                       args.dcoarse)
    return [round(float(r), 5) for r in np.concatenate([fine, coarse])]


def _f(row, name):
    v = row.get(name, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def report_one(basis, rows, spec, nroots):
    """Per-basis: validity, the crossing, and the band to first order."""
    print(f"\n{'=' * 72}\n== {basis}\n{'=' * 72}")
    good = [r for r in rows if not r["status"].startswith("fail")]
    print(f"  {len(good)} of {len(rows)} points usable"
          f"   nbf {good[0]['nbf'] if good else '?'}")
    for r in rows:
        if r["status"].startswith("fail"):
            print(f"     {_f(r, 'r_ox_A'):.3f} A: {r['status']}")
    if not good:
        return None

    print(f"\n  {'r/A':>7}{'T1(S)':>9}{'w(3A\")/eV':>11}{'root0':>8}"
          f"{'gap any':>9}{'gap A\"':>9}{'w1':>7}  status")
    for r in good:
        print(f"  {_f(r, 'r_ox_A'):>7.3f}{_f(r, 't1_s'):>9.4f}"
              f"{_f(r, 'omega_app_eV'):>11.3f}{r.get('root0_sym', '?'):>8}"
              f"{_f(r, 'gap_any_eV'):>9.3f}{_f(r, 'gap_app_eV'):>9.3f}"
              f"{_f(r, 'w1_app'):>7.3f}  {r['status']}")

    rr = np.array([_f(r, "r_ox_A") for r in good])
    om = np.array([_f(r, "omega_app_eV") for r in good])
    t1 = np.array([_f(r, "t1_s") for r in good])

    # 1. where EOM stops being trustworthy
    bad_t1 = rr[t1 >= T1_MAX]
    print(f"\n  T1(S) first reaches {T1_MAX} at "
          + (f"{bad_t1.min():.3f} A" if bad_t1.size else
             f"no radius in 1.4-{rr.max():.1f} A (max {np.nanmax(t1):.4f})"))
    low_w1 = rr[np.array([_f(r, "w1_app") for r in good]) < 0.90]
    print(f"  single-excitation weight drops below 0.90 at "
          + (f"{low_w1.min():.3f} A" if low_w1.size else "no radius"))

    # 2. the crossing
    ap = [r for r in good if r.get("root0_sym", "") and
          '"' not in r.get("root0_sym", "")]
    print(f"  lowest triplet is 3A' from "
          + (f"{min(_f(r, 'r_ox_A') for r in ap):.3f} A outward "
             f"({len(ap)} points) -- the crossing" if ap else
             "nowhere: 3A\" is lowest at every radius"))

    # 3. the band, to first order
    i_eq = int(np.argmin(np.abs(rr - spec["r_eq"])))
    vert = om[i_eq]
    slope = float(np.gradient(om, rr)[i_eq])
    print(f"\n  vertical at r = {rr[i_eq]:.3f} A: {vert:.3f} eV "
          f"= {NM_PER_EV / vert:.0f} nm, against {spec['obs_nm']:.0f} nm "
          f"measured ({100 * (NM_PER_EV / vert - spec['obs_nm']) / spec['obs_nm']:+.1f}%)")
    print(f"  slope there: {slope:.3f} eV/A  "
          f"(HOCl's EOM-CCSD reference was -6.965)")
    is_app = np.array([('"' in r.get("root0_sym", "")) for r in good])
    n_roots_below = np.array([
        sum(1 for k in range(1, nroots + 1)
            if r.get(f"sym{k}") and '"' not in r.get(f"sym{k}", "")
            and _f(r, f"omega{k}_eV") < _f(r, "omega_app_eV"))
        for r in good])
    return dict(r=rr, omega=om, t1=t1, vert=vert, slope=slope,
                is_app=is_app, n_below=n_roots_below)


def report_bases(res, bases):
    """The aug- question: an offset is uninteresting, a TREND is not."""
    if len(bases) < 2 or any(b not in res or res[b] is None for b in bases[:2]):
        return
    a, b = bases[0], bases[1]
    ra, rb = res[a]["r"], res[b]["r"]
    common = np.intersect1d(np.round(ra, 5), np.round(rb, 5))
    if common.size < 4:
        print("\n  too few shared radii to compare the bases")
        return
    ia = [int(np.where(np.round(ra, 5) == r)[0][0]) for r in common]
    ib = [int(np.where(np.round(rb, 5) == r)[0][0]) for r in common]
    d = res[b]["omega"][ib] - res[a]["omega"][ia]

    print(f"\n{'=' * 72}\n== {b} minus {a}\n{'=' * 72}")
    print(f"  {'r/A':>7}{'d(omega)/meV':>14}")
    for r, dv in zip(common, d):
        print(f"  {r:>7.3f}{dv * 1000:>14.1f}")
    slope, intercept = np.polyfit(common, d * 1000, 1)
    print(f"\n  mean offset {np.mean(d) * 1000:+.1f} meV, "
          f"spread {np.ptp(d) * 1000:.1f} meV")
    print(f"  trend with r: {slope:+.1f} meV per A")
    print("""
  Reading this. A constant offset is the two bases describing the SAME state
  at slightly different completeness, and it cancels out of a band shape. A
  difference that GROWS with r is the interesting case: that is the BSSE
  signature, since basis-set superposition is an attractive error that peaks
  where the fragments are close but separating, and it would distort the
  slope the band width depends on.""")
    if abs(slope) < 20:
        print("  -> flat within 20 meV/A: aug- is not buying anything here,\n"
              "     and the cheaper set carries the raster.")
    else:
        print(f"  -> {abs(slope):.0f} meV/A is a real trend. Check the root\n"
              "     symmetries below before concluding it is BSSE rather than\n"
              "     a diffuse root being picked up by the labelling.")

    # The risk side of aug-: extra low-lying roots to sort through. If a
    # basis puts MORE roots below the valence 3A", the labelling has more
    # chances to pick the wrong state, and the raster stores what it labels.
    print("\n  has aug- put more roots below the valence 3A\"?")
    print(f"    {'basis':>18}{'radii':>7}{'3A\" lowest':>12}"
          f"{'max roots below':>17}")
    for nm in bases[:2]:
        d = res[nm]
        print(f"    {nm:>18}{d['r'].size:>7}"
              f"{int(np.sum(d['is_app'])):>12}"
              f"{int(np.max(d['n_below'])) if d['n_below'].size else 0:>17}")
    extra = int(np.max(res[b]["n_below"])) - int(np.max(res[a]["n_below"]))
    if extra > 0:
        print(f"    -> {b} places up to {extra} more root(s) below the 3A\".\n"
              f"       That is the failure mode to weigh against any accuracy\n"
              f"       gain: a raster stores whatever its labelling picks.")
    else:
        print(f"    -> no extra low-lying roots from {b}; the diffuse-root\n"
              f"       risk did not materialise here.")


def maybe_plot(res, bases, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  (matplotlib not available, skipping the plot)")
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for nm in bases:
        if res.get(nm) is None:
            continue
        ax[0].plot(res[nm]["r"], res[nm]["omega"], "o-", ms=3, label=nm)
    ax[0].set_xlabel("r(O-X) / A")
    ax[0].set_ylabel("omega(3A\") / eV")
    ax[0].set_title("EOM-CCSD triplet along the bond")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    for nm in bases:
        if res.get(nm) is None:
            continue
        ax[1].plot(res[nm]["r"], res[nm]["t1"], "o-", ms=3, label=nm)
    ax[1].axhline(T1_MAX, color="r", ls="--", lw=1,
                  label=f"T1 = {T1_MAX} (EOM validity)")
    ax[1].set_xlabel("r(O-X) / A")
    ax[1].set_ylabel("T1(S)")
    ax[1].set_title("single-reference quality of the ground state")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\n  wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--molecule", default="HOBr", choices=sorted(MOLECULES))
    p.add_argument("--bases", nargs="+",
                   default=["cc-pvtz-dk", "aug-cc-pvtz-dk"])
    p.add_argument("--geom", nargs=2, type=float, default=None,
                   metavar=("R_OH", "THETA"),
                   help="spectator coordinates; default is the molecule's "
                        "table value. Update from 01_geometry.py.")
    p.add_argument("--rmin", type=float, default=1.50)
    p.add_argument("--rfine", type=float, default=2.60)
    p.add_argument("--rmax", type=float, default=3.20)
    p.add_argument("--dfine", type=float, default=0.05)
    p.add_argument("--dcoarse", type=float, default=0.10)
    p.add_argument("--nroots", type=int, default=NROOTS)
    p.add_argument("--csv", default=None)
    p.add_argument("--png", default=None)
    p.add_argument("--nproc", type=int, default=os.cpu_count() or 1)
    p.add_argument("--memory", type=int, default=4000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    spec = MOLECULES[args.molecule]
    r_oh, theta = args.geom if args.geom else (spec["r_oh"], spec["theta"])
    csv_path = args.csv or str(DATA / f"{args.molecule.lower()}_obr_cut.csv")
    png_path = args.png or str(DATA / f"{args.molecule.lower()}_obr_cut.png")
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    radii = build_radii(spec["r_eq"], args)
    grid = [key(b, r) for b in args.bases for r in radii]

    print("=" * 72)
    print(f"== {args.molecule} a 3A\" along O-{spec['halogen']}, "
          f"{len(args.bases)} bases")
    print(f"== CCSD(T) + EOM-EE-CCSD triplet, {args.nroots} roots, x2c, Cs")
    print(f"== spectators fixed at r(O-H) = {r_oh:.4f} A, "
          f"angle = {theta:.2f} deg")
    print("=" * 72)
    print(f"  {len(radii)} radii x {len(args.bases)} bases = {len(grid)} points")
    if args.dry_run:
        print(f"  radii: {', '.join(f'{r:.2f}' for r in radii)}")
        return

    done = load_done(csv_path, args.nroots)
    todo = [g for g in grid if g not in done]
    print(f"  {len(todo)} to compute, {len(grid) - len(todo)} already present")

    if todo and not args.report_only:
        tasks = [(spec["halogen"], r, r_oh, theta, b, args.nroots,
                  args.memory) for b, r in todo]
        exists = os.path.exists(csv_path)
        t0 = time.time()
        ctx = mp.get_context("fork")
        counter = ctx.Value("i", 0)
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields(args.nroots),
                               extrasaction="ignore")
            if not exists:
                w.writeheader()
            with ctx.Pool(min(args.nproc, len(tasks)),
                          initializer=_init_worker,
                          initargs=(counter,)) as pool:
                for i, row in enumerate(
                        pool.imap_unordered(compute_point, tasks), 1):
                    w.writerow(row)
                    fh.flush()
                    print(f"    [{i}/{len(tasks)}] {row['basis']} "
                          f"{row['r_ox_A']}: {row['status']} "
                          f"({row['wall_s']} s)")
        print(f"  done in {(time.time() - t0) / 60:.1f} min")

    done = load_done(csv_path, args.nroots)
    res = {}
    for b in args.bases:
        rows = [done[key(b, r)] for r in radii if key(b, r) in done]
        rows.sort(key=lambda r: _f(r, "r_ox_A"))
        res[b] = report_one(b, rows, spec, args.nroots) if rows else None
    report_bases(res, args.bases)
    maybe_plot(res, args.bases, png_path)
    print(f"\n  csv: {csv_path}")


if __name__ == "__main__":
    main()
