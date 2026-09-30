#!/usr/bin/env python3
"""
HOI: a 2D (r(O-I), angle) raster -- CCSD(T) ground state and EOM-CCSD
triplets -- for the 2D band model of the a 3A" band.

WHY 2D, AND WHY ONLY THIS (TASKS.md, 2026-10-01): the 1D model along O-I
gives sigma(532) = 3.6e-20, insensitive to everything beyond the scan. The
bend is the dimension a 1D model leaves out that matters most for the band's
WIDTH (for HOCl the triplet moved 0.26 eV across the angles the ground
state's zero-point motion samples), so a 2D model checks whether the 1D
shape and sigma(532) hold. r(O-H) stays frozen: its hot band holds nothing
at 295 K and HOBr's 3D run showed it adds little.

Each point is HOBr/01_method/02_obr_cut.py's calculation (imported, not
copied): x2c CCSD(T), EOM-EE-CCSD triplet roots each labelled A'/A" by its
dominant excitation, the lowest 3A" stored as omega_app. cc-pvtz-dk,
all-electron, as the 1D cut.

Grid: r(O-I) 1.70-2.80 A in 0.05 A (23) x angle 75-135 deg in 5 deg (13) =
299 points, r(O-H) 0.9694 A (01). The SOC offsets and dipoles come from a
coarser grid (01_method/07_soc_scan.py --grid2d); the fine shape comes from
here. 8 workers x 4 cores x 10 GB (the first 1D cut's lesson), ~6 min a
point: ~3.7 h.

Resumable: HOI/data/hoi_raster_2d.csv, keyed (r, angle).
    python 09_raster_2d.py 2>&1 | tee ../logs/raster_2d.log
    python 09_raster_2d.py --dry-run
"""
import argparse
import csv
import importlib.util
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HOX = Path(__file__).resolve().parents[2]
CSV = HOX / "HOI" / "data" / "hoi_raster_2d.csv"
HOBR02 = HOX / "HOBr" / "01_method" / "02_obr_cut.py"
R_OH = 0.9694                           # 01_geometry.py


def load_02():
    spec = importlib.util.spec_from_file_location("hobr02", HOBR02)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hobr02"] = mod          # the worker pool pickles by name
    spec.loader.exec_module(mod)
    return mod


m02 = load_02()                          # sets the one-thread env block first
import numpy as np  # noqa: E402


def grid(args):
    rs = np.round(np.arange(args.rmin, args.rmax + 1e-9, args.dr), 4)
    ts = np.round(np.arange(args.tmin, args.tmax + 1e-9, args.dt), 3)
    return [(float(r), float(t)) for t in ts for r in rs]


def key(r, t):
    return (round(float(r), 4), round(float(t), 3))


def point(task):
    """02's compute_point plus the angle, which 02's row does not carry."""
    row = m02.compute_point(task)
    row["theta_deg"] = task[3]
    return row


def load_done(path):
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return {}
    return {key(r["r_ox_A"], r["theta_deg"]): r for r in csv.DictReader(open(path))
            if not r["status"].startswith("fail")}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="cc-pvtz-dk")
    p.add_argument("--rmin", type=float, default=1.70)
    p.add_argument("--rmax", type=float, default=2.80)
    p.add_argument("--dr", type=float, default=0.05)
    p.add_argument("--tmin", type=float, default=75.0)
    p.add_argument("--tmax", type=float, default=135.0)
    p.add_argument("--dt", type=float, default=5.0)
    p.add_argument("--nroots", type=int, default=6)
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory", type=int, default=10000, help="MB per worker")
    p.add_argument("--csv", default=str(CSV))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    pts = grid(args)
    done = load_done(args.csv)
    todo = [g for g in pts if key(*g) not in done]
    print("=" * 76)
    print(f"== HOI 2D raster: CCSD(T) + EOM-CCSD triplets, {args.basis}, "
          f"r(O-H) {R_OH} A")
    print(f"== {len(pts)} points ({args.rmin}-{args.rmax} A x {args.tmin}-"
          f"{args.tmax} deg), {len(todo)} to compute; {args.nproc} workers x "
          f"{args.threads} threads x {args.memory} MB")
    print("=" * 76)
    if args.dry_run or not todo:
        return

    fields = m02.fields(args.nroots) + ["theta_deg"]
    new = not (os.path.exists(args.csv) and os.path.getsize(args.csv) > 0)
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    tasks = [("I", r, R_OH, t, args.basis, args.nroots, args.memory)
             for r, t in todo]
    ctx = mp.get_context("fork")
    counter = ctx.Value("i", 0)
    t0 = time.time()
    with open(args.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
            fh.flush()
        with ctx.Pool(min(args.nproc, len(tasks)), initializer=m02._init_worker,
                      initargs=(counter, args.threads)) as pool:
            for i, row in enumerate(pool.imap_unordered(point, tasks), 1):
                w.writerow(row)
                fh.flush()
                print(f"    [{i}/{len(tasks)}] r {row['r_ox_A']} angle "
                      f"{row['theta_deg']}: {row['status']} ({row['wall_s']} s)",
                      flush=True)
    print(f"  done in {(time.time() - t0) / 60:.1f} min; csv {args.csv}")


if __name__ == "__main__":
    main()
