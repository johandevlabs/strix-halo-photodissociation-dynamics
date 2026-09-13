#!/usr/bin/env python3
"""
Tier 0: machine calibration (v2).

Fixes a real bug in v1: numpy does NOT thread element-wise operations, so the
v1 STREAM and FFT tests silently measured ONE core. On a 16-core Strix Halo
part that under-reported memory bandwidth by roughly an order of magnitude
(20 GB/s observed against a ~256 GB/s bus). v1 also allocated a temporary
array inside the timed region.

Three things measured here, each bounding a different later workload:

  DGEMM       -> CCSD and (T), which are almost entirely dgemm
  bandwidth   -> the split-operator wavepacket propagator
  3D FFT      -> the propagator directly, at a realistic grid size

Usage:
    OMP_NUM_THREADS=16 OMP_PLACES=cores OMP_PROC_BIND=close \
        python3 01_calibrate.py --threads 16 --csv calibration.csv
"""
import argparse
import csv
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import numexpr
    HAVE_NUMEXPR = True
except ImportError:
    HAVE_NUMEXPR = False

try:
    import scipy.fft as sfft
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def dgemm_gflops(n: int, reps: int = 3) -> float:
    """Sustained FP64 DGEMM. Threaded by the BLAS itself, so this one was fine."""
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    a @ b
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    return (2.0 * n**3) / best / 1e9


def triad_numexpr(mb: int, threads: int, reps: int = 5) -> float:
    """
    Classic STREAM triad a = b + s*c, fused and multithreaded.
    numexpr evaluates in one pass, so traffic is a genuine 24 bytes/element
    (read b, read c, write a).
    """
    numexpr.set_num_threads(threads)
    n = (mb * 1024 * 1024) // 8 // 3
    a = np.zeros(n)
    b = np.random.rand(n)
    c = np.random.rand(n)
    s = 3.0
    numexpr.evaluate("b + s*c", out=a)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        numexpr.evaluate("b + s*c", out=a)
        best = min(best, time.perf_counter() - t0)
    return (24.0 * n) / best / 1e9


def copy_threaded(mb: int, threads: int, reps: int = 5) -> float:
    """
    Dependency-free cross-check: threaded memcpy over chunks.
    numpy releases the GIL inside ufuncs, so Python threads genuinely
    parallelise here. Traffic is 16 bytes/element (one read, one write).
    """
    n = (mb * 1024 * 1024) // 8 // 2
    src = np.random.rand(n)
    dst = np.empty(n)
    bounds = np.linspace(0, n, threads + 1).astype(int)
    chunks = list(zip(bounds[:-1], bounds[1:]))

    def work(rng):
        i, j = rng
        np.copyto(dst[i:j], src[i:j])

    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(work, chunks))
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            list(ex.map(work, chunks))
            best = min(best, time.perf_counter() - t0)
    return (16.0 * n) / best / 1e9


def fft3d_gflops(n: int, threads: int, reps: int = 3) -> float:
    """
    Complex 3D FFT on an n^3 grid: the actual shape of a triatomic
    wavepacket propagation step. scipy.fft threads via workers=;
    np.fft never does, which is what broke v1.
    """
    x = (np.random.rand(n, n, n) + 1j * np.random.rand(n, n, n))
    sfft.fftn(x, workers=threads)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        sfft.fftn(x, workers=threads)
        best = min(best, time.perf_counter() - t0)
    npts = n**3
    return (5.0 * npts * np.log2(npts)) / best / 1e9


def interpret_dgemm(gflops: float, cores: int) -> None:
    """
    Back out the implied all-core clock under each possible FP64 datapath
    width. If the 16 FLOP/cycle figure comes out above ~5.5 GHz it is
    physically impossible, which means the part has a full 512-bit datapath.
    """
    print("\n  implied sustained all-core clock:")
    for flops_per_cycle, label in [(32, "512-bit datapath (2x512 FMA)"),
                                   (16, "256-bit, AVX-512 double-pumped")]:
        ghz = gflops / (cores * flops_per_cycle)
        flag = "  <- physically impossible" if ghz > 5.5 else ""
        print(f"    {flops_per_cycle:2d} FLOP/cycle/core  "
              f"{ghz:5.2f} GHz   {label}{flag}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--cores", type=int, default=16,
                   help="physical cores, for the clock back-calculation")
    p.add_argument("--sizes", type=int, nargs="+", default=[4000, 8000])
    p.add_argument("--mb", type=int, default=4096,
                   help="working set per bandwidth test, MB (must exceed L3)")
    p.add_argument("--fft-n", type=int, default=256,
                   help="3D FFT grid dimension (n^3 complex128)")
    p.add_argument("--csv", default="calibration.csv")
    args = p.parse_args()

    omp = int(os.environ.get("OMP_NUM_THREADS", 0) or 0)
    if omp and omp != args.threads:
        raise SystemExit(
            f"refusing to run: --threads={args.threads} but "
            f"OMP_NUM_THREADS={omp}. BLAS would use {omp} threads while the "
            f"bandwidth and FFT tests use {args.threads}, mixing two thread "
            f"counts into one CSV row. Set both to the same value.")

    print(f"host        : {platform.node()} / {platform.machine()}")
    print(f"numpy       : {np.__version__}")
    print(f"threads     : {args.threads}")
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}  "
          f"OPENBLAS_CORETYPE={os.environ.get('OPENBLAS_CORETYPE', 'auto')}")
    print()

    rows = []

    best_dgemm = 0.0
    for n in args.sizes:
        g = dgemm_gflops(n)
        print(f"DGEMM n={n:<6d}        {g:8.1f} GFLOP/s")
        best_dgemm = max(best_dgemm, g)
        rows.append({"threads": args.threads, "test": f"dgemm_{n}",
                     "value": round(g, 2), "unit": "GFLOP/s"})
    if args.threads >= args.cores:
        interpret_dgemm(best_dgemm, args.cores)
    print()

    if HAVE_NUMEXPR:
        bw = triad_numexpr(args.mb, args.threads)
        print(f"STREAM triad (fused)  {bw:8.1f} GB/s")
        rows.append({"threads": args.threads, "test": "stream_triad",
                     "value": round(bw, 2), "unit": "GB/s"})
    else:
        print("STREAM triad          numexpr not installed, skipped")

    bwc = copy_threaded(args.mb, args.threads)
    print(f"threaded copy         {bwc:8.1f} GB/s")
    rows.append({"threads": args.threads, "test": "copy_threaded",
                 "value": round(bwc, 2), "unit": "GB/s"})

    if HAVE_SCIPY:
        ff = fft3d_gflops(args.fft_n, args.threads)
        print(f"FFT {args.fft_n}^3 c2c        {ff:8.1f} GFLOP/s")
        rows.append({"threads": args.threads,
                     "test": f"fft3d_{args.fft_n}",
                     "value": round(ff, 2), "unit": "GFLOP/s"})
    else:
        print("FFT 3D                scipy not installed, skipped")

    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["threads", "test", "value", "unit"])
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\nappended {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
