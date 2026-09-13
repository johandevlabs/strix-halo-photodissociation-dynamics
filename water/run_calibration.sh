#!/usr/bin/env bash
# Tier 0 sweep: 1 -> 16 cores. BLAS thread counts are read at library load
# time, so they must be exported before python starts.
#
# Note: 16, not 32. SMT hurts DGEMM because both sibling threads contend for
# the same FP pipes. Pin to physical cores.
set -euo pipefail

CSV=${1:-calibration.csv}

for T in 1 2 4 8 12 16; do
    echo "--------------------------------------------------- threads=$T"
    OMP_NUM_THREADS=$T \
    OPENBLAS_NUM_THREADS=$T \
    MKL_NUM_THREADS=$T \
    BLIS_NUM_THREADS=$T \
    NUMEXPR_NUM_THREADS=$T \
    OMP_PLACES=cores \
    OMP_PROC_BIND=close \
        python 01_calibrate.py --threads "$T" --csv "$CSV"
done
