#!/usr/bin/env bash
# Phase 0.5, step 1: install the SOC toolchain into the `qc` env.
#
# Two packages, both PySCF plugins, both source installs:
#
#   socutils  (xubwa/socutils)      X2CAMF spin-orbit integrals, spinor SCF.
#                                   Bundles x2camf and zquatev; needs a `make`
#                                   at the repo root against BLAS/LAPACK.
#   prism     (sokolov-group/prism) NEVPT2 / QD-NEVPT2 with state-interaction
#                                   SOC. Imports socutils for the integrals.
#
# Nothing here touches the water/ pipeline: both are additive imports. But it
# does build C/C++ against whatever BLAS the env has, so run it INSIDE the
# activated qc env, not outside it.
#
# Usage:
#   conda activate qc
#   bash 00_setup_soc.sh                 # clones into ~/src
#   SRC=/path/to/src bash 00_setup_soc.sh
set -euo pipefail

SRC=${SRC:-$HOME/src}
mkdir -p "$SRC"

# Refuse to run outside a conda env -- a stray pip install -e into the base
# interpreter is annoying to unpick, and the whole point is that this lands
# next to the pyscf that water/ already uses.
if [[ -z ${CONDA_PREFIX:-} ]]; then
    echo "no CONDA_PREFIX -- activate the qc env first: conda activate qc" >&2
    exit 1
fi
echo "installing into: $CONDA_PREFIX"
echo "python:          $(command -v python)"
python -c "import pyscf; print('pyscf:          ', pyscf.__version__)"
echo

clone_or_pull () {
    local url=$1 dir=$2
    if [[ -d $dir/.git ]]; then
        echo "--- updating $dir"
        git -C "$dir" pull --ff-only
    else
        echo "--- cloning $url"
        git clone "$url" "$dir"
    fi
}

# ---------------------------------------------------------------- socutils
clone_or_pull https://github.com/xubwa/socutils.git "$SRC/socutils"
echo "--- building socutils (bundled x2camf + zquatev)"
make -C "$SRC/socutils" -j"$(nproc)"
python -m pip install -e "$SRC/socutils"

# ------------------------------------------------------------------- prism
clone_or_pull https://github.com/sokolov-group/prism.git "$SRC/prism"
python -m pip install -e "$SRC/prism"

# Prism lists these as optional but wants them for SOC and for any tensor
# contraction of a size worth caring about.
python -m pip install "sympy>=1.12" opt_einsum

echo
echo "=== smoke test ==="
python - <<'PY'
import importlib
for name in ("pyscf", "socutils", "x2camf", "zquatev", "prism"):
    try:
        m = importlib.import_module(name)
        print(f"  ok      {name:10s} {getattr(m, '__version__', '(no __version__)')}"
              f"  {getattr(m, '__file__', '')}")
    except Exception as exc:
        print(f"  FAILED  {name:10s} {type(exc).__name__}: {exc}")
PY

echo
echo "=== done. next: python 01_soc_probe.py 2>&1 | tee logs/soc_probe.log ==="
