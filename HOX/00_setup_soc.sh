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
mkdir -p "$(dirname "$0")/logs"

echo "installing into: $CONDA_PREFIX"
echo "python:          $(command -v python)"
python -c "import pyscf; print('pyscf:          ', pyscf.__version__)"
echo

# --- preflight: is the conda env actually the one being imported from? ------
# A pip --user numpy shadows the conda one AND brings its own bundled
# OpenBLAS, which is the "two OpenBLAS copies" trap in water/README.md
# wearing a different hat. Warn loudly; do not silently "fix" the user's env.
python - <<'PY'
import os, sys, site
prefix = os.environ.get("CONDA_PREFIX", "")
usersite = site.getusersitepackages() if hasattr(site, "getusersitepackages") else ""
bad = []
for name in ("numpy", "scipy"):
    try:
        m = __import__(name)
    except ImportError:
        continue
    path = getattr(m, "__file__", "") or ""
    inside = path.startswith(prefix)
    print(f"  {name:6s} {m.__version__:10s} {'env' if inside else 'USER-SITE'}  {path}")
    if not inside:
        bad.append(name)
if bad:
    print()
    print("  " + "!" * 66)
    print(f"  !! {', '.join(bad)} loading from user-site, not {prefix}")
    print("  !! pip wheels bundle their own OpenBLAS -- two copies, two thread")
    print("  !! pools, and the Prescott-kernel failure mode in water/README.md.")
    print("  !! Fix before trusting any timing from this toolchain:")
    print("  !!     export PYTHONNOUSERSITE=1        # per-shell, reversible")
    print("  !!     pip uninstall --user " + " ".join(bad) + "   # permanent")
    print("  " + "!" * 66)
PY
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

# socutils' CMakeLists does a bare find_package(BLAS REQUIRED), which searches
# system paths and does NOT look inside the conda env -- so it fails with
# "Could NOT find BLAS" even though libopenblas is sitting right there. The
# Makefile forwards $CMAKE_ARGS, and the CMakeLists honours a user-supplied
# BLAS_LIBRARIES (and then assumes LAPACK is in the same library, which is
# true for OpenBLAS). So point it straight at the conda one.
find_lib () {            # find_lib <dir> <name...>  -> first match on stdout
    local dir=$1; shift
    local name cand
    for name in "$@"; do
        for cand in "$dir/$name" "$dir/$name".*; do
            [[ -e $cand ]] && { echo "$cand"; return 0; }
        done
    done
    return 1
}

echo "--- locating BLAS in the conda env"
BLAS_LIB=$(find_lib "$CONDA_PREFIX/lib" libopenblas.so libblas.so) || {
    echo "no BLAS found in $CONDA_PREFIX/lib -- install one:" >&2
    echo "  conda install -c conda-forge 'libopenblas=*=*openmp*'" >&2
    echo "(the openmp build, not pthreads -- see water/README.md Environment)" >&2
    exit 1
}
echo "    $BLAS_LIB"

# socutils assumes a PIP-layout pyscf, where the bundled dependencies sit in
# pyscf/lib/deps/lib, and looks for libcint there with NO_DEFAULT_PATH.
# Conda-forge pyscf instead links a SHARED libcint from $CONDA_PREFIX/lib, so
# that search finds nothing. The other three libs it wants (cgto, np_helper,
# ao2mo) are pyscf's own and do live in pyscf/lib, which is why only cint
# failed. find_library is a no-op when its cache variable is already set, so
# handing it the path directly skips the broken search.
echo "--- locating libcint"
PYSCF_LIB_DIR=$(python -c \
    "import os, pyscf; print(os.path.join(os.path.dirname(pyscf.__file__), 'lib'))")
CINT_LIB=$(find_lib "$CONDA_PREFIX/lib" libcint.so) \
    || CINT_LIB=$(find_lib "$PYSCF_LIB_DIR/deps/lib" libcint.so) \
    || {
    echo "no libcint found in $CONDA_PREFIX/lib or $PYSCF_LIB_DIR/deps/lib" >&2
    echo "  conda install -c conda-forge libcint" >&2
    exit 1
}
echo "    $CINT_LIB"
echo "    pyscf libs: $PYSCF_LIB_DIR"

# ...and the same split again for the HEADER. ao2mo/CMakeLists.txt hardcodes
#     target_include_directories(nrr_opt PRIVATE
#         ${PYSCF_LIB_DIR} ${PYSCF_LIB_DIR}/deps/include)
# with no override variable, so cint.h cannot be pointed at the way the
# library could. Inject it through CMAKE_C_FLAGS, which is additive to the
# per-target include dirs rather than replacing them.
echo "--- locating cint.h"
CINT_INC=""
for d in "$CONDA_PREFIX/include" "$PYSCF_LIB_DIR/deps/include"; do
    if [[ -e $d/cint.h ]]; then CINT_INC=$d; break; fi
done
if [[ -z $CINT_INC ]]; then
    echo "cint.h not found in $CONDA_PREFIX/include or $PYSCF_LIB_DIR/deps/include" >&2
    echo "  conda install -c conda-forge libcint" >&2
    echo "(conda-forge ships the headers in the same package as the library)" >&2
    exit 1
fi
echo "    $CINT_INC/cint.h"

echo "--- building socutils (bundled x2camf + zquatev)"
# A previously failed configure leaves a CMakeCache.txt that remembers the
# failure and ignores the new args. Clear it so a retry is a real retry.
rm -rf "$SRC/socutils/lib/build"
make -C "$SRC/socutils" -j"$(nproc)" \
    CMAKE_ARGS="-DBLAS_LIBRARIES=$BLAS_LIB \
                -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
                -DPYSCF_LIB_DIR=$PYSCF_LIB_DIR \
                -DPYSCF_CINT_LIB=$CINT_LIB \
                -DCMAKE_C_FLAGS=-I$CINT_INC"
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
