#!/usr/bin/env bash
# Run the whole chain for one isotopologue, into its own directory.
#
# The ELECTRONIC surfaces do not depend on nuclear mass, so nothing at that
# level is recomputed: 13_gs_well.py resumes from the shared gs_well.csv and
# 07_jacobi.py re-reads the shared pes_surface_v2.npz. Only the Jacobi
# transform, the relaxation and the propagation repeat.
#
# Shared inputs stay in the project root:
#     gs_well.csv          CCSD(T) ground-state raster (isotope independent)
#     pes_surface_v2.npz   spliced X/A surfaces + mu (isotope independent)
#     oh_curve.csv         OH diatomic curve for the asymptote
# Per-isotopologue outputs go to runs/<LABEL>/.
#
# HDO has two INEQUIVALENT channels. The Jacobi set names one atom departing
# (coordinate R) and one spectator (coordinate r). Either is a valid
# coordinate system and the TOTAL cross section must come out the same; the
# other channel simply appears as r -> large. Put the dominant channel in R
# so it gets the better grid -- for HDO that is O-H cleavage.
#
# Usage:
#   bash run_isotopologue.sh H2O  1.00782503207 1.00782503207
#   bash run_isotopologue.sh HDO  1.00782503207 2.01410177812  # H leaves, OD spectator
#   bash run_isotopologue.sh HDOd 2.01410177812 1.00782503207  # D leaves, OH spectator
#   bash run_isotopologue.sh D2O  2.01410177812 2.01410177812
#
# args: LABEL  M_DEPARTING  M_SPECTATOR   (amu)
set -euo pipefail

LAB=${1:?label}
MDEP=${2:?departing mass}
MSPEC=${3:?spectator mass}
NPROC=${NPROC:-30}
OUT="runs/$LAB"
mkdir -p "$OUT"

for f in gs_well.csv pes_surface_v2.npz oh_curve.csv; do
    [[ -f $f ]] || { echo "missing shared input: $f" >&2; exit 1; }
done

echo "=== $LAB : departing $MDEP amu, spectator $MSPEC amu -> $OUT ==="

python 13_gs_well.py --basis aug-cc-pVTZ --nproc "$NPROC" \
    --m-departing "$MDEP" --m-spectator "$MSPEC" \
    --csv gs_well.csv --npz "$OUT/jacobi_bound.npz" \
    2>&1 | tee "$OUT/gs_well.log"

python 08_relax.py --jacobi "$OUT/jacobi_bound.npz" --nstates 5 --tol 1e-8 \
    --out "$OUT/vib_states.npz" --png "$OUT/vib_states.png" \
    2>&1 | tee "$OUT/relax.log"

python 07_jacobi.py --surface pes_surface_v2.npz --oh oh_curve.csv \
    --m-departing "$MDEP" --m-spectator "$MSPEC" \
    --out "$OUT/jacobi_surface.npz" --png "$OUT/jacobi_surface.png" \
    2>&1 | tee "$OUT/jacobi.log"

python 09_propagate.py --jacobi "$OUT/jacobi_surface.npz" \
    --states "$OUT/vib_states.npz" \
    --dt 2.0 --nsteps 1000 --R-abs 8.0 --eta 0.15 --r-abs 4.0 \
    --temperatures 200 250 298 \
    --out "$OUT/cross_section.npz" --png "$OUT/cross_section.png" \
    2>&1 | tee "$OUT/propagate.log"

cat > "$OUT/masses.txt" <<EOF
label=$LAB
m_departing_amu=$MDEP
m_spectator_amu=$MSPEC
EOF

echo
echo "=== $LAB done -> $OUT ==="
