#!/usr/bin/env python3
"""
HOI step 4, the gate: is the visible band still perturbative in SOC?

HOI's visible band carries f ~1e-3 -- ~20x HOBr's, where scaling HOBr's
borrowed intensity by the spin-orbit constant squared gives ~4x -- and it
sits bluer than HOBr's, not redder (02_band_model/10_obs_bands.py). For HOCl
and HOBr the band is a triplet borrowing a little from the singlet above it:
one spin-free surface plus a borrowed dipole. This run asks whether that
still holds for iodine.

The machinery is HOBr's: HOBr/01_method/04_soc_vertical.py (SA-CASSCF in Ms=0,
spin-free and DKH1 QD-NEVPT2 on the same reference). This file only
registers HOI and runs it. The report ends in a two-state estimate of the
singlet admixture c^2 (triplet SOC shift / gap to the next spin-free state;
HOBr 2.0%) and a verdict:
    PERTURBATIVE   c^2 < 5%     HOBr's pipeline as is
    BORDERLINE     5-20%        same pipeline, SOC in the surface
    STRONGLY MIXED > 20%        coupled spin-orbit states in the dynamics

Basis cc-pvtz-dk, HOBr's. It DOES have iodine (an earlier note here said
otherwise -- wrong; the EVO resolves it, toolchain/logs/soc_atoms_i_sweep.log:
2P splitting -7.7%, against -8.4% for x2c-tzvpall, which the first gate run
used). The HOBr bridge run showed the basis change moves nothing that
matters: 429.0 nm / f 1.63e-5 in x2c-tzvpall against 426.9 / 1.54e-5.
def2-tzvp has an ECP for I and is rejected (X2CAMF needs the core).

Geometry: 01_geometry.py's CCSD(T)/x2c-tzvpall minimum (the recalled start
was within 0.001 A and 0.7 deg).

Usage (from this directory):
    python 04_soc_vertical.py --nroots 10 2>&1 | tee ../logs/soc_vertical_dk_n10.log
    python 04_soc_vertical.py --basis x2c-tzvpall --nroots 10   # the first run
"""
import importlib.util
import sys
from pathlib import Path

HOBR04 = (Path(__file__).resolve().parents[2] / "HOBr" / "01_method"
          / "04_soc_vertical.py")

# Measured: 02_band_model/10_obs_bands.py (Rowley 1999 low, Bauer 1998 high).
# Geometry: logs/geometry.log, CCSD(T)/x2c-tzvpall.
HOI = dict(halogen="I", avas=["I 5p", "O 2p"], basis="cc-pvtz-dk",
           geom=(1.9907, 0.9694, 104.65),
           obs_nm=407.0, f_obs_vis=(9.6e-4, 1.17e-3),
           uv_nm=340.0, f_obs_uv=1.85e-3)


def load():
    spec = importlib.util.spec_from_file_location("hobr04", HOBR04)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hobr04"] = mod
    spec.loader.exec_module(mod)
    mod.MOLECULES["HOI"] = HOI
    return mod


if __name__ == "__main__":
    load().main(default="HOI", doc=__doc__)
