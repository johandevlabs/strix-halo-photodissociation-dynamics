#!/usr/bin/env python3
"""
HOI step 2: the ground state and the spin-free a 3A" along O-I.

HOBr/01_method/02_obr_cut.py does the work -- CCSD(T) ground state and
EOM-EE-CCSD triplet roots at each r(O-I), every root labelled A'/A" by its
dominant excitation, spectators fixed -- and this registers HOI and runs it.
Output HOI/data/hoi_oi_cut.csv.

What the band model (02_band_model/11_band_soc_1d.py) takes from it: the
CCSD(T) ground-state curve, for the vibrational levels and as the base the
spin-orbit excitation energies of 07_soc_scan.py sit on; and the EOM a 3A"
curve as a check on 07's spin-free triplet along the cut (at r_eq the two
agree: 2.415 against 2.42 eV).

One basis, cc-pvtz-dk: for HOBr, aug- moved the band 0.9 nm. The radii
reach further out than HOBr's (r_eq is 0.16 A longer): 1.60-2.80 A in
0.05 A steps, then to 3.40 A in 0.10.

Usage (from this directory):
    python 02_oi_cut.py 2>&1 | tee ../logs/oi_cut.log
"""
import importlib.util
import sys
from pathlib import Path

HOX = Path(__file__).resolve().parents[2]
HOBR02 = HOX / "HOBr" / "01_method" / "02_obr_cut.py"

# 01_geometry.py's CCSD(T) minimum (logs/geometry.log)
HOI = dict(halogen="I", r_eq=1.9907, r_oh=0.9694, theta=104.65, obs_nm=None,
           data=HOX / "HOI" / "data", stem="hoi_oi_cut")


def load():
    spec = importlib.util.spec_from_file_location("hobr02", HOBR02)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hobr02"] = mod          # the worker pool pickles by name
    spec.loader.exec_module(mod)
    mod.MOLECULES["HOI"] = HOI
    return mod


if __name__ == "__main__":
    # 16 workers, not one per core: HOI points peak above 2.2 GB resident
    # each, and 31 of them pushed the EVO (93 GB visible) into swap.
    load().main(default="HOI", doc=__doc__, bases=["cc-pvtz-dk"],
                rmin=1.60, rfine=2.80, rmax=3.40, nproc=16)
