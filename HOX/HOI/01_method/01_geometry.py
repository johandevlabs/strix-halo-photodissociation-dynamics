#!/usr/bin/env python3
"""
HOI step 1: the equilibrium geometry and harmonic force field, from CCSD(T).

HOBr/01_method/01_geometry.py does the work (x2c CCSD(T) on a 3x3x3 grid,
full quadratic fit, frequencies from the fitted Hessian); this file
registers HOI and runs it, writing to HOI/data/hoi_geometry.csv.

Start RECALLED (r(O-I) ~1.99 A). No observed fundamentals entered: none
checked against a source yet, and a recalled number in the calc/obs column
would look like a test. Basis x2c-tzvpall (see 04_soc_vertical.py).

Usage (from this directory):
    python 01_geometry.py 2>&1 | tee ../logs/geometry.log
    python 01_geometry.py --refine 2>&1 | tee ../logs/geometry.log
"""
import importlib.util
import sys
from pathlib import Path

HOX = Path(__file__).resolve().parents[2]
HOBR01 = HOX / "HOBr" / "01_method" / "01_geometry.py"

HOI = dict(halogen="I",
           start=(1.99, 0.964, 104.0),
           obs_fundamental=None,
           obs_geom=None,
           data=HOX / "HOI" / "data")


def load():
    spec = importlib.util.spec_from_file_location("hobr01", HOBR01)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hobr01"] = mod          # the worker pool pickles by name
    spec.loader.exec_module(mod)
    mod.MOLECULES["HOI"] = HOI
    return mod


if __name__ == "__main__":
    load().main(default="HOI", default_basis="x2c-tzvpall", doc=__doc__)
