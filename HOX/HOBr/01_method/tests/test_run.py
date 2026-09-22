"""End-to-end run of 01_geometry.py with a fake quadratic 'calculation'.

Exercises the parts that are not chemistry and have broken before: the fork
pool, the resumable CSV, the grid/CSV key agreement, and --refine.
"""
import importlib.util, os, pathlib, sys, tempfile
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent / "01_geometry.py"
spec = importlib.util.spec_from_file_location("g01", HERE)
m = importlib.util.module_from_spec(spec)
sys.modules["g01"] = m
spec.loader.exec_module(m)

# deliberately OFF the default start (1.834, 0.961, 102.3), as the
# real minimum will be, so --refine has something to move to
EQ = np.array([1.8405, 0.9578, 103.10])
K = np.array([[0.691, 0.02, 0.0015],
              [0.02, 1.754, -0.0020],
              [0.0015, -0.0020, 5.06e-5]])

def fake_point(task):
    """Module-level, not a lambda: fork pools must be able to pickle it."""
    _hal, r_ox, r_oh, th, _basis, _mem = task
    d = np.array([r_ox, r_oh, th]) - EQ
    return {"r_ox_A": round(r_ox, 5), "r_oh_A": round(r_oh, 5),
            "theta_deg": round(th, 4), "nbf": 100, "e_hf_Ha": -100.0,
            "e_ccsd_Ha": -100.0, "e_ccsdt_Ha": -100.0 + 0.5 * d @ K @ d,
            "t1": 0.012, "status": "ok", "wall_s": 0.01}

m.compute_point = fake_point
csv_path = os.path.join(tempfile.mkdtemp(), "t_geom.csv")

ok = True
def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {label}: {got!r}")

print("=== first run ===")
sys.argv = ["x", "--molecule", "HOBr", "--csv", csv_path, "--nproc", "4"]
m.main()

import csv as _csv
with open(csv_path) as fh:
    rows = list(_csv.DictReader(fh))
check("27 rows written", len(rows), 27)
check("no duplicate grid points",
      len({(r["r_ox_A"], r["r_oh_A"], r["theta_deg"]) for r in rows}), 27)

print("\n=== second run: everything should already be done ===")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.main()
out = buf.getvalue()
check("resume recomputes nothing", "27 points, 0 to compute" in out, True)
with open(csv_path) as fh:
    check("still 27 rows", len(list(_csv.DictReader(fh))), 27)
check("header written exactly once",
      open(csv_path).read().count("r_ox_A,r_oh_A"), 1)

print("\n=== --refine: re-centre and add a second grid ===")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = ["x", "--molecule", "HOBr", "--csv", csv_path, "--nproc", "4",
                "--refine"]
    m.main()
out = buf.getvalue()
with open(csv_path) as fh:
    rows2 = list(_csv.DictReader(fh))
check("refine added a fresh 27-point grid", len(rows2), 54)
check("refine re-centred on the fitted minimum",
      "1.8405, 0.9578, 103.10" in out, True)

# the recovered geometry and frequencies from the real report path
print("\n=== recovered from the full pipeline ===")
for line in out.splitlines():
    if any(k in line for k in ("r(O-Br)", "r(O-H)", "angle   =", "nu3", "nu2",
                               "nu1", "rms residual")):
        print("   ", line.strip())

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
