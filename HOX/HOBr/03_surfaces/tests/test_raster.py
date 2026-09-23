"""Offline check of 05_pes_raster.py: no pyscf, a stubbed calculation.

05 is HOCl's validated 12 transformed mechanically (names, grid, defaults),
so what needs checking is that the transform broke nothing that is not
chemistry: the pool, the pilot selection, the CSV and the resume.
"""
import contextlib, csv, importlib.util, io, os, pathlib, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent.parent / "05_pes_raster.py"
spec = importlib.util.spec_from_file_location("r05", HERE)
m = importlib.util.module_from_spec(spec)
sys.modules["r05"] = m
spec.loader.exec_module(m)

ok = True
def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {label}: {got!r}")

def fake(task):
    r_ox, r_oh, th, basis, nroots, mem = task
    return {"r_obr_A": round(r_ox, 4), "r_oh_A": round(r_oh, 4),
            "theta_deg": round(th, 3), "nbf": 87, "status": "ok",
            "omega_app_eV": 2.9, "root0_sym": 'A"', "wall_s": 0.01}
m.compute_point = fake

d = tempfile.mkdtemp()
path = os.path.join(d, "not_yet", "raster.csv")     # directory does NOT exist

def run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sys.argv = ["x", "--csv", path, "--nproc", "4", *argv]
        m.main()
    return buf.getvalue()

out = run("--pilot", "12")
rows = list(csv.DictReader(open(path)))
check("missing data directory created", os.path.isdir(os.path.dirname(path)), True)
check("pilot wrote 12 or 13 points (plus equilibrium)", len(rows) in (12, 13), True)
eq = [r for r in rows if abs(float(r["r_obr_A"]) - 1.85) < 1e-6
      and abs(float(r["r_oh_A"]) - 0.95) < 1e-6
      and abs(float(r["theta_deg"]) - 100.0) < 1e-6]
check("equilibrium-nearest point included", len(eq), 1)
check("column is r_obr_A", "r_obr_A" in rows[0], True)
check("per-point cost reported", "CPU-s per point" in out, True)

out = run()
rows = list(csv.DictReader(open(path)))
check("full run completes the grid", len(rows), 2470)
check("no duplicates",
      len({(r["r_obr_A"], r["r_oh_A"], r["theta_deg"]) for r in rows}), 2470)
check("r(O-Br) spans 1.55-2.45",
      (min(float(r["r_obr_A"]) for r in rows),
       max(float(r["r_obr_A"]) for r in rows)), (1.55, 2.45))

out = run()
check("resume: nothing left to run", "nothing to do" in out, True)
check("header once", open(path).read().count("r_obr_A"), 1)

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
