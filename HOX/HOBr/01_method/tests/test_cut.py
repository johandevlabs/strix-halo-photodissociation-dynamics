"""Offline check of 02_obr_cut.py: no pyscf, a synthetic triplet manifold.

The chemistry is faked; what is tested is everything around it -- the fork
pool, the basis-aware resume, the report's four verdicts, and in particular
that a 3A' crossing and a basis-dependent trend are DETECTED rather than
merely storable.
"""
import contextlib, csv as _csv, importlib.util, io, os, pathlib, sys, tempfile
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent / "02_obr_cut.py"
spec = importlib.util.spec_from_file_location("cut02", HERE)
m = importlib.util.module_from_spec(spec)
sys.modules["cut02"] = m
spec.loader.exec_module(m)

ok = True
def check(label, got, want):
    global ok
    good = got == want
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {label}: {got!r}")


def fake_point(task):
    """A 3A" falling with r, a 3A' that crosses it at 2.40 A, and a basis
    difference that GROWS with r -- the BSSE signature the report looks for."""
    hal, r, _roh, _th, basis, nroots, _mem = task
    w_app = 3.10 - 1.9 * (r - 1.834)          # the valence 3A"
    w_ap = 4.30 - 3.1 * (r - 1.834)           # a 3A' that dives and crosses
    if basis == "aug-cc-pvtz-dk":
        w_app -= 0.030 * (r - 1.50)           # 30 meV/A trend
    roots = sorted([(w_app, 'A"'), (w_ap, "A'"),
                    (w_app + 1.1, 'A"'), (w_ap + 1.4, "A'"),
                    (w_app + 2.2, 'A"'), (w_ap + 2.6, "A'")][:nroots])
    row = {"basis": basis, "r_ox_A": round(r, 5), "nbf": 87,
           "t1_s": 0.008 + 0.02 * max(0.0, r - 2.50),
           "e_hf_Ha": -2600.0, "e_ccsd_Ha": -2601.0, "e_ccsdt_Ha": -2601.1,
           "status": "ok", "wall_s": 0.01}
    for k, (om, sym) in enumerate(roots, 1):
        row[f"omega{k}_eV"], row[f"sym{k}"], row[f"w1_{k}"] = om, sym, 0.94
    app = [k for k, (_, sym) in enumerate(roots) if '"' in sym]
    k = app[0]
    row["root0_sym"] = roots[0][1]
    row["omega_app_eV"], row["w1_app"] = roots[k][0], 0.94
    row["gap_any_eV"] = min(abs(o - roots[k][0])
                            for j, (o, _) in enumerate(roots) if j != k)
    row["gap_app_eV"] = roots[app[1]][0] - roots[k][0] if len(app) > 1 else float("nan")
    flags = []
    if row["t1_s"] >= m.T1_MAX:
        flags.append("t1")
    if k != 0:
        flags.append("Ap_below")
    if flags:
        row["status"] = "warn:" + "+".join(flags)
    return row


m.compute_point = fake_point
csv_path = os.path.join(tempfile.mkdtemp(), "cut.csv")

print("=== first run ===")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = ["x", "--csv", csv_path, "--png", csv_path + ".png", "--nproc", "4"]
    m.main()
out = buf.getvalue()
with open(csv_path) as fh:
    rows = list(_csv.DictReader(fh))
check("58 rows (29 radii x 2 bases)", len(rows), 58)
check("both bases present", {r["basis"] for r in rows},
      {"cc-pvtz-dk", "aug-cc-pvtz-dk"})
check("every root stored", all(r["sym6"] for r in rows), True)

print("\n=== the four things the report must find ===")
# Expectations are DERIVED from the synthetic surface, not written down: the
# first draft of this test hard-coded 2.40 and 2.55 by eye, both wrong, and
# the script's correct answers looked like failures.
RADII = m.build_radii(1.834, type("A", (), dict(
    rmin=1.50, rfine=2.60, rmax=3.20, dfine=0.05, dcoarse=0.10))())
# 1. the crossing: 3.10 - 1.9d == 4.30 - 3.1d  =>  d = 1.0, r = 2.834
r_cross = 1.834 + (4.30 - 3.10) / (3.1 - 1.9)
first = min(r for r in RADII if r >= r_cross)
check("crossing detected at the first grid point past it",
      f"lowest triplet is 3A' from {first:.3f} A outward" in out, True)
# 2. T1 = 0.008 + 0.02*(r - 2.50) reaches 0.02 at r = 3.10
r_t1 = 2.50 + (m.T1_MAX - 0.008) / 0.02
first_t1 = min(r for r in RADII if r >= r_t1 - 1e-9)
check("T1 limit detected",
      f"T1(S) first reaches {m.T1_MAX} at {first_t1:.3f} A" in out, True)
# 3. the basis difference, injected as -30 meV/A everywhere: it must show
#    in the exit channel as a BSSE-like slope
ex = [l for l in out.splitlines() if "offset" in l and "slope" in l][1]
val = float(ex.split("slope")[1].split()[0])
check("exit-channel slope recovered as -30 meV/A", round(val), -30)
check("exit-channel trend flagged", "that is the BSSE signature" in out, True)
# 4. no spurious extra roots in this synthetic case
check("diffuse-root risk reported as absent", "did not materialise" in out, True)

print("\n=== resume is basis-aware ===")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.main()
check("nothing recomputed", "0 to compute" in buf.getvalue(), True)

print("\n=== the REAL HOBr shape must not be called BSSE ===")
# What aug-cc-pvtz-dk actually did on 2026-09-23: -224 meV on the inner wall,
# decaying through the FC window, flat (+0..+9 meV) from 2.3 A out. The
# first report fitted one line to all of it and flagged BSSE. It must not.
csv3 = os.path.join(tempfile.mkdtemp(), "real.csv")
def real_shape(task):
    row = fake_point(task)
    r = task[1]
    if task[4] == "aug-cc-pvtz-dk":
        row["omega_app_eV"] = (row["omega_app_eV"] + 0.030 * (r - 1.50)
                               - 0.25 * np.exp(-(r - 1.50) / 0.20) + 0.009)
    return row
m.compute_point = real_shape
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = ["x", "--csv", csv3, "--png", csv3 + ".png", "--nproc", "4"]
    m.main()
o3 = buf.getvalue()
check("inner-wall difference NOT reported as BSSE",
      "no BSSE signature" in o3, True)
check("FC-window effect reported instead", "shifts the vertical by" in o3, True)

print("\n=== a flat basis difference must NOT be called a trend ===")
csv2 = os.path.join(tempfile.mkdtemp(), "flat.csv")
def flat_point(task):
    row = fake_point(task)
    if task[4] == "aug-cc-pvtz-dk":          # constant offset, no trend
        row["omega_app_eV"] = row["omega_app_eV"] + 0.030 * (task[1] - 1.50) - 0.02
    return row
m.compute_point = flat_point
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = ["x", "--csv", csv2, "--png", csv2 + ".png", "--nproc", "4"]
    m.main()
check("flat offset reported as flat", "no BSSE signature" in buf.getvalue(), True)

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
