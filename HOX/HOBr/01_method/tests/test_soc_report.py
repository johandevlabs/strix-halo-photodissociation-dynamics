"""Offline check of 04_soc_vertical.py's report, on HOCl's real Prism output.

The numbers below are copied from HOCl/logs/soc_hocl.log (HOCl's 03,
def2-TZVP, CAS(12,7), 6 Ms=0 roots). The spin-free energies are that log's
per-state diagonal NEVPT2 totals, which the real run replaces with a
spin-free QD-NEVPT2; close enough to check the bookkeeping.
"""
import contextlib, importlib.util, io, pathlib, sys, types
import numpy as np

for n in ("pyscf", "pyscf.gto", "pyscf.scf", "pyscf.mcscf", "pyscf.fci"):
    sys.modules[n] = types.ModuleType(n)
for n in ("gto", "scf", "mcscf", "fci"):
    setattr(sys.modules["pyscf"], n, sys.modules[f"pyscf.{n}"])

HERE = pathlib.Path(__file__).resolve().parent.parent / "04_soc_vertical.py"
spec = importlib.util.spec_from_file_location("s04", HERE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

e_soc = np.array([-536.734569126325, -536.607875986102, -536.607872550379,
                  -536.607859267105, -536.574268404973, -536.572203128178,
                  -536.572196788437, -536.571790766430, -536.536720932679,
                  -536.469884150137, -536.469881904188, -536.469879240663])
osc = np.array([0.0, 7e-8, 8.1e-7, 9.032e-4, 0.0, 1.6e-7, 1.9664e-4,
                5.91998e-3, 2e-8, 2.9e-7, 1.24e-6])
e_sf = np.array([-536.734549301925, -536.607432851176, -536.573830990521,
                 -536.572228161540, -536.536732101874, -536.470302271043])

ok = True
def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.report(m.MOLECULES["HOCl"], e_soc, osc, e_sf, 0.05)
out = buf.getvalue()
print(out[out.index("a 3A\" band ="):])

check("centroid 3.4477 eV (03: 3.4475-3.4480 components)", "centroid   3.4477 eV" in out)
check("summed f 8.8e-7", "summed f   8.8000e-07" in out)
check("control reproduces 03", "reproduced; the adaptation is sound" in out)
check("matched spin-free triplet 3.4590 eV", "matched spin-free triplet 3.4590 eV" in out)
check("SOC shift -11 meV", "shifts the a 3A\" band by -11 meV" in out)

print("\n  a HOBr-like case, where f misses the target by far:")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.report(m.MOLECULES["HOBr"], e_soc, osc, e_sf, 0.05)
o2 = buf.getvalue()
check("reports calc/needed against 03's floor", "calc/needed 0.01" in o2)
check("reports Gaussian f(obs)", "Gaussian estimate" in o2)
check("predicts where 03's band would move", "would sit near" in o2)

print("\n  f exactly zero must be caught, not reported as a band:")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.report(m.MOLECULES["HOBr"], e_soc, np.zeros_like(osc), e_sf, 0.05)
check("zero-f failure flagged", "EXACTLY zero" in buf.getvalue())

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
