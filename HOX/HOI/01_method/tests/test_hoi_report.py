"""Offline check of HOI's gate report: HOI registered in HOBr's 04, and the
verdict machinery on a synthetic strongly mixed case built from HOBr's real
n10 energies (HOBr/logs/soc_vertical_n10.log) -- the triplet's spin-free
state put 0.2 eV above its SOC states, the next state 0.5 eV beyond, so
c^2 = 0.2 / 0.5 = 40%. Runs without PySCF.
"""
import contextlib, importlib.util, io, pathlib, sys, types
import numpy as np

for n in ("pyscf", "pyscf.gto", "pyscf.scf", "pyscf.mcscf", "pyscf.fci"):
    sys.modules[n] = types.ModuleType(n)
for n in ("gto", "scf", "mcscf", "fci"):
    setattr(sys.modules["pyscf"], n, sys.modules[f"pyscf.{n}"])

HERE = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("hoi04", HERE / "04_soc_vertical.py")
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
m = w.load()

ok = True
def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")

check("HOI registered alongside HOBr and HOCl",
      set(m.MOLECULES) == {"HOBr", "HOCl", "HOI"})
check("HOI basis cc-pvtz-dk, HOBr's", m.MOLECULES["HOI"]["basis"] == "cc-pvtz-dk")

HA = 27.211386
rel = np.array([0.0, 2.9015, 2.9021, 2.9084, 3.5881, 3.6751, 3.6849, 3.8128,
                4.5597, 6.1845, 6.1897, 6.1966, 6.6212, 6.7031, 6.7052,
                6.7825, 7.4989, 7.7889, 7.7919, 7.8654])
e = -2680.4259403866 + rel / HA
osc = np.array([8.2403e-08, 1.1747e-06, 1.4142e-05, 4.4805e-04, 4.2621e-10,
                7.5033e-07, 8.4567e-04, 5.5220e-03, 6.2171e-06, 1.1327e-08,
                2.2622e-05, 2.9484e-03, 1.7717e-08, 9.0884e-06, 4.0254e-03,
                4.4210e-05, 3.0308e-06, 4.3609e-05, 1.1668e-05])
t_abs = e[1:4].mean() + 0.2 / HA
sf = np.array([e[0] + 0.0108 / HA, t_abs, t_abs + 0.5 / HA, t_abs + 0.6 / HA])

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.report(m.MOLECULES["HOI"], e, osc, sf, 0.05)
out = buf.getvalue()
print(out[out.index("mixing, two-state"):])
check("c^2 ~ 40%", "c^2 ~ 40." in out)
check("verdict strongly mixed", "STRONGLY MIXED" in out)
check("triplet band reported as not measured", "NOT MEASURED" in out)
check("lender band against the measured 407 nm band",
      "obs 1.2e-03 at 407 nm" in out)
check("no HOBr-only band-model line", "would sit near" not in out)
check("SOC-squared scaling from HOBr", "(xi/xi_Br)^2 x 1.5e-5 = 6.4e-05" in out)

print("\n  01_geometry registers HOI with its own data directory:")
spec = importlib.util.spec_from_file_location("hoi01", HERE / "01_geometry.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
g1 = g.load()
check("HOI in 01's MOLECULES", "HOI" in g1.MOLECULES)
check("CSV goes to HOI/data", g1.MOLECULES["HOI"]["data"].parts[-2:] == ("HOI", "data"))
check("mass of iodine present", abs(g1.MASS["I"] - 126.904473) < 1e-9)

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
