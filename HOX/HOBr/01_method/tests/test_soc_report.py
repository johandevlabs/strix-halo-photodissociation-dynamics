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

print("\n  HOBr's real n10 run (logs/soc_vertical_n10.log), a perturbative case:")
HA = 27.211386
rel10 = np.array([0.0, 2.9015, 2.9021, 2.9084, 3.5881, 3.6751, 3.6849, 3.8128,
                  4.5597, 6.1845, 6.1897, 6.1966, 6.6212, 6.7031, 6.7052,
                  6.7825, 7.4989, 7.7889, 7.7919, 7.8654])
e10 = -2680.4259403866 + rel10 / HA
osc10 = np.array([8.2403e-08, 1.1747e-06, 1.4142e-05, 4.4805e-04, 4.2621e-10,
                  7.5033e-07, 8.4567e-04, 5.5220e-03, 6.2171e-06, 1.1327e-08,
                  2.2622e-05, 2.9484e-03, 1.7717e-08, 9.0884e-06, 4.0254e-03,
                  4.4210e-05, 3.0308e-06, 4.3609e-05, 1.1668e-05])
sf10 = (-2680.4259403866 + 0.0108 / HA
        + np.array([0.0, 2.9083, 3.6555, 3.7358, 4.5510, 6.1900, 6.6726,
                    6.7076, 7.5583, 7.7710]) / HA)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.report(m.MOLECULES["HOBr"], e10, osc10, sf10, 0.05)
o2 = buf.getvalue()
print(o2[o2.index("mixing, two-state"):])
check("gap to next spin-free state 0.747 eV", "next spin-free state 0.747 eV" in o2)
check("c^2 ~ 2%", "c^2 ~ 2.0%" in o2)
check("UV lenders 346-325 nm, f 1.29e-3", "calc 1.29e-03" in o2 and
      "346, 337, 336, 325 nm" in o2)
check("visible calc/obs 0.25-0.33", "calc/obs 0.25-0.33" in o2)
check("verdict perturbative", "PERTURBATIVE" in o2)
check("predicts where 03's band would move", "would sit near" in o2)

print("\n  a synthetic strongly mixed case (triplet 0.2 eV above its SOC states,"
      " lender 0.5 eV beyond):")
t_abs = e10[1:4].mean() + 0.2 / HA
sf_mix = np.array([sf10[0], t_abs, t_abs + 0.5 / HA, t_abs + 0.6 / HA])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.report(m.MOLECULES["HOI"], e10, osc10, sf_mix, 0.05)
o3 = buf.getvalue()
check("c^2 ~ 40%", "c^2 ~ 40." in o3)
check("verdict strongly mixed", "STRONGLY MIXED" in o3)
check("no HOBr-only band-model line for HOI", "would sit near" not in o3)
check("SOC-squared scaling line for iodine", "(xi/xi_Br)^2 x 1.5e-5 = 6.4e-05" in o3)

print("\n  f exactly zero must be caught, not reported as a band:")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    m.report(m.MOLECULES["HOBr"], e_soc, np.zeros_like(osc), e_sf, 0.05)
check("zero-f failure flagged", "EXACTLY zero" in buf.getvalue())

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
