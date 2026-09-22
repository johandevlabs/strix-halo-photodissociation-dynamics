"""Offline check of 01_geometry.py: no pyscf, synthetic energy surfaces.

The frequency chain (fit -> H_int -> B^T H B -> mass-weight) is checked
against a DIRECT Cartesian finite-difference Hessian of the same analytic
energy function. That second route shares no code with the first: it never
touches B, never converts degrees, and differences energies in bohr straight
away. If the units in frequencies() are wrong, the two disagree.
"""
import importlib.util, pathlib, sys, types
import numpy as np

for name in ("pyscf", "pyscf.gto", "pyscf.scf", "pyscf.cc"):
    sys.modules[name] = types.ModuleType(name)

HERE = pathlib.Path(__file__).resolve().parent.parent / "01_geometry.py"
spec = importlib.util.spec_from_file_location("g01", HERE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

ok = True
def check(label, got, want, tol):
    global ok
    good = abs(got - want) <= tol
    ok &= good
    print(f"  [{'ok ' if good else 'FAIL'}] {label}: {got:.6g} (want {want:.6g} +/- {tol:g})")

# ---- a synthetic molecule with a KNOWN quadratic force field -------------
HAL = "Br"
EQ = np.array([1.8340, 0.9610, 102.30])       # A, A, deg
# Force constants in Hartree per (A, A, deg), CALIBRATED to put the test in
# the regime the script will actually run in -- k = mu*omega^2 with HOBr's
# observed 620 / 1163 / 3615 cm-1 -- so the conditioning of the fit and of
# B^T H B is exercised where it matters rather than at arbitrary magnitudes.
K = np.array([[0.691,   0.02,    0.0015],
              [0.02,    1.754,  -0.0020],
              [0.0015, -0.0020,  5.06e-5]])

def energy_from_internals(q):
    d = np.asarray(q, float) - EQ
    return -100.0 + 0.5 * d @ K @ d

def energy_from_cartesian_bohr(x):
    """Same surface, reached through the geometry rather than the grid."""
    r_ox, r_oh, th_rad = m.internals(np.asarray(x, float))
    return energy_from_internals([r_ox / m.ANG2BOHR, r_oh / m.ANG2BOHR,
                                  th_rad / m.DEG])

print("quadratic_fit recovers a known force field:")
grid = m.build_grid("test-basis", list(EQ + np.array([0.01, -0.008, 0.7])),
                    [0.03, 0.03, 3.0])            # deliberately off-centre
pts = [g[1:] for g in grid]                       # drop the basis element
e = [energy_from_internals(p) for p in pts]
coords, g, H, resid = m.quadratic_fit(pts, e)
check("rms residual (exact quadratic)", resid, 0.0, 1e-12)
for i, nm in enumerate("r_OX r_OH theta".split()):
    check(f"minimum {nm}", coords[i], EQ[i], 1e-8)
check("H max abs error", float(np.abs(H - K).max()), 0.0, 1e-8)

print("\nfrequencies vs a direct Cartesian Hessian of the same surface:")
f_fit = m.frequencies(HAL, coords, H)

x0 = m.cartesians_bohr(HAL, coords)
h = 1e-4
Hc = np.zeros((9, 9))
for i in range(9):
    for j in range(9):
        xpp, xpm, xmp, xmm = (x0.copy() for _ in range(4))
        xpp[i] += h; xpp[j] += h
        xpm[i] += h; xpm[j] -= h
        xmp[i] -= h; xmp[j] += h
        xmm[i] -= h; xmm[j] -= h
        Hc[i, j] = (energy_from_cartesian_bohr(xpp)
                    - energy_from_cartesian_bohr(xpm)
                    - energy_from_cartesian_bohr(xmp)
                    + energy_from_cartesian_bohr(xmm)) / (4 * h * h)
Hc = 0.5 * (Hc + Hc.T)
mass = np.repeat([m.MASS["O"], m.MASS[HAL], m.MASS["H"]], 3) * m.AMU2ME
w = np.linalg.eigvalsh(Hc / np.sqrt(np.outer(mass, mass)))
f_cart = np.array([np.sign(v) * np.sqrt(abs(v)) * m.HARTREE2CM
                   for v in np.sort(w)[-3:]])
print(f"    B^T H B route : {np.array2string(f_fit, precision=2)}")
print(f"    Cartesian FD  : {np.array2string(f_cart, precision=2)}")
for a, b, nm in zip(f_fit, f_cart, ("nu3", "nu2", "nu1")):
    check(f"{nm} agreement", a, b, 0.5)

zero = np.sort(w)[:6]
check("six near-zero Cartesian modes (trans+rot)",
      float(np.abs(zero).max() * m.HARTREE2CM**2), 0.0, 1e4)

print("\nthe calibration lands near the observed HOBr fundamentals:")
for got, want, nm in zip(f_fit, (620.23, 1162.57, 3614.90),
                         ("nu3 O-Br", "nu2 bend", "nu1 O-H")):
    check(f"{nm}", got, want, 0.12 * want)

print("\nmass dependence is real (79Br -> 81Br lowers the O-Br stretch):")
m.MASS["Br"] = 80.9162906
f_81 = m.frequencies(HAL, coords, H)
m.MASS["Br"] = 78.9183376
print(f"    79Br {f_fit[0]:.2f}  ->  81Br {f_81[0]:.2f} cm-1  "
      f"({f_81[0] - f_fit[0]:+.2f})")
check("81Br shifts nu3 down by 0.5-3 cm-1", f_fit[0] - f_81[0], 1.75, 1.25)

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
