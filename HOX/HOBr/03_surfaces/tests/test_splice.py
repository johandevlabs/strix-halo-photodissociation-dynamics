"""Offline end-to-end check of 09_splice.py on a synthetic surface.

Raster, shell and fragments are all generated from ONE analytic a 3A"
surface, with the shell shifted by a known method offset that varies
smoothly across (r_OH, angle) -- as NEVPT2's does against CCSD(T) -- so the
splice has a right answer: recover that offset per column, find the
columns' offsets constant across the overlap, agree on the slope at the
seam, land on E(Br) + V_OH, and read E(Br) from the fragments CSV.
"""
import contextlib, csv, importlib.util, io, os, pathlib, sys, tempfile
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent / "09_splice.py"
spec = importlib.util.spec_from_file_location("sp09", HERE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
H = m.HARTREE2EV

E_BR = -2604.1234567
def v_oh(h):                                   # Morse, Ha
    return -75.73 + 0.17 * (1 - np.exp(-2.3 * (h - 0.97))) ** 2
def v_t(r, h, t):                              # repulsive, onto E(Br)+V_OH
    return (E_BR + v_oh(h) + 0.60 * np.exp(-3.0 * (r - 1.60))
            * (1 + 0.002 * (t - 102)))
def v_s(r, h, t):
    return (E_BR + v_oh(h) - 0.09 + 0.09 * (1 - np.exp(-1.9 * (r - 1.84))) ** 2
            + 2e-5 * (t - 102) ** 2)
def method_offset(h, t):                       # NEVPT2 above CC, smooth
    return 0.030 + 0.001 * (h - 1.0) + 2e-5 * (t - 100)

d = tempfile.mkdtemp()
ras, she, fra = (os.path.join(d, n) for n in ("r.csv", "s.csv", "f.csv"))
grid = lambda a, b, s: np.round(np.arange(a, b + s / 2, s), 4)  # noqa: E731
R, HH, T = grid(1.55, 2.45, 0.05), grid(0.80, 1.25, 0.05), grid(75, 135, 5)
with open(ras, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["r_obr_A", "r_oh_A", "theta_deg", "e_ccsdt_Ha", "omega_app_eV", "status"])
    for r in R:
        for h in HH:
            for t in T:
                w.writerow([r, h, t, v_s(r, h, t), (v_t(r, h, t) - v_s(r, h, t)) * H, "ok"])
RS, TS = grid(2.25, 3.85, 0.10), grid(75, 135, 10)
with open(she, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["r_obr_A", "r_oh_A", "theta_deg", "e_nevpt2_Ha", "status"])
    for r in RS:
        for h in HH:
            for t in TS:
                w.writerow([r, h, t, v_t(r, h, t) + method_offset(h, t), "ok"])
with open(fra, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["r_A", "e_uccsdt_Ha", "e_nevpt2_Ha", "e_x_uccsdt_Ha", "e_x_nevpt2_Ha"])
    for h in grid(0.70, 1.40, 0.025):
        w.writerow([h, v_oh(h), v_oh(h) + 0.03, E_BR, E_BR + 0.03])

ok = True
def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")

npz, png = os.path.join(d, "o.npz"), os.path.join(d, "o.png")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    sys.argv = ["x", "--raster", ras, "--shell", she, "--fragments", fra,
                "--npz", npz, "--png", png]
    m.main()
out = buf.getvalue()
for l in out.splitlines():
    if any(k in l for k in ("E(Br) UCCSD", "offset (raster", "ACROSS", "minus E(Br)", "at 2.35")):
        print("   ", l.strip())

check("E(Br) read from the fragments CSV", f"{E_BR:.8f}" in out and "f.csv" in out)
z = np.load(npz)
check("halogen-neutral NPZ keys", {"r_ox_A", "r_ox_bound_A", "e_x_cc_Ha", "V_T_Ha"} <= set(z.files))
off = -z["offset_Ha"]                           # raster - shell = -offset
want = method_offset(HH[:, None], T[None, :])
check("per-column method offset recovered to 1 meV",
      np.nanmax(np.abs(off - want)) * H * 1000 < 1.0)
rr = z["r_ox_A"]
i6 = int(np.argmin(abs(rr - 6.0)))
check("surface lands on E(Br) + V_OH at 6 A (to 1 meV)",
      np.nanmax(np.abs(z["V_T_Ha"][i6] - (E_BR + v_oh(HH))[:, None])) * H * 1000 < 1.0)
i_fc = int(np.argmin(abs(rr - 1.85)))
check("raster region untouched at 1.85 A (to 0.1 meV)",
      np.nanmax(np.abs(z["V_T_Ha"][i_fc] - v_t(1.85, HH[:, None], T[None, :]))) * H * 1000 < 0.1)
seam = [l for l in out.splitlines() if l.strip().startswith("at 2.35")][0]
worst = float(seam.split("mismatch")[1].split("eV/A")[0])
check("slopes agree at the interior seam radius (< 0.02 eV/A)", worst < 0.02)

print("\n  two different E(Br) values in the CSV must stop the splice:")
rows = list(csv.DictReader(open(fra)))
rows[3]["e_x_uccsdt_Ha"] = str(E_BR + 0.001)
with open(fra, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        m.main()
    check("inconsistent E(Br) rejected", False)
except SystemExit as e:
    check("inconsistent E(Br) rejected", "expected one E(Br)" in str(e))

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
