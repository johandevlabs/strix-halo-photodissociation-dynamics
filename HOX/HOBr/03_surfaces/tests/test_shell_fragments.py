"""Offline checks of 07_outer_shell.py and 08_fragments.py, no pyscf.

Both were generated from HOCl's 13 and 14 by substitution. 07: the pool, the
resume, the data directory, and that the path to HOCl's 07 active-space
module resolves. 08: that E(Br) lands in every CSV row, in both methods,
because 09 now reads it from there.
"""
import contextlib, csv, importlib.util, io, os, pathlib, sys, tempfile, types

ok = True
def check(label, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")

here = pathlib.Path(__file__).resolve().parent.parent

# ---- 07: shell
spec = importlib.util.spec_from_file_location("sh07", here / "07_outer_shell.py")
sh = importlib.util.module_from_spec(spec)
sys.modules["sh07"] = sh
spec.loader.exec_module(sh)
check("path to HOCl's 07_fc_active_space.py resolves", os.path.exists(sh._FC07_PY))
def fake(task):
    r, h, t = task[:3]
    return {"r_obr_A": round(r, 4), "r_oh_A": round(h, 4), "theta_deg": round(t, 3),
            "ncas": 6, "nelecas": 10, "e_nevpt2_Ha": -2680.0, "status": "ok", "wall_s": 0.01}
sh.compute_point = fake
path = os.path.join(tempfile.mkdtemp(), "new_dir", "shell.csv")
def run(*a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sys.argv = ["x", "--csv", path, "--nproc", "4", *a]
        sh.main()
    return buf.getvalue()
run("--pilot", "10")
run()
rows = list(csv.DictReader(open(path)))
check("shell: 1190 points, into a directory that did not exist", len(rows) == 1190)
check("shell: r(O-Br) 2.25-3.85 and overlaps the raster at 2.25/2.35/2.45",
      {float(r["r_obr_A"]) for r in rows} >= {2.25, 2.35, 2.45, 3.85})
check("shell: resume runs nothing", "nothing to do" in run())

# ---- 08: fragments, with pyscf stubbed and the two calculations faked
for n in ("pyscf", "pyscf.gto", "pyscf.scf", "pyscf.cc", "pyscf.mcscf",
          "pyscf.mrpt", "pyscf.fci", "pyscf.mcscf.avas"):
    sys.modules.setdefault(n, types.ModuleType(n))
pk = sys.modules["pyscf"]
for n in ("gto", "scf", "cc", "mcscf", "mrpt", "fci"):
    setattr(pk, n, sys.modules[f"pyscf.{n}"])
sys.modules["pyscf.mcscf"].avas = sys.modules["pyscf.mcscf.avas"]
spec = importlib.util.spec_from_file_location("fr08", here / "08_fragments.py")
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)
fr.x_mol = lambda basis, verbose=0: "Br"
fr.oh_mol = lambda r, basis, verbose=0: ("OH", r)
fr.uccsd_t = lambda mol: ((-2604.5, True) if mol == "Br"
                          else (-75.7 + 0.2 * (mol[1] - 0.97) ** 2, True))
fr.casscf_nevpt2 = lambda mol, labels, minao, ss: (
    (-2604.4, 3, 5, True) if mol == "Br"
    else (-75.67 + 0.2 * (mol[1] - 0.97) ** 2, 5, 7, True))
fpath = os.path.join(tempfile.mkdtemp(), "nd", "frag.csv")
with contextlib.redirect_stdout(io.StringIO()):
    sys.argv = ["x", "--csv", fpath]
    fr.main()
rows = list(csv.DictReader(open(fpath)))
check("fragments: E(Br) UCCSD(T) in every row",
      {r["e_x_uccsdt_Ha"] for r in rows} == {"-2604.5"})
check("fragments: E(Br) NEVPT2 in every row",
      {r["e_x_nevpt2_Ha"] for r in rows} == {"-2604.4"})
check("fragments: OH curve 0.70-1.40", (float(rows[0]["r_A"]), float(rows[-1]["r_A"])) == (0.7, 1.4))

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
