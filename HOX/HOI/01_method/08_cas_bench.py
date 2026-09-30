#!/usr/bin/env python3
"""
HOX: does a larger active space recover the triplet's borrowed intensity?
Minaev's CAS against ours, on HOCl, HOBr and HOI together.

WHY. Our QD-NEVPT2 state-interaction f for the a 3A" band is 3-4x LOW for
HOBr (1.5e-5 against 4.7-6.2e-5 measured; HOCl 10-40x), while Minaev's MCSCF
quadratic response (J. Phys. Chem. A 103, 7294 (1999)) gets HOBr right
(7-8e-5). For HOI the two agree (1.5e-4 / 1.6-2.5e-4), and whether that
number is right decides whether Bauer's 532 nm null contradicts theory
(HOI/02_band_model/11_band_soc_1d.py). Already ruled out: more roots (6 to 16,
up to 10.7 eV: flat) and sigma*(O-H) (AVAS + H 1s, CAS(12,8): flat).

What Minaev's space has that ours lacks: CAS(12,9) = all valence occupied
but the lowest, plus the three lowest EMPTY orbitals, two a' and one a"
(ours: one, sigma*(O-X)). His dominant lender, 4 1A' at ~9 eV (sigma* <-
sigma), is inside our space already, so if the extra virtuals do not help,
the difference is his orbital relaxation (response), which state interaction
over a finite set of roots cannot reproduce -- and that is worth knowing too.

Runs HOBr/01_method/04_soc_vertical.py (HOI through its wrapper), all in
cc-pvtz-dk at each molecule's 01_geometry.py minimum, 12 Ms = 0 roots:

    avas    AVAS X np + O 2p            CAS(12,7), ours
    cas9    --cas 9 12, canonical       CAS(12,9), Minaev's
    cas11   --cas 11 12, canonical      CAS(12,11), two virtuals more

04 prints what each canonical window contains (A'/A", atomic make-up), so
the virtuals are named, not assumed. Order: the three CAS(12,7) and CAS(12,9)
pairs first (the comparison that matters), then CAS(12,11). Every run's log
goes to HOI/logs/cas_bench/; a finished log is not rerun. Several hours;
overnight.

MEASURED (triplet band f; lender bands f): HOCl 3.3-3.4e-5 (310 nm band
3.0e-4); HOBr 4.7-6.2e-5 (352 nm 6.5e-4); HOI triplet not measured (407 nm
band 1.17e-3). A space that brings HOCl and HOBr within ~2x earns trust for
HOI.

Usage (from this directory):
    python 08_cas_bench.py 2>&1 | tee ../logs/cas_bench.log
    python 08_cas_bench.py --summary-only
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOX = HERE.parents[1]
LOGS = HOX / "HOI" / "logs" / "cas_bench"
S04 = HOX / "HOBr" / "01_method" / "04_soc_vertical.py"
HOI04 = HERE / "04_soc_vertical.py"

MOLECULES = {
    "HOCl": dict(script=S04, args=["--molecule", "HOCl", "--geom", "1.7069",
                                   "0.9648", "101.88"],
                 obs_t=(3.3e-5, 3.4e-5), obs_l=3.0e-4),
    "HOBr": dict(script=S04, args=["--molecule", "HOBr"],
                 obs_t=(4.7e-5, 6.2e-5), obs_l=6.5e-4),
    "HOI": dict(script=HOI04, args=[], obs_t=None, obs_l=1.17e-3),
}
SPACES = {"avas": [], "cas9": ["--cas", "9", "12"], "cas11": ["--cas", "11", "12"]}
ORDER = [(m, s) for s in ("avas", "cas9") for m in ("HOBr", "HOCl", "HOI")] \
    + [(m, "cas11") for m in ("HOBr", "HOCl", "HOI")]
DONE = re.compile(r"total [\d.]+ s wall")


def log_path(mol, space):
    return LOGS / f"{mol.lower()}_{space}.log"


def run(mol, space, nroots):
    spec = MOLECULES[mol]
    cmd = ([sys.executable, str(spec["script"])] + spec["args"]
           + ["--basis", "cc-pvtz-dk", "--nroots", str(nroots)] + SPACES[space])
    path = log_path(mol, space)
    t0 = time.time()
    print(f"  {mol} {space}: {' '.join(cmd[1:])}", flush=True)
    with open(path, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    ok = DONE.search(path.read_text()) is not None
    print(f"    -> {'done' if ok else 'FAILED, see ' + path.name} "
          f"({(time.time() - t0) / 60:.1f} min)", flush=True)


def parse(path):
    if not path.exists():
        return None
    t = path.read_text()
    if not DONE.search(t):
        return dict(status="incomplete")
    g = lambda pat: (lambda m: m.group(1) if m else None)(re.search(pat, t))  # noqa: E731
    out = dict(status="ok",
               cas=g(r"(CAS\(\d+e?, ?\d+o?\))") or g(r"--cas window: (.*)"),
               centroid=g(r"centroid\s+([\d.]+) eV"),
               f_t=g(r"summed f\s+([\d.e+-]+)"),
               f_l=g(r"lender band\s+calc ([\d.e+-]+)"),
               c2=g(r"c\^2 ~ ([\d.]+)%"),
               gap=g(r"gap to the next spin-free state ([\d.]+) eV"))
    out["vir"] = re.findall(r"vir\s+\d+\s+e\s+([+-][\d.]+) Ha\s+(A'|A\")\s+(.*)", t)
    return out


def summary():
    print("\n== triplet band f and lender band f against measurement")
    print(f"  {'':6}{'space':<7}{'T/eV':>7}{'f(3A\")':>10}{'obs':>18}"
          f"{'calc/obs':>10}{'f(lender)':>11}{'obs':>9}{'c^2 %':>7}")
    for mol, spec in MOLECULES.items():
        for space in SPACES:
            p = parse(log_path(mol, space))
            if p is None:
                continue
            if p["status"] != "ok":
                print(f"  {mol:<6}{space:<7}  {p['status']}")
                continue
            ft = float(p["f_t"]) if p["f_t"] else float("nan")
            ob = spec["obs_t"]
            ratio = (f"{ft / ob[1]:.2f}-{ft / ob[0]:.2f}" if ob else "--")
            obs = f"{ob[0]:.1e}-{ob[1]:.1e}" if ob else "not measured"
            print(f"  {mol:<6}{space:<7}{float(p['centroid'] or 'nan'):7.3f}"
                  f"{ft:10.2e}{obs:>18}{ratio:>10}"
                  f"{float(p['f_l'] or 'nan'):11.2e}{spec['obs_l']:9.1e}"
                  f"{float(p['c2'] or 'nan'):7.1f}")
    print("\n== the virtuals each canonical window adds (energy, symmetry, "
          "largest shares)")
    for mol in MOLECULES:
        p = parse(log_path(mol, "cas11"))
        if p and p.get("vir"):
            print(f"  {mol}:")
            for e, sym, share in p["vir"]:
                print(f"    {e} Ha  {sym:<3} {share.strip()}")
    print("""
  Reading it. If cas9/cas11 raise HOCl's and HOBr's triplet f towards the
  measured values, the missing lenders were virtuals outside CAS(12,7), and
  the same space's HOI value is the one to set against Bauer's 532 nm bound.
  If f stays flat, the difference from Minaev is his orbital relaxation --
  response, not a bigger CAS -- and HOI's f carries HOBr's 3-4x uncertainty
  in both directions.""")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nroots", type=int, default=12)
    p.add_argument("--only", nargs="+", default=None,
                   help="subset, e.g. HOBr:cas9 HOI:cas9")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print(f"== CAS benchmark for the borrowed triplet f, {args.nroots} roots, "
          f"cc-pvtz-dk")
    print("=" * 76)
    if not args.summary_only:
        todo = ORDER if not args.only else [tuple(x.split(":")) for x in args.only]
        for mol, space in todo:
            p_ = parse(log_path(mol, space))
            if p_ and p_["status"] == "ok":
                print(f"  {mol} {space}: done already")
                continue
            run(mol, space, args.nroots)
    summary()


if __name__ == "__main__":
    main()
