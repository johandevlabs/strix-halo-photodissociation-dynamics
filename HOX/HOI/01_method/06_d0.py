#!/usr/bin/env python3
"""
HOI step 6: D0(HO-I), and whether Bauer's 532 nm null bounds the triplet band.

WHY. Bauer et al. (1998) photolysed HOI at 532 nm and saw no OH, which puts
sigma(532) < ~1e-20 cm2 -- IF HO + I is open at 532 nm (2.331 eV). Our
computed a3A" band (f 1.5e-4 near 516-560 nm) puts 1-4e-20 there, so the
bound would bite. Bauer took the threshold as 582 +- 20 nm (dHf(HOI) =
-60 +- 7 kJ/mol); IUPAC's datasheet gives 507 nm (236 kJ/mol). A band on a
repulsive state absorbs nothing below D0 from v = 0, so:

  D0 < 2.33 eV   the null bounds the band, and the computed band is 1-4x over
  D0 > 2.33 eV   the null is silent -- but then the triplet vertical (2.40 eV
                 with SOC, 04) barely clears D0, and the band must sit blue
                 of 532 nm, inside Bauer's measured range

METHOD. CCSD(T), x2c, at each molecule's CCSD(T) minimum, against OH at its
own minimum (a 5-point scan) and the halogen atom (UCCSD(T) on ROHF, one
component of 2P). Two bases, cc-pvtz-dk and cc-pvqz-dk: HF from the larger,
correlation extrapolated as X^-3. Two core treatments: 'fc' freezes O 1s and
the halogen through (n-1)d -- the valence-only bases have no core-correlating
functions -- and 'ae' correlates everything, as the HOX surfaces did; the
difference is the core error bar.

Then, all in the output:
  ZPE      HOX harmonic from its 01_geometry.py run; OH harmonic from the scan
  SO       the atoms' first-order lowering, measured: X(2P3/2) lies dE/3 below
           the 2P average (Cl 294, Br 1228, I 2534 cm-1), OH(2Pi3/2) |A|/2
           = 70 cm-1 below its average; the HOX ground state's own
           second-order lowering from 04 (HOCl 0.5, HOBr 10.8, HOI 54.4 meV,
           QD-NEVPT2 state interaction -- a floor, DKH1 SOC runs ~8% low)
  298 K    dH298 - D0 from ideal-gas translation/rotation, harmonic
           vibration and OH's 2Pi1/2 level, to compare with the dH298-based
           thresholds of Bauer and IUPAC

CONTROL: HOCl, whose D0 is measured -- 19 289.7 cm-1 (2.3916 eV) from
state-resolved overtone predissociation (Rizzo and co-workers), QUOTED FROM
MEMORY, verify before citing. HOBr is computed alongside for the trend.

Resumable: every energy goes to HOI/data/d0.csv.
    python 06_d0.py 2>&1 | tee ../logs/d0.log
    python 06_d0.py --bases cc-pvtz-dk          # TZ only, quick
    python 06_d0.py --report-only
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

HOX = Path(__file__).resolve().parents[2]
CSV = HOX / "HOI" / "data" / "d0.csv"
HARTREE2EV = 27.211386245988
HARTREE2CM = 219474.6313702
HARTREE2KJ = 2625.4996394799
CM2KJ = 0.01196265663
EV2KJ = 96.48533212
NM_PER_EV = 1239.841984
ANG2BOHR = 1.8897261254578281
AMU2ME = 1822.888486209
KT298_CM = 207.2227
R_KJ = 0.0083144626
DEG = np.pi / 180.0
M_O, M_H = 15.9949146221, 1.0078250319

# geometry: each molecule's 01_geometry.py CCSD(T) minimum (HOCl and HOBr
# cc-pvtz-dk; HOI x2c-tzvpall). harm: that run's harmonic frequencies, cm-1.
# so_mol_meV: 04's ground-state lowering by SOC. core: spatial orbitals the
# 'fc' treatment freezes on the halogen (through (n-1)d).
MOLECULES = {
    "HOCl": dict(halogen="Cl", geom=(1.7069, 0.9648, 101.88),
                 harm=(769.4, 1279.8, 3817.9), so_mol_meV=0.5,
                 obs_d0_cm=19289.7),
    "HOBr": dict(halogen="Br", geom=(1.8357, 0.9646, 102.02),
                 harm=(632.4, 1199.4, 3868.6), so_mol_meV=10.8),
    "HOI": dict(halogen="I", geom=(1.9907, 0.9694, 104.65),
                harm=(582.7, 1103.8, 3841.7), so_mol_meV=54.4,
                triplet_ev=2.403),              # 04, cc-pvtz-dk, SOC centroid
}
CORE = {"Cl": 5, "Br": 14, "I": 23}             # 1s..(n-1)p/(n-1)d
FS_2P_CM = {"Cl": 882.35, "Br": 3685.24, "I": 7602.97}   # toolchain table
OH_A_CM = -139.21          # OH X 2Pi spin-orbit constant, from memory
OH_R = (0.940, 0.955, 0.970, 0.985, 1.000)
FIELDS = ["species", "basis", "frozen", "r_A", "e_hf", "e_ccsd_corr",
          "e_t", "status", "wall_s"]
THRESH = {"Bauer 1998": 582.0, "IUPAC 2007": 507.0}


def cardinal(basis):
    b = basis.lower()
    for tag, n in (("dz", 2), ("tz", 3), ("qz", 4), ("5z", 5)):
        if tag in b:
            return n
    raise ValueError(f"no cardinal number in {basis!r}")


def hox_atoms(halogen, r_ox, r_oh, theta):
    th = theta * DEG
    return [["O", (0.0, 0.0, 0.0)], [halogen, (r_ox, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def run_cc(atoms, basis, spin, nfrozen, mem):
    """HF, CCSD correlation and (T) for one species; x2c, no symmetry."""
    from pyscf import gto, scf, cc
    mol = gto.M(atom=atoms, basis=basis, spin=spin, charge=0, symmetry=False,
                unit="Angstrom", verbose=0, max_memory=mem)
    if mol.has_ecp():
        raise RuntimeError(f"{basis} has an ECP")
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).x2c()
    mf.conv_tol = 1e-11
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF not converged")
    frozen = nfrozen or None
    mycc = cc.CCSD(mf, frozen=frozen) if spin == 0 else cc.UCCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-9
    mycc.max_cycle = 150
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD not converged")
    return float(mf.e_tot), float(mycc.e_corr), float(mycc.ccsd_t())


def load():
    if not CSV.exists():
        return {}
    with open(CSV) as fh:
        return {(r["species"], r["basis"], r["frozen"], r["r_A"]): r
                for r in csv.DictReader(fh)}


def append(row):
    first = not CSV.exists()
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if first:
            w.writeheader()
        w.writerow(row)


def tasks(molecules, bases, frozens):
    """(species, basis, frozen, r_A, atoms, spin, nfrozen) for everything."""
    out = []
    halogens = [MOLECULES[m]["halogen"] for m in molecules]
    for b in bases:
        for fz in frozens:
            for r in OH_R:
                out.append(("OH", b, fz, f"{r:.3f}",
                            [["O", (0, 0, 0)], ["H", (0, 0, r)]], 1,
                            1 if fz == "fc" else 0))
            for m in molecules:
                sp = MOLECULES[m]
                out.append((m, b, fz, "eq", hox_atoms(sp["halogen"], *sp["geom"]),
                            0, (1 + CORE[sp["halogen"]]) if fz == "fc" else 0))
            for x in halogens:
                out.append((x, b, fz, "atom", f"{x} 0 0 0", 1,
                            CORE[x] if fz == "fc" else 0))
    return out


def energy(rows, species, basis, frozen, r, parts):
    r_ = rows.get((species, basis, frozen, r))
    if r_ is None or r_["status"] != "ok":
        return None
    return sum(float(r_[p]) for p in parts)


def oh_min(rs, es):
    """Minimum energy, r_e (A) and harmonic w (cm-1) from a quartic fit."""
    p = np.polyfit(rs, es, 4)
    dp, d2p = np.polyder(p), np.polyder(p, 2)
    roots = [x.real for x in np.roots(dp) if abs(x.imag) < 1e-9
             and rs[0] <= x.real <= rs[-1]]
    re = min(roots, key=lambda x: np.polyval(p, x))
    k = np.polyval(d2p, re) / ANG2BOHR ** 2            # Ha / bohr^2
    mu = M_O * M_H / (M_O + M_H) * AMU2ME
    return float(np.polyval(p, re)), float(re), float(np.sqrt(k / mu) * HARTREE2CM)


def species_energy(rows, species, basis, frozen, cbs_pair):
    """Total energy at one basis, or CBS from a (small, large) pair:
    HF(large) + [X^3 Ec(X) - Y^3 Ec(Y)] / (X^3 - Y^3)."""
    def at(b, parts):
        if species == "OH":
            es = [energy(rows, "OH", b, frozen, f"{r:.3f}", parts) for r in OH_R]
            return None if None in es else es
        r = "eq" if species in MOLECULES else "atom"
        return energy(rows, species, b, frozen, r, parts)

    def total(b):
        return at(b, ("e_hf", "e_ccsd_corr", "e_t"))

    if basis != "CBS":
        return total(basis)
    small, large = cbs_pair
    hf = at(large, ("e_hf",))
    cs, cl = at(small, ("e_ccsd_corr", "e_t")), at(large, ("e_ccsd_corr", "e_t"))
    if hf is None or cs is None or cl is None:
        return None
    X, Y = cardinal(large), cardinal(small)
    f = lambda h, a, b: h + (X ** 3 * b - Y ** 3 * a) / (X ** 3 - Y ** 3)  # noqa: E731
    if species == "OH":
        return [f(h, a, b) for h, a, b in zip(hf, cs, cl)]
    return f(hf, cs, cl)


def thermal_298_kj(harm_hox, w_oh):
    """dH298 - D0 for HOX -> OH + X: ideal gas, harmonic vibration."""
    def hvib(ws):
        return sum(w / np.expm1(w / KT298_CM) for w in ws) * CM2KJ
    rt = R_KJ * 298.15
    x = abs(OH_A_CM) / KT298_CM                      # OH 2Pi1/2, equal g
    h_el_oh = abs(OH_A_CM) * CM2KJ * np.exp(-x) / (1 + np.exp(-x))
    h_x = 2.5 * rt                                   # atom: 3/2 RT + RT
    h_oh = 3.5 * rt + hvib([w_oh]) + h_el_oh         # + RT rotation
    h_hox = 4.0 * rt + hvib(harm_hox)                # 3/2 + 3/2 + RT
    return h_x + h_oh - h_hox


def report(rows, molecules, bases, frozens):
    levels = list(bases) + (["CBS"] if len(bases) >= 2 else [])
    pair = (bases[0], bases[-1]) if len(bases) >= 2 else None
    e532 = NM_PER_EV / 532.0
    summary = {}
    for m in molecules:
        sp = MOLECULES[m]
        x = sp["halogen"]
        print(f"\n== {m} -> OH + {x}")
        print(f"  {'level':<14}{'core':<5}{'De/eV':>9}{'OH re/A':>9}"
              f"{'w_OH':>8}{'D0/eV':>9}{'D0 kJ':>8}{'thr/nm':>8}")
        for fz in frozens:
            for lv in levels:
                e_hox = species_energy(rows, m, lv, fz, pair)
                e_x = species_energy(rows, x, lv, fz, pair)
                e_oh = species_energy(rows, "OH", lv, fz, pair)
                if None in (e_hox, e_x) or e_oh is None:
                    print(f"  {lv:<14}{fz:<5}  (incomplete)")
                    continue
                eoh, re, w_oh = oh_min(np.array(OH_R), np.array(e_oh))
                de = (eoh + e_x - e_hox) * HARTREE2EV
                zpe = (sum(sp["harm"]) - w_oh) / 2 / HARTREE2CM * HARTREE2EV
                so_atoms = (FS_2P_CM[x] / 3 + abs(OH_A_CM) / 2) / HARTREE2CM \
                    * HARTREE2EV
                d0 = de - zpe - so_atoms + sp["so_mol_meV"] / 1000
                print(f"  {lv:<14}{fz:<5}{de:9.3f}{re:9.4f}{w_oh:8.0f}"
                      f"{d0:9.3f}{d0 * EV2KJ:8.1f}{NM_PER_EV / d0:8.0f}")
                summary[(m, fz, lv)] = dict(de=de, d0=d0, zpe=zpe, w_oh=w_oh,
                                            so=so_atoms, label=f"{lv} {fz}")
        best = next((summary[(m, fz, lv)] for lv in reversed(levels)
                     for fz in ("fc", "ae") if (m, fz, lv) in summary), None)
        if best is None:
            continue
        dth = thermal_298_kj(sp["harm"], best["w_oh"])
        print(f"  best level {best['label']}; the fc-ae spread above is the "
              f"core error bar")
        print(f"  corrections: ZPE -{best['zpe']:.3f} eV, "
              f"atomic SO -{best['so']:.3f} eV, molecular SO "
              f"+{sp['so_mol_meV'] / 1000:.3f} eV; dH298 - D0 = +{dth:.1f} kJ/mol")
        print(f"  dH298 = {best['d0'] * EV2KJ + dth:.0f} kJ/mol"
              + ("   (Bauer's threshold 582 nm = 206 kJ/mol; IUPAC 236 kJ/mol"
                 " = 507 nm)" if x == "I" else ""))
        if "obs_d0_cm" in sp:
            obs = sp["obs_d0_cm"] / HARTREE2CM * HARTREE2EV
            print(f"  CONTROL: measured D0 {obs:.4f} eV (from memory, verify) "
                  f"-> calc - obs {best['d0'] - obs:+.3f} eV "
                  f"({(best['d0'] - obs) * EV2KJ:+.1f} kJ/mol)")
        if x == "I":
            d0 = best["d0"]
            print(f"\n  532 nm = {e532:.3f} eV. D0 = {d0:.3f} eV: HO + I is "
                  + ("OPEN at 532 nm -> Bauer's null bounds sigma(532) < ~1e-20"
                     if d0 < e532 - 0.03 else
                     "CLOSED at 532 nm -> Bauer's null says nothing about "
                     "absorption" if d0 > e532 + 0.03 else
                     "MARGINAL at 532 nm (within 0.03 eV) -> the null is "
                     "weak evidence either way"))
            print(f"  triplet vertical with SOC (04): {sp['triplet_ev']:.3f} eV"
                  f" -- {sp['triplet_ev'] - d0:+.3f} eV above D0 (a repulsive "
                  f"state must sit above it)")
            for k, nm in THRESH.items():
                print(f"  {k} threshold {nm:.0f} nm = {NM_PER_EV / nm:.3f} eV "
                      f"(dH298 basis); ours {NM_PER_EV / d0:.0f} nm from D0")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--molecules", nargs="+", default=list(MOLECULES),
                   choices=list(MOLECULES))
    p.add_argument("--bases", nargs="+", default=["cc-pvtz-dk", "cc-pvqz-dk"],
                   help="smallest first; two or more -> CBS from the last two"
                        " (HF from the last)")
    p.add_argument("--frozen", nargs="+", default=["fc", "ae"],
                   choices=["fc", "ae"])
    p.add_argument("--memory", type=int, default=60000, help="MB")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if len(args.bases) >= 2:
        args.bases = args.bases[-2:]

    print("=" * 76)
    print(f"== D0(HO-X), CCSD(T)/x2c, {', '.join(args.bases)}, core "
          f"{'/'.join(args.frozen)}")
    print("=" * 76)
    rows = load()
    if not args.report_only:
        todo = [t for t in tasks(args.molecules, args.bases, args.frozen)
                if (t[0], t[1], t[2], t[3]) not in rows
                or rows[(t[0], t[1], t[2], t[3])]["status"] != "ok"]
        print(f"  {len(todo)} calculations to run")
        for i, (sp, b, fz, r, atoms, spin, nfz) in enumerate(todo, 1):
            t0 = time.time()
            row = dict(species=sp, basis=b, frozen=fz, r_A=r, status="ok")
            try:
                row["e_hf"], row["e_ccsd_corr"], row["e_t"] = run_cc(
                    atoms, b, spin, nfz, args.memory)
            except Exception as exc:
                row["status"] = f"fail:{type(exc).__name__}:{exc}"[:120]
            row["wall_s"] = round(time.time() - t0, 1)
            append(row)
            rows[(sp, b, fz, r)] = {k: str(v) for k, v in row.items()}
            print(f"    [{i}/{len(todo)}] {sp:<5}{b:<12}{fz:<3}{r:>6}  "
                  f"{row['status'][:60]}  {row['wall_s']} s", flush=True)
    report(rows, args.molecules, args.bases, args.frozen)


if __name__ == "__main__":
    main()
