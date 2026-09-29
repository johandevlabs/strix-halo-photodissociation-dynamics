#!/usr/bin/env python3
"""
HOI step 5: which computed state is which measured band? EOM-CCSD verticals
across HOCl, HOBr and HOI, as an independent check on 04's energies.

WHY. 04 (QD-NEVPT2, CAS(12,7)) puts HOI's lowest states at
    T1 2.42 eV (513 nm), S1 2.88 (430), T2 3.17 (391), S2 3.72 (333)
spin-free, and with SOC four groups of states with f > 1e-4 at 520, 444,
374 and 327 nm. The measured spectrum (02_band_model/10_obs_bands.py) has
exactly TWO bands in 280-500 nm, at 407 and 340 nm, and falls to ~1e-21 cm2
at both ends -- 2e-22 at 500 nm, where 04's lowest band (f 1.7e-4) would
give ~3e-20. Either reading of the assignment breaks something:

  A. 407 nm is the triplet band, as for HOCl and HOBr. Then 04 is ~0.6 eV
     too red for every state, where for HOCl and HOBr it was 0.1-0.2 eV too
     BLUE; and HOI's triplet band sits blue of HOBr's (457 nm), against the
     trend HOCl 368 -> HOBr 457.
  B. 407 nm is the singlet 1A" band (HOBr's 352 nm band, red-shifted as the
     trend says), 340 nm the next one (HOBr's 284). Then 04's energies are
     ~0.2 eV off, as for HOBr -- but the triplet band 04 predicts at ~500 nm
     is missing from the measurements by a factor 10-100.

EOM-CCSD shares nothing with 04 but the basis and the geometry: no active
space, no state averaging, no perturbation theory on top of CASSCF. HOCl and
HOBr, whose measured bands are assigned, calibrate its errors. If EOM puts
HOI's states where 04 does, the calculation is robust and the assignment is
the question; if it puts them ~0.6 eV higher, 04 is broken for iodine.

Spin-free only -- EOM here has no SOC and no oscillator strengths. Every
root is labelled by the irrep of its dominant single excitation and the
orbitals involved (H-k -> L+m), so n -> sigma* can be told from anything else.

Runs the three molecules one after the other, each with every core.
~10-20 min on the EVO.
    python 05_eom_vertical.py 2>&1 | tee ../logs/eom_vertical.log
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

HOX = Path(__file__).resolve().parents[2]
CSV = HOX / "HOI" / "data" / "eom_vertical.csv"
HARTREE2EV = 27.211386245988
NM_PER_EV = 1239.841984
DEG = np.pi / 180.0

# CCSD(T) minima from each molecule's 01_geometry.py (HOCl: the geometry the
# HOCl work used throughout; HOI: x2c-tzvpall, HOI/logs/geometry.log).
# Measured band maxima, nm, reddest first: HOI/02_band_model/10_obs_bands.py
# and HOBr/02_band_model/10_compare_obs.py.
MOLECULES = {
    "HOCl": dict(halogen="Cl", geom=(1.6891, 0.9644, 102.96),
                 obs=(368, 310, 237)),
    "HOBr": dict(halogen="Br", geom=(1.8357, 0.9646, 102.02),
                 obs=(457, 352, 284)),          # 437 in Barnes 1996
    "HOI": dict(halogen="I", geom=(1.9907, 0.9694, 104.65),
                obs=(407, 340)),                # nothing else 280-500 nm
}
FIELDS = ["molecule", "basis", "spin", "root", "omega_eV", "nm", "sym",
          "w1", "excitation", "t1_diag", "wall_s"]


def geometry(halogen, r_ox, r_oh, theta):
    th = theta * DEG
    return [["O", (0.0, 0.0, 0.0)],
            [halogen, (r_ox, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def run(name, spec, basis, nroots, mem):
    from pyscf import gto, scf, cc, symm
    from pyscf.cc import eom_rccsd

    t0 = time.time()
    mol = gto.M(atom=geometry(spec["halogen"], *spec["geom"]), basis=basis,
                spin=0, charge=0, symmetry="Cs", unit="Angstrom", verbose=0,
                max_memory=mem)
    if mol.has_ecp():
        raise RuntimeError(f"{basis} uses an ECP for {spec['halogen']}")
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    mycc = cc.RCCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.max_cycle = 100
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    nocc = mycc.t1.shape[0]
    t1d = float(np.linalg.norm(mycc.t1) / np.sqrt(2 * nocc))
    print(f"\n  {name}: nao {mol.nao_nr()}, T1 diagnostic {t1d:.4f}, "
          f"CCSD {time.time() - t0:.0f} s")

    ids = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff,
                              s=mf.get_ovlp())
    names = dict(zip(mol.irrep_id, mol.irrep_name))
    rows = []
    for spin, cls in (("singlet", eom_rccsd.EOMEESinglet),
                      ("triplet", eom_rccsd.EOMEETriplet)):
        t1 = time.time()
        eom = cls(mycc)
        e, vecs = eom.kernel(nroots=nroots)
        e = np.atleast_1d(np.asarray(e, dtype=float))
        vecs = vecs if isinstance(vecs, (list, tuple)) else list(vecs)
        conv = np.all(np.atleast_1d(getattr(eom, "converged", True)))
        for k in np.argsort(e):
            v = vecs[k]
            r1 = np.asarray(eom.vector_to_amplitudes(v)[0])
            i, a = np.unravel_index(int(np.argmax(np.abs(r1))), r1.shape)
            w1 = float(np.sum(np.abs(r1) ** 2) / np.sum(np.abs(v) ** 2))
            om = float(e[k] * HARTREE2EV)
            rows.append(dict(
                molecule=name, basis=basis, spin=spin, root=len(rows),
                omega_eV=round(om, 4), nm=round(NM_PER_EV / om, 1),
                sym=names.get(int(ids[i]) ^ int(ids[nocc + a]), "?"),
                w1=round(w1, 3),
                excitation=(f"H-{nocc - 1 - i} ({names.get(int(ids[i]))}) -> "
                            f"L+{a} ({names.get(int(ids[nocc + a]))})"),
                t1_diag=round(t1d, 4),
                wall_s=round(time.time() - t1, 1)))
        if not conv:
            print(f"  !! {spin} EOM not fully converged")
    return rows


def report(all_rows):
    for name, spec in MOLECULES.items():
        rows = [r for r in all_rows if r["molecule"] == name]
        if not rows:
            continue
        print(f"\n== {name}   measured bands: "
              + ", ".join(f"{n} nm ({NM_PER_EV / n:.2f} eV)"
                          for n in spec["obs"]))
        print(f"  {'':8}{'eV':>8}{'nm':>8}  {'sym':<5}{'w1':>6}   excitation")
        for r in sorted(rows, key=lambda r: r["omega_eV"]):
            flag = "" if r["w1"] >= 0.85 else "   (doubles-heavy)"
            print(f"  {r['spin']:<8}{r['omega_eV']:8.3f}{r['nm']:8.1f}  "
                  f"{r['sym']:<5}{r['w1']:6.2f}   {r['excitation']}{flag}")

    # the trend, lowest root of each (spin, symmetry)
    print("\n== lowest root of each kind, eV")
    kinds = [("triplet", 'A"'), ("singlet", 'A"'), ("triplet", "A'"),
             ("singlet", "A'")]
    print(f"  {'':8}" + "".join(f"{s[0].upper() + ' ' + y:>8}"
                                for s, y in kinds))
    for name in MOLECULES:
        rows = [r for r in all_rows if r["molecule"] == name]
        if not rows:
            continue
        cells = []
        for s, y in kinds:
            m = [r["omega_eV"] for r in rows if r["spin"] == s
                 and r["sym"].replace("''", '"') == y]
            cells.append(f"{min(m):8.3f}" if m else f"{'--':>8}")
        print(f"  {name:<8}" + "".join(cells))
    print("""
  Reading it. For HOCl and HOBr, T A" is the visible band and S A" the
  next; their EOM-minus-measured offsets calibrate the method. Apply the
  same offsets to HOI: if T A" lands near 407 nm, reading A holds and 04's
  energies are too low for iodine; if S A" lands near 407 nm, reading B
  holds and the missing ~500 nm triplet band is the question. 04 for HOI:
  T1 2.42, S1 2.88, T2 3.17, S2 3.72 eV (x2c-tzvpall, QD-NEVPT2).""")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--molecules", nargs="+", default=list(MOLECULES),
                   choices=list(MOLECULES))
    p.add_argument("--basis", default="cc-pvtz-dk",
                   help="has I (toolchain sweep: -7.7%% on the 2P splitting, "
                        "the same as Br's), and is HOBr's production basis")
    p.add_argument("--nroots", type=int, default=6)
    p.add_argument("--memory", type=int, default=60000, help="MB")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()

    print("=" * 72)
    print(f"== EOM-CCSD verticals, singlets and triplets, {args.basis}, x2c, "
          f"{args.nroots} roots each")
    print("=" * 72)
    rows = []
    if CSV.exists():
        with open(CSV) as fh:
            rows = [r for r in csv.DictReader(fh) if r["basis"] == args.basis]
        for r in rows:
            for k in ("omega_eV", "nm", "w1"):
                r[k] = float(r[k])
    done = {r["molecule"] for r in rows}
    if not args.report_only:
        CSV.parent.mkdir(parents=True, exist_ok=True)
        for name in args.molecules:
            if name in done:
                print(f"  {name}: in {CSV.name} already, skipped")
                continue
            try:
                new = run(name, MOLECULES[name], args.basis, args.nroots,
                          args.memory)
            except Exception as exc:
                print(f"  {name} FAILED: {type(exc).__name__}: {exc}")
                continue
            first = not CSV.exists()
            with open(CSV, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if first:
                    w.writeheader()
                w.writerows(new)
            rows += new
    report(rows)


if __name__ == "__main__":
    main()
