#!/usr/bin/env python3
"""
Phase 0.5, step 9: is the 0.1 eV vertical-energy offset the NEVPT2 contraction?

07 chose CAS(10,6) for the triplet surface and left one discrepancy. At
equilibrium, PySCF's NEVPT2 puts a 3A" <- X 1A' at 3.550 eV, while the two
independent references agree with each other:

    3.4477 eV   03, Prism QD-NEVPT2 with SOC, CAS(12,7)
    3.4142 eV   04, dCCSD(T)

State-averaging imbalance was ruled out: going from 4 to 2 triplet roots moved
the NEVPT2 vertical energy by +15 meV, the wrong direction and a tenth of the
gap. The leading explanation left is how NEVPT2 is contracted. PySCF's
mrpt.NEVPT is STRONGLY contracted (SC). Prism's is FULLY internally contracted
(FIC; its log reads "Internal contraction: Full (= Partial)"). But 03 also
differed in other ways (quasi-degenerate, six roots, mixed spin, CAS(12,7)),
so nothing so far isolates the contraction.

This script does: SC and FIC NEVPT2 on IDENTICAL references. Prism's own
example examples/nevpt/04-nevpt2_sc_vs_fic.py makes exactly this comparison --
FIC from Prism, SC from pyscf.mrpt.NEVPT, same orbitals -- so the method is
the developers', not an improvisation.

Two answers per geometry, at r_eq and r_eq +/- 0.05 A:

  OFFSET  Does FIC land near 3.41-3.45 eV at equilibrium? If so, the offset
          was the contraction, not the active space or the orbitals.
  SLOPE   d(E_T)/dr through the Franck-Condon window. By the reflection
          principle the band WIDTH scales with this slope, so it matters more
          for the spectrum than a constant offset does. If SC and FIC slopes
          agree, the cheap SC surface has the right shape and only its
          absolute position needs correcting (a dCCSD(T) splice, or a shift).
          If they differ, the surface itself needs FIC.

REFERENCES. One set of orbitals serves both states: the 3A" orbitals from a
4-root state-averaged CASSCF, built exactly as in 07 (fixed CAS(10,6),
AVAS-style canonicalisation, Cs forced). The singlet is a CASCI on those same
orbitals rather than its own CASSCF. That is cheaper (07's singlet CASSCF took
~103 s of each ~131 s point) and gives both states the same orbitals, so the
SC/FIC difference is not mixed up with orbital relaxation.

NO SYMMETRY INSIDE THE NEVPT2 STEP. Every Prism example runs without molecular
symmetry, and its interface's handling of symmetry could not be inspected.
Rather than depend on it, Cs is used only to OPTIMISE the orbitals. The CASCI
references that both NEVPT2 variants read are then rebuilt on a copy of the
molecule with symmetry off -- identical atoms (in bohr, so the Cs frame is
kept), identical basis. Three checks guard this, each fatal to the point if
it fails:

  - the AO overlap matrices of the two molecules agree, so the orbitals
    transfer;
  - the unsymmetric triplet CASCI root 0 reproduces the symmetry-forced 3A"
    root 0 energy, so it is the same state and not a lower 3A';
  - Prism's reference energy (e_tot - e_corr) reproduces the CASCI energy it
    was given, so Prism used this reference and not one of its own.

ALIASING. PySCF's NEVPT rewrites the CASCI object's CI vector in place (this
corrupted 06's dipoles). So Prism and PySCF each get their own CASCI object,
diagonalised separately from the same orbitals, and the two are checked to
agree before either NEVPT2 runs.

Usage:
    python 08_nevpt2_contraction.py 2>&1 | tee logs/nevpt2_contraction.log
    python 08_nevpt2_contraction.py --kmin 0 --kmax 0     # equilibrium only
"""
import argparse
import csv
import importlib.util
import os
import time
import traceback

import numpy as np

from pyscf import gto, scf, mcscf, mrpt, fci

# The active-space machinery is 07's, loaded from that file so both scripts
# select, canonicalise and symmetrise orbitals with the same code. (Script
# names start with a digit, so a plain import is not possible.)
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "fc07", os.path.join(_here, "07_fc_active_space.py"))
fc07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fc07)

HARTREE2EV = fc07.HARTREE2EV
R_OCL_EQ_ANG = fc07.R_OCL_EQ_ANG
REF_03_EV, REF_04_EV = fc07.REF_03_EV, fc07.REF_04_EV
REF_MEAN_EV = 0.5 * (REF_03_EV + REF_04_EV)

TOL_SAME_STATE_HA = 1e-6
SLOPE_TOL = 0.03          # SC/FIC slopes within 3% count as the same shape


def nosym_copy(mol, spin):
    """Same atoms, same frame, same basis, symmetry off."""
    return gto.M(atom=mol._atom, unit="Bohr", basis=mol.basis, spin=spin,
                 charge=0, symmetry=False, verbose=mol.verbose,
                 max_memory=mol.max_memory)


def casci_on(mf, ncas, nelecas, mo, spin):
    """Single-root CASCI on given orbitals, spin held fixed."""
    mc = mcscf.CASCI(mf, ncas, nelecas)
    S = spin / 2.0
    fci.addons.fix_spin_(mc.fcisolver, ss=S * (S + 1.0))
    mc.verbose = 0
    mc.kernel(mo)
    return mc


def fic_nevpt2(mf, mc, verbose):
    """Prism FIC-NEVPT2 on a single-root CASCI. Returns (e_tot, e_ref)."""
    import prism.interface
    import prism.nevpt

    interface = prism.interface.PYSCF(mf, mc, backend="opt_einsum")
    nevpt = prism.nevpt.NEVPT(interface)
    # Settings from Prism's own SC-vs-FIC example.
    nevpt.compute_singles_amplitudes = False
    nevpt.s_thresh_singles = 1e-10
    nevpt.s_thresh_doubles = 1e-10
    nevpt.verbose = verbose
    e_tot, e_corr, _ = nevpt.kernel()
    e_tot = float(np.atleast_1d(e_tot)[0])
    e_corr = float(np.atleast_1d(e_corr)[0])
    return e_tot, e_tot - e_corr


def run_geometry(r, args):
    row = {"r": r}
    t0, c0 = time.time(), time.process_time()
    laps = {"t": time.time()}

    def lap(name):
        now = time.time()
        row[f"t_{name}"] = now - laps["t"]
        laps["t"] = now

    n_occ, n_vir = args.n_occ, args.n_vir

    # ---- orbitals: 07's fixed CAS(10,6), then the 3A" SA-CASSCF, Cs forced
    mol_s = fc07.build_mol(r, 0, args.basis, args.verbose)
    mf_s = scf.RHF(mol_s).x2c()
    mf_s.conv_tol = 1e-10
    mf_s.kernel()
    proj = fc07.Projector(mol_s, args.avas_aos, args.minao)
    mo, ncore, ncas, nelecas = fc07.select_fixed(mf_s, proj, n_occ, n_vir)
    mo = fc07.canonicalize_like_avas(mol_s, mf_s, mo, ncore, ncas)
    mo = fc07.symmetrize_blocks(mol_s, mo, mf_s.get_ovlp(), ncore, ncas)
    row["space"] = f"({nelecas},{ncas})"

    mol_t = fc07.build_mol(r, 2, args.basis, args.verbose)
    mf_t = scf.ROHF(mol_t).x2c()
    mf_t.conv_tol = 1e-10
    mf_t.kernel()
    nt = args.nroots_triplet
    mc_t = mcscf.CASSCF(mf_t, ncas, nelecas)
    mc_t.fcisolver = fc07.triplet_solver(mol_t, nt)
    if nt > 1:
        mc_t = mc_t.state_average_(np.ones(nt) / nt)
    mc_t.conv_tol, mc_t.max_cycle_macro, mc_t.verbose = 1e-8, args.max_cycle, 0
    mc_t.kernel(mo)
    row["conv_t"] = bool(mc_t.converged)
    mo_t = mc_t.mo_coeff
    # the symmetry-forced 3A" root 0 energy, as the identity target
    ci_sym = mcscf.CASCI(mf_t, ncas, nelecas)
    ci_sym.fcisolver, ci_sym.verbose = fc07.triplet_solver(mol_t, nt), 0
    ci_sym.kernel(mo_t)
    e_target = float(np.atleast_1d(ci_sym.e_tot)[0])
    lap("orbitals")

    # ---- symmetry-free copies for the NEVPT2 step
    mol_tn, mol_sn = nosym_copy(mol_t, 2), nosym_copy(mol_s, 0)
    mf_tn = scf.ROHF(mol_tn).x2c()
    mf_tn.conv_tol = 1e-10
    mf_tn.kernel()
    mf_sn = scf.RHF(mol_sn).x2c()
    mf_sn.conv_tol = 1e-10
    mf_sn.kernel()
    if not np.allclose(mf_tn.get_ovlp(), mf_t.get_ovlp(), atol=1e-10):
        raise RuntimeError("AO overlap differs between Cs and C1 copies: "
                           "orbitals would not transfer")
    lap("copies")

    # Separate CASCI objects for Prism and PySCF (PySCF's NEVPT mutates ci).
    t_fic = casci_on(mf_tn, ncas, nelecas, mo_t, 2)
    t_sc = casci_on(mf_tn, ncas, nelecas, mo_t, 2)
    s_fic = casci_on(mf_sn, ncas, nelecas, mo_t, 0)
    s_sc = casci_on(mf_sn, ncas, nelecas, mo_t, 0)
    row["e_cas_t"], row["e_cas_s"] = float(t_fic.e_tot), float(s_fic.e_tot)
    row["same_state_ha"] = abs(row["e_cas_t"] - e_target)
    if row["same_state_ha"] > TOL_SAME_STATE_HA:
        raise RuntimeError(
            f"unsymmetric triplet root 0 is {row['same_state_ha']:.2e} Ha from "
            f"the 3A\" root 0: a different state, not comparable")
    if (abs(t_fic.e_tot - t_sc.e_tot) > 1e-8
            or abs(s_fic.e_tot - s_sc.e_tot) > 1e-8):
        raise RuntimeError("the duplicate CASCI objects disagree")
    lap("casci")

    # ---- FIC first (Prism), then SC (PySCF)
    row["e_fic_t"], ref_t = fic_nevpt2(mf_tn, t_fic, args.prism_verbose)
    row["e_fic_s"], ref_s = fic_nevpt2(mf_sn, s_fic, args.prism_verbose)
    row["prism_ref_dev_ha"] = max(abs(ref_t - row["e_cas_t"]),
                                  abs(ref_s - row["e_cas_s"]))
    lap("fic")
    row["e_sc_t"] = row["e_cas_t"] + float(mrpt.NEVPT(t_sc).kernel())
    row["e_sc_s"] = row["e_cas_s"] + float(mrpt.NEVPT(s_sc).kernel())
    lap("sc")

    row["de_cas"] = (row["e_cas_t"] - row["e_cas_s"]) * HARTREE2EV
    row["de_sc"] = (row["e_sc_t"] - row["e_sc_s"]) * HARTREE2EV
    row["de_fic"] = (row["e_fic_t"] - row["e_fic_s"]) * HARTREE2EV
    row["wall"], row["cpu"] = time.time() - t0, time.process_time() - c0
    return row


def slope_ev_per_ang(rows, key):
    """Least-squares d(value)/dr over the rows, value in eV."""
    r = np.array([x["r"] for x in rows])
    v = np.array([x[key] for x in rows])
    if key.startswith("e_"):
        v = v * HARTREE2EV
    return float(np.polyfit(r - r.mean(), v - v.mean(), 1)[0])


def verdict(rows, args):
    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    bad_ref = [x["r"] for x in rows if x["prism_ref_dev_ha"] > 1e-6]
    if bad_ref:
        print(f"  !! Prism's reference energy differs from the CASCI it was "
              f"given at r = {', '.join(f'{v:.4f}' for v in bad_ref)} A.")
        print("     Prism did not use this reference there; its FIC numbers "
              "are not comparable.")
    unconv = [x["r"] for x in rows if not x["conv_t"]]
    if unconv:
        print(f"  !! triplet SA-CASSCF unconverged at r = "
              f"{', '.join(f'{v:.4f}' for v in unconv)} A")

    eq = next((x for x in rows if abs(x["r"] - R_OCL_EQ_ANG) < 1e-6), None)
    if eq is not None:
        print(f"\n  OFFSET at equilibrium (vertical a 3A\" <- X 1A'):")
        print(f"    CASCI      {eq['de_cas']:.4f} eV")
        print(f"    SC-NEVPT2  {eq['de_sc']:.4f} eV   "
              f"(07, own singlet CASSCF: 3.5502)")
        print(f"    FIC-NEVPT2 {eq['de_fic']:.4f} eV")
        print(f"    references 03 {REF_03_EV:.4f}, 04 {REF_04_EV:.4f}, "
              f"mean {REF_MEAN_EV:.4f} eV")
        off_sc, off_fic = eq["de_sc"] - REF_MEAN_EV, eq["de_fic"] - REF_MEAN_EV
        print(f"    SC {off_sc:+.3f} eV and FIC {off_fic:+.3f} eV from the "
              f"reference mean; FIC - SC = {eq['de_fic'] - eq['de_sc']:+.3f} eV")
        if abs(off_fic) < 0.05 and abs(off_sc) > 0.08:
            print("    -> The contraction explains the offset: FIC lands on "
                  "the references, SC does not.")
        elif abs(eq["de_fic"] - eq["de_sc"]) < 0.03:
            print("    -> The contraction does NOT explain it: SC and FIC "
                  "agree, so the offset comes\n       from the reference "
                  "(active space, orbitals) rather than the perturbation "
                  "theory.")
        else:
            print("    -> The contraction accounts for part of it: FIC moves "
                  "towards the references\n       but the remainder "
                  "has another source.")

    if len(rows) >= 2:
        s_sc = slope_ev_per_ang(rows, "e_sc_t")
        s_fic = slope_ev_per_ang(rows, "e_fic_t")
        d_sc = slope_ev_per_ang(rows, "de_sc")
        d_fic = slope_ev_per_ang(rows, "de_fic")
        ratio = s_fic / s_sc if s_sc else float("nan")
        span = f"{rows[0]['r']:.3f}-{rows[-1]['r']:.3f} A"
        print(f"\n  SLOPE of the triplet energy over {span} "
              f"(band width scales with it):")
        print(f"    SC  {s_sc:+.3f} eV/A    FIC {s_fic:+.3f} eV/A    "
              f"FIC/SC = {ratio:.3f}")
        print(f"    (vertical-energy slope: SC {d_sc:+.3f}, "
              f"FIC {d_fic:+.3f} eV/A)")
        if abs(ratio - 1.0) < SLOPE_TOL:
            print(f"    -> Same shape within {SLOPE_TOL:.0%}. The SC surface "
                  f"is fine for the band width;\n       only its absolute "
                  f"position needs correcting (dCCSD(T) splice or a shift).")
        else:
            print(f"    -> Shapes differ by {abs(ratio - 1.0):.1%}, beyond "
                  f"{SLOPE_TOL:.0%}. A band computed on the SC\n       surface "
                  f"would be that much too wide or narrow: the surface "
                  f"itself needs FIC.")
    else:
        print("\n  SLOPE: needs at least two geometries (use the default "
              "kmin/kmax).")

    t_fic = float(np.mean([x["t_fic"] for x in rows]))
    t_sc = float(np.mean([x["t_sc"] for x in rows]))
    t_orb = float(np.mean([x["t_orbitals"] for x in rows]))
    print(f"\n  COST per geometry, wall: orbitals {t_orb:.0f} s, FIC {t_fic:.0f} s "
          f"(two states), SC {t_sc:.0f} s (two states)")
    print("  For a raster only the triplet is needed per point, so roughly "
          "half of each NEVPT2 figure.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--avas-aos", nargs="+", default=["Cl 3p", "O 2p"])
    p.add_argument("--minao", default="ano")
    p.add_argument("--n-occ", type=int, default=5)
    p.add_argument("--n-vir", type=int, default=1)
    p.add_argument("--nroots-triplet", type=int, default=4)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--kmin", type=int, default=-1)
    p.add_argument("--kmax", type=int, default=1)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--csv", default="hocl_nevpt2_contraction.csv")
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--prism-verbose", type=int, default=0)
    args = p.parse_args()

    ks = list(range(args.kmin, args.kmax + 1))
    rs = [R_OCL_EQ_ANG + k * args.step for k in ks]

    print("=" * 78)
    print("== HOX Phase 0.5 -- SC vs FIC NEVPT2 on identical references")
    print(f"== CAS({2 * args.n_occ},{args.n_occ + args.n_vir}), {args.basis}, "
          f"triplet orbitals from a {args.nroots_triplet}-root 3A\" SA-CASSCF")
    print(f"== r = {', '.join(f'{r:.4f}' for r in rs)} A")
    print("=" * 78)
    print(f"\n  {'r/A':>7}{'dE CAS':>9}{'dE SC':>9}{'dE FIC':>9}{'FIC-SC':>9}"
          f"{'same-state':>12}{'prism ref':>11}{'t/s':>6}")

    rows = []
    for r in rs:
        try:
            row = run_geometry(r, args)
        except Exception as exc:
            print(f"  {r:7.4f}  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        rows.append(row)
        print(f"  {r:7.4f}{row['de_cas']:9.4f}{row['de_sc']:9.4f}"
              f"{row['de_fic']:9.4f}{row['de_fic'] - row['de_sc']:+9.4f}"
              f"{row['same_state_ha']:12.1e}{row['prism_ref_dev_ha']:11.1e}"
              f"{row['wall']:6.0f}")
        print(f"  {'':7}  stages/s: orbitals {row['t_orbitals']:.0f}, "
              f"C1 copies {row['t_copies']:.0f}, CASCI {row['t_casci']:.0f}, "
              f"FIC {row['t_fic']:.0f}, SC {row['t_sc']:.0f}"
              f"{'' if row['conv_t'] else '   !! triplet SA-CASSCF unconverged'}")

    if not rows:
        print("\n  no geometry succeeded")
        return

    cols = ["r", "space", "conv_t", "e_cas_s", "e_cas_t", "e_sc_s", "e_sc_t",
            "e_fic_s", "e_fic_t", "de_cas", "de_sc", "de_fic",
            "same_state_ha", "prism_ref_dev_ha", "wall", "cpu",
            "t_orbitals", "t_copies", "t_casci", "t_fic", "t_sc"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(f"\n  wrote {args.csv}")
    verdict(rows, args)


if __name__ == "__main__":
    main()
