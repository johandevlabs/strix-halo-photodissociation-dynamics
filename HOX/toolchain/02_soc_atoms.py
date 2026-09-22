#!/usr/bin/env python3
"""
Phase 0.5, step 3: validate the SOC toolchain on halogen fine structure.

Why this test rather than reproducing a table from the Prism paper: the
halogen ground term is 2P, and spin-orbit coupling splits it into 2P_3/2 and
2P_1/2 with a separation known experimentally to six figures. It is the
cheapest calculation that exercises exactly the machinery HOX needs -- the
atomic SOC on the halogen is what lends the a 3A" band its intensity in the
first place. Cl and Br in one run also show how cost and error scale with
nuclear charge, which is what decides whether HOBr and HOI are affordable.

Three independent checks come out of one calculation:

  1. DEGENERACY. Six spin-orbit states must come out as 4 + 2. If they do
     not, the state interaction is wrong regardless of the splitting.
  2. ORDERING. The halogen multiplet is INVERTED (p^5 is one hole, more than
     half full), so 2P_3/2 lies BELOW 2P_1/2. Getting the sign right is a
     real check, not a formality.
  3. MAGNITUDE against experiment.

Expected accuracy: a few percent on Cl, worse on Br and worse again on I,
where basis quality and the treatment of scalar relativity start to dominate.
A 30%+ error, a broken degeneracy pattern, or an inverted sign means the
toolchain is misconfigured rather than merely approximate.

The API follows prism/examples/soc/01-qdnevpt2-SOC.py verbatim; the return
signature (e_tot, e_corr, osc) is from 04-qdnevpt2-SUS.py. Oscillator
strengths coming back from kernel() is the thing HOX actually depends on, so
this prints them even though they are not meaningful for a free atom.

Usage:
    python 02_soc_atoms.py 2>&1 | tee logs/soc_atoms.log
    python 02_soc_atoms.py --atoms Cl              # fastest single check
    python 02_soc_atoms.py --atoms Cl --no-nevpt2  # CASSCF-level SOC only
    python 02_soc_atoms.py --soc breit-pauli
"""
import argparse
import time
import traceback

import numpy as np

from pyscf import gto, scf, mcscf, fci

HARTREE2CM = 219474.6313702

# Experimental 2P_1/2 - 2P_3/2 separations, cm^-1, ground configuration np^5.
# NIST Atomic Spectra Database values, quoted from memory during the work --
# CHECK THESE against NIST ASD before they appear anywhere that matters.
# (Same caveat as the observed vibrational fundamentals in water/README.md.)
FINE_STRUCTURE_CM = {
    "F":  404.14,
    "Cl": 882.35,
    "Br": 3685.24,
    "I":  7602.97,
}

# def2-TZVP is what every shipped Prism SOC example uses, and it is
# all-electron through Kr -- fine for Cl and Br. For I it carries an ECP,
# which X2CAMF cannot use because the SOC integrals need the core; the run
# checks mol.has_ecp() rather than trusting the basis name.
DEFAULT_BASIS = "def2-tzvp"


def build_casscf(atom, basis, ncas=3, nelecas=5, nstates=3, verbose=0):
    """SA-CASSCF over the three spatial components of the 2P term.

    p^5 is one hole in three p orbitals: CAS(5e,3o). The 2P term is spatially
    threefold degenerate, so averaging three doublet roots is the correct
    reference for a subsequent state interaction.

    Note the spin constraint here is belt-and-braces rather than load-bearing:
    five electrons in three orbitals is a single hole, so only doublets exist
    and the quartet the water pipeline had to guard against cannot arise. It
    is kept because this function is the template for the molecular case,
    where it very much is load-bearing (water/README.md Gotchas).
    """
    mol = gto.M(atom=f"{atom} 0 0 0", basis=basis, spin=1, charge=0,
                symmetry=False, verbose=verbose)
    if mol.has_ecp():
        raise RuntimeError(
            f"{atom}/{basis} uses an ECP. X2CAMF needs the core, so the SOC "
            f"integrals would be wrong. Use an all-electron relativistic set "
            f"(ANO-RCC, dyall, sapporo -- most need "
            f"`pip install basis-set-exchange`).")

    mf = scf.ROHF(mol).x2c()        # scalar relativity; SOC added by Prism
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{atom}: ROHF did not converge")

    weights = np.ones(nstates) / nstates
    mc = mcscf.CASSCF(mf, ncas, nelecas).state_average_(weights)
    fci.addons.fix_spin_(mc.fcisolver, ss=0.75)          # doublet, S(S+1)=3/4
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-6
    mc.mc1step()
    if not mc.converged:
        print(f"  WARNING: {atom}: CASSCF did not converge")
    return mol, mf, mc


def _unpack(res):
    """Prism's kernel returns (e_tot, e_corr, osc); run_soc may differ."""
    osc = None
    if isinstance(res, tuple):
        if len(res) == 3:
            e_tot, _, osc = res
        else:
            e_tot = res[0]
    else:
        e_tot = res
    return np.atleast_1d(np.asarray(e_tot, dtype=float)).ravel(), osc


def run_soc(mf, mc, soc="DKH1", use_nevpt2=True, verbose=4):
    """State-interaction SOC, returning (energies_hartree, oscillator_strengths).

    Call sequence from prism/examples/soc/. Two routes:
      use_nevpt2=True   QD-NEVPT2 then state interaction (the real method)
      use_nevpt2=False  state interaction over CASSCF only -- much cheaper,
                        and isolates whether a failure is in the SOC step or
                        in the perturbation on top of it.
    """
    import prism.interface
    import prism.nevpt

    interface = prism.interface.PYSCF(mf, mc, backend="opt_einsum")

    if not use_nevpt2:
        interface.soc = soc
        return _unpack(interface.run_soc())

    nevpt = prism.nevpt.NEVPT(interface)
    nevpt.method = "nevpt2"
    nevpt.method_type = "qd"          # quasi-degenerate: required for SOC
    nevpt.soc = soc
    nevpt.verbose = verbose
    return _unpack(nevpt.kernel())


def analyse(atom, energies):
    """Degeneracy, ordering and magnitude checks on the SOC manifold."""
    n = len(energies)
    rel = (energies - energies[0]) * HARTREE2CM
    print(f"  {n} spin-orbit states:")
    for i, e in enumerate(rel):
        print(f"      {i}  {energies[i]:18.10f} Ha   {e:12.2f} cm-1")

    if n < 6:
        print(f"  !! expected 6 spin-orbit states from a 2P term, got {n}. "
              f"The state interaction is not doing what we think.")
        return None

    tol = 1.0                                    # cm^-1
    groups, current = [], [rel[0]]
    for e in rel[1:]:
        if abs(e - current[-1]) < tol:
            current.append(e)
        else:
            groups.append(current)
            current = [e]
    groups.append(current)
    pattern = [len(g) for g in groups]
    print(f"  degeneracy pattern: {pattern}   (want [4, 2])")

    if pattern[:2] != [4, 2]:
        print("  !! NOT the 4+2 pattern of an inverted 2P term.")
        if pattern[:2] == [2, 4]:
            print("     [2, 4] means the multiplet came out NORMAL rather than")
            print("     inverted -- the J=1/2 pair is below the J=3/2 quartet.")
            print("     For p^5 that is wrong: check the sign of the SOC term.")
        return None

    return float(np.mean(groups[1]) - np.mean(groups[0]))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", nargs="+", default=["Cl", "Br"],
                   help="halogens to run (default: Cl Br). I needs an "
                        "all-electron basis; def2 carries an ECP there.")
    p.add_argument("--basis", default=DEFAULT_BASIS,
                   help=f"all-electron relativistic basis (default "
                        f"{DEFAULT_BASIS}, as used by every Prism SOC example)")
    p.add_argument("--soc", default="DKH1",
                   help="SOC Hamiltonian: DKH1 (= x2c-1, matches the .x2c() "
                        "mean field) or breit-pauli. Running both and "
                        "comparing is itself a check.")
    p.add_argument("--no-nevpt2", action="store_true",
                   help="state interaction over CASSCF only, skipping the "
                        "QD-NEVPT2 correction")
    p.add_argument("--verbose", type=int, default=0,
                   help="pyscf verbosity (prism is told 4 regardless)")
    args = p.parse_args()

    print("=" * 72)
    print("== HOX Phase 0.5 -- halogen fine structure as an SOC validation")
    print(f"== basis {args.basis}, soc {args.soc}, "
          f"{'CASSCF state interaction' if args.no_nevpt2 else 'QD-NEVPT2 + SOC'}")
    print("=" * 72)

    results, timings = {}, {}
    for atom in args.atoms:
        print(f"\n--- {atom} " + "-" * 60)
        t0 = time.time()
        try:
            mol, mf, mc = build_casscf(atom, args.basis, verbose=args.verbose)
            t_ref = time.time() - t0
            print(f"  reference done in {t_ref:7.1f} s "
                  f"(nao {mol.nao_nr()}, CAS(5,3), 3 roots)")

            energies, osc = run_soc(mf, mc, soc=args.soc,
                                    use_nevpt2=not args.no_nevpt2)
            if osc is not None:
                arr = np.atleast_1d(np.asarray(osc)).ravel()
                print(f"  oscillator strengths returned: {arr.size} values, "
                      f"max {np.max(np.abs(arr)):.3e}")
                print("      (meaningless for a free atom -- what matters is "
                      "that kernel() returns them at all, since that is the "
                      "borrowed intensity HOX needs)")
            results[atom] = analyse(atom, energies)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[atom] = None
        timings[atom] = time.time() - t0
        print(f"  wall time {timings[atom]:7.1f} s")

    print("\n" + "=" * 72)
    print("== fine structure 2P_1/2 - 2P_3/2")
    print("=" * 72)
    print(f"{'atom':>6}{'calc/cm-1':>14}{'obs/cm-1':>12}{'err':>10}{'wall/s':>10}")
    for atom in args.atoms:
        obs = FINE_STRUCTURE_CM.get(atom, float("nan"))
        calc = results.get(atom)
        wall = timings.get(atom, float("nan"))
        if calc is None:
            print(f"{atom:>6}{'--':>14}{obs:12.2f}{'--':>10}{wall:10.1f}")
        else:
            print(f"{atom:>6}{calc:14.2f}{obs:12.2f}"
                  f"{100.0 * (calc - obs) / obs:9.1f}%{wall:10.1f}")

    print("""
Reading the result:
  within a few %          toolchain is sound; go on to the molecular test.
  right sign, 20-30% off  plausible for Br at this basis -- basis quality and
                          scalar relativity, not a broken pipeline.
  wrong degeneracy, or
  wrong sign              misconfigured. Do not build on it.

The wall times feed the Phase 1 go/no-go, but only weakly: an atom is far
cheaper than a triatomic, and the raster cost is set by the molecule.
03_soc_hocl_vertical.py measures the real thing. What these times DO tell you
is how steeply cost climbs from Cl to Br, which is the relevant gradient for
whether HOBr and eventually HOI stay affordable.
""")


if __name__ == "__main__":
    main()
