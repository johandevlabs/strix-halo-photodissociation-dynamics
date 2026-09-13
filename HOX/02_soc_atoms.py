#!/usr/bin/env python3
"""
Phase 0.5, step 3: validate the SOC toolchain on halogen fine structure.

Why this test rather than reproducing a table from the Prism paper: the
halogen ground term is 2P, and spin-orbit coupling splits it into 2P_3/2 and
2P_1/2 with a separation that is known experimentally to six figures. It is
the cheapest possible calculation that exercises exactly the machinery HOX
needs -- the atomic SOC on the halogen is what lends the a 3A" band its
intensity in the first place. Cl, Br and I in one run also show how the cost
and the error scale with nuclear charge, which is the thing that decides
whether HOBr and HOI are affordable.

Three independent checks come out of one calculation:

  1. DEGENERACY. Six spin-orbit states must come out as 4 + 2. If they do
     not, the state-interaction step is wrong regardless of the splitting.
  2. ORDERING. The halogen multiplet is INVERTED (p^5, more than half full),
     so 2P_3/2 lies below 2P_1/2. Getting the sign right is a real check.
  3. MAGNITUDE against experiment.

Expected accuracy: a few percent on Cl, worse on I where the treatment of
scalar relativity and basis quality starts to matter. An error of 30%+, or a
broken degeneracy pattern, means the toolchain is misconfigured rather than
merely approximate.

Also reports wall time per atom, which is the input to the raster cost
estimate that decides Phase 1.

Usage:
    python 02_soc_atoms.py 2>&1 | tee logs/soc_atoms.log
    python 02_soc_atoms.py --atoms Cl Br --basis ano-rcc-vdzp
    python 02_soc_atoms.py --atoms Cl --no-nevpt2   # CASSCF-level SOC only
"""
import argparse
import time
import traceback

import numpy as np

from pyscf import gto, scf, mcscf
from pyscf import fci

HARTREE2CM = 219474.6313702

# Experimental 2P_1/2 - 2P_3/2 separations, cm^-1, ground configuration np^5.
# NIST Atomic Spectra Database values, quoted from memory during the work --
# CHECK THESE against NIST ASD before they appear in anything that matters.
# (Same caveat as the observed vibrational fundamentals in water/README.md.)
FINE_STRUCTURE_CM = {
    "F":  404.14,
    "Cl": 882.35,
    "Br": 3685.24,
    "I":  7602.97,
}

# Default basis per atom. ANO-RCC is an all-electron relativistic set, which
# is what X2CAMF wants; ECP sets are NOT appropriate here because the SOC
# integrals need the core.
DEFAULT_BASIS = "ano-rcc-vdzp"


def build_casscf(atom, basis, x2c_scalar=True, verbose=0):
    """SA-CASSCF(5e,3o) over the three components of the 2P term.

    The p^5 shell gives one hole in three p orbitals: CAS(5,3). The 2P term
    is spatially threefold degenerate, so state-averaging over three doublet
    roots is the correct reference for a subsequent state interaction. Spin
    is constrained explicitly -- without it the solver is free to return a
    quartet and every SOC matrix element involving it is meaningless. Same
    trap as the water pipeline (see water/README.md Gotchas).
    """
    mol = gto.M(atom=f"{atom} 0 0 0", basis=basis, spin=1, charge=0,
                symmetry=False, verbose=verbose)

    mf = scf.ROHF(mol)
    if x2c_scalar:
        mf = mf.x2c()          # scalar relativity; SOC comes from X2CAMF later
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{atom}: ROHF did not converge")

    mc = mcscf.CASSCF(mf, 3, 5)
    mc.fcisolver = fci.direct_spin1.FCI(mol)
    fci.addons.fix_spin_(mc.fcisolver, ss=0.75)      # doublet: S(S+1) = 3/4
    mc.state_average_([1.0 / 3] * 3)
    mc.kernel()
    if not mc.converged:
        print(f"  WARNING: {atom}: CASSCF did not converge")
    return mol, mf, mc


# ---------------------------------------------------------------------------
#  ADAPT THIS AFTER 01_soc_probe.py
# ---------------------------------------------------------------------------
#  Everything above and below is API-independent and should be correct as
#  written. The call sequence inside run_soc() is the ONE piece written
#  without having seen Prism's API, inferred from its documentation:
#
#     "State-interaction spin-orbit coupling with Breit-Pauli and exact
#      two-component Douglas-Kroll-Hess Hamiltonians" for NEVPT2/QD-NEVPT2
#
#  Replace the body with whatever prism/examples/soc/*.py actually shows.
#  Keep the return contract: (energies_hartree, info_dict).
# ---------------------------------------------------------------------------
def run_soc(mol, mf, mc, use_nevpt2=True):
    """Return spin-orbit state energies in hartree, lowest first."""
    import prism
    import prism.interface

    interface = prism.interface.PYSCF(mf, mc, opt_einsum=True)

    if use_nevpt2:
        import prism.qdnevpt as method_mod
        method = method_mod.QDNEVPT(interface)
    else:
        import prism.nevpt as method_mod
        method = method_mod.NEVPT(interface)

    # Ask for the state interaction. The attribute name is the guess most
    # likely to be wrong; if it raises, the except branch prints every public
    # knob on the object so the fix is one edit rather than another round trip.
    method.compute_soc = True

    try:
        energies = method.kernel()
    except Exception:
        print("\n  !! method.kernel() failed. Public attributes on "
              f"{type(method).__name__}:")
        for attr in sorted(a for a in dir(method) if not a.startswith("_")):
            print(f"       {attr}")
        raise

    energies = np.atleast_1d(np.asarray(energies, dtype=float)).ravel()
    return np.sort(energies), {"method": type(method).__name__}


def analyse(atom, energies):
    """Degeneracy, ordering and magnitude checks on the SOC manifold."""
    n = len(energies)
    print(f"  {n} spin-orbit states:")
    rel = (energies - energies[0]) * HARTREE2CM
    for i, e in enumerate(rel):
        print(f"      {i}  {energies[i]:18.10f} Ha   {e:12.2f} cm-1")

    if n < 6:
        print(f"  !! expected 6 spin-orbit states from a 2P term, got {n}."
              " The state interaction is not doing what we think.")
        return None

    # Group by energy. A correct 2P manifold is 4 (J=3/2) then 2 (J=1/2).
    tol = 1.0                                   # cm^-1
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
        print("     If it is [2, 4] the multiplet came out normal rather than")
        print("     inverted -- check the sign convention and that the hole,")
        print("     not the electron, is being described.")
        return None

    split = float(np.mean(groups[1]) - np.mean(groups[0]))
    return split


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--atoms", nargs="+", default=["Cl", "Br"],
                   help="halogens to run (default: Cl Br). Add I once Br "
                        "works; drop to Cl alone for the fastest check.")
    p.add_argument("--basis", default=DEFAULT_BASIS,
                   help=f"all-electron relativistic basis (default "
                        f"{DEFAULT_BASIS}). Do NOT use an ECP set: X2CAMF "
                        f"needs the core.")
    p.add_argument("--no-nevpt2", action="store_true",
                   help="state interaction over CASSCF only, skipping the "
                        "NEVPT2 correction. Faster, and isolates whether a "
                        "failure is in the SOC step or the perturbation.")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    print("=" * 72)
    print("== HOX Phase 0.5 -- halogen fine structure as an SOC validation")
    print(f"== basis {args.basis}, "
          f"{'CASSCF-level SOC' if args.no_nevpt2 else 'QD-NEVPT2 + SOC'}")
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

            energies, info = run_soc(mol, mf, mc,
                                     use_nevpt2=not args.no_nevpt2)
            split = analyse(atom, energies)
            results[atom] = split
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
        obs = FINE_STRUCTURE_CM.get(atom)
        calc = results.get(atom)
        if calc is None:
            print(f"{atom:>6}{'--':>14}{obs:12.2f}{'--':>10}"
                  f"{timings.get(atom, float('nan')):10.1f}")
            continue
        err = 100.0 * (calc - obs) / obs
        print(f"{atom:>6}{calc:14.2f}{obs:12.2f}{err:9.1f}%"
              f"{timings[atom]:10.1f}")

    print("""
Reading the result:
  within a few %      toolchain is sound; proceed to the molecular test.
  right sign, ~20-30% off on I only   expected; scalar relativity and basis
                      quality, not a broken pipeline.
  wrong degeneracy pattern, or sign   misconfigured. Do not build on it.

The wall times feed the Phase 1 go/no-go. A 3D HOCl raster is ~3000 points,
so a molecular SO-QDNEVPT2 point of t seconds costs roughly 3000*t/30 s of
wall clock across 30 processes. Atoms are much cheaper than the molecule --
03 measures the real thing -- but if Br is already slow here, that is the
signal to fall back to SOMF-QDNEVPT2 or geometry-independent SOC.
""")


if __name__ == "__main__":
    main()
