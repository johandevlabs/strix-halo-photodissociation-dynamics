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
    python 02_soc_atoms.py --sweep --atoms Br          # which basis for HOBr
    python 02_soc_atoms.py --sweep --atoms Br --soc DKH1 breit-pauli
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

# Candidates for the sweep, in rough order of expected cost. The Phase 0.5
# finding is that def2-TZVP is all-electron for Br but NON-RELATIVISTICALLY
# CONTRACTED, and the SOC operator samples precisely the near-nuclear region
# that contraction gets wrong -- decontracting it recovered more than half the
# error. So the sweep is mostly about what a basis does near the nucleus:
#
#   unc-*            the same functions, contraction removed (the 7.6% fix)
#   x2c-*all         Pollak-Weigend, contracted FOR an x2c Hamiltonian, which
#                    is the one we actually run
#   *-dk / *-dkh     contracted for Douglas-Kroll-Hess
#   ano-rcc, dyall   built for relativistic all-electron work from the start
#
# Availability differs between PySCF's bundled library and basis-set-exchange,
# and neither is worth guessing at from a laptop -- the sweep probes each name
# and reports where it came from, so one run on the EVO settles it.
CANDIDATE_BASES = [
    "def2-tzvp",            # the baseline, and what every Prism example uses
    "unc-def2-tzvp",        # Phase 0.5: -16.6% -> -7.6%
    "def2-qzvp",
    "unc-def2-qzvp",
    "x2c-tzvpall",
    "x2c-qzvpall",
    "cc-pvtz-dk",
    "aug-cc-pvtz-dk",
    "sapporo-dkh3-tzp",
    "ano-rcc",
    "dyall-v3z",
    "jorge-tzp-dkh",
]


class BasisUnavailable(Exception):
    """The name is not in PySCF's library and basis-set-exchange cannot help."""


def resolve_basis(atom, name):
    """Return (basis_for_gto, where_it_came_from).

    PySCF's bundled library first, then basis-set-exchange if installed.
    Raises BasisUnavailable rather than letting a sweep die on one name.
    """
    try:
        gto.M(atom=f"{atom} 0 0 0", basis=name, spin=1, verbose=0)
        return name, "pyscf"
    except BasisUnavailable:
        raise
    except Exception as exc:
        first = exc

    try:
        import basis_set_exchange as bse
    except ImportError:
        raise BasisUnavailable(
            f"not in pyscf ({type(first).__name__}) and basis-set-exchange "
            f"is not installed (pip install basis-set-exchange)")

    try:
        txt = bse.get_basis(name, elements=[atom], fmt="nwchem",
                            header=False)
        parsed = {atom: gto.basis.parse(txt)}
        gto.M(atom=f"{atom} 0 0 0", basis=parsed, spin=1, verbose=0)
        return parsed, "bse"
    except Exception as exc:
        raise BasisUnavailable(
            f"pyscf: {type(first).__name__}; bse: {type(exc).__name__}: {exc}")


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


def measure(atom, basis, soc, use_nevpt2=True, verbose=0, quiet=False):
    """One (atom, basis, soc) point: returns (splitting_cm, nao, wall_s).

    splitting_cm is None if the degeneracy or ordering check failed, which is
    a different outcome from a crash and is reported as such.
    """
    t0 = time.time()
    mol, mf, mc = build_casscf(atom, basis, verbose=verbose)
    nao = mol.nao_nr()
    energies, osc = run_soc(mf, mc, soc=soc, use_nevpt2=use_nevpt2,
                            verbose=0 if quiet else 4)
    if not quiet and osc is not None:
        arr = np.atleast_1d(np.asarray(osc)).ravel()
        print(f"  oscillator strengths returned: {arr.size} values, "
              f"max {np.max(np.abs(arr)):.3e}")
        print("      (meaningless for a free atom -- what matters is that "
              "kernel() returns them at all, since that is the borrowed "
              "intensity HOX needs)")
    split = analyse(atom, energies) if not quiet else _quiet_analyse(energies)
    return split, nao, time.time() - t0


def _quiet_analyse(energies):
    """analyse() without the per-state dump, for the sweep's inner loop."""
    rel = (np.asarray(energies) - energies[0]) * HARTREE2CM
    if len(rel) < 6:
        return None
    groups, current = [], [rel[0]]
    for e in rel[1:]:
        if abs(e - current[-1]) < 1.0:
            current.append(e)
        else:
            groups.append(current)
            current = [e]
    groups.append(current)
    if [len(g) for g in groups][:2] != [4, 2]:
        return None
    return float(np.mean(groups[1]) - np.mean(groups[0]))


def sweep(atoms, bases, socs, use_nevpt2=True, verbose=0):
    """Basis sweep: which all-electron set reproduces the fine structure.

    Written for Br, where def2-TZVP is 16.6% low and the HOBr band exists ONLY
    through SOC borrowing, so this error propagates straight into the
    intensity. Fixing it at the atom costs minutes; discovering it in a 3D
    raster costs days.
    """
    rows = []
    for atom in atoms:
        obs = FINE_STRUCTURE_CM.get(atom, float("nan"))
        print(f"\n{'=' * 72}\n== {atom}: basis sweep against "
              f"{obs:.2f} cm-1 observed\n{'=' * 72}")
        for name in bases:
            try:
                basis, where = resolve_basis(atom, name)
            except BasisUnavailable as exc:
                print(f"  {name:20s} unavailable: {exc}")
                rows.append((atom, name, None, None, None, None, "unavailable"))
                continue
            for soc in socs:
                tag = f"{name} [{where}] {soc}"
                try:
                    split, nao, wall = measure(atom, basis, soc,
                                               use_nevpt2=use_nevpt2,
                                               verbose=verbose, quiet=True)
                except Exception as exc:
                    print(f"  {tag:40s} FAILED {type(exc).__name__}: {exc}")
                    rows.append((atom, name, soc, None, None, None,
                                 f"fail:{type(exc).__name__}"))
                    continue
                if split is None:
                    print(f"  {tag:40s} bad degeneracy/ordering -- not a "
                          f"usable number")
                    rows.append((atom, name, soc, None, nao, wall,
                                 "bad-pattern"))
                    continue
                err = 100.0 * (split - obs) / obs
                print(f"  {tag:40s} nao {nao:4d}  {split:9.2f} cm-1  "
                      f"{err:+6.1f}%  {wall:6.1f} s")
                rows.append((atom, name, soc, split, nao, wall, "ok"))
    return rows


def report_sweep(rows):
    """Rank by |error|, and say what it means for the molecular cost."""
    print("\n" + "=" * 72)
    print("== sweep summary, ranked by |error|")
    print("=" * 72)
    print(f"{'atom':>5} {'basis':>18} {'soc':>12} {'nao':>5} "
          f"{'calc/cm-1':>11} {'err':>8} {'wall/s':>8}")
    ok = [r for r in rows if r[6] == "ok"]
    for atom, name, soc, split, nao, wall, _ in sorted(
            ok, key=lambda r: abs(r[3] - FINE_STRUCTURE_CM[r[0]])):
        obs = FINE_STRUCTURE_CM[atom]
        print(f"{atom:>5} {name:>18} {soc:>12} {nao:>5} {split:11.2f} "
              f"{100.0 * (split - obs) / obs:+7.1f}% {wall:8.1f}")
    bad = [r for r in rows if r[6] != "ok"]
    if bad:
        print("\n  not usable:")
        for atom, name, soc, _, _, _, why in bad:
            print(f"    {atom:>3} {name:20s} {str(soc):12s} {why}")
    if not ok:
        return
    print("""
Choosing from this table. The winner is not simply the smallest error: the
basis has to be affordable in the MOLECULE, where nao roughly triples (H + O
on top of the halogen) and the raster is thousands of points. A set that is
2% better and 4x more expensive is the wrong trade for a band whose intensity
is already uncertain by an order of magnitude (03's f was 10-25x low for
HOCl). Take the cheapest basis inside a few percent, and record the rest as
the error bar on the SOC.

Comparing DKH1 against breit-pauli on the SAME basis separates the two error
sources: if they agree, the residual belongs to the basis, and a better basis
is the only way forward. If they differ by as much as the error itself, the
SOC operator is also in play.""")


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
    p.add_argument("--soc", nargs="+", default=["DKH1"],
                   help="SOC Hamiltonian(s): DKH1 (= x2c-1, matches the "
                        ".x2c() mean field) and/or breit-pauli. Giving both "
                        "is itself a check: it separates basis error from "
                        "operator error.")
    p.add_argument("--sweep", action="store_true",
                   help="sweep candidate all-electron bases instead of "
                        "running one. This is how the Br basis gets chosen "
                        "before any HOBr surface is built.")
    p.add_argument("--sweep-bases", nargs="+", default=CANDIDATE_BASES,
                   help="override the candidate list")
    p.add_argument("--no-nevpt2", action="store_true",
                   help="state interaction over CASSCF only, skipping the "
                        "QD-NEVPT2 correction")
    p.add_argument("--verbose", type=int, default=0,
                   help="pyscf verbosity (prism is told 4 regardless)")
    args = p.parse_args()

    if args.sweep:
        rows = sweep(args.atoms, args.sweep_bases, args.soc,
                     use_nevpt2=not args.no_nevpt2, verbose=args.verbose)
        report_sweep(rows)
        return

    print("=" * 72)
    print("== HOX Phase 0.5 -- halogen fine structure as an SOC validation")
    print(f"== basis {args.basis}, soc {', '.join(args.soc)}, "
          f"{'CASSCF state interaction' if args.no_nevpt2 else 'QD-NEVPT2 + SOC'}")
    print("=" * 72)

    results, timings = {}, {}
    for atom in args.atoms:
        for soc in args.soc:
            key = (atom, soc)
            print(f"\n--- {atom}, {soc} " + "-" * 50)
            t0 = time.time()
            try:
                split, nao, _ = measure(atom, args.basis, soc,
                                        use_nevpt2=not args.no_nevpt2,
                                        verbose=args.verbose)
                print(f"  nao {nao}, CAS(5,3), 3 roots")
                results[key] = split
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                results[key] = None
            timings[key] = time.time() - t0
            print(f"  wall time {timings[key]:7.1f} s")

    print("\n" + "=" * 72)
    print("== fine structure 2P_1/2 - 2P_3/2")
    print("=" * 72)
    print(f"{'atom':>6}{'soc':>14}{'calc/cm-1':>14}{'obs/cm-1':>12}"
          f"{'err':>10}{'wall/s':>10}")
    for atom in args.atoms:
        for soc in args.soc:
            obs = FINE_STRUCTURE_CM.get(atom, float("nan"))
            calc = results.get((atom, soc))
            wall = timings.get((atom, soc), float("nan"))
            if calc is None:
                print(f"{atom:>6}{soc:>14}{'--':>14}{obs:12.2f}"
                      f"{'--':>10}{wall:10.1f}")
            else:
                print(f"{atom:>6}{soc:>14}{calc:14.2f}{obs:12.2f}"
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
