#!/usr/bin/env python3
"""
Phase 0.5, step 4: HOCl vertical a 3A" <- X 1A', and what a raster would cost.

02_soc_atoms.py established that the toolchain reproduces atomic fine
structure. That test does NOT exercise the thing HOX actually depends on:
INTENSITY BORROWING between states of different multiplicity. Within a single
2P term there is nothing to borrow from, which is why the oscillator strengths
came back at 1e-24. Here the triplet is dark on its own and acquires its
intensity entirely through spin-orbit mixing with the bright singlets. If that
does not work, the whole HOX plan does not work.

Two things come out of this script.

  1. PHYSICS. The vertical a 3A" excitation energy, and a NONZERO
     SOC-borrowed oscillator strength. HOCl has a measured triplet band at
     380 nm (3.26 eV) with sigma ~ 4e-21 cm^2 and a tail to 480 nm, so there
     is a real number to miss. At this level expect the energy within a few
     tenths of an eV; f is the harder quantity, as water found for mu.

     The decisive qualitative check is f > 0 but small (~1e-7 to 1e-5). An f
     of exactly zero means the borrowing is not happening -- the direct
     analogue of the mu = 0.0000 CASSCF failure in water/README.md, which
     also produced a clean-looking run with no error message.

  2. COST. Wall time for ONE molecular point, split into reference and
     SOC-NEVPT2, extrapolated to a 3D raster. This is the Phase 1 go/no-go:
     the water raster was 3289 points, and if a point costs minutes rather
     than seconds the SOMF variant or geometry-independent SOC is the
     fallback, not the full treatment.

MIXED MULTIPLICITIES, and why it is not state_average_mix_. Prism's interface
takes ONE mc and uses a single global mc.nelecas for every state -- passing
state_average_mix_ dies in make_rdm1 with a CI vector of the wrong length,
because a triplet (7,5) and a singlet (6,6) have different (nalpha, nbeta).

But it also computes each state's spin from its own CI vector, via
compute_spin_square(). That is the opening: singlet and triplet states share
the same (nalpha, nbeta) as long as both are taken from the **Ms = 0
determinant space**, which spans S = 0, 1, 2 ... at once. So the reference is
a single ordinary state-averaged CASSCF at mol.spin = 0, with two departures
from the water pipeline:

  - fci.direct_spin1 rather than direct_spin0. PySCF picks direct_spin0 for a
    closed-shell molecule, and that projects onto singlets, which would throw
    away the triplet entirely.
  - NO fix_spin_. water/README.md lists "constrain the spin in CASSCF" as a
    gotcha, because there the solver returning a triplet as an excited root
    was the bug. Here it is the entire point, and constraining S would remove
    the state whose intensity we are trying to compute.

Because the roots come out mixed, you cannot ask for a given number of each:
you take the lowest --nroots and read off what they are. The script prints
<S^2> per root before doing any SOC work, so a reference containing no triplet
is caught immediately rather than after the expensive step.

Usage:
    python 03_soc_hocl_vertical.py 2>&1 | tee logs/soc_hocl.log
    python 03_soc_hocl_vertical.py --triplet-only     # fallback, timing only
    python 03_soc_hocl_vertical.py --no-nevpt2        # CASSCF-level SOC
    python 03_soc_hocl_vertical.py --cas 10 8         # override AVAS
"""
import argparse
import os
import time
import traceback

import numpy as np

from pyscf import gto, scf, mcscf, fci

HARTREE2EV = 27.211386245988
HARTREE2CM = 219474.6313702
NM_PER_EV = 1239.841984

# HOCl equilibrium geometry. Quoted from memory during the work -- CHECK
# against a spectroscopic source (or Peterson's PES papers) before this
# appears anywhere that matters, same caveat as the vibrational fundamentals
# in water/README.md.
R_OH_ANG = 0.9644
R_OCL_ANG = 1.6891
ANGLE_HOCL_DEG = 102.96

# Measured HOCl triplet band, for comparison. From the 1998 JPCA study cited
# in docs/deep-review-claude.md: peak 380 nm, sigma ~ 4e-21 cm^2, tail to
# 480 nm.
OBS_PEAK_NM = 380.0
OBS_SIGMA_CM2 = 4e-21


def hocl_geometry(r_oh=R_OH_ANG, r_ocl=R_OCL_ANG, angle=ANGLE_HOCL_DEG):
    """Planar HOCl: O at origin, Cl along +x, H in the xy plane."""
    th = np.deg2rad(angle)
    return [["O", (0.0, 0.0, 0.0)],
            ["Cl", (r_ocl, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def report_reference_spins(mc):
    """<S^2> per CASSCF root, before any SOC work is done.

    In the Ms = 0 space the multiplicities come out mixed and unordered, so
    this is how you find out what the reference actually contains. A reference
    with no triplet cannot borrow intensity, and finding that out here costs
    seconds instead of after the SOC step.
    """
    from pyscf.fci import spin_op

    cis = mc.ci if isinstance(mc.ci, (list, tuple)) else [mc.ci]
    energies = getattr(mc, "e_states", None)
    print(f"\n  {'root':>5}{'E / Ha':>20}{'<S^2>':>10}{'2S+1':>8}  assignment")
    mults = []
    for i, c in enumerate(cis):
        # Use the bare FCI routine, NOT mc.fcisolver.spin_square. After
        # state_average_ the solver is a wrapper whose spin_square expects the
        # whole list of CI vectors and iterates over it, so handing it one
        # vector makes it treat each ROW as a state: "cannot reshape array of
        # size 7 into shape (7,7)". spin_square0 is correct here because alpha
        # and beta share the same spatial orbitals.
        ss, mult = spin_op.spin_square0(c, mc.ncas, mc.nelecas)
        m = int(round(mult))
        mults.append(m)
        label = {1: "singlet", 3: "triplet", 5: "quintet"}.get(
            m, f"multiplicity {mult:.2f}")
        e = energies[i] if energies is not None else float("nan")
        print(f"  {i:>5}{e:20.10f}{ss:10.4f}{mult:8.2f}  {label}")

    if 3 not in mults:
        print("\n  !! no triplet among the reference states. There is nothing")
        print("     for the singlets to lend intensity to, so any f computed")
        print("     below is meaningless. Raise --nroots, or check that the")
        print("     solver is direct_spin1 and that fix_spin_ is NOT applied.")
    elif 1 not in mults:
        print("\n  !! no singlet among the reference states -- nothing to")
        print("     borrow FROM. Raise --nroots.")
    return mults


def build_reference(basis, cas=None, nroots=4, max_cycle=100,
                    triplet_only=False, verbose=0):
    """Mean field plus a state-averaged CASSCF spanning the states we need.

    The a 3A" <- X 1A' transition is essentially a Cl lone pair promoted into
    sigma*(O-Cl), so the active space has to hold the O-Cl sigma/sigma* pair
    and the Cl and O p lone pairs. AVAS built from 'Cl 3p' and 'O 2p' selects
    exactly that, and -- following water/README.md -- it is rebuilt here from
    the mean field rather than seeded from anything, and uses minao='ano'
    because the default minao silently drops orbitals it has no reference for.
    """
    spin = 2 if triplet_only else 0
    mol = gto.M(atom=hocl_geometry(), basis=basis, spin=spin, charge=0,
                symmetry=False, unit="Angstrom", verbose=verbose)
    if mol.has_ecp():
        raise RuntimeError(f"{basis} uses an ECP; X2CAMF needs the core.")

    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).x2c()
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    if cas is not None:
        ncas, nelecas = cas
        mo = mf.mo_coeff
    else:
        from pyscf.mcscf import avas
        ncas, nelecas, mo = avas.avas(mf, ["Cl 3p", "O 2p"], minao="ano")
        print(f"  AVAS selected CAS({nelecas}e, {ncas}o)")

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-6
    mc.max_cycle_macro = max_cycle

    if triplet_only:
        mc.fcisolver = fci.direct_spin1.FCI(mol)
        fci.addons.fix_spin_(mc.fcisolver, ss=2.0)          # triplet S(S+1)=2
        mc = mc.state_average_(np.ones(nroots) / nroots)
    else:
        # direct_spin1 in the Ms = 0 space, deliberately UNCONSTRAINED in S:
        # that space holds singlets and triplets together, and they share one
        # (nalpha, nbeta), which is what Prism's single global nelecas needs.
        # direct_spin0 would project the triplet away; fix_spin_ would too.
        mc.fcisolver = fci.direct_spin1.FCI(mol)
        mc = mc.state_average_(np.ones(nroots) / nroots)

    mc.kernel(mo)
    if not mc.converged:
        print("  WARNING: CASSCF did not converge -- raise --max-cycle, or "
              "try a smaller\n           active space with --cas. SOC results "
              "from an unconverged reference\n           are not trustworthy.")
    return mol, mf, mc


def run_soc(mf, mc, soc="DKH1", use_nevpt2=True, verbose=4):
    """State-interaction SOC. Returns (energies_hartree, oscillator_strengths)."""
    import prism.interface
    import prism.nevpt

    interface = prism.interface.PYSCF(mf, mc, backend="opt_einsum")
    if not use_nevpt2:
        interface.soc = soc
        res = interface.run_soc()
    else:
        nevpt = prism.nevpt.NEVPT(interface)
        nevpt.method = "nevpt2"
        nevpt.method_type = "qd"
        nevpt.soc = soc
        nevpt.verbose = verbose
        res = nevpt.kernel()

    osc = None
    if isinstance(res, tuple):
        e_tot, osc = (res[0], res[2]) if len(res) == 3 else (res[0], None)
    else:
        e_tot = res
    return np.atleast_1d(np.asarray(e_tot, dtype=float)).ravel(), osc


def cluster_states(rel_ev, tol=0.02):
    """Group SOC states into near-degenerate clusters.

    A singlet-derived state stands alone; a triplet-derived one appears as
    three components split only by SOC (a few cm-1 here). So cluster size is
    the multiplicity label, which the SOC states do not otherwise carry --
    they are mixed by construction.
    """
    clusters, cur = [], [0]
    for i in range(1, len(rel_ev)):
        if rel_ev[i] - rel_ev[cur[-1]] < tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return clusters


def report(energies, osc):
    """Excitation energies and the borrowed intensity."""
    e0 = energies[0]
    rel_ev = (energies - e0) * HARTREE2EV
    osc_arr = (np.atleast_1d(np.asarray(osc, dtype=float)).ravel()
               if osc is not None else None)

    print(f"\n  {len(energies)} spin-orbit states:")
    print(f"  {'i':>3}{'E / Ha':>20}{'dE / eV':>10}{'dE / nm':>10}{'f':>14}")
    for i, e in enumerate(energies):
        nm = f"{NM_PER_EV / rel_ev[i]:10.1f}" if rel_ev[i] > 1e-9 else f"{'--':>10}"
        # Prism lists oscillator strengths for transitions FROM the ground
        # state, so entry i-1 belongs to state i.
        f = ""
        if osc_arr is not None and 0 < i <= osc_arr.size:
            f = f"{osc_arr[i - 1]:.4e}"
        print(f"  {i:>3}{e:20.10f}{rel_ev[i]:10.4f}{nm}{f:>14}")

    if osc_arr is None or osc_arr.size == 0:
        print("\n  !! no oscillator strengths returned -- without them there is "
              "no borrowed intensity to read, and the HOX plan needs them.")
        return None

    fmax = float(np.max(np.abs(osc_arr)))
    print(f"\n  max |f| over excited states: {fmax:.4e}")
    if fmax == 0.0:
        print("  !! f is EXACTLY zero. The triplet is not borrowing any")
        print("     intensity, which is the singlet-triplet analogue of the")
        print("     mu = 0.0000 CASSCF failure in water/README.md -- a clean")
        print("     run with no error and no physics. Check that the state")
        print("     interaction really spans both multiplicities.")
    elif fmax < 1e-12:
        print("  !! f is nonzero but absurdly small; suspect the same problem.")
    else:
        print("  f is small and nonzero, which is what a SOC-borrowed")
        print("  singlet-triplet transition should look like.")

    # Max |f| finds the bright SINGLET, which is not the quantity of interest.
    # Cluster instead: the a 3A" band is a three-component cluster and its
    # intensity is the SUM over those components.
    clusters = cluster_states(rel_ev)
    print(f"\n  {'group':>6}{'n':>4}{'dE / eV':>10}{'dE / nm':>10}"
          f"{'sum f':>13}   assignment")
    triplets = []
    for k, idx in enumerate(clusters):
        ev = float(np.mean([rel_ev[i] for i in idx]))
        fsum = float(sum(osc_arr[i - 1] for i in idx
                         if 0 < i <= osc_arr.size))
        if len(idx) == 3:
            label = "TRIPLET-derived (borrowed)"
            triplets.append((ev, fsum))
        elif len(idx) == 1:
            label = "singlet-derived" + (" (ground)" if idx[0] == 0 else "")
        else:
            label = f"{len(idx)} components -- unexpected, check --nroots"
        nm = f"{NM_PER_EV / ev:10.1f}" if ev > 1e-9 else f"{'--':>10}"
        print(f"  {k:>6}{len(idx):>4}{ev:10.4f}{nm}{fsum:13.4e}   {label}")

    if not triplets:
        print("\n  !! no three-component cluster: no triplet-derived band was")
        print("     resolved. Nothing here is the a 3A\" transition.")
        return fmax

    ev, fsum = triplets[0]
    nm = NM_PER_EV / ev
    obs_ev = NM_PER_EV / OBS_PEAK_NM
    print(f"\n  lowest triplet-derived band (the a 3A\" candidate):")
    print(f"      vertical   {ev:.4f} eV = {nm:.1f} nm,  summed f = {fsum:.4e}")
    print(f"      measured   {obs_ev:.4f} eV = {OBS_PEAK_NM:.0f} nm,  "
          f"sigma ~ {OBS_SIGMA_CM2:.0e} cm^2")
    print(f"      difference {ev - obs_ev:+.4f} eV ({nm - OBS_PEAK_NM:+.1f} nm)")

    # Very rough band-integrated f from the measured peak cross section, to
    # put the computed f on a comparable scale:
    #     f = 1.13e12 * Int sigma dnu~   (sigma in cm^2, nu~ in cm^-1)
    # The width is a guess, so this is an order-of-magnitude check only.
    for width_cm in (2000.0, 5000.0):
        f_obs = 1.13e12 * OBS_SIGMA_CM2 * width_cm
        print(f"      f(obs) ~ {f_obs:.2e} if the band is {width_cm:.0f} cm-1 wide"
              f"   -> calc/obs = {fsum / f_obs:.2f}")
    print("""      The width is assumed, not measured, so treat the ratio as an
      order of magnitude. A vertical f is also not a band-integrated f: mu
      varies along the dissociation coordinate, which is what the propagation
      in Phase 1 would actually integrate over. water/ found the analogous
      quantity low by a constant 1.29, so a shortfall here is expected in kind
      if not in size.""")
    return fmax


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--soc", default="DKH1", help="DKH1 or breit-pauli")
    p.add_argument("--cas", nargs=2, type=int, metavar=("NCAS", "NELEC"),
                   default=None, help="override the AVAS active space")
    p.add_argument("--nroots", type=int, default=6,
                   help="CASSCF roots in the Ms=0 space (default 6). The "
                        "multiplicities come out mixed and cannot be "
                        "requested individually; raise this if no triplet "
                        "appears among them.")
    p.add_argument("--max-cycle", type=int, default=100,
                   help="CASSCF macro iterations (default 100)")
    p.add_argument("--triplet-only", action="store_true",
                   help="spin-constrained triplets only -- no borrowing, but "
                        "it still gives the timing if the mixed reference "
                        "misbehaves")
    p.add_argument("--no-nevpt2", action="store_true",
                   help="CASSCF-level state interaction, no perturbation")
    p.add_argument("--raster-points", type=int, default=3289,
                   help="grid size for the cost estimate (default 3289, the "
                        "water raster)")
    p.add_argument("--nproc", type=int, default=30,
                   help="processes the raster would use (default 30, as in "
                        "water/04_pes_grid.py)")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    cas = tuple(args.cas) if args.cas else None

    print("=" * 72)
    print("== HOX Phase 0.5 -- HOCl vertical a 3A\" and raster cost")
    print(f"== basis {args.basis}, soc {args.soc}, "
          f"{'triplet only' if args.triplet_only else 'singlet + triplet'}, "
          f"{'CASSCF SI' if args.no_nevpt2 else 'QD-NEVPT2 + SOC'}")
    print("=" * 72)

    t0, c0 = time.time(), time.process_time()
    try:
        mol, mf, mc = build_reference(
            args.basis, cas=cas, nroots=args.nroots,
            max_cycle=args.max_cycle, triplet_only=args.triplet_only,
            verbose=args.verbose)
        t_ref = time.time() - t0
        print(f"  reference done in {t_ref:8.1f} s  (nao {mol.nao_nr()})")
        report_reference_spins(mc)

        t1 = time.time()
        energies, osc = run_soc(mf, mc, soc=args.soc,
                                use_nevpt2=not args.no_nevpt2)
        t_soc = time.time() - t1
        print(f"  SOC step done in  {t_soc:8.1f} s")
        cpu_total = time.process_time() - c0
        report(energies, osc)
    except Exception as exc:
        print(f"\n  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if not args.triplet_only:
            print("\n  If this is the mixed-multiplicity setup, retry with")
            print("      python 03_soc_hocl_vertical.py --triplet-only")
            print("  which uses one multiplicity and still gives the timing.")
        return

    wall_total = time.time() - t0
    par = cpu_total / wall_total if wall_total > 0 else float("nan")
    ncpu = os.cpu_count() or 1
    npts = args.raster_points

    # The cost of a raster is TOTAL CPU WORK divided by the throughput of the
    # machine. It is NOT wall-time-per-point times points over processes: this
    # point used every core, so 30 concurrent copies of it cannot each run at
    # this speed. Whether the work is arranged as 30 single-threaded workers
    # or as one all-core job at a time barely matters -- the machine has a
    # fixed number of cores either way, and that is what sets the answer.
    cpu_hours = cpu_total * npts / 3600.0
    phys = ncpu // 2 if ncpu > 1 else 1          # assume SMT; cores, not threads

    print("\n" + "=" * 72)
    print("== raster cost estimate")
    print("=" * 72)
    print(f"  one point, wall      {wall_total:10.1f} s   "
          f"(reference {t_ref:.1f} + SOC {t_soc:.1f})")
    print(f"  one point, CPU       {cpu_total:10.1f} s   "
          f"observed parallel factor {par:.1f}x over {ncpu} logical CPUs")
    print(f"\n  x {npts} points")
    print(f"  total work           {cpu_hours:10.1f} CPU-hours")
    print(f"  wall on {phys:2d} cores    {cpu_hours / phys:10.1f} h"
          f"   <- the realistic figure")
    print(f"  wall on {ncpu:2d} threads  {cpu_hours / ncpu:10.1f} h"
          f"   <- only if SMT scaled perfectly, which it does not")
    print(f"""
  Naively dividing the 155 s wall time by {args.nproc} processes would give
  {wall_total * npts / args.nproc / 3600:.1f} h, and that is wrong: this point already
  consumed the whole machine at a {par:.0f}x parallel factor. Total CPU work over
  available cores is the honest measure, and it is the figure above.

  For scale, water's raster was {npts} points of SA-CASSCF+NEVPT2 in 41 minutes
  of wall clock on the same machine.

    under ~12 h    comparable in spirit to water; affordable.
    12-72 h        a weekend job. Viable for HOCl once, but reconsider before
                   HOBr, where the basis is larger and the SOC step heavier,
                   and before any thought of repeating it per temperature.
    over ~72 h     do not raster this. Fall back to SOMF-QDNEVPT2, a smaller
                   active space or basis, or treat SOC as geometry-independent
                   and compute it once near equilibrium -- physically
                   defensible, since the SOC constant is dominated by the
                   halogen core and varies weakly with bond length.

  Two things still push the real cost UP from this estimate. A dissociating
  geometry converges worse than equilibrium, and this reference needed 141 s
  of the 155 s total -- CASSCF convergence, not the SOC step, is the
  bottleneck, and it is the part that degrades away from equilibrium. Against
  that, a raster can reuse converged orbitals from neighbouring geometries,
  which water did not need to do but which would help a lot here.
""")


if __name__ == "__main__":
    main()
