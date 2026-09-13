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

MIXED MULTIPLICITIES -- the part written without a worked example. Every
shipped Prism SOC example uses one mol.spin and one SA-CASSCF over roots of a
single multiplicity. Singlet-plus-triplet needs PySCF's state_average_mix_
with two FCI solvers. Prism reporting per-state active electrons as
[(3, 2), (3, 2), (3, 2)] and counting "reference microstates" separately says
it tracks (nalpha, nbeta) per state, which is what that requires -- but it is
inference, not a documented API. --triplet-only is the graceful fallback: it
runs the molecular SOC path with a single multiplicity, which still gives the
timing even if the mixed setup needs work.

Usage:
    python 03_soc_hocl_vertical.py 2>&1 | tee logs/soc_hocl.log
    python 03_soc_hocl_vertical.py --triplet-only     # fallback, timing only
    python 03_soc_hocl_vertical.py --no-nevpt2        # CASSCF-level SOC
    python 03_soc_hocl_vertical.py --cas 10 8         # override AVAS
"""
import argparse
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


def build_reference(basis, cas=None, nsinglet=1, ntriplet=1,
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

    if triplet_only:
        weights = np.ones(ntriplet) / ntriplet
        mc = mc.state_average_(weights)
        fci.addons.fix_spin_(mc.fcisolver, ss=2.0)          # triplet S(S+1)=2
    else:
        # Two solvers, one per multiplicity. This is what lets the state
        # interaction mix them; a single solver cannot represent both.
        solver_s = fci.direct_spin0.FCI(mol)
        solver_s.spin = 0
        solver_s.nroots = nsinglet
        fci.addons.fix_spin_(solver_s, ss=0.0)

        solver_t = fci.direct_spin1.FCI(mol)
        solver_t.spin = 2
        solver_t.nroots = ntriplet
        fci.addons.fix_spin_(solver_t, ss=2.0)

        n = nsinglet + ntriplet
        mcscf.state_average_mix_(mc, [solver_s, solver_t], np.ones(n) / n)

    mc.kernel(mo)
    if not mc.converged:
        print("  WARNING: CASSCF did not converge")
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

    i = int(np.argmax(np.abs(osc_arr))) + 1
    ev = rel_ev[i]
    print(f"\n  brightest excited state: #{i}, {ev:.4f} eV "
          f"= {NM_PER_EV / ev:.1f} nm, f = {osc_arr[i - 1]:.4e}")
    print(f"  measured HOCl triplet band: {OBS_PEAK_NM:.0f} nm "
          f"({NM_PER_EV / OBS_PEAK_NM:.3f} eV), sigma ~ {OBS_SIGMA_CM2:.0e} cm^2")
    print(f"  vertical energy is NOT the band peak -- the peak sits below it "
          f"by the reorganisation\n  along the dissociation coordinate, so a "
          f"vertical value above 3.26 eV is expected.")
    return fmax


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--soc", default="DKH1", help="DKH1 or breit-pauli")
    p.add_argument("--cas", nargs=2, type=int, metavar=("NCAS", "NELEC"),
                   default=None, help="override the AVAS active space")
    p.add_argument("--nsinglet", type=int, default=2,
                   help="singlet roots (default 2: X 1A' plus the bright "
                        "1A\" that the triplet borrows from)")
    p.add_argument("--ntriplet", type=int, default=2,
                   help="triplet roots (default 2: a 3A\" and its partner)")
    p.add_argument("--triplet-only", action="store_true",
                   help="single multiplicity -- fallback that still gives the "
                        "timing if the mixed-multiplicity setup needs work")
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

    t0 = time.time()
    try:
        mol, mf, mc = build_reference(
            args.basis, cas=cas, nsinglet=args.nsinglet,
            ntriplet=args.ntriplet, triplet_only=args.triplet_only,
            verbose=args.verbose)
        t_ref = time.time() - t0
        print(f"  reference done in {t_ref:8.1f} s  (nao {mol.nao_nr()})")

        t1 = time.time()
        energies, osc = run_soc(mf, mc, soc=args.soc,
                                use_nevpt2=not args.no_nevpt2)
        t_soc = time.time() - t1
        print(f"  SOC step done in  {t_soc:8.1f} s")
        report(energies, osc)
    except Exception as exc:
        print(f"\n  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        if not args.triplet_only:
            print("\n  If this is the mixed-multiplicity setup, retry with")
            print("      python 03_soc_hocl_vertical.py --triplet-only")
            print("  which uses one multiplicity and still gives the timing.")
        return

    total = time.time() - t0
    per_point = total
    serial = per_point * args.raster_points
    wall = serial / args.nproc

    print("\n" + "=" * 72)
    print("== raster cost estimate")
    print("=" * 72)
    print(f"  one point            {per_point:10.1f} s  "
          f"(reference {t_ref:.1f} + SOC {t_soc:.1f})")
    print(f"  x {args.raster_points} points        {serial / 3600:10.1f} core-hours")
    print(f"  / {args.nproc} processes       {wall / 3600:10.1f} hours wall")
    print(f"""
  The water raster was {args.raster_points} points of SA-CASSCF+NEVPT2 in 41 minutes.
  Read this estimate against that:

    under ~6 h     comparable to water; the full treatment is affordable.
    6-24 h         viable but do HOCl only, and reconsider before HOBr, where
                   the basis is larger and the SOC step heavier.
    over ~24 h     fall back to SOMF-QDNEVPT2, or treat SOC as geometry
                   independent and compute it once at equilibrium.

  Two caveats on this number. The raster runs single-threaded workers, but
  this point had all cores available -- the Cl SOC step showed 31x parallel
  speedup, so a single-threaded point will be substantially slower than the
  wall time above. And a dissociating geometry converges worse than
  equilibrium, so the average point costs more than this one.
""")


if __name__ == "__main__":
    main()
