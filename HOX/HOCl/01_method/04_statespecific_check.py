#!/usr/bin/env python3
"""
Phase 0.5, step 5: can the two surfaces be built WITHOUT excited-state theory?

03 established that the physics works and that it is unaffordable: 5783
CPU-seconds for one point, 5283 CPU-hours for a 3289-point raster, ~14 days on
16 cores. Water's entire raster was 41 minutes.

The way out is that the two states HOX needs are each the LOWEST of their own
spin manifold:

    X 1A'    the global ground state         -- closed shell, CCSD(T)
    a 3A"    the lowest triplet              -- state-specific, spin = 2

Neither needs state averaging, and neither needs a multi-root solve. That
deletes the 4-6 root SA-CASSCF which, per 03, is where essentially all the
time went (172 s of the 183 s total, and it was the part that failed to
converge).

What this does NOT give you is the intensity. a 3A" <- X 1A' is spin-forbidden
and borrows its entire oscillator strength from bright singlets through SOC,
so f requires the multi-state treatment. But it is only needed in the
Franck-Condon window, exactly as in water/09_propagate.py:

    "The dipole enters only at t = 0, which is why mu(R) was only ever needed
     across the Franck-Condon window."

So the proposed split is:

    V(X 1A')   CCSD(T), state-specific        full bound region    cheap
    V(a 3A")   state-specific, lowest triplet full 3D raster       cheap
    mu_SOC(R)  6-state QD-NEVPT2 + SOC        FC window only       expensive

This script tests the one assumption that split rests on: that a state-specific
triplet reproduces the multi-state answer. 03 gives the reference values at the
same geometry and basis:

    6 roots, converged      3.4477 eV   (359.6 nm)
    4 roots, NOT converged  3.4495 eV   (359.4 nm)

Agreement within ~0.05 eV would justify the whole restructuring on one data
point. A large disagreement means the triplet has multireference character that
a state-specific treatment misses, and the expensive route is the only route.

Two methods, deliberately unrelated, in the same spirit as water's OH curve
where NEVPT2 gave De 4.628 eV and UCCSD(T) 4.631 -- two methods agreeing to
3 meV is what made that surface trustworthy:

    ΔCCSD(T)          E(triplet, UCCSD(T)) - E(singlet, CCSD(T))
    ΔCASSCF+NEVPT2    single-root CASSCF then PySCF's own SC-NEVPT2

SYMMETRY. "Lowest triplet" is only well defined if a 3A' cannot overtake
a 3A" along the dissociation coordinate. Within the A" irrep it is the ground
state of its block and any crossing is harmless. At equilibrium the lowest
triplet is a 3A" anyway, so --symmetry is off by default here; turn it on for
the raster, where water/README.md's "force the point group, don't detect it"
becomes load-bearing rather than tidy.

Usage:
    python 04_statespecific_check.py 2>&1 | tee logs/statespecific.log
    python 04_statespecific_check.py --no-nevpt2       # CCSD(T) only, fastest
    python 04_statespecific_check.py --symmetry Cs
"""
import argparse
import os
import time
import traceback

import numpy as np

from pyscf import gto, scf, cc, mcscf, mrpt, fci

HARTREE2EV = 27.211386245988
NM_PER_EV = 1239.841984

# Same geometry as 03, same caveat: quoted from memory, verify before citing.
R_OH_ANG = 0.9644
R_OCL_ANG = 1.6891
ANGLE_HOCL_DEG = 102.96

# From 03 at this geometry and basis, for comparison.
REF_6ROOT_EV = 3.4477
REF_4ROOT_EV = 3.4495
OBS_PEAK_NM = 380.0


def hocl_geometry():
    th = np.deg2rad(ANGLE_HOCL_DEG)
    return [["O", (0.0, 0.0, 0.0)],
            ["Cl", (R_OCL_ANG, 0.0, 0.0)],
            ["H", (R_OH_ANG * np.cos(th), R_OH_ANG * np.sin(th), 0.0)]]


def build_mol(spin, basis, symmetry, verbose=0):
    return gto.M(atom=hocl_geometry(), basis=basis, spin=spin, charge=0,
                 symmetry=symmetry, unit="Angstrom", verbose=verbose)


def run_ccsd_t(spin, basis, symmetry, verbose=0):
    """CCSD(T) on the lowest state of this spin. Returns (energy, info)."""
    mol = build_mol(spin, basis, symmetry, verbose)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).x2c()
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"spin={spin}: SCF did not converge")

    ss = ""
    if spin != 0:
        s2, mult = mf.spin_square()
        ss = f", <S^2> = {s2:.4f} (2S+1 = {mult:.2f}, want 3.00)"

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-9
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError(f"spin={spin}: CCSD did not converge")
    e_t = mycc.ccsd_t()
    return mycc.e_tot + e_t, f"SCF {mf.e_tot:.8f}{ss}", True


def determine_cas(basis, symmetry, cas=None, verbose=0):
    """Pick ONE active space, from the singlet, for both spins to share.

    Running AVAS separately per spin is wrong and was the original bug here:
    it chose CAS(12e,7o) for the singlet and CAS(10e,6o) for the triplet, and
    the difference of two energies computed in different active spaces is not
    an excitation energy. The orbitals are returned too, so both CASSCF runs
    start from the same space and only the occupation differs.
    """
    mol = build_mol(0, basis, symmetry, verbose)
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-11
    mf.kernel()
    if cas is not None:
        ncas, nelecas = cas
        return ncas, nelecas, mf.mo_coeff
    from pyscf.mcscf import avas
    ncas, nelecas, mo = avas.avas(mf, ["Cl 3p", "O 2p"], minao="ano")
    return ncas, nelecas, mo


def run_casscf_nevpt2(spin, basis, symmetry, cas_data, max_cycle=200,
                      verbose=0):
    """Single-root CASSCF then SC-NEVPT2 -- the method water used."""
    ncas, nelecas, mo_ref = cas_data
    mol = build_mol(spin, basis, symmetry, verbose)
    mf = (scf.RHF(mol) if spin == 0 else scf.ROHF(mol)).x2c()
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"spin={spin}: SCF did not converge")

    # Same ncas and same TOTAL active electrons for both spins; PySCF splits
    # nelecas into (na, nb) according to mol.spin, so the triplet gets (7,5)
    # where the singlet gets (6,6) -- same space, different occupation, which
    # is exactly what a vertical excitation energy requires.
    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = 1e-10
    mc.max_cycle_macro = max_cycle
    # One root, spin-constrained: this is a state-SPECIFIC calculation, the
    # whole point of the exercise. fix_spin_ is correct here precisely because
    # we are NOT trying to reach a different multiplicity than mol.spin.
    S = spin / 2.0
    fci.addons.fix_spin_(mc.fcisolver, ss=S * (S + 1.0))
    mc.kernel(mo_ref)
    conv = bool(mc.converged)
    if not conv:
        print(f"    WARNING: spin={spin}: CASSCF did not converge in "
              f"{max_cycle} macro cycles")
    e_pt = mrpt.NEVPT(mc).kernel()
    na, nb = mc.nelecas
    return (mc.e_tot + e_pt,
            f"CAS({nelecas}e,{ncas}o) as ({na},{nb}), CASSCF {mc.e_tot:.8f}",
            conv)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--symmetry", default=None,
                   help="point group to FORCE, e.g. Cs. Off by default: at "
                        "equilibrium the lowest triplet is a 3A\" anyway. Turn "
                        "it on for a raster.")
    p.add_argument("--cas", nargs=2, type=int, metavar=("NCAS", "NELEC"),
                   default=None)
    p.add_argument("--no-nevpt2", action="store_true",
                   help="CCSD(T) only -- fastest check")
    p.add_argument("--max-cycle", type=int, default=200,
                   help="CASSCF macro iterations (default 200; both spins "
                        "hit the old 50-cycle default)")
    p.add_argument("--raster-points", type=int, default=3289)
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    sym = args.symmetry if args.symmetry else False
    cas = tuple(args.cas) if args.cas else None

    print("=" * 72)
    print("== HOX Phase 0.5 -- state-specific surfaces, without excited states")
    print(f"== basis {args.basis}, symmetry {sym}")
    print("=" * 72)

    results, timings, converged = {}, {}, {}

    methods = [("dCCSD(T)", lambda s: run_ccsd_t(s, args.basis, sym,
                                                 args.verbose))]
    if not args.no_nevpt2:
        # One active space, chosen once, shared by both spins.
        cas_data = determine_cas(args.basis, sym, cas, args.verbose)
        print(f"\n  active space for both spins: "
              f"CAS({cas_data[1]}e, {cas_data[0]}o)")
        methods.append(("dCASSCF+NEVPT2",
                        lambda s: run_casscf_nevpt2(s, args.basis, sym,
                                                    cas_data, args.max_cycle,
                                                    args.verbose)))

    for name, fn in methods:
        print(f"\n--- {name} " + "-" * (60 - len(name)))
        # Time each METHOD separately: lumping them together let two broken
        # CASSCF runs poison the cost estimate for the one that worked.
        c_m, t_m = time.process_time(), time.time()
        try:
            ts = time.time()
            e_s, info_s, conv_s = fn(0)
            print(f"  singlet X 1A'   {e_s:18.10f} Ha   {info_s}"
                  f"   [{time.time() - ts:.1f} s]")

            tt = time.time()
            e_t, info_t, conv_t = fn(2)
            print(f"  triplet a 3A\"   {e_t:18.10f} Ha   {info_t}"
                  f"   [{time.time() - tt:.1f} s]")

            de = (e_t - e_s) * HARTREE2EV
            results[name] = de
            converged[name] = conv_s and conv_t
            print(f"  vertical T <- S {de:8.4f} eV = {NM_PER_EV / de:.1f} nm")
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results[name] = None
            converged[name] = False
        timings[name] = (time.time() - t_m, time.process_time() - c_m)

    print("\n" + "=" * 72)
    print("== vertical a 3A\" <- X 1A', against the multi-state result")
    print("=" * 72)
    print(f"  {'method':>18}{'dE / eV':>11}{'nm':>9}{'vs 6-root':>12}  conv")
    print(f"  {'03, 6 roots':>18}{REF_6ROOT_EV:11.4f}"
          f"{NM_PER_EV / REF_6ROOT_EV:9.1f}{'--':>12}   yes")
    print(f"  {'03, 4 roots*':>18}{REF_4ROOT_EV:11.4f}"
          f"{NM_PER_EV / REF_4ROOT_EV:9.1f}"
          f"{REF_4ROOT_EV - REF_6ROOT_EV:+12.4f}   NO")
    for name, de in results.items():
        ok = "yes" if converged.get(name) else "NO"
        if de is None:
            print(f"  {name:>18}{'--':>11}{'--':>9}{'--':>12}   {ok}")
        else:
            print(f"  {name:>18}{de:11.4f}{NM_PER_EV / de:9.1f}"
                  f"{de - REF_6ROOT_EV:+12.4f}   {ok}")
    print("  * unconverged reference; shown for scale, not as a target")
    if any(not converged.get(n) for n in results):
        print("\n  Rows marked NO did not converge. Their numbers are not")
        print("  evidence of anything -- do not read them as a method")
        print("  comparison. Raise --max-cycle or change the active space.")

    # Only converged methods count towards the verdict.
    good = [d for n, d in results.items()
            if d is not None and converged.get(n)]
    if good:
        worst = max(abs(d - REF_6ROOT_EV) for d in good)
        print(f"\n  largest deviation from the 6-root answer: {worst:.4f} eV")
        if worst < 0.05:
            verdict = ("state-specific reproduces the multi-state result. The "
                       "restructuring\n  is justified: build both surfaces "
                       "state-specifically and spend the\n  expensive "
                       "multi-state machinery only on mu in the FC window.")
        elif worst < 0.15:
            verdict = ("close but not tight. Usable for a raster if the same "
                       "offset holds\n  along the dissociation coordinate -- "
                       "check that at a stretched geometry\n  before "
                       "committing.")
        else:
            verdict = ("too large. The triplet likely has multireference "
                       "character that a\n  state-specific treatment misses, "
                       "and the expensive route may be the\n  only honest one.")
        print(f"  -> {verdict}")

    npts = args.raster_points
    ncpu = os.cpu_count() or 1
    phys = ncpu // 2 if ncpu > 1 else 1
    # Per method, since each is a complete route to both surfaces and they
    # cost wildly different amounts.
    print("\n" + "=" * 72)
    print("== raster cost, per method (both surfaces, one geometry)")
    print("=" * 72)
    print(f"  {'method':>18}{'wall/s':>10}{'CPU/s':>10}{'par':>7}"
          f"{'CPU-h':>12}{'h/16 cores':>12}")
    for name, (w, c) in timings.items():
        par = c / w if w > 0 else float("nan")
        ch = c * npts / 3600.0
        print(f"  {name:>18}{w:10.1f}{c:10.1f}{par:6.1f}x"
              f"{ch:12.1f}{ch / phys:12.1f}")
    print(f"""
  Compare 03, the full multi-state point: 5783 CPU-s, 5283 CPU-hours,
  330 h on 16 cores. The speedup here is the whole argument.

  Remember this does NOT include mu_SOC, which still needs the multi-state
  treatment over the Franck-Condon window. At 5783 CPU-s per point, a few
  hundred FC points is {5783 * 300 / 3600 / phys:.0f} h on {phys} cores -- so budget for that
  separately, and consider whether mu can be sampled coarsely and
  interpolated. water/10_mu_sensitivity.py measured what freezing mu costs:
  the peak moved 4 nm and the band narrowed 15%, so mu is not negligible but
  it is also not needed on the full raster grid.
""")


if __name__ == "__main__":
    main()
