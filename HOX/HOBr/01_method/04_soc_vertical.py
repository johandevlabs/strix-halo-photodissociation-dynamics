#!/usr/bin/env python3
"""
HOBr step 4: the SOC-borrowed intensity and the spin-orbit shift of the band.

Descends from HOCl/01_method/03_soc_hocl_vertical.py: SA-CASSCF in the Ms = 0
determinant space (direct_spin1, NO fix_spin_, so singlets and triplets share
one (nalpha, nbeta) as Prism's single global nelecas requires), then DKH1
QD-NEVPT2 with state-interaction SOC. That script's reasoning about mixed
multiplicities, and its checks (<S^2> per root before the SOC step; f
exactly zero is the failure mode, not a small f), carry over unchanged.

Two questions, both set by 03's band model:

  1. THE INTENSITY. The a 3A" band is dark by spin and borrows everything
     through SOC. 03 found the f that would reproduce Ingham et al.'s peak
     sigma (2.3e-20 cm2 at 457 nm) is ~1.3e-4, a floor. HOCl's equivalent
     comparison came out 10-25x low. Br's SOC is ~4x Cl's and f goes as its
     square, so HOBr should borrow far more; whether it reaches the target is
     the question.

  2. THE SPIN-ORBIT SHIFT OF THE BAND. The spin-free vertical sits ~20 nm
     blue of the measured band (02: 432 nm; 03's 1D peak 437 nm). SOC pushes
     the triplet components down, at second order. For HOCl, 03's log shows
     the components ~12 meV below the spin-free triplet. Second order scales
     as the coupling squared, so HOBr's could be ~0.1-0.2 eV -- the size of
     the whole gap. This is measured, not assumed: QD-NEVPT2 is run twice on
     the same reference, without SOC and with it, and the band shift is
     (E_T - E_0) with SOC minus (E_T - E_0) without.

     This matters for the deliverable directly: 03's temperature ratio was
     read with the model band shifted rigidly onto the measured peak, and
     that alignment is an assumption this run can replace.

Basis cc-pvtz-dk and DKH1, from the toolchain sweep: the converged Br value
of the 2P splitting is -7.2% with DKH1, so the SOC here is expected ~7% low
and f ~14% low -- carried as an error bar, not corrected.

CONTROL: `--molecule HOCl --basis def2-tzvp` should reproduce HOCl's 03 --
vertical 3.4475 eV, summed f ~8.9e-7 -- and checks the adaptation before the
HOBr numbers are trusted. ~3 min.

Usage:
    python 04_soc_vertical.py 2>&1 | tee ../logs/soc_vertical.log
    python 04_soc_vertical.py --molecule HOCl --basis def2-tzvp   # control
    python 04_soc_vertical.py --cas 7 12                          # CAS(12,7)
    python 04_soc_vertical.py --avas "Br 4p" "O 2p" "H 1s" --nroots 12
"""
import argparse
import time
import traceback

import numpy as np

from pyscf import gto, scf, mcscf, fci

HARTREE2EV = 27.211386245988
NM_PER_EV = 1239.841984

MOLECULES = {
    # 01_geometry.py, CCSD(T)/cc-pvtz-dk
    "HOBr": dict(halogen="Br", avas=["Br 4p", "O 2p"],
                 geom=(1.8357, 0.9646, 102.02),
                 obs_nm=457.0, obs_sigma=2.3e-20,
                 f_target=1.3e-4,            # 03's band model, a floor
                 fwhm_ev=0.566),             # 03's 1D band, for the f(obs) estimate
    # HOCl's 03, for the control run
    "HOCl": dict(halogen="Cl", avas=["Cl 3p", "O 2p"],
                 geom=(1.6891, 0.9644, 102.96),
                 obs_nm=380.0, obs_sigma=4e-21,
                 f_target=None, fwhm_ev=None,
                 ref_vertical=3.4475, ref_f=8.9e-7),
}


def geometry(halogen, r_ox, r_oh, theta):
    th = np.deg2rad(theta)
    return [["O", (0.0, 0.0, 0.0)],
            [halogen, (r_ox, 0.0, 0.0)],
            ["H", (r_oh * np.cos(th), r_oh * np.sin(th), 0.0)]]


def report_reference_spins(mc):
    """<S^2> per root BEFORE the SOC step: a reference with no triplet cannot
    borrow, and finding that out here costs seconds."""
    from pyscf.fci import spin_op
    cis = mc.ci if isinstance(mc.ci, (list, tuple)) else [mc.ci]
    energies = getattr(mc, "e_states", None)
    print(f"\n  {'root':>5}{'E / Ha':>20}{'<S^2>':>10}{'2S+1':>8}")
    mults = []
    for i, c in enumerate(cis):
        # spin_square0, not mc.fcisolver.spin_square: after state_average_ the
        # solver wrapper expects the whole list (HOCl 03's note).
        ss, mult = spin_op.spin_square0(c, mc.ncas, mc.nelecas)
        mults.append(int(round(mult)))
        e = energies[i] if energies is not None else float("nan")
        print(f"  {i:>5}{e:20.10f}{ss:10.4f}{mult:8.2f}")
    if 3 not in mults:
        print("\n  !! no triplet among the reference roots: nothing to borrow "
              "into. Raise --nroots.")
    if 1 not in mults:
        print("\n  !! no singlet among the reference roots: nothing to borrow "
              "from. Raise --nroots.")
    return mults


def build_reference(spec, basis, geom, cas, nroots, max_cycle, minao, verbose):
    mol = gto.M(atom=geometry(spec["halogen"], *geom), basis=basis, spin=0,
                charge=0, symmetry=False, unit="Angstrom", verbose=verbose)
    if mol.has_ecp():
        raise RuntimeError(f"{basis} uses an ECP; X2CAMF needs the core.")
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-12
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")

    if cas is not None:
        ncas, nelecas = cas
        mo = mf.mo_coeff
    else:
        from pyscf.mcscf import avas
        try:
            ncas, nelecas, mo = avas.avas(mf, spec["avas"], minao=minao)
        except Exception as exc:
            # 'ano' is what HOCl used; if it has no Br reference the fallback
            # is MINAO, and saying so matters, since it changes the selection.
            print(f"  AVAS with minao={minao!r} failed ({type(exc).__name__}: "
                  f"{exc}); retrying with minao='minao'")
            ncas, nelecas, mo = avas.avas(mf, spec["avas"], minao="minao")
        print(f"  AVAS {spec['avas']} selected CAS({nelecas}e, {ncas}o)")

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    mc.conv_tol = 1e-11
    mc.conv_tol_grad = 1e-6
    mc.max_cycle_macro = max_cycle
    mc.fcisolver = fci.direct_spin1.FCI(mol)       # Ms = 0, S unconstrained
    mc = mc.state_average_(np.ones(nroots) / nroots)
    mc.kernel(mo)
    if not mc.converged:
        print("  WARNING: CASSCF did not converge; SOC results from an "
              "unconverged reference are not trustworthy.")
    return mol, mf, mc


def qd_nevpt2(mf, mc, soc):
    """QD-NEVPT2 on the given reference, with SOC (soc='DKH1' ...) or without
    (soc=None). Returns (energies, osc)."""
    import prism.interface
    import prism.nevpt
    interface = prism.interface.PYSCF(mf, mc, backend="opt_einsum")
    nevpt = prism.nevpt.NEVPT(interface)
    nevpt.method = "nevpt2"
    nevpt.method_type = "qd"
    if soc is not None:
        nevpt.soc = soc
    nevpt.verbose = 4
    res = nevpt.kernel()
    osc = None
    if isinstance(res, tuple):
        e_tot, osc = (res[0], res[2]) if len(res) == 3 else (res[0], None)
    else:
        e_tot = res
    e = np.sort(np.atleast_1d(np.asarray(e_tot, dtype=float)).ravel())
    return e, osc


def cluster_states(rel_ev, tol):
    clusters, cur = [], [0]
    for i in range(1, len(rel_ev)):
        if rel_ev[i] - rel_ev[cur[-1]] < tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return clusters


def report(spec, e_soc, osc, e_sf, tol):
    e0 = e_soc[0]
    rel = (e_soc - e0) * HARTREE2EV
    f = (np.atleast_1d(np.asarray(osc, dtype=float)).ravel()
         if osc is not None else None)

    print(f"\n  {len(e_soc)} spin-orbit states:")
    print(f"  {'i':>3}{'E / Ha':>20}{'dE / eV':>10}{'dE / nm':>10}{'f':>14}")
    for i, e in enumerate(e_soc):
        nm = f"{NM_PER_EV / rel[i]:10.1f}" if rel[i] > 1e-9 else f"{'--':>10}"
        fs = f"{f[i - 1]:.4e}" if f is not None and 0 < i <= f.size else ""
        print(f"  {i:>3}{e:20.10f}{rel[i]:10.4f}{nm}{fs:>14}")
    if f is None or f.size == 0:
        print("\n  !! no oscillator strengths returned.")
        return
    if float(np.max(np.abs(f))) == 0.0:
        print("\n  !! f is EXACTLY zero: nothing is being borrowed -- the "
              "singlet-triplet analogue\n     of water's mu = 0.0000 failure.")
        return

    # Cluster table, as HOCl's 03 -- informative, but for Br the triplet's
    # own SOC splitting can exceed a fixed tolerance, so the a 3A" band is
    # taken below as the LOWEST THREE EXCITED states, with their spread shown,
    # rather than trusted to the clustering.
    print(f"\n  clusters (tol {tol} eV):")
    for k, idx in enumerate(cluster_states(rel, tol)):
        ev = float(np.mean(rel[idx]))
        fs = float(sum(f[i - 1] for i in idx if 0 < i <= f.size))
        nm = f"{NM_PER_EV / ev:9.1f}" if ev > 1e-9 else f"{'--':>9}"
        print(f"    {k:>3}  n={len(idx)}  {ev:8.4f} eV {nm} nm  sum f {fs:.4e}")

    comp = [1, 2, 3]
    ev_c = rel[comp]
    f_c = float(sum(f[i - 1] for i in comp))
    centroid = float(np.mean(ev_c))
    spread_cm = (ev_c.max() - ev_c.min()) / HARTREE2EV * 219474.63
    gap_next = rel[4] - ev_c.max() if len(rel) > 4 else float("nan")
    print(f"\n  a 3A\" band = the lowest three excited SOC states:")
    print(f"      energies   {', '.join(f'{v:.4f}' for v in ev_c)} eV  "
          f"(spread {spread_cm:.0f} cm-1)")
    print(f"      centroid   {centroid:.4f} eV = {NM_PER_EV / centroid:.1f} nm")
    print(f"      summed f   {f_c:.4e}")
    print(f"      gap to the next state {gap_next:.3f} eV"
          + ("   !! small: components may be mixed with a neighbour"
             if gap_next < 3 * (ev_c.max() - ev_c.min()) + 0.05 else ""))

    # ---- spin-orbit shift of the band
    if e_sf is not None and e_sf.size:
        e0_sf = float(e_sf.min())
        t_abs = centroid / HARTREE2EV + e0
        i_t = int(np.argmin(np.abs(e_sf - t_abs)))
        vert_sf = (e_sf[i_t] - e0_sf) * HARTREE2EV
        match = abs(e_sf[i_t] - t_abs) * HARTREE2EV
        shift = centroid - vert_sf
        print(f"\n  spin-free QD-NEVPT2, same reference: "
              f"{', '.join(f'{(v - e0_sf) * HARTREE2EV:.4f}' for v in e_sf)} eV")
        print(f"      matched spin-free triplet {vert_sf:.4f} eV "
              f"(absolute match {match * 1000:.0f} meV"
              + ("  !! poor match, check by hand" if match > 0.3 else "") + ")")
        print(f"      ground state lowered by SOC "
              f"{(e0 - e0_sf) * HARTREE2EV * 1000:+.1f} meV")
        print(f"  -> SOC shifts the a 3A\" band by {shift * 1000:+.0f} meV: "
              f"{NM_PER_EV / vert_sf:.1f} -> {NM_PER_EV / centroid:.1f} nm "
              f"({NM_PER_EV / centroid - NM_PER_EV / vert_sf:+.1f} nm)")
        if spec["f_target"] is not None:
            print(f"     03's 1D band peaked at 437 nm with a spin-free surface;"
                  f" moved by this shift\n     it would sit near "
                  f"{NM_PER_EV / (NM_PER_EV / 437.0 + shift):.0f} nm, "
                  f"against {spec['obs_nm']:.0f} nm measured.")

    # ---- the intensity against what the band needs
    print()
    if spec["f_target"] is not None:
        wcm = spec["fwhm_ev"] / HARTREE2EV * 219474.63
        f_obs = 1.1296e12 * 1.0645 * spec["obs_sigma"] * wcm
        print(f"  f against measurement:")
        print(f"      computed (sum of 3 components)   {f_c:.3e}")
        print(f"      needed by 03's band model         {spec['f_target']:.1e}"
              f"  (a floor)   calc/needed {f_c / spec['f_target']:.2f}")
        print(f"      Gaussian estimate, Ingham peak x 03's width "
              f"{f_obs:.2e}   calc/obs {f_c / f_obs:.2f}")
        print("      (the SOC is ~7% low at this basis and operator -- the "
              "toolchain sweep --\n       so f carries ~14% from that alone. "
              "HOCl's equivalent came out 10-25x low.)")
    else:
        print(f"  CONTROL against HOCl's 03: vertical {centroid:.4f} eV "
              f"(03: {spec['ref_vertical']}), summed f {f_c:.2e} "
              f"(03: {spec['ref_f']:.1e})")
        ok = (abs(centroid - spec["ref_vertical"]) < 0.01
              and 0.8 < f_c / spec["ref_f"] < 1.25)
        print("  -> reproduced; the adaptation is sound." if ok else
              "  -> NOT reproduced. Check basis and active space against 03's "
              "run before trusting HOBr.")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--molecule", default="HOBr", choices=sorted(MOLECULES))
    p.add_argument("--basis", default="cc-pvtz-dk")
    p.add_argument("--soc", default="DKH1")
    p.add_argument("--geom", nargs=3, type=float, default=None,
                   metavar=("R_OX", "R_OH", "THETA"))
    p.add_argument("--cas", nargs=2, type=int, default=None,
                   metavar=("NCAS", "NELEC"))
    p.add_argument("--nroots", type=int, default=6)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--minao", default="ano")
    p.add_argument("--avas", nargs="+", default=None, metavar="LABEL",
                   help="override the AVAS labels. Full valence, which adds "
                        "sigma/sigma*(O-H): --avas 'Br 4p' 'O 2p' 'H 1s'. The "
                        "default CAS(12,7) has ONE virtual, so every state the "
                        "triplet can borrow from is n/pi -> sigma*(O-Br); the "
                        "~10x f deficit in both HOCl and HOBr is the reason to "
                        "try a larger space")
    p.add_argument("--cluster-tol", type=float, default=0.05)
    p.add_argument("--no-spin-free", action="store_true",
                   help="skip the spin-free QD-NEVPT2 run (no SOC shift)")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    spec = dict(MOLECULES[args.molecule])
    if args.avas:
        spec["avas"] = list(args.avas)
    geom = tuple(args.geom) if args.geom else spec["geom"]
    print("=" * 72)
    print(f"== {args.molecule} vertical a 3A\" with SOC: borrowed f and the "
          f"spin-orbit shift")
    print(f"== {args.basis}, {args.soc}-QD-NEVPT2, {args.nroots} Ms=0 roots, "
          f"geometry {geom}")
    print("=" * 72)
    t0 = time.time()
    try:
        mol, mf, mc = build_reference(spec, args.basis, geom,
                                      tuple(args.cas) if args.cas else None,
                                      args.nroots, args.max_cycle, args.minao,
                                      args.verbose)
        print(f"  reference {time.time() - t0:.1f} s (nao {mol.nao_nr()})")
        report_reference_spins(mc)

        e_sf = None
        if not args.no_spin_free:
            t1 = time.time()
            try:
                e_sf, _ = qd_nevpt2(mf, mc, soc=None)
                print(f"  spin-free QD-NEVPT2 {time.time() - t1:.1f} s")
            except Exception as exc:
                print(f"  spin-free QD-NEVPT2 FAILED ({type(exc).__name__}: "
                      f"{exc}); continuing without the SOC shift. Prism's log "
                      f"above still lists each state's spin-free NEVPT2 "
                      f"energy and multiplicity.")
        t1 = time.time()
        e_soc, osc = qd_nevpt2(mf, mc, soc=args.soc)
        print(f"  SOC QD-NEVPT2 {time.time() - t1:.1f} s")
        report(spec, e_soc, osc, e_sf, args.cluster_tol)
    except Exception as exc:
        print(f"\n  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    print(f"\n  total {time.time() - t0:.1f} s wall")


if __name__ == "__main__":
    main()
