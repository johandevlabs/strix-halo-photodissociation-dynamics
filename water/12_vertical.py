#!/usr/bin/env python3
"""
Tier 3f: is the A-X vertical excitation energy converged?

The computed band sits 0.067 eV below the measurement. About a third of that
is the ground-state ZPE coming out 174 cm-1 too high (fix the ground surface,
not the spectrum). The rest is the vertical excitation energy itself, and
this script tests whether it is converged with respect to the three things
that could be limiting it:

    BASIS         aug-cc-pVDZ -> pVTZ -> pVQZ
    ACTIVE SPACE  full valence CAS(8,6) -> adding Rydberg/correlating orbitals
    METHOD        SA-CASSCF, SC-NEVPT2, EE-ADC(2), EE-ADC(3), EOM-EE-CCSD

EOM-CCSD is the useful independent check here. It gives no transition moments
in PySCF (which is why it was not used for mu), but the A state is dominated
by a single excitation at the equilibrium geometry, so its ENERGY is reliable
and it shares no machinery with CASSCF.

If the answer drifts upward with basis or active space, the production
surface is not converged and the deficit is a real limitation. If it is flat,
the remaining offset is something else -- most likely the shape of the
surface away from the vertical point, which the band maximum also samples.

Note: the band maximum is NOT the vertical excitation energy. The packet has
zero-point spread and the surface is steeply repulsive, so the peak sits
above V_ex at the minimum by the amount the spread samples. Compare trends
here, not absolute agreement with 7.47 eV.

Usage:
    python 12_vertical.py
    python 12_vertical.py --bases aug-cc-pVDZ aug-cc-pVTZ --quick
"""
import argparse
import time

import numpy as np

from pyscf import gto, scf, mcscf, fci, mrpt, adc, cc
from pyscf.mcscf import avas

HARTREE2EV = 27.211386245988
DEG = np.pi / 180.0

# AVAS target orbitals -> active space. Full valence is (8,6) for water;
# adding O 3s picks up the Rydberg character the A state partly carries in a
# diffuse basis, which CAS(8,6) cannot describe.
# With minao='ano' these give CAS(8,6), CAS(8,7) and CAS(8,10). With the
# AVAS default minao='minao' all three collapse to (8,6), because that
# reference basis has no O 3s to project onto and the request is dropped
# without warning.
AVAS_SETS = {
    "valence": ["O 2s", "O 2p", "H 1s"],
    "+O3s": ["O 2s", "O 2p", "O 3s", "H 1s"],
    "+O3s3p": ["O 2s", "O 2p", "O 3s", "O 3p", "H 1s"],
}


def build(r1, r2, theta, basis):
    a = theta * DEG
    atom = [["O", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, r1)],
            ["H", (0.0, r2 * np.sin(a), r2 * np.cos(a))]]
    return gto.M(atom=atom, basis=basis, unit="Angstrom",
                 symmetry="Cs", verbose=0, max_memory=8000)


def casscf_nevpt2(mol, aos, minao):
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()
    ncas, ne, orbs = avas.avas(mf, aos, canonicalize=False,
                               verbose=0, minao=minao)
    mc = mcscf.CASSCF(mf, ncas, ne)
    s0 = fci.direct_spin0_symm.FCI(mol)
    s0.wfnsym, s0.nroots = "A'", 1
    s1 = fci.direct_spin0_symm.FCI(mol)
    s1.wfnsym, s1.nroots = 'A"', 1
    mcscf.state_average_mix_(mc, [s0, s1], [0.5, 0.5])
    mc.verbose = 0
    mc.conv_tol = 1e-8
    mc.kernel(orbs)

    e_cas, e_nev = [], []
    for sym in ("A'", 'A"'):
        mci = mcscf.CASCI(mf, ncas, ne)
        mci.fcisolver = fci.direct_spin0_symm.FCI(mol)
        mci.fcisolver.wfnsym = sym
        mci.verbose = 0
        mci.kernel(mc.mo_coeff)
        e_cas.append(mci.e_tot)
        e_nev.append(mci.e_tot + mrpt.NEVPT(mci).kernel())
    return ((e_cas[1] - e_cas[0]) * HARTREE2EV,
            (e_nev[1] - e_nev[0]) * HARTREE2EV, ncas, ne, mf)


def adc_vertical(mf, method, nroots=3):
    a = adc.ADC(mf)
    a.method = method
    a.method_type = "ee"
    a.verbose = 0
    e, v, p, x = a.kernel(nroots=nroots)
    return e[0] * HARTREE2EV          # lowest excited singlet == A state


def eomccsd_vertical(mf, nroots=3):
    mycc = cc.CCSD(mf)
    mycc.verbose = 0
    mycc.kernel()
    e = mycc.eomee_ccsd_singlet(nroots=nroots)[0]
    e = np.atleast_1d(e)
    return float(np.min(e)) * HARTREE2EV


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bases", nargs="+",
                   default=["aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ"])
    p.add_argument("--spaces", nargs="+", default=list(AVAS_SETS),
                   choices=list(AVAS_SETS))
    p.add_argument("--r", type=float, default=0.9578,
                   help="equilibrium OH bond length, Angstrom")
    p.add_argument("--theta", type=float, default=104.48)
    p.add_argument("--no-cc", action="store_true",
                   help="skip EOM-CCSD (the expensive one at pVQZ)")
    p.add_argument("--minao", default="ano",
                   help="reference basis AVAS projects onto. The DEFAULT "
                        "'minao' has no O 3s, so asking for Rydberg orbitals "
                        "silently returns the same valence CAS(8,6); 'ano' "
                        "gives (8,7) and (8,10) as intended.")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.bases = ["aug-cc-pVDZ"]
        args.spaces = ["valence"]

    print(f"geometry r(OH) = {args.r} A, angle = {args.theta} deg\n")
    print(f"{'basis':>14}{'space':>14}{'CAS':>9}{'SA-CASSCF':>11}"
          f"{'NEVPT2':>9}{'ADC(2)':>9}{'ADC(3)':>9}{'EOM-CCSD':>10}{'t/s':>7}")

    for basis in args.bases:
        mol = build(args.r, args.r, args.theta, basis)
        e_adc2 = e_adc3 = e_cc = np.nan
        first = True
        for sname in args.spaces:
            t0 = time.perf_counter()
            try:
                e_cas, e_nev, ncas, ne, mf = casscf_nevpt2(
                    mol, AVAS_SETS[sname], args.minao)
            except Exception as ex:
                print(f"{basis:>14}{sname:>14}  failed: {type(ex).__name__}")
                continue

            # method comparisons are geometry- and basis-dependent only,
            # so compute them once per basis
            if first:
                try:
                    e_adc2 = adc_vertical(mf, "adc(2)")
                except Exception:
                    pass
                try:
                    e_adc3 = adc_vertical(mf, "adc(3)")
                except Exception:
                    pass
                if not args.no_cc:
                    try:
                        e_cc = eomccsd_vertical(mf)
                    except Exception:
                        pass
                first = False

            print(f"{basis:>14}{sname:>14}{f'({ne},{ncas})':>9}"
                  f"{e_cas:11.3f}{e_nev:9.3f}{e_adc2:9.3f}{e_adc3:9.3f}"
                  f"{e_cc:10.3f}{time.perf_counter()-t0:7.1f}", flush=True)

    print("\nproduction surface gives 7.579 eV at its own minimum "
          "(aug-cc-pVTZ, CAS(8,6), NEVPT2)")
    print("band maximum: computed 7.402 eV, measured 7.469 eV")
    print("\nThe band maximum is not the vertical energy -- zero-point spread")
    print("on a repulsive surface puts the peak above V_ex(r_e). Read the")
    print("TRENDS across basis and active space, not absolute agreement.")


if __name__ == "__main__":
    main()
