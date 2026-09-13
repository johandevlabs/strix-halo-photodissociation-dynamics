#!/usr/bin/env python3
"""
Tier 2a: 1D cut through the Franck-Condon region of H2O.

Question this answers, before any raster is built: do SA-CASSCF and EE-ADC(2)
agree on the SHAPE of the transition dipole mu(r) across the FC window, not
just its value at equilibrium? For a total cross section only mu(r_eq) really
matters. For isotope effects the SLOPE matters, because the whole signal comes
from different isotopologues sampling the same surfaces through different
zero-point spreads.

Scans ONE OH bond with the other held fixed: the asymmetric coordinate,
which is the actual dissociation path (H + OH) and the one the isotope
effect lives in. Everything runs in Cs. A symmetric stretch would head
toward O + 2H instead, a channel the photochemistry never visits.

Computes at each geometry:
  SA-CASSCF(ne,no)  excitation energy + transition dipole (from the active
                    space transition 1-RDM; the core contributes nothing to
                    a transition density between orthogonal states)
  SC-NEVPT2         excitation energy on the SA-CASSCF reference
  EE-ADC(2)         excitation energy + oscillator strength

Headless-safe: matplotlib Agg, writes PNG + CSV, never opens a display.

Usage:
    python 03_tdm_check.py --basis aug-cc-pVDZ --npts 15
    python 03_tdm_check.py --quick          # 3 points, 6-31G, for smoke testing
"""
import argparse
import csv

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless server: no X, no Qt
import matplotlib.pyplot as plt

from pyscf import gto, scf, mcscf, mrpt, fci, adc
from pyscf.mcscf import avas

HARTREE2EV = 27.211386245988
DEG = np.pi / 180.0


def build_mol(r1, r2, angle_deg, basis, verbose=0):
    """
    O at the origin, H1 along z at r1, H2 at the bond angle at r2.

    symmetry="Cs" FORCES the subgroup rather than letting PySCF detect the
    point group. At r1 == r2 it would otherwise find C2v, the A'/A" labels
    would disappear mid-scan, and the irrep targeting would crash. Cs is
    also the symmetry of the real problem: photodissociation breaks one
    bond (H + OH), and HDO is Cs at every geometry.
    """
    a = angle_deg * DEG
    atom = [["O", (0.0, 0.0, 0.0)],
            ["H", (0.0, 0.0, r1)],
            ["H", (0.0, r2 * np.sin(a), r2 * np.cos(a))]]
    return gto.M(atom=atom, basis=basis, unit="Angstrom",
                 symmetry="Cs", verbose=verbose, max_memory=8000)


def casscf_point(mol, avas_aos, gs_sym, ex_sym):
    """
    Symmetry-targeted SA-CASSCF, averaging one root of each irrep.

    The state is fixed by irrep, and the ACTIVE SPACE is rebuilt from atomic
    character (AVAS) at every geometry rather than seeded from the previous
    point. Seeding looked sensible and was not: in a diffuse basis the aug
    functions lie close to sigma*, CASSCF rotates them into the active space,
    and the damage compounds along the scan. Symptoms were negative excitation
    energies and |mu| collapsing to 1e-4. AVAS is deterministic given the AO
    list, so there is nothing to propagate and nothing to degrade.

    Returns (e_gs, e_ex, |mu|, [casci_gs, casci_ex], mf, mo) with the two
    single-root CASCI objects that NEVPT2 will accept.
    """
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.kernel()

    ncas, nelecas, orbs = avas.avas(mf, avas_aos,
                                    canonicalize=False, verbose=0)

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    s_gs = fci.direct_spin0_symm.FCI(mol)
    s_gs.wfnsym, s_gs.nroots = gs_sym, 1
    s_ex = fci.direct_spin0_symm.FCI(mol)
    s_ex.wfnsym, s_ex.nroots = ex_sym, 1
    mcscf.state_average_mix_(mc, [s_gs, s_ex], [0.5, 0.5])
    mc.conv_tol = 1e-8
    mc.max_cycle_macro = 100
    mc.verbose = 0
    mc.kernel(orbs)

    # NEVPT2 refuses state-averaged solvers, so re-diagonalise each irrep
    # separately on the optimised SA orbitals: same orbitals, single root.
    cas, energies, civecs = [], [], []
    for sym in (gs_sym, ex_sym):
        mci = mcscf.CASCI(mf, ncas, nelecas)
        mci.fcisolver = fci.direct_spin0_symm.FCI(mol)
        mci.fcisolver.wfnsym = sym
        mci.verbose = 0
        mci.kernel(mc.mo_coeff)
        cas.append(mci)
        energies.append(mci.e_tot)
        civecs.append(mci.ci)

    ncore = cas[0].ncore
    mo_cas = mc.mo_coeff[:, ncore:ncore + ncas]
    t_dm1 = fci.direct_spin1.trans_rdm1(
        civecs[0], civecs[1], ncas, cas[0].nelecas)
    with mol.with_common_orig(mol.atom_coords().mean(axis=0)):
        dip_ao = mol.intor("int1e_r", comp=3)
    mu = -np.einsum("xij,ji->x", dip_ao, mo_cas @ t_dm1 @ mo_cas.T)

    return energies[0], energies[1], float(np.linalg.norm(mu)), cas, mf, ncas


def adc_point(mf, nroots):
    """
    EE-ADC(2): excitation energies (Ha) and oscillator strengths.

    Take the LOWEST root, not the brightest. The A state is the lowest
    excited singlet but is weak (f ~ 0.05); the B state above it is far
    brighter, so argmax(f) picks B every time and silently compares two
    different states.
    """
    myadc = adc.ADC(mf)
    myadc.method = "adc(2)"
    myadc.method_type = "ee"
    myadc.verbose = 0
    e, v, p, x = myadc.kernel(nroots=nroots)
    return np.asarray(e), np.asarray(p)


def osc_to_mu(f, de):
    """|mu| from oscillator strength: f = (2/3) dE |mu|^2, atomic units."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.clip(1.5 * f / de, 0.0, None))


def fc_sigma(omega_cm, red_mass_amu):
    """
    Gaussian width of the ground vibrational wavefunction in the stretch,
    sigma = sqrt(hbar / 2*mu*omega), returned in Angstrom.

    This sets how much of the scan actually matters. chi_0 has real amplitude
    over roughly r_eq +/- sigma; the dipole's behaviour outside that window is
    irrelevant to the cross section no matter how dramatic it looks on a plot.
    """
    omega_ha = omega_cm * 4.5563352812e-6
    mu_me = red_mass_amu * 1822.888486
    sigma_bohr = np.sqrt(1.0 / (2.0 * mu_me * omega_ha))
    return sigma_bohr * 0.529177210903


def log_slope(r, mu, mask=None):
    """
    d(ln mu)/dr. The RIGHT diagnostic: a constant factor on mu rescales the
    absolute cross section but leaves the band shape and any isotope
    fractionation untouched, so only FRACTIONAL variation across the FC
    window is physically meaningful. An absolute dmu/dr comparison between
    two methods whose magnitudes differ is misleading.
    """
    if mask is not None:
        r, mu = r[mask], mu[mask]
    if len(r) < 2 or np.any(mu <= 0):
        return np.nan
    return float(np.polyfit(r, np.log(mu), 1)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--basis", default="aug-cc-pVDZ")
    p.add_argument("--rmin", type=float, default=0.80)
    p.add_argument("--rmax", type=float, default=1.20)
    p.add_argument("--npts", type=int, default=15)
    p.add_argument("--angle", type=float, default=104.48)
    p.add_argument("--r2", type=float, default=0.958,
                   help="the SPECTATOR OH bond, held fixed while r1 scans")
    p.add_argument("--avas-aos", nargs="+",
                   default=["O 2s", "O 2p", "H 1s"],
                   help="atomic orbitals defining the AVAS active space; the "
                        "default gives full valence CAS(8,6) for water")
    p.add_argument("--gs-sym", default="A'", help="ground state irrep")
    p.add_argument("--ex-sym", default='A"',
                   help="excited state irrep; the A state is the lowest 1A\" "
                        "in Cs (the 1b1 hole is out of plane).")
    p.add_argument("--adc-roots", type=int, default=3)
    p.add_argument("--no-adc", action="store_true")
    p.add_argument("--no-nevpt2", action="store_true")
    p.add_argument("--quick", action="store_true",
                   help="3 points / 6-31G, just to check the machinery runs")
    p.add_argument("--r-eq", type=float, default=0.958,
                   help="equilibrium bond length, centre of the FC window")
    p.add_argument("--omega", type=float, default=3700.0,
                   help="stretch frequency in cm-1, for the FC window width")
    p.add_argument("--red-mass", type=float, default=0.9481,
                   help="reduced mass in amu (OH default; O-D is 1.789)")
    p.add_argument("--csv", default="tdm_scan.csv")
    p.add_argument("--png", default="tdm_scan.png")
    args = p.parse_args()

    if args.quick:
        args.basis, args.npts = "6-31G", 3

    rs = np.linspace(args.rmin, args.rmax, args.npts)

    print(f"basis={args.basis}  AVAS[{' '.join(args.avas_aos)}]  "
          f"{args.gs_sym}->{args.ex_sym}  angle={args.angle} deg")
    print(f"scanning r1 {args.rmin}-{args.rmax} A, {args.npts} points, "
          f"r2 fixed at {args.r2} A\n")
    print(f"{'r/A':>6} {'E_cas/eV':>9} {'E_nev/eV':>9} {'E_adc/eV':>9} "
          f"{'mu_cas':>8} {'mu_adc':>8}")

    rows = []
    for r in rs:
        mol = build_mol(r, args.r2, args.angle, args.basis)
        e_gs, e_ex, mu_cas, cas, mf, ncas_used = casscf_point(
            mol, args.avas_aos, args.gs_sym, args.ex_sym)

        de_cas = (e_ex - e_gs) * HARTREE2EV

        de_nev = np.nan
        if not args.no_nevpt2:
            c_gs = mrpt.NEVPT(cas[0]).kernel()
            c_ex = mrpt.NEVPT(cas[1]).kernel()
            de_nev = ((e_ex + c_ex) - (e_gs + c_gs)) * HARTREE2EV

        de_adc, mu_adc = np.nan, np.nan
        if not args.no_adc:
            e_a, f_a = adc_point(mf, args.adc_roots)
            k = 0  # lowest excited singlet == the A state
            de_adc = e_a[k] * HARTREE2EV
            mu_adc = float(osc_to_mu(f_a[k], e_a[k]))

        print(f"{r:6.3f} {de_cas:9.3f} {de_nev:9.3f} {de_adc:9.3f} "
              f"{mu_cas:8.4f} {mu_adc:8.4f}", flush=True)

        rows.append({"r_A": round(float(r), 4),
                     "e_casscf_eV": round(float(de_cas), 5),
                     "e_nevpt2_eV": round(float(de_nev), 5),
                     "e_adc2_eV": round(float(de_adc), 5),
                     "mu_casscf_au": round(float(mu_cas), 5),
                     "mu_adc2_au": round(float(mu_adc), 5)})

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.csv}")

    # --- plots -----------------------------------------------------------
    r = np.array([x["r_A"] for x in rows])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(r, [x["e_casscf_eV"] for x in rows], "o-", label="SA-CASSCF")
    if not args.no_nevpt2:
        ax[0].plot(r, [x["e_nevpt2_eV"] for x in rows], "s-", label="SC-NEVPT2")
    if not args.no_adc:
        ax[0].plot(r, [x["e_adc2_eV"] for x in rows], "^-", label="EE-ADC(2)")
    ax[0].set_xlabel("r1(OH) / A")
    ax[0].set_ylabel("excitation energy / eV")
    ax[0].set_title("Vertical excitation, FC window")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    mu_c = np.array([x["mu_casscf_au"] for x in rows])
    ax[1].plot(r, mu_c, "o-", label="SA-CASSCF")
    if not args.no_adc:
        mu_a = np.array([x["mu_adc2_au"] for x in rows])
        ax[1].plot(r, mu_a, "^-", label="EE-ADC(2)")
    ax[1].set_xlabel("r1(OH) / A")
    ax[1].set_ylabel(r"$|\mu_{0\to A}|$ / a.u.")
    ax[1].set_title("Transition dipole (shape is what matters)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"wrote {args.png}")

    # --- slope comparison -------------------------------------------------
    sigma = fc_sigma(args.omega, args.red_mass)
    lo, hi = args.r_eq - sigma, args.r_eq + sigma
    fc = (r >= lo) & (r <= hi)

    print(f"\nFC window: r_eq {args.r_eq} +/- sigma {sigma:.4f} A "
          f"=> [{lo:.3f}, {hi:.3f}]  ({fc.sum()} of {len(r)} scan points)")
    if fc.sum() < 3:
        print("  warning: few points inside the FC window; "
              "narrow --rmin/--rmax or raise --npts for a meaningful fit")

    print(f"\n{'':12s}{'full scan':>22s}{'FC window':>22s}")
    print(f"{'':12s}{'dlnmu/dr':>11s}{'delta mu':>11s}"
          f"{'dlnmu/dr':>11s}{'delta mu':>11s}")

    curves = [("SA-CASSCF", mu_c)]
    if not args.no_adc:
        curves.append(("EE-ADC(2)", np.array([x["mu_adc2_au"] for x in rows])))

    slopes = {}
    for name, mu in curves:
        if not np.all(np.isfinite(mu)) or np.any(mu <= 0):
            continue
        s_all, s_fc = log_slope(r, mu), log_slope(r, mu, fc)
        d_all = mu[-1] / mu[0] - 1.0
        d_fc = (mu[fc][-1] / mu[fc][0] - 1.0) if fc.sum() >= 2 else np.nan
        slopes[name] = s_fc
        print(f"{name:12s}{s_all:10.3f}{100*d_all:10.1f}%"
              f"{s_fc:11.3f}{100*d_fc:10.1f}%")

    if len(slopes) == 2:
        a, b = list(slopes.values())
        if abs(b) > 1e-10:
            print(f"\nratio of FC-window log slopes: {a / b:.2f}")
            print("Near 1.0 means the methods agree on the SHAPE of mu where "
                  "chi_0 lives.\nA large ratio is only a problem if it moves "
                  "the observable: test that\nby recomputing the cross section "
                  "with constant mu and with each slope.")


if __name__ == "__main__":
    main()
