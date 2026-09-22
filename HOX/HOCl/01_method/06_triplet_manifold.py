#!/usr/bin/env python3
"""
Phase 0.5, step 7: is the kink in the a 3A" curve an avoided crossing?

05 found the state-specific UCCSD(T) triplet going down, up and down again
between 2.2 and 2.8 A, with T1 spiking to ~0.10. Pinning the irrep occupation
changed nothing, and unconstrained aufbau never left 3A", so the kink is not an
A'/A" flip. Looking at the plots, Johan's reading was that a HIGHER 3A" state
appears to be coming down and meeting ours. That fits everything 05 showed:

  - any state that crowds ours must also be 3A" (a 3A' would CROSS, aufbau
    would have jumped to it, and --pin-irrep would have flagged it), and
    same-symmetry states repel rather than cross;
  - T1 settles at a DIFFERENT level on each side (0.024-0.030 inside,
    0.014-0.018 outside), which is what a change of dominant configuration
    looks like, not a method gradually failing;
  - OH(2Pi) + Cl(2P) gives 3 x 2 = 6 spatial states, each singlet and triplet,
    so several 3A" states converge on the same asymptote.

A single-determinant method cannot follow an avoided crossing: it describes
one configuration, gets stuck halfway through the swap, then describes the
other. So the kink tells us the method failed there, not what the true curve
looks like. This script computes the true curves directly.

METHOD. Several 3A" roots TOGETHER along the same O-Cl cut: state-averaged
CASSCF with the wavefunction symmetry forced to A" and the spin held at
S = 1, then SC-NEVPT2 on each root. Built exactly as water/03_tdm_check.py
builds its states, which ran thousands of points on this machine:

  - AVAS rebuilds the active space at EVERY geometry from atomic character,
    never seeded from the previous point (water/README.md: seeding rotated
    diffuse functions into the active space and compounded along the scan).
  - Symmetry is FORCED to Cs rather than detected.
  - NEVPT2 refuses state-averaged solvers, so each root is re-diagonalised as
    a CASCI on the optimised state-averaged orbitals.

AVAS runs on the closed-shell RHF of the same geometry and its orbitals start
the triplet calculation, so every point uses one well-defined active space.
Its orbitals are then re-symmetrised block by block, because near-degenerate
Cl 3p functions at long range can come out of AVAS mixed across irreps, which
a symmetry-forced CASSCF rejects.

WHAT IT MEASURES, all independent of how the active orbitals happen to be
ordered at each geometry (which changes along a scan):

  gap(r)       E(root 1) - E(root 0). An avoided crossing is a minimum.
  weight       c^2 of each root's dominant configuration. At a crossing the
               two roots become mixtures, so both weights dip together.
  |dipole|     of roots 0 and 1. Two diabatic states have different charge
               distributions; if the roots EXCHANGE dipoles across the gap
               minimum, they have exchanged character. That is the most
               direct evidence of an avoided crossing available from energies
               and densities alone.

WHY IT MATTERS FOR THE SPECTRUM. A wavepacket arriving fast at a narrowly
avoided crossing tends to stay on its diabatic state and jump the gap rather
than follow the lower adiabat (the Landau-Zener picture). The script gives a
rough Landau-Zener probability from the computed gap, the diabatic slopes and
the kinetic energy gained falling from the Franck-Condon region. A large
probability says a single adiabatic surface is the wrong thing to propagate
on, however accurately it is computed.

COST. SA-CASSCF over 4 roots plus 4 NEVPT2 solves per point. Roughly an hour
for the default 16 points; --no-nevpt2 roughly halves it and is enough to see
whether a crossing is there at all.

Usage:
    python 06_triplet_manifold.py 2>&1 | tee logs/triplet_manifold.log
    python 06_triplet_manifold.py --no-nevpt2                  # quick look
    python 06_triplet_manifold.py --rmin 2.3 --rmax 2.7 --npoints 17
    python 06_triplet_manifold.py --avas-aos "Cl 3p" "O 2p" "H 1s"
"""
import argparse
import csv
import os
import time
import traceback

import numpy as np

from pyscf import gto, scf, mcscf, mrpt, fci, symm
from pyscf.mcscf import avas
from pyscf.fci import spin_op
from pyscf.scf import hf as scf_hf
from pathlib import Path

# Data files live in HOCl/data/, one level up from this approach directory,
# so the defaults below work no matter where the script is invoked from.
DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2EV = 27.211386245988
BOHR_PER_ANG = 1.0 / 0.529177210903
AMU2AU = 1822.888486209

# Same cut as 05: O-Cl stretched at fixed r(OH) and angle. Geometry quoted from
# memory during the work -- verify before citing, as elsewhere in HOX/.
R_OH_ANG = 0.9644
R_OCL_EQ_ANG = 1.6891
ANGLE_HOCL_DEG = 102.96

M_O, M_H, M_CL = 15.99491462, 1.00782503, 34.96885268


def geometry(r_ocl):
    th = np.deg2rad(ANGLE_HOCL_DEG)
    return [["O", (0.0, 0.0, 0.0)],
            ["Cl", (r_ocl, 0.0, 0.0)],
            ["H", (R_OH_ANG * np.cos(th), R_OH_ANG * np.sin(th), 0.0)]]


def build_mol(r, spin, basis, verbose):
    return gto.M(atom=geometry(r), basis=basis, spin=spin, charge=0,
                 symmetry="Cs", unit="Angstrom", verbose=verbose,
                 max_memory=16000)


def symmetrize_blocks(mol, mo, s, ncore, ncas):
    """Re-symmetrise core, active and virtual blocks separately.

    Rotating within a block leaves the CASSCF problem unchanged, but makes
    every orbital pure A' or A". Raises if a block is not closed under the
    mirror plane, which would mean AVAS split a degenerate pair between
    active and inactive -- worth knowing about, not papering over.
    """
    out = mo.copy()
    nmo = mo.shape[1]
    for sl in (slice(0, ncore), slice(ncore, ncore + ncas),
               slice(ncore + ncas, nmo)):
        if out[:, sl].shape[1]:
            out[:, sl] = symm.symmetrize_space(mol, out[:, sl], s=s)
    return out


def triplet_solver(mol, nroots):
    solver = fci.direct_spin1_symm.FCI(mol)
    solver.wfnsym = 'A"'
    solver.nroots = nroots
    fci.addons.fix_spin_(solver, ss=2.0)          # S = 1 only
    return solver


def dominant_weight(ci, ncas, nelecas):
    try:
        big = fci.addons.large_ci(ci, ncas, nelecas, tol=0.05,
                                  return_strs=False)
        return float(max(c * c for c, *_ in big)) if big else float("nan")
    except Exception:
        return float("nan")


def run_point(r, args):
    """Several 3A" roots at one geometry. Returns a dict."""
    c0, t0 = time.process_time(), time.time()
    row = {"r": r}

    # Active space from the closed-shell reference at this geometry.
    mol_s = build_mol(r, 0, args.basis, args.verbose)
    mf_s = scf.RHF(mol_s).x2c()
    mf_s.conv_tol = 1e-10
    mf_s.kernel()
    ncas, nelecas, orbs = avas.avas(mf_s, args.avas_aos, minao=args.minao,
                                    verbose=0)
    ncore = (mol_s.nelectron - nelecas) // 2
    try:
        orbs = symmetrize_blocks(mol_s, orbs, mf_s.get_ovlp(), ncore, ncas)
    except Exception as exc:
        row["note"] = f"symmetrize failed: {exc}"
    row["ncas"], row["nelecas"] = ncas, nelecas

    # Triplet on the same geometry. Symmetry detection depends only on the
    # nuclei, so the frame is the singlet's and the AVAS orbitals transfer;
    # checked rather than assumed.
    mol_t = build_mol(r, 2, args.basis, args.verbose)
    if not np.allclose(mol_s.atom_coords(), mol_t.atom_coords(), atol=1e-8):
        raise RuntimeError("singlet and triplet frames differ; the AVAS "
                           "orbitals would not transfer")
    mf_t = scf.ROHF(mol_t).x2c()
    mf_t.conv_tol = 1e-10
    mf_t.kernel()

    n = args.nroots
    mc = mcscf.CASSCF(mf_t, ncas, nelecas)
    mc.fcisolver = triplet_solver(mol_t, n)
    mc = mc.state_average_(np.ones(n) / n)
    mc.conv_tol = 1e-8
    mc.max_cycle_macro = args.max_cycle
    mc.verbose = 0
    mc.kernel(orbs)
    row["conv"] = bool(mc.converged)

    # NEVPT2 refuses state-averaged solvers: re-diagonalise on the SA orbitals.
    mci = mcscf.CASCI(mf_t, ncas, nelecas)
    mci.fcisolver = triplet_solver(mol_t, n)
    mci.verbose = 0
    mci.kernel(mc.mo_coeff)
    e_cas = np.atleast_1d(np.asarray(mci.e_tot, dtype=float))
    cis = mci.ci if isinstance(mci.ci, (list, tuple)) else [mci.ci]
    nel = mci.nelecas

    # Everything derived from the CI vectors is computed BEFORE any NEVPT2.
    # PySCF's NEVPT copies the CASCI object's attributes by reference
    # (self.__dict__.update(mc.__dict__)) and its kernel then writes
    # self.ci[root] = <CI in rotated natural orbitals> into that shared list,
    # while the CASCI object keeps the old orbitals. A density built after
    # NEVPT2 therefore pairs new CI with old orbitals. In the first version of
    # this script the dipoles were computed after the loop and came out wrong
    # -- rising linearly to 6 D at 3.6 A -- while the CASSCF-only run, which
    # never calls NEVPT, gave a sensible ~1.8 D. Energies were unaffected:
    # each root's NEVPT2 reads its own untouched vector and the orbitals.
    for i in range(n):
        row[f"e_cas_{i}"] = float(e_cas[i])
        row[f"s2_{i}"] = float(spin_op.spin_square0(cis[i], ncas, nel)[0])
        row[f"w_{i}"] = dominant_weight(cis[i], ncas, nel)
    for i in (0, 1):
        dm = mci.make_rdm1(ci=cis[i])
        d = scf_hf.dip_moment(mol_t, dm, unit="Debye", verbose=0)
        row[f"dip_{i}"] = float(np.linalg.norm(d))

    if not args.no_nevpt2:
        for i in range(n):
            row[f"e_nev_{i}"] = float(e_cas[i]
                                      + mrpt.NEVPT(mci, root=i).kernel())

    row["wall"], row["cpu"] = time.time() - t0, time.process_time() - c0
    return row


def energies(rows, i, key):
    return np.array([r[f"{key}_{i}"] for r in rows])


def landau_zener(rows, key, imin):
    """Rough 1D Landau-Zener probability of staying diabatic at the gap minimum.

    P = exp(-2 pi H12^2 / (v |dF|)), atomic units, with
      H12 = gap_min / 2,
      dF  = difference of the two diabatic slopes, taken from roots 0 and 1 on
            the INNER side of the minimum, where each is still one diabat,
      v   = classical speed along r(O-Cl) after falling from the first grid
            point to the crossing, with OH moving as a unit against Cl.
    """
    if imin < 2:
        return None, "gap minimum too close to the inner edge for slopes"
    r_b = np.array([r["r"] for r in rows]) * BOHR_PER_ANG
    e0, e1 = energies(rows, 0, key), energies(rows, 1, key)
    j, k = imin - 2, imin - 1
    f0 = (e0[k] - e0[j]) / (r_b[k] - r_b[j])
    f1 = (e1[k] - e1[j]) / (r_b[k] - r_b[j])
    df = abs(f0 - f1)
    h12 = (e1[imin] - e0[imin]) / 2.0
    e_cross = (e0[imin] + e1[imin]) / 2.0
    e_kin = e0[0] - e_cross
    if e_kin <= 0 or df == 0:
        return None, "no kinetic energy or parallel diabats at the crossing"
    m_oh = M_O + M_H
    mu = m_oh * M_CL / (m_oh + M_CL) * AMU2AU
    v = np.sqrt(2.0 * e_kin / mu)
    p = float(np.exp(-2.0 * np.pi * h12 ** 2 / (v * df)))
    detail = (f"H12 {h12 * HARTREE2EV * 1000:.1f} meV, dF {df:.4f} Ha/bohr, "
              f"E_kin {e_kin * HARTREE2EV:.2f} eV, v {v:.2e} au")
    return p, detail


def load_compare(path):
    if not path or not os.path.exists(path):
        return None
    out = []
    with open(path) as fh:
        for rec in csv.DictReader(fh):
            try:
                out.append((float(rec["r"]), float(rec["e_t"]),
                            float(rec["t1_t"])))
            except (KeyError, ValueError):
                continue
    return out or None


def make_plot(rows, key, n, path, compare):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.array([x["r"] for x in rows])
    ref = rows[-1][f"{key}_0"]
    fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1, 1]})
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"]
    for i in range(n):
        ax[0].plot(r, (energies(rows, i, key) - ref) * HARTREE2EV, "o-",
                   ms=4, color=colors[i % len(colors)],
                   label=f'3A" root {i}')
    if compare:
        # Different method: shift UCCSD(T) to coincide with root 0 at the
        # outermost common radius, so only the SHAPE is compared.
        cr = np.array([c[0] for c in compare])
        ce = np.array([c[1] for c in compare])
        j = int(np.argmin(np.abs(cr - r[-1])))
        ax[0].plot(cr, (ce - ce[j]) * HARTREE2EV, "k--", lw=1,
                   label="05 UCCSD(T), shifted")
        ct1 = np.array([c[2] for c in compare])
        ax[0].axvline(cr[int(np.argmax(ct1))], color="grey", ls=":", lw=1,
                      label="05 max T1")
    ax[0].axvline(R_OCL_EQ_ANG, color="k", lw=0.6)
    ax[0].set_ylabel(f"E - E(root 0, r_max) / eV  [{key}]")
    ax[0].legend(fontsize=7)
    ax[0].set_title('HOCl 3A" manifold along r(O-Cl)')

    gap = (energies(rows, 1, key) - energies(rows, 0, key)) * HARTREE2EV
    ax[1].plot(r, gap, "s-", color="k")
    ax[1].set_ylabel("gap 1-0 / eV")

    ax[2].plot(r, [x["dip_0"] for x in rows], "o-", color=colors[0],
               label="|mu| root 0")
    ax[2].plot(r, [x["dip_1"] for x in rows], "o-", color=colors[1],
               label="|mu| root 1")
    ax[2].set_ylabel("|dipole| / D")
    ax[2].set_xlabel("r(O-Cl) / A")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--avas-aos", nargs="+", default=["Cl 3p", "O 2p"])
    p.add_argument("--minao", default="ano",
                   help="AVAS reference basis. The default minao silently "
                        "drops orbitals it has no reference for "
                        "(water/README.md)")
    p.add_argument("--nroots", type=int, default=4)
    p.add_argument("--rmin", type=float, default=1.7)
    p.add_argument("--rmax", type=float, default=3.2)
    p.add_argument("--npoints", type=int, default=16)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--no-nevpt2", action="store_true",
                   help="CASSCF energies only; enough to see a crossing")
    p.add_argument("--compare", default=str(DATA / "hocl_scan_pin.csv"),
                   help="05 output to overlay (shape only)")
    p.add_argument("--csv", default=str(DATA / "hocl_triplet_manifold.csv"))
    p.add_argument("--png", default=str(DATA / "hocl_triplet_manifold.png"))
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    if args.nroots < 2:
        p.error("--nroots must be at least 2: a crossing needs two states")
    key = "e_cas" if args.no_nevpt2 else "e_nev"
    n = args.nroots
    rs = np.linspace(args.rmin, args.rmax, args.npoints)

    print("=" * 78)
    print('== HOX Phase 0.5 -- the 3A" manifold along the O-Cl cut')
    print(f"== {n} roots, AVAS {args.avas_aos} (minao {args.minao}), "
          f"{args.basis}, {'CASSCF' if args.no_nevpt2 else 'SC-NEVPT2'}")
    print(f"== r = {args.rmin} to {args.rmax} A in {args.npoints} points")
    print("=" * 78)
    print(f"\n  {'r/A':>6} {'CAS':>8} {'conv':>4}"
          + "".join(f"{'E' + str(i) + '/Ha':>17}" for i in range(n))
          + f"{'gap/eV':>8}{'w0':>6}{'w1':>6}{'|mu0|':>7}{'|mu1|':>7}{'t/s':>6}")

    rows, t_start, c_start = [], time.time(), time.process_time()
    for r in rs:
        try:
            row = run_point(r, args)
        except Exception as exc:
            print(f"  {r:6.3f}  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        rows.append(row)
        gap = (row[f"{key}_1"] - row[f"{key}_0"]) * HARTREE2EV
        cas = f"({row['nelecas']},{row['ncas']})"
        flags = []
        if any(abs(row[f"s2_{i}"] - 2.0) > 0.01 for i in range(n)):
            flags.append("S2!")
        if row.get("note"):
            flags.append("sym!")
        print(f"  {r:6.3f} {cas:>8} {'yes' if row['conv'] else 'NO':>4}"
              + "".join(f"{row[f'{key}_{i}']:17.8f}" for i in range(n))
              + f"{gap:8.3f}{row['w_0']:6.2f}{row['w_1']:6.2f}"
              f"{row['dip_0']:7.2f}{row['dip_1']:7.2f}{row['wall']:6.0f}"
              f"  {' '.join(flags)}")

    if len(rows) < 3:
        print("\n  too few successful points to analyse")
        return

    # ------------------------------------------------------------------ csv
    cols = (["r", "ncas", "nelecas", "conv"]
            + [f"e_cas_{i}" for i in range(n)]
            + ([] if args.no_nevpt2 else [f"e_nev_{i}" for i in range(n)])
            + [f"s2_{i}" for i in range(n)] + [f"w_{i}" for i in range(n)]
            + ["dip_0", "dip_1", "wall", "cpu", "note"])
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})
    print(f"\n  wrote {args.csv}")
    compare = load_compare(args.compare)
    try:
        make_plot(rows, key, n, args.png, compare)
    except Exception as exc:
        print(f"  plot FAILED: {type(exc).__name__}: {exc}")

    # -------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    unconv = [x["r"] for x in rows if not x["conv"]]
    if unconv:
        print(f"  !! CASSCF unconverged at r = "
              f"{', '.join(f'{v:.2f}' for v in unconv)} A")

    # Analyse only the points sharing the most common active space. A point
    # with a different space sits on a different energy scale. In the first
    # run that was the innermost point, 1.7 A, at (12,7) against (10,6)
    # everywhere else -- and the Landau-Zener estimate took its starting
    # energy from exactly that point.
    sizes = {}
    for x in rows:
        sz = (int(x["nelecas"]), int(x["ncas"]))
        sizes[sz] = sizes.get(sz, 0) + 1
    common = max(sizes, key=sizes.get)
    ana = [x for x in rows
           if (int(x["nelecas"]), int(x["ncas"])) == common]
    if len(sizes) > 1:
        dropped = [x["r"] for x in rows if x not in ana]
        print(f"  !! AVAS active space varies along the scan "
              f"(CAS(nelec,norb): points) {sizes}.")
        print(f"     Analysing only the CAS{common} points; excluded r = "
              f"{', '.join(f'{v:.2f}' for v in dropped)} A.")
    if len(ana) < 3:
        print("  too few points share one active space to analyse")
        return

    r = np.array([x["r"] for x in ana])
    gap = (energies(ana, 1, key) - energies(ana, 0, key)) * HARTREE2EV
    imin = int(np.argmin(gap))
    w0 = np.array([x["w_0"] for x in ana])
    w1 = np.array([x["w_1"] for x in ana])
    d0 = np.array([x["dip_0"] for x in ana])
    d1 = np.array([x["dip_1"] for x in ana])

    print(f"  smallest root 1 - root 0 gap: {gap[imin]:.3f} eV at "
          f"r = {r[imin]:.2f} A  (grid spacing {r[1] - r[0]:.3f} A)")
    lo, hi = max(imin - 2, 0), min(imin + 2, len(r) - 1)
    swap = (d0[lo] - d1[lo]) * (d0[hi] - d1[hi]) < 0
    dip_note = (f"root 0 |mu| {d0[lo]:.2f} -> {d0[hi]:.2f} D, "
                f"root 1 {d1[lo]:.2f} -> {d1[hi]:.2f} D "
                f"across r = {r[lo]:.2f}-{r[hi]:.2f} A")
    print(f"  dipoles: {dip_note}"
          f"  -> {'EXCHANGED' if swap else 'not exchanged'}")
    # Mixing is judged as a DIP relative to each root's own typical weight,
    # not against a fixed cutoff: a correlated CASSCF root sits well below
    # c^2 = 1 even far from any crossing, and strongly coupled states are
    # mixed everywhere, which is not the same thing as mixing AT a crossing.
    dip0 = float(np.nanmedian(w0) - np.nanmin(w0[lo:hi + 1]))
    dip1 = float(np.nanmedian(w1) - np.nanmin(w1[lo:hi + 1]))
    mixed = dip0 > 0.15 and dip1 > 0.15
    print(f"  dominant-configuration weight dip near the minimum: root 0 "
          f"{dip0:.2f}, root 1 {dip1:.2f} below their scan medians"
          f"  -> {'both mix there' if mixed else 'no localised mixing'}")

    if compare:
        cr = np.array([c[0] for c in compare])
        ct1 = np.array([c[2] for c in compare])
        r_t1 = cr[int(np.argmax(ct1))]
        print(f"  05 UCCSD(T) max T1 at r = {r_t1:.2f} A, "
              f"{abs(r_t1 - r[imin]):.2f} A from the gap minimum")

    # A minimum on the edge of the grid is not a minimum: the gap is still
    # heading somewhere the scan did not reach. The first version reported
    # this as INCONCLUSIVE and suggested a narrower window around the edge,
    # which only moves the edge; two follow-up runs chased it from 3.2 to
    # 3.6 A with the gap still closing. Say what it actually is instead.
    if imin == len(r) - 1:
        e_last = [ana[-1][f"{key}_{i}"] for i in range(n)]
        near = [i for i in range(n)
                if (e_last[i] - e_last[0]) * HARTREE2EV < 0.15]
        print(f"\n  -> NO AVOIDED CROSSING between {r[0]:.2f} and {r[-1]:.2f} A."
              f" The gap closes\n     monotonically all the way to the outer "
              f"edge, so the minimum is the grid\n     running out, not a "
              f"feature. At r = {r[-1]:.2f} A roots {near} lie within "
              f"0.15 eV\n     of root 0: they are converging on a shared "
              f"asymptote. OH(2Pi) + Cl(2P)\n     gives exactly three 3A\" "
              f"states. Scanning further out only follows that\n     "
              f"convergence; it will not find a minimum.")
        print("\n  Landau-Zener: not applicable (no interior gap minimum).")
    elif imin == 0:
        print(f"\n  -> NO AVOIDED CROSSING between {r[0]:.2f} and {r[-1]:.2f} A."
              f" The gap is smallest at\n     the inner edge and opens "
              f"outward.")
        print("\n  Landau-Zener: not applicable (no interior gap minimum).")
    elif gap[imin] < 0.3 and (swap or mixed):
        print("\n  -> AVOIDED CROSSING. A higher 3A\" state comes down and the "
              "two exchange\n     character at the gap minimum. The kink "
              "in 05 is the single-reference\n     method failing to follow "
              "that exchange, not a feature of the true curve.")
    elif gap[imin] > 0.5 and swap:
        print("\n  -> BROAD, STRONGLY AVOIDED CROSSING. The roots exchange "
              "character but the\n     gap stays large, so the lower curve "
              "changes character smoothly and a\n     packet follows it. The "
              "kink in 05 is the single-reference method failing\n     "
              "through that smooth change.")
    elif gap[imin] > 0.5:
        print("\n  -> NO nearby 3A\" state. The gap stays open and the roots "
              "keep their\n     character, so the kink in 05 has no crossing "
              "behind it. Look at the SCF\n     reference instead.")
    else:
        print("\n  -> INCONCLUSIVE at this resolution. Rerun on a narrow "
              f"window around {r[imin]:.2f} A,\n     e.g. --rmin "
              f"{max(r[imin] - 0.2, args.rmin):.2f} --rmax "
              f"{r[imin] + 0.2:.2f} --npoints 17.")

    # Landau-Zener only means something at an interior gap minimum; the edge
    # cases above have already said it does not apply.
    if 0 < imin < len(r) - 1:
        p_lz, detail = landau_zener(ana, key, imin)
        if p_lz is None:
            print(f"\n  Landau-Zener: not estimated ({detail})")
        else:
            print(f"\n  Landau-Zener estimate of staying DIABATIC: "
                  f"P = {p_lz:.2f}\n     ({detail})")
            print("     Rough, 1D and classical. The sampled gap can only be "
                  "LARGER than the\n     true minimum on this grid, so the "
                  "true P is at least this large.")
            if p_lz > 0.5:
                print("     A fast packet mostly jumps the gap: propagate on "
                      "the DIABATIC\n     continuation, or on both coupled "
                      "states, not on the lower adiabat.")
            elif p_lz < 0.1:
                print("     The packet mostly follows the lower adiabat: a "
                      "single smooth adiabatic\n     surface is the right "
                      "object, provided it is computed with this method.")
            else:
                print("     Neither limit holds: this region needs the two "
                      "states propagated\n     together with their coupling.")

    wall, cpu = time.time() - t_start, time.process_time() - c_start
    print(f"\n  cost: {cpu:.0f} CPU-s over {len(rows)} points "
          f"({cpu / len(rows):.0f} CPU-s/point, {wall / 60:.1f} min wall)")


if __name__ == "__main__":
    main()
