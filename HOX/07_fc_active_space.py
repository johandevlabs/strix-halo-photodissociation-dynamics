#!/usr/bin/env python3
"""
Phase 0.5, step 8: one active space through the Franck-Condon window, and the
vertical energy on that footing.

06 settled the triplet manifold: a smooth multireference lower curve, three
3A" states converging on OH(2Pi) + Cl(2P), and SA-CASSCF + SC-NEVPT2 over
3A" roots at 624 CPU-s/point -- cheaper than dCCSD(T). It also exposed the
last thing standing between that and a raster: AVAS picks CAS(12e,7o) at
1.6891 and 1.7 A but CAS(10e,6o) from 1.8 A outward. The switch falls inside
the Franck-Condon window, where the band position and width are decided, and
energies from different active spaces are not on the same scale. A raster
needs ONE size throughout.

This script does three things.

  1. WHY AVAS SWITCHES. AVAS ranks occupied and virtual orbitals by how much
     of each projects onto the requested atomic orbitals (Cl 3p, O 2p) and
     keeps those above a threshold, 0.2 by default. At every geometry the
     projection eigenvalues are printed, together with the atomic character
     of the borderline orbitals. If the 6th occupied eigenvalue sits near 0.2
     and drifts through it, the switch is a threshold artefact and says
     nothing about the physics; the character shows which orbital it is.

  2. FIXED SIZES. PySCF's AVAS selects by threshold only, so a fixed number of
     orbitals needs the selection reimplemented: the same projector, the same
     eigenvectors, but the top N occupied and top M virtual kept regardless of
     the threshold. To make sure the reimplementation matches PySCF, the
     threshold-based count is also computed from these eigenvalues and checked
     against a real avas.avas call at every geometry. Any mismatch is printed.

  3. VERTICAL ENERGY, SAME FOOTING. For each fixed space across the window:
     the singlet X 1A' (CASSCF + SC-NEVPT2) and the triplet a 3A" (SA-CASSCF
     over several 3A" roots, as in 06, then SC-NEVPT2 on root 0), with the same
     active space and the same starting orbitals. The vertical energy at
     equilibrium is compared with the two references already in hand:

         3.4477 eV   03, six-root QD-NEVPT2 with SOC (CAS(12,7), mixed S/T)
         3.4142 eV   04, dCCSD(T)

     Neither is exact, and they disagree by 0.034 eV, so closeness to them is
     informative rather than decisive. What decides is below.

HOW IT RECOMMENDS. A space is usable only if every CASSCF in the window
converged and its triplet curve has no kink. If both are usable and their
equilibrium vertical energies agree within 0.05 eV, the extra orbital does
not change the physics and the smaller space wins: cheaper, and already
validated out to 3.6 A in 06. If they disagree by more, the extra orbital
matters, but AVAS dropped it beyond 1.8 A because its projection is weak, so
it must be re-validated along the full dissociation cut before a raster uses
it.

One imbalance to keep in mind: the triplet orbitals are averaged over several
3A" roots (single-root triplet CASSCF did not converge in 04), while the
singlet is a single root. Orbitals optimised for one state lower that state
slightly, which biases the vertical energy upward. NEVPT2 reduces the effect.
--nroots-singlet averages the singlet too, as a check.

COST. RHF and projections are seconds. Each (space, geometry) is a singlet
CASSCF, a 4-root triplet SA-CASSCF, a CASCI and two NEVPT2 solves: roughly
20 s wall, so 2 spaces x 8 geometries is about 6 minutes.

Usage:
    python 07_fc_active_space.py 2>&1 | tee logs/fc_active_space.log
    python 07_fc_active_space.py --spaces 5,1 6,1 6,2
    python 07_fc_active_space.py --nroots-singlet 4
"""
import argparse
import csv
import time
import traceback

import numpy as np
from scipy import linalg

from pyscf import gto, scf, mcscf, mrpt, fci, symm
from pyscf.mcscf import avas

HARTREE2EV = 27.211386245988
NM_PER_EV = 1239.841984

R_OH_ANG = 0.9644
R_OCL_EQ_ANG = 1.6891
ANGLE_HOCL_DEG = 102.96

REF_03_EV = 3.4477      # six-root QD-NEVPT2 + SOC, CAS(12,7)
REF_04_EV = 3.4142      # dCCSD(T)


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
    """Make every orbital pure A' or A" without mixing core/active/virtual."""
    out = mo.copy()
    for sl in (slice(0, ncore), slice(ncore, ncore + ncas),
               slice(ncore + ncas, mo.shape[1])):
        if out[:, sl].shape[1]:
            out[:, sl] = symm.symmetrize_space(mol, out[:, sl], s=s)
    return out


def triplet_solver(mol, nroots):
    solver = fci.direct_spin1_symm.FCI(mol)
    solver.wfnsym = 'A"'
    solver.nroots = nroots
    fci.addons.fix_spin_(solver, ss=2.0)
    return solver


class Projector:
    """The AVAS projection, kept so a FIXED number of orbitals can be taken.

    Mirrors pyscf.mcscf.avas for a closed-shell reference with no frozen core:
    the reference molecule carries the minao basis in the SAME frame as the
    target (its atoms are taken from mol._atom in bohr, symmetry off, so the
    Cs reorientation of the target is not undone), the requested AO labels
    define the subspace, and occupied and virtual MOs are rotated separately
    to diagonalise their projection onto it.
    """

    def __init__(self, mol, aolabels, minao):
        pmol = mol.copy()
        pmol.atom = mol._atom
        pmol.unit = "B"
        pmol.symmetry = False
        pmol.basis = minao
        pmol.build(False, False)
        self.pmol = pmol
        self.sel = pmol.search_ao_label(aolabels)
        s_all = pmol.intor_symmetric("int1e_ovlp")
        x_all = gto.intor_cross("int1e_ovlp", pmol, mol)
        self.s2 = s_all[np.ix_(self.sel, self.sel)]
        self.s21 = x_all[self.sel]
        self.s_all, self.x_all = s_all, x_all
        self.groups = self._groups(pmol)

    @staticmethod
    def _groups(pmol):
        keys = []
        for lab in pmol.ao_labels(fmt=False):
            if isinstance(lab, (tuple, list)) and len(lab) >= 3:
                keys.append(f"{lab[1]} {lab[2]}")
            else:                                   # "0 O 2px" style
                parts = str(lab).split()
                keys.append(f"{parts[1]} {parts[2][:2]}")
        return keys

    def eig(self, c):
        """Projection eigenvalues (descending) and rotation for MOs c."""
        s21c = self.s21 @ c
        sa = s21c.T @ linalg.solve(self.s2, s21c, assume_a="pos")
        w, u = linalg.eigh(sa)
        return w[::-1], u[:, ::-1]

    def character(self, c, top=3):
        """Atomic make-up of one MO against the FULL minao basis."""
        b = self.x_all @ c
        a = linalg.solve(self.s_all, b, assume_a="pos")
        contrib = {}
        for k, v in zip(self.groups, a * b):
            contrib[k] = contrib.get(k, 0.0) + float(v)
        best = sorted(contrib.items(), key=lambda kv: -kv[1])[:top]
        return ", ".join(f"{k} {v:.2f}" for k, v in best)


def select_fixed(mf, proj, n_occ, n_vir):
    """Top n_occ occupied and top n_vir virtual by projection, in AVAS order:
    [core | active occupied | active virtual | remaining virtual]."""
    occ = mf.mo_occ > 0
    c_occ, c_vir = mf.mo_coeff[:, occ], mf.mo_coeff[:, ~occ]
    _, u_o = proj.eig(c_occ)
    _, u_v = proj.eig(c_vir)
    c_occ, c_vir = c_occ @ u_o, c_vir @ u_v
    mo = np.hstack([c_occ[:, n_occ:], c_occ[:, :n_occ],
                    c_vir[:, :n_vir], c_vir[:, n_vir:]])
    ncore = c_occ.shape[1] - n_occ
    return mo, ncore, n_occ + n_vir, 2 * n_occ


def diagnose(r, args):
    """RHF, projection eigenvalues, AVAS's own choice. Returns (ctx, diag)."""
    mol = build_mol(r, 0, args.basis, args.verbose)
    mf = scf.RHF(mol).x2c()
    mf.conv_tol = 1e-10
    mf.kernel()
    proj = Projector(mol, args.avas_aos, args.minao)

    occ = mf.mo_occ > 0
    w_o, u_o = proj.eig(mf.mo_coeff[:, occ])
    w_v, u_v = proj.eig(mf.mo_coeff[:, ~occ])
    thr = args.threshold
    n_occ_thr, n_vir_thr = int((w_o > thr).sum()), int((w_v > thr).sum())

    ncas_py, nelec_py, _ = avas.avas(mf, args.avas_aos, minao=args.minao,
                                     threshold=thr, verbose=0)
    k = max(s[0] for s in args.spaces_parsed) - 1
    border_occ = mf.mo_coeff[:, occ] @ u_o[:, k]
    border_vir = mf.mo_coeff[:, ~occ] @ u_v[:, 0]
    diag = {
        "r": r, "w_occ": w_o[:8].tolist(), "w_vir": w_v[:3].tolist(),
        "mine": (2 * n_occ_thr, n_occ_thr + n_vir_thr),
        "pyscf": (int(nelec_py), int(ncas_py)),
        "char_occ": proj.character(border_occ), "k_occ": k + 1,
        "char_vir": proj.character(border_vir),
    }
    return {"mol": mol, "mf": mf, "proj": proj}, diag


def solve_space(r, ctx, n_occ, n_vir, args):
    """Singlet and triplet, same active space and starting orbitals."""
    c0, t0 = time.process_time(), time.time()
    mol_s, mf_s, proj = ctx["mol"], ctx["mf"], ctx["proj"]
    mo, ncore, ncas, nelecas = select_fixed(mf_s, proj, n_occ, n_vir)
    row = {"r": r, "space": f"({nelecas},{ncas})"}
    try:
        mo = symmetrize_blocks(mol_s, mo, mf_s.get_ovlp(), ncore, ncas)
    except Exception as exc:
        row["note"] = f"symmetrize failed: {exc}"

    # ---- singlet X 1A'
    ns = args.nroots_singlet
    mc_s = mcscf.CASSCF(mf_s, ncas, nelecas)
    s_solver = fci.direct_spin0_symm.FCI(mol_s)
    s_solver.wfnsym, s_solver.nroots = "A'", ns
    mc_s.fcisolver = s_solver
    if ns > 1:
        mc_s = mc_s.state_average_(np.ones(ns) / ns)
    mc_s.conv_tol, mc_s.max_cycle_macro, mc_s.verbose = 1e-8, args.max_cycle, 0
    mc_s.kernel(mo)
    row["conv_s"] = bool(mc_s.converged)
    if ns > 1:
        mci_s = mcscf.CASCI(mf_s, ncas, nelecas)
        sol = fci.direct_spin0_symm.FCI(mol_s)
        sol.wfnsym, sol.nroots = "A'", ns
        mci_s.fcisolver, mci_s.verbose = sol, 0
        mci_s.kernel(mc_s.mo_coeff)
        row["e_cas_s"] = float(np.atleast_1d(mci_s.e_tot)[0])
        row["e_nev_s"] = row["e_cas_s"] + float(
            mrpt.NEVPT(mci_s, root=0).kernel())
    else:
        row["e_cas_s"] = float(mc_s.e_tot)
        row["e_nev_s"] = row["e_cas_s"] + float(mrpt.NEVPT(mc_s).kernel())

    # ---- triplet a 3A", root 0 of an SA over 3A" roots (06's recipe)
    mol_t = build_mol(r, 2, args.basis, args.verbose)
    if not np.allclose(mol_s.atom_coords(), mol_t.atom_coords(), atol=1e-8):
        raise RuntimeError("singlet and triplet frames differ")
    mf_t = scf.ROHF(mol_t).x2c()
    mf_t.conv_tol = 1e-10
    mf_t.kernel()
    nt = args.nroots_triplet
    mc_t = mcscf.CASSCF(mf_t, ncas, nelecas)
    mc_t.fcisolver = triplet_solver(mol_t, nt)
    mc_t = mc_t.state_average_(np.ones(nt) / nt)
    mc_t.conv_tol, mc_t.max_cycle_macro, mc_t.verbose = 1e-8, args.max_cycle, 0
    mc_t.kernel(mo)
    row["conv_t"] = bool(mc_t.converged)
    mci_t = mcscf.CASCI(mf_t, ncas, nelecas)
    mci_t.fcisolver, mci_t.verbose = triplet_solver(mol_t, nt), 0
    mci_t.kernel(mc_t.mo_coeff)
    row["e_cas_t"] = float(np.atleast_1d(mci_t.e_tot)[0])
    row["e_nev_t"] = row["e_cas_t"] + float(mrpt.NEVPT(mci_t, root=0).kernel())

    row["de_cas"] = (row["e_cas_t"] - row["e_cas_s"]) * HARTREE2EV
    row["de_nev"] = (row["e_nev_t"] - row["e_nev_s"]) * HARTREE2EV
    row["wall"], row["cpu"] = time.time() - t0, time.process_time() - c0
    return row


ROUGH_MEV = 10.0


def roughness_mev(r, values_ev, deg=4):
    """Largest deviation, in meV, from a smooth quartic through the window.

    Not a ratio of second differences. That was the first version, and on a
    nearly straight curve -- which a vertical energy across 0.35 A is -- the
    median second difference is numerical noise, so the ratio explodes and
    every space looks kinked. A deviation in energy units has a scale: a
    quartic follows the repulsive curvature to a few meV over this window,
    while an active-space jump or a switch to another CASSCF solution leaves
    a residual.

    Sensitivity, measured on synthetic curves: a steep exponential wall with
    0.2 meV noise gives 0.2-0.4 meV; a 50 meV step across the window gives
    13 meV, because the quartic absorbs part of any step. So steps below
    roughly 40 meV will pass this test. The convergence flags and the
    per-point table are the backstop for anything subtler.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(values_ev, dtype=float)
    if v.size < deg + 2:
        return float("nan")
    x, y = r - r.mean(), v - v.mean()
    res = y - np.polyval(np.polyfit(x, y, deg), x)
    return float(np.abs(res).max() * 1000.0)


def make_plot(diags, results, spaces, args, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for j, sp in enumerate(spaces):
        rows = results[sp]
        if rows:
            ax[0].plot([x["r"] for x in rows], [x["de_nev"] for x in rows],
                       "o-", color=colors[j % 4], label=f"CAS{sp} SC-NEVPT2")
    ax[0].plot([R_OCL_EQ_ANG], [REF_03_EV], "k*", ms=11,
               label="03 QD-NEVPT2+SOC (12,7)")
    ax[0].plot([R_OCL_EQ_ANG], [REF_04_EV], "kD", ms=7, label="04 dCCSD(T)")
    ax[0].set_ylabel("E(a 3A\") - E(X 1A') / eV")
    ax[0].legend(fontsize=7)
    ax[0].set_title("HOCl vertical triplet energy across the FC window")

    r = [d["r"] for d in diags]
    kmax = len(diags[0]["w_occ"])
    for k in range(max(0, kmax - 4), kmax):
        ax[1].plot(r, [d["w_occ"][k] for d in diags], "o-", ms=4,
                   label=f"occupied #{k + 1}")
    ax[1].plot(r, [d["w_vir"][0] for d in diags], "s--", ms=4,
               label="virtual #1")
    ax[1].axhline(args.threshold, color="k", ls=":", lw=1,
                  label=f"AVAS threshold {args.threshold}")
    for axis in ax:
        axis.axvline(R_OCL_EQ_ANG, color="grey", lw=0.6)
    ax[1].set_xlabel("r(O-Cl) / A")
    ax[1].set_ylabel("projection eigenvalue")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"  wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-tzvp")
    p.add_argument("--avas-aos", nargs="+", default=["Cl 3p", "O 2p"])
    p.add_argument("--minao", default="ano")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="AVAS threshold, for the diagnostic only")
    p.add_argument("--spaces", nargs="+", default=["5,1", "6,1"],
                   help="fixed spaces as N_OCC,N_VIR active orbitals "
                        "(default 5,1 = CAS(10,6) and 6,1 = CAS(12,7))")
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--kmin", type=int, default=-3)
    p.add_argument("--kmax", type=int, default=4,
                   help="grid is r_eq + k*step for k in [kmin, kmax]; "
                        "uniform, and always contains equilibrium")
    p.add_argument("--nroots-triplet", type=int, default=4)
    p.add_argument("--nroots-singlet", type=int, default=1)
    p.add_argument("--max-cycle", type=int, default=100)
    p.add_argument("--csv", default="hocl_fc_active_space.csv")
    p.add_argument("--png", default="hocl_fc_active_space.png")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    args.spaces_parsed = [tuple(int(x) for x in s.split(","))
                          for s in args.spaces]
    spaces = [f"({2 * no},{no + nv})" for no, nv in args.spaces_parsed]
    ks = list(range(args.kmin, args.kmax + 1))
    if 0 not in ks:
        p.error("the grid must contain equilibrium: kmin <= 0 <= kmax")
    rs = [R_OCL_EQ_ANG + k * args.step for k in ks]
    i_eq = ks.index(0)

    print("=" * 78)
    print("== HOX Phase 0.5 -- one active space through the Franck-Condon window")
    print(f"== {args.basis}, AVAS {args.avas_aos} (minao {args.minao}), "
          f"spaces {', '.join('CAS' + s for s in spaces)}")
    print(f"== r = {rs[0]:.4f} to {rs[-1]:.4f} A, step {args.step}, "
          f"equilibrium {R_OCL_EQ_ANG} A; triplet SA over "
          f"{args.nroots_triplet} 3A\" roots, singlet {args.nroots_singlet}")
    print("=" * 78)

    # ---------------------------------------------------------- 1. diagnose
    print("\n  1. AVAS projection eigenvalues (occupied top 8 | virtual top 3)")
    print(f"     AVAS keeps those above {args.threshold}. 'mine' is the count "
          f"from these eigenvalues,\n     'pyscf' a real avas.avas call; "
          f"they must agree.\n")
    diags, ctxs, t_start = [], {}, time.time()
    for r in rs:
        try:
            ctx, d = diagnose(r, args)
        except Exception as exc:
            print(f"  {r:.4f}  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            continue
        ctxs[r] = ctx
        diags.append(d)
        occ = " ".join(f"{w:.2f}" for w in d["w_occ"])
        vir = " ".join(f"{w:.2f}" for w in d["w_vir"])
        ok = "" if d["mine"] == d["pyscf"] else "  !! MISMATCH"
        print(f"  {r:.4f}  {occ} | {vir}   mine CAS{d['mine']} "
              f"pyscf CAS{d['pyscf']}{ok}")
    if diags:
        print(f"\n     character of occupied #{diags[0]['k_occ']} "
              f"and virtual #1 (against the full minao basis):")
        for d in diags:
            print(f"  {d['r']:.4f}  occ #{d['k_occ']}: {d['char_occ']:<34}"
                  f"  vir #1: {d['char_vir']}")
    mism = [d["r"] for d in diags if d["mine"] != d["pyscf"]]

    # ------------------------------------------------------ 2+3. per space
    results = {s: [] for s in spaces}
    for (no, nv), sp in zip(args.spaces_parsed, spaces):
        print(f"\n  CAS{sp}: {no} occupied + {nv} virtual active")
        print(f"  {'r/A':>7}{'conv S/T':>9}{'E(S) NEV/Ha':>17}"
              f"{'E(T) NEV/Ha':>17}{'dE CAS':>8}{'dE NEV':>8}{'nm':>7}{'t/s':>6}")
        for r in rs:
            if r not in ctxs:
                continue
            try:
                row = solve_space(r, ctxs[r], no, nv, args)
            except Exception as exc:
                print(f"  {r:7.4f}  FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                continue
            results[sp].append(row)
            conv = (f"{'y' if row['conv_s'] else 'N'}/"
                    f"{'y' if row['conv_t'] else 'N'}")
            flag = "  sym!" if row.get("note") else ""
            print(f"  {r:7.4f}{conv:>9}{row['e_nev_s']:17.8f}"
                  f"{row['e_nev_t']:17.8f}{row['de_cas']:8.3f}"
                  f"{row['de_nev']:8.3f}{NM_PER_EV / row['de_nev']:7.1f}"
                  f"{row['wall']:6.0f}{flag}")

    # ------------------------------------------------------------ outputs
    cols = ["space", "r", "conv_s", "conv_t", "e_cas_s", "e_nev_s",
            "e_cas_t", "e_nev_t", "de_cas", "de_nev", "wall", "cpu", "note"]
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for sp in spaces:
            for row in results[sp]:
                w.writerow({c: row.get(c, "") for c in cols})
    print(f"\n  wrote {args.csv}")
    if diags:
        try:
            make_plot(diags, results, spaces, args, args.png)
        except Exception as exc:
            print(f"  plot FAILED: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ verdict
    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    if mism:
        print(f"  !! the fixed-count selection disagrees with PySCF's AVAS at "
              f"r = {', '.join(f'{v:.4f}' for v in mism)} A.")
        print("     The reimplementation is not reproducing AVAS there, so the "
              "fixed spaces\n     below may not be the orbitals AVAS would "
              "choose. Treat them with care.")
    if diags:
        sizes = sorted({d["pyscf"] for d in diags})
        if len(sizes) > 1:
            k = diags[0]["k_occ"] - 1
            near = [(d["r"], d["w_occ"][k]) for d in diags]
            print(f"  AVAS switches between {sizes} in this window. Occupied "
                  f"#{k + 1} eigenvalue: "
                  + ", ".join(f"{r:.2f}->{v:.2f}" for r, v in near))
            print(f"  against a threshold of {args.threshold}: a threshold "
                  f"crossing, not a change in the physics.")

    summary = {}
    print(f"\n  {'space':>8}{'converged':>11}{'rough E(T)':>12}{'rough dE':>10}"
          f"{'dE eq NEV':>11}{'vs 03':>8}{'vs 04':>8}{'CPU-s/pt':>10}")
    for sp in spaces:
        rows = results[sp]
        if not rows:
            print(f"  {sp:>8}  no results")
            continue
        n_ok = sum(1 for x in rows if x["conv_s"] and x["conv_t"])
        complete = len(rows) == len(rs)
        rr = [x["r"] for x in rows]
        et_ev = [x["e_nev_t"] * HARTREE2EV for x in rows]
        rt = roughness_mev(rr, et_ev) if complete else float("nan")
        rd = (roughness_mev(rr, [x["de_nev"] for x in rows])
              if complete else float("nan"))
        eq = next((x for x in rows if abs(x["r"] - R_OCL_EQ_ANG) < 1e-6), None)
        de = eq["de_nev"] if eq else float("nan")
        cpu = float(np.mean([x["cpu"] for x in rows]))
        usable = (complete and n_ok == len(rows) and eq is not None
                  and rt < ROUGH_MEV and rd < ROUGH_MEV)
        ncas = int(sp.strip("()").split(",")[1])
        summary[sp] = {"usable": usable, "de": de, "ncas": ncas}
        print(f"  {sp:>8}{f'{n_ok}/{len(rs)}':>11}{rt:12.1f}{rd:10.1f}"
              f"{de:11.4f}{de - REF_03_EV:+8.3f}{de - REF_04_EV:+8.3f}"
              f"{cpu:10.0f}")
    print(f"  rough = largest deviation in meV from a smooth quartic through "
          f"the window;\n  above {ROUGH_MEV:.0f} meV is treated as a kink.")

    usable = [s for s in spaces if summary.get(s, {}).get("usable")]
    print()
    if not usable:
        print("  -> NEITHER space is usable across the window as it stands: "
              "each has an\n     unconverged point, a missing point or a "
              "kink. See the tables above.")
    elif len(usable) == 1:
        print(f"  -> Use CAS{usable[0]}: the only space that converges "
              f"everywhere in the window\n     without a kink.")
    else:
        des = [summary[s]["de"] for s in usable]
        spread = max(des) - min(des)
        smallest = min(usable, key=lambda s: summary[s]["ncas"])
        largest = max(usable, key=lambda s: summary[s]["ncas"])
        print(f"  Both usable. Equilibrium vertical energies span "
              f"{spread:.3f} eV across spaces.")
        if spread < 0.05:
            print(f"  -> Use CAS{smallest}. The extra orbital moves the vertical "
                  f"energy by less than\n     0.05 eV, so it does not change "
                  f"the physics, and the smaller space is\n     cheaper"
                  + (" and already validated out to 3.6 A in 06."
                     if smallest == "(10,6)" else "."))
        else:
            print(f"  -> The extra orbital matters ({spread:.3f} eV). CAS{largest} "
                  f"is the more\n     complete description, but AVAS drops "
                  f"that orbital beyond ~1.8 A because\n     its projection "
                  f"is weak, so re-validate CAS{largest} along the whole "
                  f"dissociation\n     cut (06 with a fixed space) before a "
                  f"raster relies on it.")

    print(f"\n  References are not exact: 03 and 04 differ from each other by "
          f"{REF_03_EV - REF_04_EV:.3f} eV.")
    if args.nroots_singlet == 1:
        print("  The triplet is state-averaged and the singlet is not, which "
              "biases the vertical\n  energy slightly upward; rerun with "
              "--nroots-singlet 4 to see how much.")
    print(f"\n  total wall {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
