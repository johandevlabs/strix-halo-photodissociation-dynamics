#!/usr/bin/env python3
"""
HOI: the a 3A" band in 2D -- r(O-I) and the angle -- on the triplet
spin-orbit states. Does the 1D band, and sigma(532 nm), survive the bend?

WHY (TASKS.md, 2026-10-01). The 1D model (11_band_soc_1d.py) put the band at
514-520 nm raw (~555 nm calibrated) with sigma(532) = 3.6e-20, 1.5-6x over
Bauer's 532 nm bound once the f uncertainty is included. The bend is the
dimension it leaves out that most affects the band's WIDTH, and width is
what moves sigma at a fixed wavelength. This checks it, before any decision
about a full multi-state HOI treatment.

COORDINATES. Jacobi, r(O-H) frozen at 0.9694 A, J = 0: R from the OH centre
of mass to I, gamma the angle between O->H and that vector. The kinetic
energy is
    T = -1/(2 mu_R) d2/dR2 + [1/(2 mu_R R^2) + 1/(2 mu_OH r_OH^2)] j^2,
sinc-DVR/FFT in R, Gauss-Legendre DVR in cos(gamma) (j^2 exact in the
Legendre basis). Split-operator, the angular step taken in the j basis with
its R-dependent prefactor, Strang-symmetric around the FFT step.

SURFACES, on the valence grid (r(O-I), angle), interpolated to Jacobi:
    V_S     CCSD(T) (03_surfaces/09_raster_2d.py) + the ground state's SOC
            lowering
    V_k     V_S + SO_ground + omega_EOM(a3A") + [dE_k - T1_sf]
            -- 11's EOM-anchored form: the EOM spin-free triplet, shifted to
            each SOC component by QD-NEVPT2's offset
    mu_k    sqrt(3 f_k / 2 dE_k), non-Condon
The offsets, SO_ground and ln f_k come from the fine 1D SOC scan along r
(07, at the equilibrium angle), corrected in angle from the coarse 2D SOC
grid (07 --grid2d): X(r, a) = X_1D(r) + [X_2D(r_c, a) - X_2D(r_c, a_eq)],
r_c = r clipped to the 2D grid. Outside the valence grid: linear in r, the
angle clamped with a steep wall (the packet must not go there, and the
report says if it does).

Each of the three triplet components is propagated from the lowest 2D
vibrational states (O-I stretch x bend), Boltzmann-weighted at 295 and 220
K, and summed. The same surfaces with the angle FROZEN at equilibrium (a 1D
run) are the comparison, so the table shows exactly what the bend changes.
The -0.18 eV calibration is 11's (Bauer's two measured bands).

EVO for production (the grid is 384 x 64 and three states x ~6 levels x two
temperatures); --test runs a coarse grid anywhere.
    python 12_band_soc_2d.py 2>&1 | tee ../logs/band_soc_2d.log
    python 12_band_soc_2d.py --test
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss, legval
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.linalg import eigh

HOX = Path(__file__).resolve().parents[2]
DATA = HOX / "HOI" / "data"
HARTREE2EV = 27.211386245988
HARTREE2CM = 219474.6313702
BOHR = 1.0 / 0.529177210903               # bohr per A
AMU = 1822.888486209
C_AU = 137.035999084
BOHR2_TO_CM2 = 2.8002852e-17
KB_AU = 3.166811563e-6
NM_EV = 1239.841984
M_O, M_H, M_I = 15.99491462, 1.00782503, 126.904473
MU_R = (M_O + M_H) * M_I / (M_O + M_H + M_I) * AMU
MU_OH = M_O * M_H / (M_O + M_H) * AMU
R_OH = 0.9694
D_G = M_H / (M_O + M_H) * R_OH             # O -> OH centre of mass, A
R_EQ, TH_EQ = 1.9907, 104.65
SHIFT_EV = -0.18                           # 11: obs - calc on Bauer's bands
BOUND_532 = 1.0e-20
NK = 3                                     # the a 3A" components


# ---------------------------------------------------------------- data
def read(path):
    return [r for r in csv.DictReader(open(path)) if not r["status"].startswith("fail")]


def regular(rows, fx, fy, fz):
    """Rows on a rectangular (x, y) grid -> x, y, Z[x, y]; raises if holes."""
    xs = sorted({round(fx(r), 4) for r in rows})
    ys = sorted({round(fy(r), 3) for r in rows})
    Z = np.full((len(xs), len(ys)), np.nan)
    for r in rows:
        Z[xs.index(round(fx(r), 4)), ys.index(round(fy(r), 3))] = fz(r)
    if np.isnan(Z).any():
        miss = [(xs[i], ys[j]) for i, j in zip(*np.where(np.isnan(Z)))]
        raise SystemExit(f"grid has {len(miss)} holes, e.g. {miss[:4]}")
    return np.array(xs), np.array(ys), Z


class Field:
    """Bicubic inside the valence grid; linear in r beyond it; the angle
    clamped (the wall is added to the potentials, not here)."""

    def __init__(self, r, a, Z):
        self.r0, self.r1, self.a0, self.a1 = r[0], r[-1], a[0], a[-1]
        k = min(3, len(r) - 1), min(3, len(a) - 1)
        self.s = RectBivariateSpline(r, a, Z, kx=k[0], ky=k[1])

    def __call__(self, r, a):
        rc = np.clip(r, self.r0, self.r1)
        ac = np.clip(a, self.a0, self.a1)
        return self.s.ev(rc, ac) + self.s.ev(rc, ac, dx=1) * (r - rc)


def surfaces(args):
    cut = read(args.raster)
    fr, fa = (lambda x: float(x["r_ox_A"])), (lambda x: float(x["theta_deg"]))
    r, a, VS = regular(cut, fr, fa, lambda x: float(x["e_ccsdt_Ha"]))
    _, _, OM = regular(cut, fr, fa, lambda x: float(x["omega_app_eV"]) / HARTREE2EV)

    one = sorted(read(args.soc1d), key=lambda x: float(x["r_ox_A"]))
    r1 = np.array([float(x["r_ox_A"]) for x in one])

    def q1(x, k):                  # per-row quantities, eV / ln f
        so = (float(x["e_soc0_Ha"]) - float(x["e_sf0_Ha"])) * HARTREE2EV
        if k == "so":
            return so
        kk = int(k[1:])
        if k[0] == "o":
            return float(x[f"dE_{kk}"]) - float(x["sf_1"])
        return np.log(max(float(x[f"f_{kk}"]), 1e-14))

    two = read(args.soc2d)
    names = ["so"] + [f"o{k}" for k in range(1, NK + 1)] + \
        [f"f{k}" for k in range(1, NK + 1)] + [f"e{k}" for k in range(1, NK + 1)]

    def q(x, k):
        if k[0] == "e":
            return float(x[f"dE_{k[1:]}"])
        return q1(x, k)

    one_d = {k: CubicSpline(r1, [q(x, k) for x in one], bc_type="natural")
             for k in names}
    r2, a2, _ = regular(two, fr, fa, lambda x: 0.0)
    two_d = {k: Field(r2, a2, regular(two, fr, fa, lambda x, k=k: q(x, k))[2])
             for k in names}

    def corrected(k, rr, aa):
        """X_1D(r) + [X_2D(r_c, a) - X_2D(r_c, a_eq)]; linear beyond the 1D
        scan in r."""
        s = one_d[k]
        rc1 = np.clip(rr, r1[0], r1[-1])
        base = s(rc1) + s(rc1, 1) * (rr - rc1)
        rc2 = np.clip(rr, r2[0], r2[-1])
        return base + two_d[k](rc2, aa) - two_d[k](rc2, np.full_like(aa, TH_EQ))

    return dict(VS=Field(r, a, VS), OM=Field(r, a, OM), corr=corrected,
                grid=(r, a), grid2=(r2, a2), r1=r1)


# ---------------------------------------------------------------- grids
def jacobi_to_valence(R_ang, gamma):
    """O at the origin, H on +x; G at D_G; I at G + R(cos g, sin g)."""
    x = D_G + R_ang * np.cos(gamma)
    y = R_ang * np.sin(gamma)
    return np.hypot(x, y), np.degrees(np.arctan2(y, x))


def angular(ng):
    x, w = leggauss(ng)                    # nodes in cos(gamma), ascending
    j = np.arange(ng)
    P = np.array([legval(x, np.eye(ng)[jj]) * np.sqrt((2 * jj + 1) / 2.0)
                  for jj in j])            # (j, node)
    U = P * np.sqrt(w)[None, :]            # orthogonal: j-basis <- DVR
    return np.arccos(x), U, j * (j + 1.0)


def potentials(S, R_ang, gam, args):
    RR, GG = np.meshgrid(R_ang, gam, indexing="ij")
    r, th = jacobi_to_valence(RR, GG)
    a0, a1 = S["grid"][1][0], S["grid"][1][-1]
    wall = args.wall * np.radians(np.clip(a0 - th, 0, None)
                                  + np.clip(th - a1, 0, None)) ** 2
    so = S["corr"]("so", r, th) / HARTREE2EV
    VS = S["VS"](r, th) + so + wall
    Vk, muk = [], []
    for k in range(1, NK + 1):
        off = S["corr"](f"o{k}", r, th) / HARTREE2EV
        V = S["VS"](r, th) + so + S["OM"](r, th) + off + wall
        dE = np.clip(S["corr"](f"e{k}", r, th), 0.3, None) / HARTREE2EV
        f = np.exp(S["corr"](f"f{k}", r, th))
        mu = np.sqrt(3.0 * f / (2.0 * dE))
        if args.condon:
            i0 = np.unravel_index(np.argmin((r - R_EQ) ** 2 + ((th - TH_EQ) / 50) ** 2),
                                  r.shape)
            mu = np.full_like(mu, mu[i0])
        Vk.append(V)
        muk.append(mu)
    return VS, Vk, muk, r, th


# ---------------------------------------------------------------- physics
def ground_states(VS, R_ang, lam_j, U, nlev, sub):
    """Direct diagonalisation on the R sub-window holding the well."""
    idx = np.where((R_ang >= sub[0]) & (R_ang <= sub[1]))[0]
    Rb = R_ang[idx] * BOHR
    dx = Rb[1] - Rb[0]
    n, ng = idx.size, U.shape[0]
    i = np.arange(n)
    d = i[:, None] - i[None, :]
    with np.errstate(divide="ignore"):
        TR = np.where(d == 0, np.pi ** 2 / 3.0, 2.0 / d.astype(float) ** 2)
    TR = TR * (-1.0) ** np.abs(d) / (2.0 * MU_R * dx ** 2)
    Tg = U.T @ np.diag(lam_j) @ U
    c = 1.0 / (2 * MU_R * Rb ** 2) + 1.0 / (2 * MU_OH * (R_OH * BOHR) ** 2)
    H = np.kron(TR, np.eye(ng)) + np.kron(np.diag(c), Tg) + np.diag(VS[idx].ravel())
    e, v = eigh(H, subset_by_index=[0, nlev - 1])
    full = np.zeros((nlev, R_ang.size, ng))
    for k in range(nlev):
        full[k, idx] = v[:, k].reshape(n, ng) / np.sqrt(dx)
    return e, full


def propagate(psi0, V, W, R_ang, U, lam_j, dt, nsteps):
    Rb = R_ang * BOHR
    dx = Rb[1] - Rb[0]
    kR = 2 * np.pi * np.fft.fftfreq(Rb.size, d=dx)
    eR = np.exp(-1j * dt * kR ** 2 / (2 * MU_R))[:, None]
    c = 1.0 / (2 * MU_R * Rb ** 2) + 1.0 / (2 * MU_OH * (R_OH * BOHR) ** 2)
    eA = np.exp(-0.5j * dt * c[:, None] * lam_j[None, :])
    eV = np.exp(-0.5j * dt * (V - 1j * W))
    psi = psi0.astype(complex)
    S = np.empty(nsteps + 1, complex)
    S[0] = np.vdot(psi0, psi) * dx
    for it in range(1, nsteps + 1):
        psi = eV * psi
        psi = ((psi @ U.T) * eA) @ U           # angular half step in j
        psi = np.fft.ifft(eR * np.fft.fft(psi, axis=0), axis=0)
        psi = ((psi @ U.T) * eA) @ U
        psi = eV * psi
        S[it] = np.vdot(psi0, psi) * dx
    return S, float(np.sum(np.abs(psi) ** 2) * dx)


def spectrum(S, e0, dt, E, stride=4):
    S = S[::stride]
    t = np.arange(S.size) * dt * stride
    win = np.cos(0.5 * np.pi * t / t[-1]) ** 2
    ph = np.exp(1j * (e0 + E[:, None]) * t[None, :])
    integ = np.trapezoid(ph * (S * win)[None, :], t, axis=1)
    return np.clip((4 * np.pi * E / (3 * C_AU)) * np.real(integ) * BOHR2_TO_CM2,
                   0, None)


def stats(E_ev, s):
    i = int(np.argmax(s))
    above = E_ev[s >= 0.5 * s[i]]
    return E_ev[i], above.max() - above.min()


def f_of(E_ev, s):
    return 1.1296e12 * np.trapezoid(s, E_ev * 8065.544)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raster", default=str(DATA / "hoi_raster_2d.csv"))
    p.add_argument("--soc1d", default=str(DATA / "hoi_soc_scan.csv"))
    p.add_argument("--soc2d", default=str(DATA / "hoi_soc_2d.csv"))
    p.add_argument("--nR", type=int, default=384)
    p.add_argument("--Rmin", type=float, default=1.55)
    p.add_argument("--Rmax", type=float, default=3.80)
    p.add_argument("--ngamma", type=int, default=64)
    p.add_argument("--well", type=float, nargs=2, default=[1.72, 2.35],
                   help="R window for the ground-state diagonalisation, A")
    p.add_argument("--nlev", type=int, default=6)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--nsteps", type=int, default=6000)
    p.add_argument("--r-abs", type=float, default=2.90, help="absorber onset in R, A")
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--wall", type=float, default=2.0, help="Ha/rad^2 outside the angles")
    p.add_argument("--temperatures", type=float, nargs="+", default=[295.0, 220.0])
    p.add_argument("--condon", action="store_true")
    p.add_argument("--test", action="store_true", help="coarse grid, quick")
    p.add_argument("--csv", default=str(DATA / "hoi_band_soc_2d.csv"))
    args = p.parse_args()
    if args.test:
        args.nR, args.ngamma, args.nsteps, args.nlev = 192, 32, 3000, 4

    S = surfaces(args)
    R_ang = np.linspace(args.Rmin, args.Rmax, args.nR)
    gam, U, lam = angular(args.ngamma)
    VS, Vk, muk, rv, thv = potentials(S, R_ang, gam, args)
    # energies relative to the ground-state minimum: absolute (-7188 Ha)
    # phases would alias when S(t) is sampled every few au
    ref = float(VS[(rv > 1.8) & (rv < 2.3)].min())
    VS = VS - ref
    Vk = [V - ref for V in Vk]
    dxb = (R_ang[1] - R_ang[0]) * BOHR

    print("=" * 78)
    print("== HOI a 3A\" band in 2D (r(O-I), angle), SOC components, EOM-anchored")
    print("=" * 78)
    print(f"  grid {args.nR} x {args.ngamma} (R {args.Rmin}-{args.Rmax} A, "
          f"Gauss-Legendre in cos gamma); dt {args.dt} au x {args.nsteps}")
    e, chi = ground_states(VS, R_ang, lam, U, args.nlev, args.well)
    e_cm = (e - e[0]) * HARTREE2CM
    print(f"  2D levels above v=0 (cm-1): " + ", ".join(f"{x:.0f}" for x in e_cm[1:]))
    print("  (O-I stretch 583, bend 1104 harmonic in 3D from 01; 1D stretch 583)")
    for k in range(args.nlev):
        w = np.abs(chi[k]) ** 2 * dxb
        th_mean = float((w * thv).sum() / w.sum())
        th_sd = float(np.sqrt((w * (thv - th_mean) ** 2).sum() / w.sum()))
        r_mean = float((w * rv).sum() / w.sum())
        print(f"    level {k}: <r(O-I)> {r_mean:.3f} A, <angle> {th_mean:.1f} "
              f"+/- {th_sd:.1f} deg")
    a_lo, a_hi = S["grid"][1][0], S["grid"][1][-1]
    out_frac = float((np.abs(chi[0]) ** 2 * dxb)[(thv < a_lo) | (thv > a_hi)].sum())
    print(f"  v=0 outside the computed angles {a_lo:.0f}-{a_hi:.0f}: {out_frac:.1e}")

    E = np.linspace(1.6, 3.4, 1200) / HARTREE2EV
    E_ev = E * HARTREE2EV
    lamnm = NM_EV / E_ev
    i532 = int(np.argmin(np.abs(lamnm - 532)))
    W = args.eta * np.clip((R_ang - args.r_abs) / 0.4, 0, 1)[:, None] ** 3 * np.ones_like(VS)

    # 2D, and the same surfaces with the angle frozen (1D along R at gamma_eq)
    jeq = int(np.argmin(np.abs(np.degrees(gam) - float(
        np.degrees(np.arctan2(R_EQ * np.sin(np.radians(TH_EQ)),
                              R_EQ * np.cos(np.radians(TH_EQ)) - D_G))))))
    # the angle frozen at gamma_eq: the same surfaces as a 1D problem in R,
    # with its own stretch levels (slices of the 2D levels would carry the
    # bend's nodes)
    U1, lam1 = np.ones((1, 1)), np.zeros(1)
    n1 = min(3, args.nlev)
    e1, chi1 = ground_states(VS[:, [jeq]], R_ang, lam1, U1, n1, args.well)

    def run(Vs, mus, Ws, levels, energies, Um, lm):
        out = {T: np.zeros_like(E) for T in args.temperatures}
        worst = 0.0
        for V, mu in zip(Vs, mus):
            spec_v = []
            for chi_v, ev in zip(levels, energies):
                phi = mu * chi_v
                Sv, left = propagate(phi, V, Ws, R_ang, Um, lm, args.dt, args.nsteps)
                worst = max(worst, left / max(np.sum(np.abs(phi) ** 2) * dxb, 1e-300))
                spec_v.append(spectrum(Sv, ev, args.dt, E))
            for T in args.temperatures:
                wT = np.exp(-(energies - energies[0]) / (KB_AU * T))
                wT /= wT.sum()
                out[T] += (wT[:, None] * np.array(spec_v)).sum(0)
        return out, worst

    sig, worst = run(Vk, muk, W, chi, e, U, lam)
    sig1, worst1 = run([V[:, [jeq]] for V in Vk], [m[:, [jeq]] for m in muk],
                       W[:, [jeq]], chi1, e1, U1, lam1)
    print(f"  largest unabsorbed fraction: 2D {worst:.1e}, angle frozen {worst1:.1e}")

    # sum rule, 2D: Int sigma/E dE = 4 pi^2 <mu^2>_T / 3c
    T0 = args.temperatures[0]
    wT = np.exp(-(e - e[0]) / (KB_AU * T0))
    wT /= wT.sum()
    m2 = sum(float((wT * np.array([np.sum((muk[k] * chi[v]) ** 2) * dxb
                                   for v in range(args.nlev)])).sum())
             for k in range(NK))
    lhs = np.trapezoid(sig[T0] / BOHR2_TO_CM2 / E, E)
    print(f"  sum rule (2D, {T0:.0f} K): {lhs / (4 * np.pi ** 2 * m2 / (3 * C_AU)):.3f}"
          f"  (1.000 = the whole band is on the energy grid)")

    print(f"\n  the a 3A\" band (components 1-{NK}), {T0:.0f} K:")
    print(f"  {'':22}{'peak/nm':>9}{'FWHM/eV':>9}{'f':>10}{'s(532) raw':>12}"
          f"{'calibrated':>12}")
    for name, s_ in (("2D (r, angle)", sig[T0]), ("angle frozen (1D)", sig1[T0])):
        pk, fw = stats(E_ev, s_)
        cal = float(np.interp(NM_EV / 532.0 - SHIFT_EV, E_ev, s_))
        print(f"  {name:<22}{NM_EV / pk:9.0f}{fw:9.3f}{f_of(E_ev, s_):10.2e}"
              f"{s_[i532]:12.2e}{cal:12.2e}")
    print(f"  11 (1D, EOM-anchored): 520 nm, FWHM 0.391 eV, f 1.45e-4, "
          f"s(532) 3.60e-20 raw; Bauer's bound {BOUND_532:.0e}")
    if len(args.temperatures) > 1:
        T1 = args.temperatures[1]
        print(f"\n  sigma({T1:.0f} K) / sigma({T0:.0f} K), 2D, calibrated wavelengths:")
        for nm in (500, 532, 560, 590):
            Eq = NM_EV / nm - SHIFT_EV
            a_, b_ = np.interp(Eq, E_ev, sig[T1]), np.interp(Eq, E_ev, sig[T0])
            print(f"    {nm} nm: {a_ / b_:.3f}")

    with open(args.csv, "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["E_eV", "lambda_nm"] + [f"sigma2d_{int(T)}K" for T in args.temperatures]
                    + [f"sigma1d_{int(T)}K" for T in args.temperatures])
        for i in range(E.size):
            w_.writerow([f"{E_ev[i]:.5f}", f"{lamnm[i]:.2f}"]
                        + [f"{sig[T][i]:.4e}" for T in args.temperatures]
                        + [f"{sig1[T][i]:.4e}" for T in args.temperatures])
    print(f"\n  csv: {args.csv}")


if __name__ == "__main__":
    main()
