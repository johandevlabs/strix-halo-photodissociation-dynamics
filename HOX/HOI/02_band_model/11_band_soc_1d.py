#!/usr/bin/env python3
"""
HOI: a 1D band model along O-I on the SPIN-ORBIT states -- sigma(532 nm) as
a prediction, to set against Bauer's 532 nm photolysis null.

WHY. D0(HO-I) = 2.07-2.18 eV (06), so HO + I is open at 532 nm, and Bauer
et al. (1998) saw no OH there: sigma(532) < ~1e-20 cm2. Our a 3A" band (f
1.5e-4) and Minaev's (1.6-2.5e-4) would put 3-4e-20 there -- but that number
came from a Gaussian of assumed width at an assumed position. This model
computes the band from the surfaces and turns "3-4x over the bound" into a
number with a width, a position and a calibration behind it.

THE MODEL. For HOBr one spin-free triplet surface carried the band (c^2
~2%). For HOI (c^2 ~20%) each spin-orbit state k gets its own curve and its
own dipole, from 07's scan (the lowest 8 excited SOC states, QD-NEVPT2 +
DKH1 SOC, fixed CAS(12,7)):

    V_k(r)   = V_S(r) + dE_k(r)        V_S: CCSD(T), 02's cut
    mu_k(r)  = sqrt(3 f_k / 2 dE_k)    non-Condon, per state

and each is propagated on its own (no coupling between the SOC adiabats),
Boltzmann-averaged over the O-I stretch levels at 295 K, and summed.
Conventions and propagator are HOBr/02_band_model/03_band_1d.py's (imported):
split-operator, S_v(t) = <mu chi_v | mu chi_v(t)>, sigma = (4 pi E / 3c)
Re Int S(t) e^{i(E_v+E)t} w(t) dt, checked against the exact sum rule
Int sigma/E dE = 4 pi^2 <chi|mu^2|chi> / 3c.

THE CALIBRATION IS THE POINT. States 4-8 make the two bands Bauer DID
measure (407 nm: 1A" + 3A'; 340 nm: 1A'). Their computed peaks against
Bauer's give the model's energy error on this molecule, with this method,
along this cut -- and that shift, applied to the triplet (states 1-3), is
the calibrated prediction at 532 nm. The EOM-anchored variant (triplet
curves = EOM a 3A" from 02 + 07's SOC splitting) is a second estimate of
the triplet's position.

WHAT IT CANNOT SAY. OH and the bend frozen (for HOBr the 3D band came out
4% wider than 1D); no coupling between SOC states; adiabatic states
interpolated in r, so where two approach, character passes between them
smoothly. A first look with a calibration, not a result.

Local: numpy + scipy.
    python 11_band_soc_1d.py 2>&1 | tee ../logs/band_soc_1d.log
"""
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

HOX = Path(__file__).resolve().parents[2]
DATA = HOX / "HOI" / "data"
_s = importlib.util.spec_from_file_location(
    "b03", HOX / "HOBr" / "02_band_model" / "03_band_1d.py")
b03 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(b03)

HARTREE2EV = b03.HARTREE2EV
BOHR_PER_ANG = b03.BOHR_PER_ANG
AMU2AU = b03.AMU2AU
C_AU = b03.C_AU
BOHR2_TO_CM2 = b03.BOHR2_TO_CM2
KB_AU = b03.KB_AU
NM_EV = b03.NM_EV
M_O, M_H, M_I = 15.99491462, 1.00782503, 126.904473
MU_AU = (M_O + M_H) * M_I / (M_O + M_H + M_I) * AMU2AU
R_EQ = 1.9907
NK = 8
S_MAX_ANG = 2.60          # ground-state points used for the levels
R_MIN_ANG, R_MAX_ANG = 1.55, 3.60
# Bauer et al. (1998), Table 1 (295 K): the measured bands
BAUER = dict(vis=(406.4, 3.30e-19, 170.2), uv=(340.4, 3.85e-19, 162.8))
F_OBS = dict(vis=1.17e-3, uv=1.85e-3)          # 10_obs_bands.py
BOUND_532 = 1.0e-20                              # Bauer, sigma(532) upper


def bauer(lam):
    out = np.zeros_like(lam, dtype=float)
    for lmax, smax, fw in BAUER.values():
        out += smax * np.exp(-fw * np.log(lmax / lam) ** 2)
    return out


def load_cut(path):
    rs, es, om = [], [], []
    for rec in csv.DictReader(open(path)):
        if rec["status"].startswith("fail"):
            continue
        try:
            r, e = float(rec["r_ox_A"]), float(rec["e_ccsdt_Ha"])
        except (KeyError, ValueError):
            continue
        rs.append(r)
        es.append(e)
        try:
            om.append(float(rec["omega_app_eV"]))
        except (KeyError, ValueError):
            om.append(np.nan)
    o = np.argsort(rs)
    return np.array(rs)[o], np.array(es)[o], np.array(om)[o]


def load_scan(path):
    rows = [r for r in csv.DictReader(open(path))
            if not r["status"].startswith("fail")]
    rows.sort(key=lambda r: float(r["r_ox_A"]))
    r = np.array([float(x["r_ox_A"]) for x in rows])
    dE = np.array([[float(x[f"dE_{k}"]) for k in range(1, NK + 1)] for x in rows])
    f = np.array([[float(x[f"f_{k}"]) for k in range(1, NK + 1)] for x in rows])
    sf1 = np.array([float(x["sf_1"]) for x in rows])
    so = np.array([(float(x["e_soc0_Ha"]) - float(x["e_sf0_Ha"])) * HARTREE2EV
                   for x in rows])
    return r, dE, f, sf1, so


def extend(x, r, y):
    """Cubic spline inside the scan, linear beyond it (last slopes)."""
    sp = CubicSpline(r, y, bc_type="natural")
    out = sp(np.clip(x, r[0], r[-1]))
    lo, hi = x < r[0], x > r[-1]
    out[lo] += sp(r[0], 1) * (x[lo] - r[0])
    out[hi] += sp(r[-1], 1) * (x[hi] - r[-1])
    return out


def run_state(x_ang, V, mu, chis, e_lev, W, args, E):
    x = x_ang * BOHR_PER_ANG
    dx = x[1] - x[0]
    S_all, left, m2 = [], [], []
    for chi in chis:
        phi0 = mu * chi
        m2.append(float(np.sum(np.abs(phi0) ** 2) * dx))
        S, rem = b03.propagate(x, V, W, phi0, args.dt, args.nsteps, MU_AU)
        S_all.append(S)
        left.append(rem / max(m2[-1], 1e-300))
    sig = b03.cross_sections(np.array(S_all), e_lev, args.dt, E)
    return sig, np.array(m2), max(left)


def stats(E_ev, s):
    if s.max() <= 0:
        return np.nan, 0.0, np.nan
    i = int(np.argmax(s))
    above = E_ev[s >= 0.5 * s[i]]
    return E_ev[i], s[i], above.max() - above.min()


def f_of(E_ev, s):
    """f from Int sigma d(nu~): f = 1.1296e12 Int sigma dnu~ (cm2, cm-1)."""
    nu = E_ev * 8065.544
    o = np.argsort(nu)
    return 1.1296e12 * np.trapezoid(s[o], nu[o])


def model(args):
    r, eS_all, om = load_cut(args.cut)
    rk, dE, fk, sf1, so = load_scan(args.scan)
    # The ground state's own SOC lowering grows along O-I (-54 meV at r_eq
    # towards the atom's -314): CCSD(T) is spin-free, so add it, or the
    # ground well -- and every V_k built on it -- tilts the wrong way.
    if args.no_so_ground:
        so = np.zeros_like(so)
    eS_all = eS_all + extend(r, rk, so) / HARTREE2EV
    keep = r <= S_MAX_ANG + 1e-9
    e_ref = float(eS_all[keep].min())
    rS, eS = r[keep], eS_all[keep] - e_ref

    print("=" * 78)
    print("== HOI 1D band model on the spin-orbit states, along r(O-I)")
    print("=" * 78)
    print(f"  ground: {rS.size} CCSD(T) points {rS.min():.2f}-{rS.max():.2f} A;"
          f" SOC scan {rk.size} points {rk.min():.2f}-{rk.max():.2f} A;"
          f" reduced mass OH-I {MU_AU / AMU2AU:.3f} amu")
    print(f"  ground-state SOC lowering {so[0] * 1000:.0f} meV at {rk[0]:.2f} A"
          f" to {so[-1] * 1000:.0f} meV at {rk[-1]:.2f} A"
          + (" -- NOT applied (--no-so-ground)" if args.no_so_ground else
             ", added to V_S"))

    r_dvr, e_lev, c_lev = b03.vib_levels(rS, eS, args.nlev, MU_AU)
    print(f"  O-I stretch v=0->1 {(e_lev[1] - e_lev[0]) * b03.HARTREE2CM:.0f} "
          f"cm-1 (3D harmonic nu3 583 from 01)")
    x_ang = np.linspace(R_MIN_ANG, R_MAX_ANG, args.npts)
    dxb = (x_ang[1] - x_ang[0]) * BOHR_PER_ANG
    chis = []
    for v in range(args.nlev):
        c = np.interp(x_ang, r_dvr, c_lev[:, v], left=0.0, right=0.0)
        chis.append(c / np.sqrt(np.sum(c ** 2) * dxb))
    V_S = extend(x_ang, r, eS_all - e_ref)

    E = np.linspace(1.5, 4.8, 2400) / HARTREE2EV
    E_ev = E * HARTREE2EV
    lam = NM_EV / E_ev
    W = b03.absorber(x_ang, args.r_abs, args.eta, 3, 0.4)
    T = args.temperature
    wT = np.exp(-(e_lev - e_lev[0]) / (KB_AU * T))
    wT /= wT.sum()

    ok = np.isfinite(om)
    om_x = extend(x_ang, r[ok], om[ok])

    def curves(k, anchor, tail="linear"):
        dk = extend(x_ang, rk, dE[:, k])
        if tail == "eom" and k < 3:
            # beyond the scan (2.70 A) follow the EOM a 3A" shape, carrying
            # the SOC offset of state k at the last scan point; blended over
            # 0.1 A. EOM stays single-reference-valid (w1 >= 0.90) to 2.60 A
            # and is a shape proxy beyond.
            off = float(dE[-1, k] - np.interp(rk[-1], r[ok], om[ok]))
            u = np.clip((x_ang - (rk[-1] - 0.1)) / 0.1, 0.0, 1.0)
            wgt = 0.5 - 0.5 * np.cos(np.pi * u)
            dk = (1 - wgt) * dk + wgt * (om_x + off)
        if anchor == "eom" and k < 3:
            # EOM a 3A" + 07's SOC offset of state k from its spin-free T1
            dk = om_x + extend(x_ang, rk, dE[:, k] - sf1)
        V = V_S + dk / HARTREE2EV
        # f spans 1e-8..1e-2 along the scan: interpolate ln f, so the spline
        # cannot undershoot below zero between points
        fint = np.exp(extend(x_ang, rk, np.log(np.clip(fk[:, k], 1e-14, None))))
        mu2 = 3.0 * fint / (2.0 * np.clip(dk, 0.3, None) / HARTREE2EV)
        mu = np.sqrt(mu2)
        if args.condon:
            mu = np.full_like(mu, float(np.interp(R_EQ, x_ang, mu)))
        return V, mu

    out = {}
    for anchor in (("nevpt2", "eom") if not args.no_eom else ("nevpt2",)):
        sig_k, m2_k = [], []
        for k in range(NK):
            if anchor == "eom" and k >= 3:
                sig_k.append(out["nevpt2"]["sig_k"][k])
                m2_k.append(out["nevpt2"]["m2_k"][k])
                continue
            V, mu = curves(k, anchor)
            s_v, m2, left = run_state(x_ang, V, mu, chis, e_lev, W, args, E)
            sig_k.append((wT[:, None] * s_v).sum(axis=0))
            m2_k.append(float((wT * m2).sum()))
            if left > 1e-3:
                print(f"  !! state {k + 1} ({anchor}): {left:.1e} of the "
                      f"packet unabsorbed")
        out[anchor] = dict(sig_k=np.array(sig_k), m2_k=np.array(m2_k))

    # ---- sum rule, state by state (exact for mu(r))
    s = out["nevpt2"]
    print("\n  sum rule Int sigma/E dE = 4 pi^2 <mu^2> / 3c, per state:")
    for k in range(NK):
        o = np.argsort(E)
        lhs = np.trapezoid(s["sig_k"][k][o] / BOHR2_TO_CM2 / E[o], E[o])
        rhs = 4 * np.pi ** 2 * s["m2_k"][k] / (3 * C_AU)
        print(f"    state {k + 1}: {lhs / rhs:.3f}", end="")
    print("   (1.000 = the band is all on the grid)")

    # ---- the states' bands
    print(f"\n  per state, {T:.0f} K (nevpt2 curves):")
    print(f"  {'k':>3}{'dE(r_eq)/eV':>12}{'peak/eV':>9}{'peak/nm':>9}"
          f"{'FWHM/eV':>9}{'f':>10}{'sigma(532)':>12}")
    i532 = int(np.argmin(np.abs(lam - 532.0)))
    for k in range(NK):
        pk, _, fw = stats(E_ev, s["sig_k"][k])
        print(f"  {k + 1:3d}{float(np.interp(R_EQ, rk, dE[:, k])):12.3f}"
              f"{pk:9.3f}{NM_EV / pk:9.1f}{fw:9.3f}{f_of(E_ev, s['sig_k'][k]):10.2e}"
              f"{s['sig_k'][k][i532]:12.2e}")

    # ---- calibration on the measured bands
    def centroid(sig):
        w = sig / E_ev                    # ~ |mu|^2 density
        return float((E_ev * w).sum() / w.sum())

    # Group by where each state's band sits. The measured 407 band is 1A" +
    # 3A' (05, Minaev); NEVPT2 puts those two 0.55 eV apart (443 and 368 nm,
    # straddling the measured dip, where EOM has them 0.18 eV apart), so a
    # single PEAK is meaningless for the group -- its f-weighted CENTROID is
    # compared, and the split itself is reported.
    peaks = [stats(E_ev, s["sig_k"][k])[0] for k in range(NK)]
    vis = [k for k in range(3, NK) if 2.6 < peaks[k] < 3.45]
    uv = [k for k in range(3, NK) if peaks[k] >= 3.45]
    trip = [0, 1, 2]
    s_vis, s_uv = s["sig_k"][vis].sum(0), s["sig_k"][uv].sum(0)
    cv, cu = centroid(s_vis), centroid(s_uv)
    shift_v = NM_EV / BAUER["vis"][0] - cv
    shift_u = NM_EV / BAUER["uv"][0] - cu
    print(f"\n  CALIBRATION against Bauer's measured bands (f-weighted centroids):")
    print(f"    407 band = states {[k + 1 for k in vis]}: calc centroid "
          f"{NM_EV / cv:.0f} nm (bands at "
          + ", ".join(f"{NM_EV / peaks[k]:.0f}" for k in vis if f_of(E_ev, s['sig_k'][k]) > 1e-4)
          + f" nm), f {f_of(E_ev, s_vis):.2e}; obs 406 nm, f "
          f"{F_OBS['vis']:.2e}; shift obs-calc {shift_v:+.3f} eV")
    print(f"    340 band = states {[k + 1 for k in uv]}: calc centroid "
          f"{NM_EV / cu:.0f} nm, f {f_of(E_ev, s_uv):.2e}; obs 340 nm, f "
          f"{F_OBS['uv']:.2e}; shift obs-calc {shift_u:+.3f} eV")
    shifts = sorted([shift_v, shift_u])

    # ---- the triplet band and 532 nm
    print(f"\n  THE a 3A\" BAND (states 1-3) and sigma(532 nm); Bauer's bound "
          f"{BOUND_532:.0e} cm2:")
    for anchor in out:
        st = out[anchor]["sig_k"][trip].sum(0)
        pk, smax, fw = stats(E_ev, st)
        line = (f"    {anchor:<7} peak {NM_EV / pk:.0f} nm, FWHM {fw:.3f} eV, f "
                f"{f_of(E_ev, st):.2e}, sigma(532) raw {st[i532]:.2e}")
        cal = []
        for sh in shifts:
            # move the band by the calibration shift: sigma_cal(E) = sigma(E - sh)
            cal.append(float(np.interp(NM_EV / 532.0 - sh, E_ev, st)))
        print(line + f"; calibrated {min(cal):.2e}-{max(cal):.2e}")
    # ---- how much does sigma(532) depend on what lies beyond the scan?
    # The spin-free a 3A" has a shallow well at ~2.4 A along this cut
    # (EOM: 1.97 eV there, 2.23 at 3.2 A); SOC mostly removes it. The
    # absorber and the tail decide what happens to slow parts of the packet.
    print(f"\n  sensitivity of the a 3A\" band (states 1-3) to the tail and "
          f"the absorber:")
    print(f"    {'tail':<7}{'absorber/A':>11}{'peak/nm':>9}{'FWHM/eV':>9}"
          f"{'sigma(532)':>12}{'unabsorbed':>12}")
    for tail, ra in (("linear", args.r_abs), ("linear", 3.0), ("eom", args.r_abs),
                     ("eom", 3.0), ("eom", 3.3)):
        Wt = b03.absorber(x_ang, ra, args.eta, 3, 0.4)
        tot, lmax = np.zeros_like(E), 0.0
        for k in trip:
            V, mu = curves(k, "nevpt2", tail)
            s_v, _, left = run_state(x_ang, V, mu, chis, e_lev, Wt, args, E)
            tot += (wT[:, None] * s_v).sum(axis=0)
            lmax = max(lmax, left)
        pk, _, fw = stats(E_ev, tot)
        print(f"    {tail:<7}{ra:11.2f}{NM_EV / pk:9.0f}{fw:9.3f}"
              f"{tot[i532]:12.2e}{lmax:12.1e}")
    print("    (unabsorbed > 1e-2: part of the packet is held in the well --"
          " 1D resonances, below\n     ~2.2 eV, i.e. red of ~560 nm; 532 nm "
          "lies above them)")

    st = out["nevpt2"]["sig_k"][trip].sum(0)
    s_all = s["sig_k"].sum(0)

    # ---- the whole computed spectrum against Bauer where he measured
    print(f"\n  the whole computed spectrum (states 1-{NK}) against Bauer's fit:")
    print(f"  {'nm':>6}{'calc':>11}{'triplet':>11}{'Bauer':>11}{'calc/Bauer':>12}")
    for nm in (340, 380, 407, 440, 460, 480, 490, 500, 532, 560, 600):
        i = int(np.argmin(np.abs(lam - nm)))
        b = float(bauer(np.array([float(nm)]))[0])
        print(f"  {nm:6d}{s_all[i]:11.2e}{st[i]:11.2e}"
              f"{(b if nm <= 490 else float('nan')):11.2e}"
              f"{(s_all[i] / b if nm <= 490 and b > 0 else float('nan')):12.2f}")

    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["E_eV", "lambda_nm", "sigma_total", "sigma_triplet"]
                   + [f"sigma_{k + 1}" for k in range(NK)])
        for i in range(E_ev.size):
            w.writerow([f"{E_ev[i]:.5f}", f"{lam[i]:.2f}", f"{s_all[i]:.4e}",
                        f"{st[i]:.4e}"] + [f"{s['sig_k'][k][i]:.4e}"
                                            for k in range(NK)])
    print(f"\n  csv: {args.csv}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.6))
        ax.semilogy(lam, s_all, "k", label="computed, states 1-8")
        ax.semilogy(lam, st, "C3", label='computed a 3A" (1-3), raw')
        sh = 0.5 * (shift_v + shift_u)
        ax.semilogy(lam, np.interp(E_ev - sh, E_ev, st), "C3--",
                    label=f'a 3A" calibrated ({sh:+.2f} eV)')
        lb = np.linspace(280, 490, 200)
        ax.semilogy(lb, bauer(lb), "C0--", label="Bauer 1998 (fit)")
        ax.axvline(532, color="0.6", lw=0.8)
        ax.plot([532], [BOUND_532], "v", color="C2", label="Bauer 532 nm bound")
        ax.set_xlim(280, 650)
        ax.set_ylim(1e-22, 2e-18)
        ax.set_xlabel("wavelength / nm")
        ax.set_ylabel("sigma / cm2")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.png, dpi=130)
        print(f"  png: {args.png}")
    except Exception as exc:
        print(f"  (no plot: {exc})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut", default=str(DATA / "hoi_oi_cut.csv"))
    p.add_argument("--scan", default=str(DATA / "hoi_soc_scan.csv"))
    p.add_argument("--npts", type=int, default=1024)
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--nsteps", type=int, default=6000)
    p.add_argument("--nlev", type=int, default=3)
    p.add_argument("--r-abs", type=float, default=2.75,
                   help="absorber onset, A; the scan ends at 2.70")
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=295.0)
    p.add_argument("--condon", action="store_true")
    p.add_argument("--no-eom", action="store_true")
    p.add_argument("--no-so-ground", action="store_true",
                   help="leave out the ground state's r-dependent SOC lowering")
    p.add_argument("--csv", default=str(DATA / "hoi_band_soc_1d.csv"))
    p.add_argument("--png", default=str(DATA / "hoi_band_soc_1d.png"))
    model(p.parse_args())


if __name__ == "__main__":
    main()
