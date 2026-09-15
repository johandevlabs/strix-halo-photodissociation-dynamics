#!/usr/bin/env python3
"""
Phase 1 scoping: a 1D wavepacket model of the HOCl a 3A" band along O-Cl.

Phase 0.5 left two facts that decide how Phase 1 is built:

  - CCSD(T) + EOM-CCSD triplet is the most reliable surface in the
    Franck-Condon window (09: smooth to 0.1 meV, slope confirmed by UCCSD(T)
    to 0.5%), but only out to ~2.29 A.
  - Near 2.35-2.4 A a second triplet, probably 3A', crosses the lowest one,
    and nothing computed so far describes that region cleanly.

So the question before any 3D raster is: does the absorption band care about
the surface beyond ~2.3 A? The autocorrelation S(t) decays as the packet
leaves the Franck-Condon region, and a direct dissociation leaves it within a
few femtoseconds. If an absorbing potential placed from ~2.3 A leaves the band
unchanged, the band needs only the region CCSD(T) + EOM-CCSD handles, and the
crossing is a question for product branching, not for sigma(lambda, T).

This script answers that in one dimension, from data already in the repo --
no new electronic structure, NumPy only, seconds of CPU:

  V_S(r)  CCSD(T) singlet along r(O-Cl), OH and angle fixed         (05, 09)
  V_T(r)  E_CCSD(T) + omega(EOM-CCSD triplet), trusted to 2.29 A    (09)
          anchored at 1.40 A by the UCCSD(T) triplet, shifted onto
          EOM's scale, since the two agree in shape to +/-2 meV     (05)
  tail    beyond 2.29 A, one of:
            nevpt2  06's symmetry-forced 3A" SC-NEVPT2 root 0, shifted to
                    join continuously. The right STATE in Cs -- a 3A' does
                    not couple to 3A" without SOC -- and its shape agreed
                    with UCCSD(T) from 2.0 to 2.2 A to 3 meV
            flat    constant at V_T(2.29 A)
            linear  EOM's last slope continued

It then varies what should NOT matter -- where the absorber starts (3.0 A in
to 2.0 A), which tail lies beyond 2.29 A, and the absorber's strength -- and
reports how much the band moves.

Also reported, as a first look rather than a result:
  - the O-Cl stretch frequency of the 1D ground well, as a sanity check;
  - band peak and width against the measured HOCl triplet band (380 nm);
  - hot-band temperature dependence from the O-Cl stretch alone.

WHAT A 1D MODEL CANNOT SAY. OH and the bend are frozen, so their
contributions to the width and to hot bands are missing (HOCl's bend, at
~1240 cm-1, contributes less Boltzmann population than the O-Cl stretch at
~725 cm-1, but it is not zero). The reduced mass treats OH as a rigid unit
moving against Cl. The transition dipole is Condon, sized from 03's
SOC-borrowed oscillator strength, which 03 itself suggested is well below
experiment, so ABSOLUTE sigma is the least reliable number here. The
absorber test is the purpose; everything else is orientation.

Conventions are water/09_propagate.py's: split-operator propagation,
S_v(t) = <mu chi_v | mu chi_v(t)>,
sigma_v(E) = (4 pi E / 3c) * 2 Re Int_0^T exp(i(E_v + E)t) S_v(t) w(t) dt,
with a cos^2 window w(t), and a Boltzmann average over v.

Usage:
    python 10_band_1d.py 2>&1 | tee logs/band_1d.log
"""
import argparse
import csv
import os

import numpy as np
from scipy.interpolate import CubicSpline

HARTREE2EV = 27.211386245988
BOHR_PER_ANG = 1.0 / 0.529177210903
AMU2AU = 1822.888486209
AU_TIME_FS = 0.02418884326
C_AU = 137.035999084
BOHR2_TO_CM2 = 2.8002852e-17
KB_AU = 3.166811563e-6
HARTREE2CM = 219474.6313702
NM_EV = 1239.841984

M_O, M_H, M_CL = 15.99491462, 1.00782503, 34.96885268
MU_AU = (M_O + M_H) * M_CL / (M_O + M_H + M_CL) * AMU2AU

# 03: SOC-borrowed f of the a 3A" band at its vertical energy, CAS(12,7)
# QD-NEVPT2. Used only to put an absolute scale on sigma (Condon).
F_03, DE_03_EV = 8.92e-7, 3.4477
OBS_PEAK_NM = 380.0            # measured HOCl triplet band; quoted, verify

EOM_TRUST_ANG = 2.2891         # last EOM-CCSD point before the crossing
R_MIN_ANG, R_MAX_ANG = 1.40, 3.60


def read(path, rkey, ekey, extra=None, filt=None):
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as fh:
        for rec in csv.DictReader(fh):
            if filt is not None and not filt(rec):
                continue
            try:
                e = float(rec[ekey]) + (float(rec[extra]) if extra else 0.0)
                out[round(float(rec["r" if rkey is None else rkey]), 4)] = e
            except (KeyError, ValueError):
                continue
    return out


def merged(*dicts):
    """Union of {r: E}, dropping any r within 1 mA of one already taken."""
    out = {}
    for d in dicts:
        for r, e in d.items():
            if all(abs(r - q) > 1e-3 for q in out):
                out[r] = e
    rs = np.array(sorted(out))
    return rs, np.array([out[r] for r in rs])


def build_surfaces(args):
    # ---- singlet: every CCSD(T) point up to 2.45 A
    s_fc = read(args.scan_fc, "r", "e_s")
    s_pin = read(args.scan_pin, "r", "e_s")
    s_eom = read(args.eom, "r", "e_ccsdt")
    s_out = read(args.eom_outer, "r", "e_ccsdt")
    rS, eS = merged(s_fc, s_eom, s_out, s_pin)
    keep = (rS >= R_MIN_ANG - 1e-6) & (rS <= 2.45)
    rS, eS = rS[keep], eS[keep]

    # ---- triplet: EOM-CCSD on the CCSD(T) ground state, trusted range only
    t_eom = read(args.eom, "r", "e_ccsdt", extra="w_0")
    t_out = read(args.eom_outer, "r", "e_ccsdt", extra="w_0",
                 filt=lambda rec: float(rec["r"]) <= EOM_TRUST_ANG + 1e-6)
    rT, eT = merged(t_eom, t_out)

    # Anchor the inner wall with UCCSD(T) at 1.40 A, moved onto EOM's scale by
    # the median offset over the shared fine grid. The median ignores the
    # single 8 meV UCCSD(T) glitch at 1.6391 A.
    u_fc = read(args.scan_fc, "r", "e_t")
    u_pin = read(args.scan_pin, "r", "e_t")
    shared = [r for r in t_eom if r in u_fc]
    offset = float(np.median([t_eom[r] - u_fc[r] for r in shared]))
    anchors = {r: e + offset for r, e in u_pin.items()
               if R_MIN_ANG - 1e-6 <= r < min(rT) - 1e-3}
    rT, eT = merged({r: e for r, e in zip(rT, eT)}, anchors)

    # ---- tail candidate: 06's 3A" SC-NEVPT2 root 0, CAS(10,6) points only
    nev = {}
    for path in args.nevpt2_tail:
        nev.update(read(path, "r", "e_nev_0",
                        filt=lambda rec: rec.get("ncas") == "6"))
    rN, eN = merged(nev)
    return (rS, eS), (rT, eT), (rN, eN), offset


def triplet_curve(r_ang, T, N, tail, r_join=EOM_TRUST_ANG):
    """V_T on r (A): the EOM surface up to r_join, then the chosen tail,
    blended over the 0.1 A before r_join. Moving r_join inward with a flat
    tail REMOVES real surface, which is the direct test of which part of V_T
    the band depends on."""
    rT, eT = T
    inner = CubicSpline(rT, eT, bc_type="natural")
    r0, r1 = r_join - 0.1, r_join
    e1, s1 = float(inner(r1)), float(inner(r1, 1))
    if tail == "nevpt2":
        rN, eN = N
        spl = CubicSpline(rN, eN, bc_type="natural")
        shift = e1 - float(spl(r1))
        lo, hi = float(rN.min()), float(rN.max())

        def outer(x):
            x = np.clip(x, lo, hi)
            return spl(x) + shift
    elif tail == "flat":
        def outer(x):
            return np.full_like(np.asarray(x, dtype=float), e1)
    elif tail == "linear":
        def outer(x):
            return e1 + s1 * (np.asarray(x, dtype=float) - r1)
    else:
        raise ValueError(tail)

    x = np.asarray(r_ang, dtype=float)
    v_in = inner(np.clip(x, rT.min(), r1))
    v_out = outer(x)
    s = np.clip((x - r0) / (r1 - r0), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * s)          # 0 below r0, 1 above r1
    return (1.0 - w) * v_in + w * v_out


def vib_levels(S, nlev, npts=500):
    """Colbert-Miller sinc-DVR on the CCSD(T) singlet."""
    rS, eS = S
    spl = CubicSpline(rS, eS, bc_type="natural")
    r_ang = np.linspace(rS.min(), rS.max(), npts)
    x = r_ang * BOHR_PER_ANG
    dx = x[1] - x[0]
    i = np.arange(npts)
    d = i[:, None] - i[None, :]
    with np.errstate(divide="ignore"):
        T = np.where(d == 0, np.pi ** 2 / 3.0, 2.0 / d.astype(float) ** 2)
    T = T * ((-1.0) ** np.abs(d)) / (2.0 * MU_AU * dx ** 2)
    H = T + np.diag(spl(r_ang))
    e, c = np.linalg.eigh(H)
    return r_ang, e[:nlev], c[:, :nlev] / np.sqrt(dx)


def propagate(x, V, W, phi0, dt, nsteps):
    n = x.size
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    eT = np.exp(-1j * dt * k ** 2 / (2.0 * MU_AU))
    eV = np.exp(-0.5j * dt * (V - 1j * W))
    psi = phi0.astype(complex).copy()
    S = np.empty(nsteps + 1, dtype=complex)
    S[0] = np.vdot(phi0, psi) * dx
    for it in range(1, nsteps + 1):
        psi = eV * psi
        psi = np.fft.ifft(eT * np.fft.fft(psi))
        psi = eV * psi
        S[it] = np.vdot(phi0, psi) * dx
    return S, float(np.sum(np.abs(psi) ** 2) * dx)


def cross_sections(S_all, e_levels, dt, E, dt_max=4.0):
    # Thin S(t) for the transform only. A spacing of 4 au still resolves
    # energies up to pi/4 Ha ~ 21 eV, far beyond this band, and keeps the
    # E x t phase matrix at tens of MB instead of ~190 MB per level.
    stride = max(1, int(dt_max // dt))
    S_all = S_all[:, ::stride]
    dt = dt * stride
    t = np.arange(S_all.shape[1]) * dt
    win = np.cos(0.5 * np.pi * t / t[-1]) ** 2
    sig = np.zeros((len(S_all), E.size))
    for v, S in enumerate(S_all):
        phase = np.exp(1j * (e_levels[v] + E[:, None]) * t[None, :])
        integ = np.trapezoid(phase * (S * win)[None, :], t, axis=1)
        sig[v] = (4.0 * np.pi * E / (3.0 * C_AU)) * 2.0 * np.real(integ)
    return np.clip(sig * BOHR2_TO_CM2, 0.0, None)


def band_stats(E, s):
    i = int(np.argmax(s))
    above = E[s >= 0.5 * s[i]] * HARTREE2EV
    fwhm = above.max() - above.min() if above.size else float("nan")
    return E[i] * HARTREE2EV, s[i], fwhm


def absorber(x_ang, r_abs, eta, power, length):
    """Ramps from 0 at r_abs to full strength at r_abs + length, then flat.

    Not water's ramp over the whole remaining grid. With power 3 over the
    ~1.6 A from 2.0 A to the grid end, the potential 0.1 A past the onset is
    at ~1e-4 of eta, so an absorber "from 2.0 A" really absorbs from ~2.6 A,
    and the first run of this script reported no sensitivity even there --
    which said nothing. A fixed-length ramp makes the onset mean what it says.
    """
    s = np.clip((x_ang - r_abs) / length, 0.0, 1.0)
    return eta * s ** power


def run_setting(x_ang, V_T, chis, e_lev, mu, W, dt, nsteps, E):
    x = x_ang * BOHR_PER_ANG
    S_all, left = [], []
    for chi in chis:
        phi0 = mu * chi
        n0 = float(np.sum(np.abs(phi0) ** 2) * (x[1] - x[0]))
        S, rem = propagate(x, V_T, W, phi0, dt, nsteps)
        S_all.append(S)
        left.append(rem / n0)
    return cross_sections(np.array(S_all), e_lev, dt, E), max(left)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scan-fc", default="hocl_scan_fc.csv")
    p.add_argument("--scan-pin", default="hocl_scan_pin.csv")
    p.add_argument("--eom", default="hocl_eom_triplet.csv")
    p.add_argument("--eom-outer", default="hocl_eom_outer.csv")
    p.add_argument("--nevpt2-tail", nargs="+",
                   default=["hocl_triplet_manifold_cas2.csv",
                            "hocl_triplet_manifold_cas3.csv",
                            "hocl_triplet_manifold_cas4.csv"])
    p.add_argument("--npts", type=int, default=1024)
    p.add_argument("--dt", type=float, default=1.0, help="atomic units")
    p.add_argument("--nsteps", type=int, default=6000)
    p.add_argument("--eta", type=float, default=0.05, help="absorber, Ha")
    p.add_argument("--power", type=int, default=3)
    p.add_argument("--cap-length", type=float, default=0.4,
                   help="A from absorber onset to full strength")
    p.add_argument("--nlev", type=int, default=3)
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[200.0, 250.0, 298.0])
    p.add_argument("--csv", default="hocl_band_1d.csv")
    p.add_argument("--png", default="hocl_band_1d.png")
    args = p.parse_args()

    S, T, N, offset = build_surfaces(args)
    e_ref = float(S[1].min())            # shift everything: small phases
    S = (S[0], S[1] - e_ref)
    T = (T[0], T[1] - e_ref)
    N = (N[0], N[1] - e_ref) if N[0].size else N

    print("=" * 78)
    print("== HOX Phase 1 scoping -- 1D a 3A\" band along r(O-Cl)")
    print("=" * 78)
    print(f"  singlet points {S[0].size} over {S[0].min():.3f}-{S[0].max():.3f} A;"
          f"  triplet points {T[0].size} over {T[0].min():.3f}-{T[0].max():.3f} A")
    print(f"  UCCSD(T) -> EOM scale offset {offset * HARTREE2EV * 1000:+.1f} meV "
          f"(median over the shared fine grid)")
    print(f"  NEVPT2 tail points {N[0].size}"
          + (f" over {N[0].min():.3f}-{N[0].max():.3f} A" if N[0].size else
             "  -- not found, the nevpt2 tail is unavailable"))
    print(f"  reduced mass OH-Cl {MU_AU / AMU2AU:.3f} amu")

    # ---- ground-state levels
    r_dvr, e_lev, c_lev = vib_levels(S, args.nlev)
    nu01 = (e_lev[1] - e_lev[0]) * HARTREE2CM
    i_min = int(np.argmin(CubicSpline(*S)(r_dvr)))
    print(f"\n  1D ground well: minimum {r_dvr[i_min]:.4f} A, "
          f"v=0->1 {nu01:.0f} cm-1, v=1->2 "
          f"{(e_lev[2] - e_lev[1]) * HARTREE2CM:.0f} cm-1")
    print("  (HOCl's O-Cl stretch is ~725 cm-1 experimentally -- quoted from "
          "memory, verify;\n   a 1D cut with OH and the angle frozen is not "
          "expected to match it exactly)")

    # ---- propagation grid and initial states
    x_ang = np.linspace(R_MIN_ANG, R_MAX_ANG, args.npts)
    chis = []
    for v in range(args.nlev):
        c = np.interp(x_ang, r_dvr, c_lev[:, v], left=0.0, right=0.0)
        c /= np.sqrt(np.sum(c ** 2) * (x_ang[1] - x_ang[0]) * BOHR_PER_ANG)
        chis.append(c)
    mu = np.sqrt(3.0 * F_03 / (2.0 * DE_03_EV / HARTREE2EV))
    E = np.linspace(1.5, 5.5, 2000) / HARTREE2EV
    t_end = args.nsteps * args.dt
    print(f"\n  grid {args.npts} points {R_MIN_ANG}-{R_MAX_ANG} A, dt {args.dt} au, "
          f"{args.nsteps} steps = {t_end * AU_TIME_FS:.0f} fs, resolution "
          f"{2 * np.pi / t_end * HARTREE2EV * 1000:.0f} meV")
    print(f"  Condon |mu| = {mu:.3e} au from 03's f = {F_03:.2e}")

    tails = ["nevpt2", "flat", "linear"] if N[0].size else ["flat", "linear"]
    ref_tail = tails[0]
    L = args.cap_length
    J = EOM_TRUST_ANG
    # (tail, join, absorber onset, eta, ramp length)
    settings = [(ref_tail, J, 3.0, args.eta, L)]
    # absorber onset series. 1.7 and 1.6 A sit INSIDE chi_0 and are positive
    # controls: they must change the band.
    absorber_series = (2.6, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6)
    settings += [(ref_tail, J, ra, args.eta, L) for ra in absorber_series]
    # what lies beyond the EOM range, and how the absorber is built
    settings += [(tl, J, 2.3, args.eta, L) for tl in tails[1:]]
    settings += [(ref_tail, J, 2.3, args.eta * 0.4, L),
                 (ref_tail, J, 2.3, args.eta * 4, L),
                 (ref_tail, J, 2.3, args.eta, 2 * L)]
    # surface-cut series: flatten V_T beyond r_join, absorber far out. Tests
    # directly which part of the surface the band depends on; 1.78 A, on the
    # Franck-Condon slope, is the positive control.
    cut_series = (2.1, 1.95, 1.85, 1.78)
    settings += [("flat", rj, 3.0, args.eta, L) for rj in cut_series]

    V_cache = {}
    for tl, rj, *_ in settings:
        if (tl, rj) not in V_cache:
            V_cache[(tl, rj)] = triplet_curve(x_ang, T, N, tl, rj)
    w0 = np.exp(-(e_lev - e_lev[0]) / (KB_AU * 298.0))
    w0 /= w0.sum()

    print(f"\n  absorber: ramp of power {args.power} from onset to full strength "
          f"over the ramp length, flat beyond")
    print(f"  {'tail':>7}{'join/A':>8}{'onset':>7}{'full':>6}{'eta':>7}"
          f"{'peak/eV':>9}{'peak/nm':>9}{'FWHM/eV':>9}{'sigma/cm2':>12}"
          f"{'max dev':>9}{'norm left':>11}")
    results = {}
    for key in settings:
        tl, rj, ra, eta, ln = key
        W = absorber(x_ang, ra, eta, args.power, ln)
        sig_v, left = run_setting(x_ang, V_cache[(tl, rj)], chis, e_lev, mu, W,
                                  args.dt, args.nsteps, E)
        s298 = (w0[:, None] * sig_v).sum(axis=0)
        results[key] = (sig_v, s298, left)
    ref = results[settings[0]][1]
    band = ref > 0.1 * ref.max()
    devs = {}
    for key in settings:
        sig_v, s298, left = results[key]
        pk, smax, fw = band_stats(E, s298)
        dev = float(np.max(np.abs(s298[band] - ref[band])) / ref.max())
        devs[key] = dev
        tl, rj, ra, eta, ln = key
        print(f"  {tl:>7}{rj:8.3f}{ra:7.2f}{ra + ln:6.2f}{eta:7.3f}{pk:9.3f}"
              f"{NM_EV / pk:9.1f}{fw:9.3f}{smax:12.3e}{dev:9.2%}{left:11.1e}")
    print("  max dev = largest change in sigma over the band (sigma > 10% of "
          "peak), relative to\n  the peak of the reference setting (first "
          "row); norm left = unabsorbed fraction at the end.")

    # ---- temperature dependence on the reference setting
    sig_v = results[settings[0]][0]
    print(f"\n  temperature dependence (O-Cl stretch hot bands only), "
          f"reference setting:")
    lams = np.array([340.0, 380.0, 420.0, 450.0, 480.0])
    Ei = NM_EV / lams / HARTREE2EV
    rows_T = {}
    for Temp in args.temperatures:
        w = np.exp(-(e_lev - e_lev[0]) / (KB_AU * Temp))
        w /= w.sum()
        s = (w[:, None] * sig_v).sum(axis=0)
        rows_T[Temp] = s
        pk, smax, fw = band_stats(E, s)
        print(f"    {Temp:5.0f} K  v=1 pop {w[1]:.3%}  peak {NM_EV / pk:6.1f} nm  "
              f"FWHM {fw:.3f} eV")
    lo_T, hi_T = min(args.temperatures), max(args.temperatures)
    print(f"    sigma({hi_T:.0f} K) / sigma({lo_T:.0f} K) at "
          + ", ".join(f"{l:.0f} nm: {np.interp(e, E, rows_T[hi_T]) / max(np.interp(e, E, rows_T[lo_T]), 1e-300):.3f}"
                      for l, e in zip(lams, Ei)))

    # ---- verdict
    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    outer = [k for k in settings
             if k[1] >= J - 1e-9 and k[2] >= 2.3 - 1e-9]
    worst_outer = max(devs[k] for k in outer)
    print(f"  Largest band change with the EOM surface intact to {J:.2f} A and an "
          f"absorber from\n  2.3 A outward, over all tails, strengths and ramp "
          f"lengths: {worst_outer:.2%}")
    print("\n  band change against absorber onset (EOM surface intact):")
    for ra in absorber_series:
        k = (ref_tail, J, ra, args.eta, L)
        print(f"    onset {ra:.1f} A, full strength at {ra + L:.1f} A: "
              f"{devs[k]:7.2%}")
    print("  band change against where the surface is cut (flat beyond):")
    for rj in cut_series:
        k = ("flat", rj, 3.0, args.eta, L)
        print(f"    V_T flattened beyond {rj:.2f} A: {devs[k]:7.2%}")

    ctrl_abs = devs[(ref_tail, J, 1.6, args.eta, L)]
    ctrl_cut = devs[("flat", cut_series[-1], 3.0, args.eta, L)]
    thresh = 0.01
    reach_abs = max([ra for ra in absorber_series
                     if devs[(ref_tail, J, ra, args.eta, L)] >= thresh],
                    default=None)
    reach_cut = max([rj for rj in cut_series
                     if devs[("flat", rj, 3.0, args.eta, L)] >= thresh],
                    default=None)
    print()
    if ctrl_abs < 0.05 or ctrl_cut < 0.05:
        print(f"  !! POSITIVE CONTROL FAILED (absorber from 1.6 A: {ctrl_abs:.1%}, "
              f"surface cut at {cut_series[-1]} A: {ctrl_cut:.1%}; each must be "
              f"above 5%).\n     The test cannot detect sensitivity, so no "
              f"conclusion below is supported.")
    else:
        print(f"  Positive controls pass: absorber from 1.6 A changes the band "
              f"{ctrl_abs:.0%},\n  flattening beyond {cut_series[-1]} A changes it "
              f"{ctrl_cut:.0%}. The test can see sensitivity.")
        print(f"  The band starts to respond (>= {thresh:.0%}) to an absorber from "
              f"{reach_abs} A and to a surface\n  cut at {reach_cut} A.")
    if ctrl_abs >= 0.05 and ctrl_cut >= 0.05 and worst_outer < 0.02:
        print("  -> The band does not depend on the surface beyond ~2.3 A in "
              "this model. CCSD(T) +\n     EOM-CCSD out to 2.3 A with an "
              "absorber beyond is enough for sigma(lambda, T);\n     the "
              "3A'/3A\" crossing belongs to product branching.")
    else:
        print("  -> The band DOES change with what lies beyond 2.3 A. The "
              "outer surface cannot\n     be left out, even for the band.")
    pk, _, _ = band_stats(E, ref)
    print(f"\n  Orientation only: 1D peak {NM_EV / pk:.0f} nm against the measured "
          f"~{OBS_PEAK_NM:.0f} nm. Frozen OH and bend,\n  Condon, and an "
          f"absolute sigma sized from 03's f, which is itself uncertain.")

    # ---- outputs
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["E_eV", "lambda_nm"] + [f"sigma_{t:.0f}K" for t in args.temperatures])
        for i in range(E.size):
            w.writerow([E[i] * HARTREE2EV, NM_EV / (E[i] * HARTREE2EV)]
                       + [rows_T[t][i] for t in args.temperatures])
    print(f"\n  wrote {args.csv}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(7, 10))
        ax[0].plot(S[0], S[1] * HARTREE2EV, "o", ms=3, color="#1f77b4",
                   label="CCSD(T) singlet")
        for tl in tails:
            ax[0].plot(x_ang, V_cache[(tl, J)] * HARTREE2EV, lw=1,
                       label=f'3A" tail: {tl}')
        ax[0].plot(x_ang, V_cache[("flat", cut_series[-1])] * HARTREE2EV,
                   lw=1, ls="--", color="purple",
                   label=f"control: flat beyond {cut_series[-1]} A")
        ax[0].plot(T[0], T[1] * HARTREE2EV, "s", ms=3, color="k",
                   label="EOM-CCSD / anchor points")
        for v in range(args.nlev):
            # abs().max(), not max(): eigenvector signs are arbitrary, and a
            # mostly negative chi has max() ~ 1e-10 from its tail, which
            # scaled that noise up into a visible jag in the first plot.
            ax[0].plot(x_ang, e_lev[v] * HARTREE2EV
                       + 0.3 * chis[v] / np.abs(chis[v]).max(),
                       color="grey", lw=0.8)
        for ra in (2.0, 2.3, 3.0):
            ax[0].axvline(ra, color="grey", ls=":", lw=0.8)
        ax[0].axvline(EOM_TRUST_ANG, color="red", ls="--", lw=0.8,
                      label="EOM trusted to here")
        ax[0].set_xlabel("r(O-Cl) / A")
        ax[0].set_ylabel("E / eV")
        ax[0].set_ylim(-0.5, 6.0)
        ax[0].legend(fontsize=7)
        lam = NM_EV / (E * HARTREE2EV)
        # The reference, the two positive controls, and one of each variant
        # kind; the full set is in the table.
        shown = [settings[0],
                 (ref_tail, J, 2.3, args.eta, L),
                 (ref_tail, J, 1.7, args.eta, L),
                 (ref_tail, J, 1.6, args.eta, L),
                 ("flat", 1.95, 3.0, args.eta, L),
                 ("flat", cut_series[-1], 3.0, args.eta, L)]
        for key in shown:
            tl, rj, ra, eta, ln = key
            ax[1].plot(lam, results[key][1], lw=1,
                       label=f"{tl} from {rj:.2f} A, absorber {ra} A")
        ax[1].axvline(OBS_PEAK_NM, color="k", ls=":", lw=0.8)
        ax[1].set_xlim(250, 550)
        ax[1].set_xlabel("wavelength / nm")
        ax[1].set_ylabel("sigma / cm^2, 298 K")
        ax[1].legend(fontsize=6)
        for Temp in args.temperatures:
            ax[2].plot(lam, rows_T[Temp], lw=1, label=f"{Temp:.0f} K")
        ax[2].set_yscale("log")
        ax[2].set_xlim(300, 550)
        ax[2].set_ylim(ref.max() * 1e-4, ref.max() * 2)
        ax[2].set_xlabel("wavelength / nm")
        ax[2].set_ylabel("sigma / cm^2")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.png, dpi=140)
        print(f"  wrote {args.png}")
    except Exception as exc:
        print(f"  plot skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
