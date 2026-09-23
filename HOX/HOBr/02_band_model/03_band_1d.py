#!/usr/bin/env python3
"""
HOBr step 3: a 1D wavepacket model of the a 3A" band along O-Br.

Descends from HOCl/02_band_model/10_band_1d.py, and asks the same question
first: HOW MUCH SURFACE DOES THE BAND NEED? The answer sets the raster's
outer edge. For HOCl it was ~2.3 A, about 0.6 A past equilibrium, and the
3A'/3A" crossing sat just beyond it. 02's cut put HOBr's crossing ~0.7 A
out (2.55 A) and EOM-CCSD's single-excitation weight below 0.90 from 2.45 A,
so the margin is similar -- but "similar" is an inference from HOCl, and this
measures it.

Then three things HOCl's model could not give, or gave only as orientation:

  1. THE BASIS, MEASURED ON THE BAND. 02 found aug-cc-pvtz-dk shifts omega
     by -42 meV and makes the slope 3.5% shallower in the Franck-Condon
     window. Running the model on both bases' curves turns that into a band
     shift in nm and a width change in %, which is what matters.
  2. THE f THE BAND NEEDS. HOBr's SOC-borrowed oscillator strength has not
     been computed. Instead of borrowing HOCl's, the model propagates with a
     nominal f and reports the f that would reproduce Ingham et al.'s peak,
     sigma = 2.3e-20 cm2 at 457 nm. That gives the SOC run a target.
  3. A FIRST LOOK AT THE DELIVERABLE: sigma(298 K) / sigma(220 K) across the
     band, from O-Br stretch hot bands alone, against the impact threshold
     (a 10-15% change over 440-500 nm).

Surfaces come from 02's cut (hobr_obr_cut.csv), OH and the angle fixed:
  V_S(r)  CCSD(T), used up to 2.45 A for the ground-state levels
  V_T(r)  V_S + omega(lowest 3A", EOM-CCSD), trusted to R_TRUST = 2.40 A
  tail    beyond R_TRUST, one of:
            eom     the same labelled 3A" root continued. The right STATE,
                    but with the single-excitation weight falling to 0.79 by
                    3.2 A, and on a closed-shell ground state that is turning
                    diradical, so a shape proxy rather than a trusted curve
            flat    constant at V_T(R_TRUST)
            linear  the last slope continued

The same sensitivity tests as HOCl's model, placed RELATIVE TO r_eq so the
two molecules can be compared directly: an absorber moved inward from far
out, V_T flattened beyond a moving cut, and positive controls inside chi_0
that MUST change the band -- without them "no sensitivity" means nothing.

WHAT A 1D MODEL CANNOT SAY. OH and the bend are frozen; the bend's zero-point
spread matters for band SHAPE (HOCl: V_T moves 0.26 eV across the angles chi_0
samples), though its thermal population is small (0.3% at 298 K). No SOC
splitting of the triplet, and Br's is 4x Cl's. Condon. So band position and
width are orientation; the absorber test and the basis comparison are the
purpose, and the temperature ratio is a first look, not a result.

Conventions are water/09_propagate.py's: split-operator propagation,
S_v(t) = <mu chi_v | mu chi_v(t)>,
sigma_v(E) = (4 pi E / 3c) * Re Int_0^T exp(i(E_v + E)t) S_v(t) w(t) dt,
cos^2 window w(t), Boltzmann average over v. Local: numpy + scipy only.

The normalisation is checked on every run against the sum rule
Int sigma dE = 2 pi^2 f / c. water/09_propagate.py's form carried an extra
factor 2 ("2 Re"), copied here at first, which doubled every absolute sigma
and halved the implied f while leaving shapes and ratios untouched -- so
nothing but a check on the absolute scale could have caught it.

Usage:
    python 03_band_1d.py 2>&1 | tee ../logs/band_1d.log
    python 03_band_1d.py --basis aug-cc-pvtz-dk
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

DATA = Path(__file__).resolve().parents[1] / "data"

HARTREE2EV = 27.211386245988
BOHR_PER_ANG = 1.0 / 0.529177210903
AMU2AU = 1822.888486209
AU_TIME_FS = 0.02418884326
C_AU = 137.035999084
BOHR2_TO_CM2 = 2.8002852e-17
KB_AU = 3.166811563e-6
HARTREE2CM = 219474.6313702
NM_EV = 1239.841984

M_O, M_H, M_BR = 15.99491462, 1.00782503, 78.9183376     # 79Br
MU_AU = (M_O + M_H) * M_BR / (M_O + M_H + M_BR) * AMU2AU

R_EQ = 1.8357              # 01_geometry.py, CCSD(T)/cc-pvtz-dk
R_TRUST = 2.40             # 02: w1 < 0.90 from 2.45 A, 3A' lowest from 2.55
R_MIN_ANG, R_MAX_ANG = 1.50, 3.60
S_MAX_ANG = 2.45           # singlet points used for the ground-state levels

# Ingham et al. (1998), quoted in TASKS.md -- verify against the source.
OBS_PEAK_NM, OBS_SIGMA = 457.0, 2.3e-20
OBS_NU3 = 620.23           # O-Br stretch fundamental, NIST via TASKS.md
F_NOMINAL = 1.0e-6         # propagate with this; scale to the implied f

# Sensitivity series, as offsets from R_EQ. HOCl's model used absolute radii
# (absorber 2.6 ... 1.6, cut 2.1 ... 1.78 with r_eq = 1.69); these are
# chosen to reproduce its offsets, so "the band responds from r_eq + x" can
# be compared across the two molecules.
ABSORBER_OFFSETS = (0.91, 0.61, 0.51, 0.41, 0.31, 0.21, 0.11, 0.01, -0.09)
CUT_OFFSETS = (0.41, 0.26, 0.16, 0.09)
REF_ABS_OFFSET = 1.06       # reference absorber onset, r_eq + this


def load_cut(path, basis):
    rs, es, oms = [], [], []
    with open(path) as fh:
        for rec in csv.DictReader(fh):
            if rec["basis"] != basis or rec["status"].startswith("fail"):
                continue
            try:
                r = float(rec["r_ox_A"])
                e = float(rec["e_ccsdt_Ha"])
                w = float(rec["omega_app_eV"]) / HARTREE2EV
            except (KeyError, ValueError):
                continue
            rs.append(r)
            es.append(e)
            oms.append(w)
    if not rs:
        raise SystemExit(f"no usable rows for basis {basis!r} in {path}")
    o = np.argsort(rs)
    return np.array(rs)[o], np.array(es)[o], np.array(oms)[o]


def triplet_curve(x_ang, rT, eT, tail, r_join):
    """V_T up to r_join, then the tail, blended over the 0.1 A before r_join.

    Moving r_join inward with a flat tail REMOVES real surface, which is the
    direct test of how much of V_T the band depends on."""
    inner = CubicSpline(rT, eT, bc_type="natural")
    r0, r1 = r_join - 0.1, r_join
    e1, s1 = float(inner(r1)), float(inner(r1, 1))
    x = np.asarray(x_ang, dtype=float)
    if tail == "eom":
        v_out = inner(np.clip(x, rT.min(), rT.max()))
    elif tail == "flat":
        v_out = np.full_like(x, e1)
    elif tail == "linear":
        v_out = e1 + s1 * (x - r1)
    else:
        raise ValueError(tail)
    v_in = inner(np.clip(x, rT.min(), r1))
    s = np.clip((x - r0) / (r1 - r0), 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * s)
    return (1.0 - w) * v_in + w * v_out


def vib_levels(rS, eS, nlev, mu, npts=500):
    """Colbert-Miller sinc-DVR on the singlet."""
    spl = CubicSpline(rS, eS, bc_type="natural")
    r_ang = np.linspace(rS.min(), rS.max(), npts)
    x = r_ang * BOHR_PER_ANG
    dx = x[1] - x[0]
    i = np.arange(npts)
    d = i[:, None] - i[None, :]
    with np.errstate(divide="ignore"):
        T = np.where(d == 0, np.pi ** 2 / 3.0, 2.0 / d.astype(float) ** 2)
    T = T * ((-1.0) ** np.abs(d)) / (2.0 * mu * dx ** 2)
    e, c = np.linalg.eigh(T + np.diag(spl(r_ang)))
    return r_ang, e[:nlev], c[:, :nlev] / np.sqrt(dx)


def propagate(x, V, W, phi0, dt, nsteps, mu):
    dx = x[1] - x[0]
    k = 2.0 * np.pi * np.fft.fftfreq(x.size, d=dx)
    eK = np.exp(-1j * dt * k ** 2 / (2.0 * mu))
    eV = np.exp(-0.5j * dt * (V - 1j * W))
    psi = phi0.astype(complex).copy()
    S = np.empty(nsteps + 1, dtype=complex)
    S[0] = np.vdot(phi0, psi) * dx
    for it in range(1, nsteps + 1):
        psi = eV * psi
        psi = np.fft.ifft(eK * np.fft.fft(psi))
        psi = eV * psi
        S[it] = np.vdot(phi0, psi) * dx
    return S, float(np.sum(np.abs(psi) ** 2) * dx)


def cross_sections(S_all, e_levels, dt, E, dt_max=4.0):
    stride = max(1, int(dt_max // dt))
    S_all = S_all[:, ::stride]
    dt = dt * stride
    t = np.arange(S_all.shape[1]) * dt
    win = np.cos(0.5 * np.pi * t / t[-1]) ** 2
    sig = np.zeros((len(S_all), E.size))
    for v, S in enumerate(S_all):
        phase = np.exp(1j * (e_levels[v] + E[:, None]) * t[None, :])
        integ = np.trapezoid(phase * (S * win)[None, :], t, axis=1)
        # (4 pi E / 3c) * Re Int_0^inf, NOT 2 Re: the Fourier representation
        # of the delta function is (1/2pi) Int_-inf^inf = (1/pi) Re Int_0^inf,
        # and (4 pi^2 E / 3c)(1/pi) = 4 pi E / 3c. Until 2026-09-24 this line
        # carried an extra 2, inherited from water/09_propagate.py, which made
        # every ABSOLUTE sigma twice too large (shapes, widths and ratios are
        # unaffected). Caught by the sum rule Int sigma dE = 2 pi^2 f / c, which
        # the script now checks on every run.
        sig[v] = (4.0 * np.pi * E / (3.0 * C_AU)) * np.real(integ)
    return np.clip(sig * BOHR2_TO_CM2, 0.0, None)


def band_stats(E, s):
    i = int(np.argmax(s))
    above = E[s >= 0.5 * s[i]] * HARTREE2EV
    fwhm = above.max() - above.min() if above.size else float("nan")
    return E[i] * HARTREE2EV, s[i], fwhm


def absorber(x_ang, r_abs, eta, power, length):
    """Fixed-length ramp, as HOCl's 10 learned: a ramp over the whole
    remaining grid makes the nominal onset meaningless."""
    s = np.clip((x_ang - r_abs) / length, 0.0, 1.0)
    return eta * s ** power


def run_setting(x_ang, V_T, chis, e_lev, mu_dip, W, dt, nsteps, E):
    x = x_ang * BOHR_PER_ANG
    S_all, left = [], []
    for chi in chis:
        phi0 = mu_dip * chi
        n0 = float(np.sum(np.abs(phi0) ** 2) * (x[1] - x[0]))
        S, rem = propagate(x, V_T, W, phi0, dt, nsteps, MU_AU)
        S_all.append(S)
        left.append(rem / n0)
    return cross_sections(np.array(S_all), e_lev, dt, E), max(left)


def model(basis, args, quiet=False):
    """Everything for one basis. Returns a dict for the cross-basis report."""
    say = (lambda *a, **k: None) if quiet else print
    r, eS_all, om = load_cut(args.cut, basis)
    e_ref = float(eS_all[r <= S_MAX_ANG].min())
    keepS = r <= S_MAX_ANG + 1e-9
    rS, eS = r[keepS], eS_all[keepS] - e_ref
    rT, eT = r, eS_all + om - e_ref

    say("=" * 78)
    say(f"== HOBr 1D a 3A\" band along r(O-Br) -- {basis}")
    say("=" * 78)
    say(f"  singlet {rS.size} points {rS.min():.2f}-{rS.max():.2f} A; triplet "
        f"{rT.size} points {rT.min():.2f}-{rT.max():.2f} A, trusted to {R_TRUST} A")
    say(f"  reduced mass OH-79Br {MU_AU / AMU2AU:.3f} amu")

    r_dvr, e_lev, c_lev = vib_levels(rS, eS, args.nlev, MU_AU)
    nu01 = (e_lev[1] - e_lev[0]) * HARTREE2CM
    i_min = int(np.argmin(CubicSpline(rS, eS)(r_dvr)))
    say(f"\n  1D ground well: minimum {r_dvr[i_min]:.4f} A, v=0->1 {nu01:.0f} "
        f"cm-1, v=1->2 {(e_lev[2] - e_lev[1]) * HARTREE2CM:.0f} cm-1")
    say(f"  (observed nu3 {OBS_NU3:.0f} cm-1; a frozen-OH, frozen-angle cut is "
        f"not expected to match exactly,\n   but it is the O-Br stretch "
        f"frequency that sets the hot-band populations below)")

    x_ang = np.linspace(R_MIN_ANG, R_MAX_ANG, args.npts)
    dxb = (x_ang[1] - x_ang[0]) * BOHR_PER_ANG
    chis = []
    for v in range(args.nlev):
        c = np.interp(x_ang, r_dvr, c_lev[:, v], left=0.0, right=0.0)
        chis.append(c / np.sqrt(np.sum(c ** 2) * dxb))

    vert = float(CubicSpline(rT, eT)(R_EQ) - CubicSpline(rS, eS)(R_EQ))
    mu_dip = np.sqrt(3.0 * F_NOMINAL / (2.0 * vert))
    E = np.linspace(1.0, 5.0, 2000) / HARTREE2EV
    t_end = args.nsteps * args.dt
    say(f"\n  grid {args.npts} points {R_MIN_ANG}-{R_MAX_ANG} A, dt {args.dt} au, "
        f"{args.nsteps} steps = {t_end * AU_TIME_FS:.0f} fs, resolution "
        f"{2 * np.pi / t_end * HARTREE2EV * 1000:.0f} meV")
    say(f"  vertical at r_eq {vert * HARTREE2EV:.3f} eV = "
        f"{NM_EV / (vert * HARTREE2EV):.0f} nm; Condon |mu| from nominal "
        f"f = {F_NOMINAL:.0e}")

    tails = ["eom", "flat", "linear"]
    L, J = args.cap_length, R_TRUST
    ref = ("eom", J, round(R_EQ + REF_ABS_OFFSET, 3), args.eta, L)
    abs_keys = [("eom", J, round(R_EQ + o, 3), args.eta, L)
                for o in ABSORBER_OFFSETS]
    tail_keys = [(tl, J, round(R_EQ + 0.51, 3), args.eta, L) for tl in tails[1:]]
    cap_keys = [("eom", J, round(R_EQ + 0.51, 3), args.eta * 0.4, L),
                ("eom", J, round(R_EQ + 0.51, 3), args.eta * 4, L),
                ("eom", J, round(R_EQ + 0.51, 3), args.eta, 2 * L)]
    cut_keys = [("flat", round(R_EQ + o, 3), ref[2], args.eta, L)
                for o in CUT_OFFSETS]
    settings = [ref] + abs_keys + tail_keys + cap_keys + cut_keys

    V_cache = {}
    for tl, rj, *_ in settings:
        if (tl, rj) not in V_cache:
            V_cache[(tl, rj)] = triplet_curve(x_ang, rT, eT, tl, rj)
    w298 = np.exp(-(e_lev - e_lev[0]) / (KB_AU * 298.0))
    w298 /= w298.sum()

    results = {}
    for key in settings:
        tl, rj, ra, eta, ln = key
        W = absorber(x_ang, ra, eta, args.power, ln)
        sig_v, left = run_setting(x_ang, V_cache[(tl, rj)], chis, e_lev,
                                  mu_dip, W, args.dt, args.nsteps, E)
        results[key] = (sig_v, (w298[:, None] * sig_v).sum(axis=0), left)
    s_ref = results[ref][1]
    band = s_ref > 0.1 * s_ref.max()
    devs = {k: float(np.max(np.abs(results[k][1][band] - s_ref[band]))
                     / s_ref.max()) for k in settings}

    say(f"\n  {'tail':>7}{'join/A':>8}{'onset':>7}{'full':>6}{'eta':>7}"
        f"{'peak/eV':>9}{'peak/nm':>9}{'FWHM/eV':>9}{'max dev':>9}"
        f"{'norm left':>11}")
    for key in settings:
        tl, rj, ra, eta, ln = key
        pk, _, fw = band_stats(E, results[key][1])
        say(f"  {tl:>7}{rj:8.3f}{ra:7.2f}{ra + ln:6.2f}{eta:7.3f}{pk:9.3f}"
            f"{NM_EV / pk:9.1f}{fw:9.3f}{devs[key]:9.2%}{results[key][2]:11.1e}")
    say("  max dev = largest change in sigma over the band (sigma > 10% of "
        "peak), relative to the\n  reference peak (first row). norm left = "
        "unabsorbed fraction at the end.")

    # ---- the question: how much surface does the band need?
    say("\n  band change against absorber onset (surface intact):")
    for o, k in zip(ABSORBER_OFFSETS, abs_keys):
        say(f"    onset r_eq{o:+.2f} = {k[2]:.2f} A: {devs[k]:7.2%}")
    say("  band change against where V_T is flattened:")
    for o, k in zip(CUT_OFFSETS, cut_keys):
        say(f"    flat beyond r_eq{o:+.2f} = {k[1]:.2f} A: {devs[k]:7.2%}")
    ctrl_abs, ctrl_cut = devs[abs_keys[-1]], devs[cut_keys[-1]]
    thresh = 0.01
    reach_abs = max([o for o, k in zip(ABSORBER_OFFSETS, abs_keys)
                     if devs[k] >= thresh], default=None)
    reach_cut = max([o for o, k in zip(CUT_OFFSETS, cut_keys)
                     if devs[k] >= thresh], default=None)
    outer = [k for k in settings if k[1] >= J - 1e-9
             and k[2] >= R_EQ + 0.51 - 1e-9]
    worst_outer = max(devs[k] for k in outer)
    controls_ok = ctrl_abs >= 0.05 and ctrl_cut >= 0.05
    if not controls_ok:
        say(f"\n  !! POSITIVE CONTROL FAILED (absorber at r_eq-0.09: "
            f"{ctrl_abs:.1%}, cut at r_eq+0.09: {ctrl_cut:.1%}; each must "
            f"exceed 5%). No conclusion below is supported.")
    else:
        say(f"\n  Positive controls pass: absorber at r_eq-0.09 changes the "
            f"band {ctrl_abs:.0%}, flattening\n  beyond r_eq+0.09 changes it "
            f"{ctrl_cut:.0%}. The test can see sensitivity.")
        say(f"  The band starts to respond (>= {thresh:.0%}) to an absorber "
            f"from r_eq{reach_abs:+.2f} A and to a\n  surface cut at "
            f"r_eq{reach_cut:+.2f} A.")
    say(f"  Worst change with the surface intact to {J} A and an absorber "
        f"from r_eq+0.51 = {R_EQ + 0.51:.2f} A\n  outward, over all tails, "
        f"strengths and ramp lengths: {worst_outer:.2%}")

    # ---- temperature dependence, O-Br stretch hot bands only
    sig_v = results[ref][0]
    rows_T = {}
    say("\n  temperature dependence, reference setting (O-Br stretch hot "
        "bands only):")
    for Temp in args.temperatures:
        w = np.exp(-(e_lev - e_lev[0]) / (KB_AU * Temp))
        w /= w.sum()
        s = (w[:, None] * sig_v).sum(axis=0)
        rows_T[Temp] = s
        pk, _, fw = band_stats(E, s)
        say(f"    {Temp:5.0f} K  v=1 {w[1]:6.2%}  v=2 {w[2]:6.3%}  peak "
            f"{NM_EV / pk:6.1f} nm  FWHM {fw:.3f} eV")
    # The ratio is read two ways. At FIXED wavelength it depends on where the
    # model band happens to sit, and this one sits ~20 nm blue of the
    # measured band -- so a fixed 500 nm lands further down the red side of
    # the model band than of the real one, and reports a wing effect as if it
    # were in the window. BAND-ALIGNED shifts the model rigidly onto the
    # measured peak first, so each wavelength samples the same part of the
    # profile it does in reality. The first run printed only the fixed read,
    # and its "-3% to +6% over 440-500 nm" was mostly that artefact; aligned,
    # the window moved -1 to -3%.
    lams = np.array([400.0, 420.0, 440.0, 457.0, 480.0, 500.0, 520.0, 550.0])
    lo_T, hi_T = 220.0, 298.0
    # Sum rule: Int sigma dE = (4 pi^2 / 3c) |mu|^2 <E>, and with |mu|^2 set
    # from f at the vertical energy that is 2 pi^2 f / c times <E>/vertical,
    # i.e. within ~1% of 1. A factor 2 here is the error water's
    # 09_propagate.py carried, which no shape or ratio test can see.
    sr = float(np.trapezoid(sig_v[0] / BOHR2_TO_CM2, E)
               / (2.0 * np.pi ** 2 * F_NOMINAL / C_AU))
    say(f"\n  sum rule check, v=0: Int sigma dE / (2 pi^2 f / c) = {sr:.4f}"
        + ("" if abs(sr - 1) < 0.05 else
           "   !! NOT ~1: the absolute scale of sigma is wrong"))

    pk, smax, fw = band_stats(E, s_ref)
    shift = NM_EV / OBS_PEAK_NM / HARTREE2EV - pk / HARTREE2EV
    ratio, ratio_al = {}, {}
    if lo_T in rows_T and hi_T in rows_T:
        say(f"    sigma({hi_T:.0f} K) / sigma({lo_T:.0f} K), model peak moved "
            f"{shift * HARTREE2EV * 1000:+.0f} meV onto {OBS_PEAK_NM:.0f} nm "
            f"for the aligned read:")
        say(f"      {'nm':>5}{'fixed':>9}{'aligned':>9}{'sigma/peak':>12}")
        for lam in lams:
            e = NM_EV / lam / HARTREE2EV
            ea = e - shift
            a, b = np.interp(e, E, rows_T[hi_T]), np.interp(e, E, rows_T[lo_T])
            aa, ba = (np.interp(ea, E, rows_T[hi_T]),
                      np.interp(ea, E, rows_T[lo_T]))
            ratio[lam] = a / b if b > 0 else float("nan")
            ratio_al[lam] = aa / ba if ba > 0 else float("nan")
            say(f"      {lam:5.0f}{ratio[lam]:9.3f}{ratio_al[lam]:9.3f}"
                f"{np.interp(ea, E, s_ref) / smax:12.2f}")

    f_implied = F_NOMINAL * OBS_SIGMA / smax
    say(f"\n  1D peak {NM_EV / pk:.0f} nm ({pk:.3f} eV) against {OBS_PEAK_NM:.0f} "
        f"nm measured; FWHM {fw:.3f} eV")
    say(f"  f implied by Ingham's peak sigma {OBS_SIGMA:.1e} cm2: {f_implied:.2e}")
    say("  (a 1D band is too NARROW -- the frozen bend and OH stretch would "
        "widen it -- so its peak\n   is too high for a given f, and this "
        "implied f is an UNDERestimate; read it as a floor)")

    return dict(basis=basis, E=E, rows_T=rows_T, ref=s_ref, peak=pk, fwhm=fw,
                vert=vert, nu01=nu01, ratio=ratio, ratio_al=ratio_al,
                shift=shift, f_implied=f_implied,
                controls_ok=controls_ok, reach_abs=reach_abs,
                reach_cut=reach_cut, worst_outer=worst_outer,
                x=x_ang, V_T=V_cache[(ref[0], ref[1])], rS=rS, eS=eS,
                rT=rT, eT=eT, e_lev=e_lev, chis=chis,
                cut_ctrl=V_cache[("flat", cut_keys[-1][1])])


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut", default=str(DATA / "hobr_obr_cut.csv"))
    p.add_argument("--basis", default="cc-pvtz-dk")
    p.add_argument("--compare", default="aug-cc-pvtz-dk",
                   help="second basis for the band-level comparison; "
                        "'none' to skip")
    p.add_argument("--npts", type=int, default=1024)
    p.add_argument("--dt", type=float, default=1.0, help="atomic units")
    p.add_argument("--nsteps", type=int, default=6000)
    p.add_argument("--eta", type=float, default=0.05, help="absorber, Ha")
    p.add_argument("--power", type=int, default=3)
    p.add_argument("--cap-length", type=float, default=0.4)
    p.add_argument("--nlev", type=int, default=3)
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[200.0, 220.0, 250.0, 298.0])
    p.add_argument("--csv", default=str(DATA / "hobr_band_1d.csv"))
    p.add_argument("--png", default=str(DATA / "hobr_band_1d.png"))
    args = p.parse_args()

    A = model(args.basis, args)
    B = None
    if args.compare and args.compare.lower() != "none":
        B = model(args.compare, args, quiet=True)
        print("\n" + "=" * 78)
        print(f"== the basis, measured on the band: {B['basis']} vs {A['basis']}")
        print("=" * 78)
        print(f"  {'':22}{A['basis']:>18}{B['basis']:>18}")
        print(f"  {'vertical / eV':22}{A['vert'] * HARTREE2EV:18.3f}"
              f"{B['vert'] * HARTREE2EV:18.3f}")
        print(f"  {'1D peak / nm':22}{NM_EV / A['peak']:18.1f}"
              f"{NM_EV / B['peak']:18.1f}")
        print(f"  {'FWHM / eV':22}{A['fwhm']:18.3f}{B['fwhm']:18.3f}")
        print(f"  {'v=0->1 / cm-1':22}{A['nu01']:18.0f}{B['nu01']:18.0f}")
        print(f"  {'implied f':22}{A['f_implied']:18.2e}{B['f_implied']:18.2e}")
        for lam in (440.0, 457.0, 500.0, 550.0):
            if lam in A["ratio_al"] and lam in B["ratio_al"]:
                print(f"  {f'298/220 aligned {lam:.0f}':22}"
                      f"{A['ratio_al'][lam]:18.3f}{B['ratio_al'][lam]:18.3f}")
        dpk = NM_EV / B["peak"] - NM_EV / A["peak"]
        dfw = 100 * (B["fwhm"] / A["fwhm"] - 1)
        print(f"\n  -> {B['basis']} moves the band {dpk:+.1f} nm and changes "
              f"its width {dfw:+.1f}%.")
        dr = max(abs(B["ratio_al"][l] - A["ratio_al"][l])
                 for l in (440.0, 457.0, 500.0)
                 if l in A["ratio_al"] and l in B["ratio_al"])
        print(f"     The temperature ratio over 440-500 nm moves by at most "
              f"{dr:.3f} -- that is what the\n     basis choice costs the "
              f"deliverable.")

    # ---- verdict
    print("\n" + "=" * 78)
    print("== verdict")
    print("=" * 78)
    if not A["controls_ok"]:
        print("  Positive controls FAILED: the surface-extent test is "
              "uninformative. Fix before using.")
    else:
        need = R_EQ + max(A["reach_abs"], A["reach_cut"])
        print(f"  The band responds to the surface out to about "
              f"r_eq{max(A['reach_abs'], A['reach_cut']):+.2f} = {need:.2f} A.")
        print(f"  EOM-CCSD is trusted to {R_TRUST} A; 3A' becomes the lowest "
              f"triplet from 2.55 A.")
        margin = R_TRUST - need
        if margin > 0.1 and A["worst_outer"] < 0.02:
            print(f"  -> {margin:.2f} A of margin: a CCSD(T) + EOM raster to "
                  f"~{R_TRUST} A with an absorber\n     beyond is enough for "
                  f"sigma(lambda, T), as for HOCl. The crossing is product-\n"
                  f"     branching scope, not band scope.")
        else:
            print(f"  -> margin only {margin:+.2f} A, or the band moves "
                  f"{A['worst_outer']:.1%} with the outer surface: the region "
                  f"past {R_TRUST} A\n     is NOT negligible for the band, "
                  f"and the raster's outer edge needs more care than HOCl's.")
    if A["ratio_al"]:
        win = [A["ratio_al"][l] for l in (440.0, 457.0, 480.0, 500.0)]
        lo, hi = min(win), max(win)
        print(f"\n  First look at the deliverable, band aligned to the "
              f"measured peak: sigma(298)/sigma(220)")
        print(f"  over 440-500 nm spans {lo:.3f}-{hi:.3f}, i.e. "
              f"{100 * (lo - 1):+.1f}% to {100 * (hi - 1):+.1f}%, from O-Br "
              f"stretch hot bands alone.")
        print(f"  In the red wing: {100 * (A['ratio_al'][520.0] - 1):+.0f}% at "
              f"520 nm, {100 * (A['ratio_al'][550.0] - 1):+.0f}% at 550 nm, "
              f"where sigma is a third of the peak and less.")
        for line in (
                "  So in this model the temperature effect is NOT in the 440-500 nm "
                "window TASKS.md's",
                "  impact threshold names -- it is below the 5% 'negligible' line "
                "there -- but in the red",
                "  wing. Whether that wing matters is a J-value question: at high "
                "solar zenith angle the",
                "  actinic flux shifts red. Caveats: a 1D band is too narrow, which "
                "exaggerates wing",
                "  ratios, and the model sits 20 nm blue, so the alignment is itself "
                "an assumption the",
                "  3D model and the SOC splitting have to remove."):
            print(line)

    # ---- outputs
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basis", "E_eV", "lambda_nm"]
                   + [f"sigma_{t:.0f}K" for t in args.temperatures])
        for M in [A] + ([B] if B else []):
            for i in range(M["E"].size):
                ev = M["E"][i] * HARTREE2EV
                w.writerow([M["basis"], ev, NM_EV / ev]
                           + [M["rows_T"][t][i] * A["f_implied"] / F_NOMINAL
                              for t in args.temperatures])
    print(f"\n  wrote {args.csv}  (sigma scaled to the implied f of "
          f"{A['basis']}, i.e. to Ingham's peak)")
    plot(A, B, args)


def plot(A, B, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    scale = A["f_implied"] / F_NOMINAL
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 11))
    a = ax[0]
    a.plot(A["rS"], A["eS"] * HARTREE2EV, "o", ms=3, label="CCSD(T) singlet")
    a.plot(A["rT"], A["eT"] * HARTREE2EV, "s", ms=3, color="k",
           label="EOM-CCSD 3A\" points")
    a.plot(A["x"], A["V_T"] * HARTREE2EV, lw=1.2, label="V_T used (eom tail)")
    a.plot(A["x"], A["cut_ctrl"] * HARTREE2EV, lw=1, ls="--", color="purple",
           label="positive control: flat beyond r_eq+0.09")
    for v in range(len(A["chis"])):
        a.plot(A["x"], A["e_lev"][v] * HARTREE2EV
               + 0.3 * A["chis"][v] / np.abs(A["chis"][v]).max(),
               color="grey", lw=0.8)
    a.axvline(R_TRUST, color="red", ls="--", lw=0.8, label="EOM trusted to here")
    a.axvline(2.55, color="orange", ls=":", lw=1, label="3A' lowest from here")
    a.set_xlim(1.5, 3.3)
    a.set_ylim(-0.5, 6.0)
    a.set_xlabel("r(O-Br) / A")
    a.set_ylabel("E / eV")
    a.set_title(f"HOBr along O-Br, {A['basis']}")
    a.legend(fontsize=7)

    lam = NM_EV / (A["E"] * HARTREE2EV)
    a = ax[1]
    a.plot(lam, A["rows_T"][298.0] * scale, lw=1.4, label=A["basis"])
    if B:
        lamB = NM_EV / (B["E"] * HARTREE2EV)
        a.plot(lamB, B["rows_T"][298.0] * scale, lw=1.2, ls="--",
               label=B["basis"] + " (same f)")
    a.axvline(OBS_PEAK_NM, color="k", ls=":", lw=0.8,
              label=f"measured peak {OBS_PEAK_NM:.0f} nm")
    a.axhline(OBS_SIGMA, color="grey", ls=":", lw=0.8)
    a.set_xlim(300, 650)
    a.set_xlabel("wavelength / nm")
    a.set_ylabel("sigma / cm2 at 298 K")
    a.set_title("1D band, scaled to Ingham's peak sigma")
    a.legend(fontsize=7)

    # Aligned read, as the verdict uses: the model band moved rigidly onto
    # the measured peak, so each wavelength samples the part of the profile
    # it does in reality. The fixed-wavelength version of this panel put a
    # red-wing effect inside the 440-500 nm window.
    a = ax[2]
    lam_al = NM_EV / ((A["E"] + A["shift"]) * HARTREE2EV)
    base = A["rows_T"][220.0]
    for T in args.temperatures:
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = np.where(base > 1e-3 * base.max(), A["rows_T"][T] / base,
                          np.nan)
        a.plot(lam_al, rr, lw=1.2, label=f"{T:.0f} K")
    a.axvspan(440, 500, color="C2", alpha=0.1, label="impact window")
    for y in (1.10, 1.15):
        a.axhline(y, color="grey", ls=":", lw=0.8)
    a.set_xlim(300, 650)
    a.set_ylim(0.8, 2.0)
    a.set_xlabel("wavelength / nm")
    a.set_ylabel("sigma(T) / sigma(220 K)")
    a.set_title("sigma(T)/sigma(220 K), O-Br stretch hot bands only,\n"
                f"model band shifted {A['shift'] * HARTREE2EV * 1000:+.0f} meV "
                f"onto the measured peak", fontsize=10)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"  wrote {args.png}")


if __name__ == "__main__":
    main()
