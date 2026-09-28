#!/usr/bin/env python3
"""
HOBr step 13: 3D wavepacket propagation, sigma(lambda, T), and the check
the 1D model could not make.

The propagator is water's: this imports water/09_propagate.py's RTProp --
the real-time split-operator step in Jacobi coordinates, with its absorbing
potential -- rather than copying it. Water needed absorbers in BOTH radial
coordinates because either OH bond can break; HOBr has one channel, Br + OH,
so the absorber is in R only (--eta-r 0).

WHAT IS NOT WATER'S: the cross section. water/09 computes
sigma = (4 pi E / 3c) * 2 Re Int_0^inf ..., twice the correct value; the
factor was found by the sum rule in HOBr's 03 on 2026-09-24 and is fixed
here, with the sum rule Int sigma dE = 2 pi^2 f / c checked every run.

The transition dipole is Condon. Its size is set from 04's computed f
(1.5e-5, 3-4x below the measured band's 4.7-6.2e-5, see 10_compare_obs) and
the absolute sigma is reported both as computed and as the scale that would
match experiment. Every temperature RATIO is independent of it.

What this run is for, in order:

  1. The WIDTH, in 3D. The 1D band came out 437 nm and 0.57 eV wide -- Barnes
     1996's band, not the recommended Ingham/JPL/IUPAC one (457 nm, 0.28 eV).
     The bend and the O-H stretch can only ADD width, so if 3D stays near
     0.57 eV the disagreement with the recommended data stands; if it
     somehow narrows, it doesn't.
  2. sigma(298)/sigma(220), now with the bend's hot band and 2nu3 in the
     Boltzmann sum and the bend's zero-point spread in every state.
  3. The comparison with 03's 1D band, which isolates what the extra two
     dimensions do.

Usage:
    python 13_propagate.py 2>&1 | tee ../logs/propagate.log
"""
import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parents[1] / "data"
WATER = HERE.parents[3] / "water"
HARTREE2EV = 27.211386245988
HARTREE2CM = 219474.6313702
AU_TIME_FS = 0.02418884326
C_AU = 137.035999084
BOHR2_TO_CM2 = 2.8002852e-17
KB_AU = 3.166811563e-6
NM_EV = 1239.841984
F_CALC = 1.51e-5            # 04_soc_vertical.py, CAS(12,7), summed components
F_OBS = {"Barnes 1996": 4.7e-5, "Ingham 1998": 6.2e-5}      # 10_compare_obs

_spec = importlib.util.spec_from_file_location("water09", WATER / "09_propagate.py")
water09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(water09)
RTProp = water09.RTProp


def band_stats(E_ev, s):
    i = int(np.argmax(s))
    lo = E_ev[:i][s[:i] <= s[i] / 2]
    hi = E_ev[i:][s[i:] <= s[i] / 2]
    lo = lo.max() if lo.size else np.nan
    hi = hi.min() if hi.size else np.nan
    return E_ev[i], s[i], hi - lo, E_ev[i] - lo


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jacobi", default=str(DATA / "jacobi_hobr.npz"))
    p.add_argument("--states", default=str(DATA / "vib_states_hobr.npz"))
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--nsteps", type=int, default=6000)
    p.add_argument("--R-abs", type=float, default=7.5,
                   help="bohr; 03 found the band blind to the surface beyond "
                        "r_eq + 0.5 A (~4.5 bohr in R)")
    p.add_argument("--eta", type=float, default=0.02)
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[200.0, 220.0, 250.0, 298.0])
    p.add_argument("--out", default=str(DATA / "sigma_hobr_3d.npz"))
    p.add_argument("--png", default=str(DATA / "sigma_hobr_3d.png"))
    args = p.parse_args()

    d = np.load(args.jacobi)
    sd = np.load(args.states)
    prop = RTProp(d, R_abs=args.R_abs, eta=args.eta, eta_r=0.0)
    energies = sd["energies"]
    chis = sd["states"]
    labels = list(sd["labels"]) if "labels" in sd.files else [f"v{n}" for n in range(len(energies))]
    n = len(energies)
    T_end = args.nsteps * args.dt
    print(f"grid {prop.shape}; {n} states ({', '.join(labels)}); "
          f"{args.nsteps} steps of {args.dt} au = {T_end * AU_TIME_FS:.0f} fs; "
          f"absorber in R from {args.R_abs} bohr")

    # Condon mu from 04's f at the FC-averaged vertical energy of v0
    V_ex, V_gs = d["V_ex"], d["V_gs"]
    rho0 = chis[0] ** 2 * prop.wvol
    rho0 /= rho0.sum()
    vert = float((rho0 * V_ex).sum() - energies[0])
    mu = np.sqrt(3.0 * F_CALC / (2.0 * vert))
    print(f"FC-averaged vertical from v0 {vert * HARTREE2EV:.3f} eV = "
          f"{NM_EV / (vert * HARTREE2EV):.0f} nm; Condon |mu| from f {F_CALC:.2e}")

    S_all = []
    for v in range(n):
        phi0 = (mu * chis[v]).astype(complex)
        psi = phi0.copy()
        S = np.empty(args.nsteps + 1, dtype=complex)
        S[0] = prop.dot(phi0, psi)
        for it in range(1, args.nsteps + 1):
            psi = prop.step(psi, args.dt)
            S[it] = prop.dot(phi0, psi)
        tail = abs(S[-1]) / abs(S[0])
        print(f"  {labels[v]:20s} |S(T)|/|S(0)| = {tail:.1e}"
              + ("   !! S(t) has not decayed: lengthen --nsteps" if tail > 1e-3 else ""),
              flush=True)
        S_all.append(S)
    S_all = np.array(S_all)

    t = np.arange(args.nsteps + 1) * args.dt
    stride = max(1, int(4.0 // args.dt))
    tt, SS = t[::stride], S_all[:, ::stride]
    win = np.cos(0.5 * np.pi * tt / tt[-1]) ** 2
    E = np.linspace(1.5, 4.5, 3000) / HARTREE2EV
    sig_v = np.zeros((n, E.size))
    for v in range(n):
        phase = np.exp(1j * (energies[v] + E[:, None]) * tt[None, :])
        integ = np.trapezoid(phase * (SS[v] * win)[None, :], tt, axis=1)
        # Re, NOT 2 Re -- water/09's factor of 2, fixed (see docstring)
        sig_v[v] = (4.0 * np.pi * E / (3.0 * C_AU)) * np.real(integ)
    # Exact form of the sum rule: Int (sigma/E) dE = 4 pi^2 |mu|^2 / 3c,
    # because the Franck-Condon density integrates to S(0) = |mu|^2. The
    # form with f, 2 pi^2 f / c, needs an energy to turn f into |mu|^2, and
    # using <V_ex> - E0 for it (no kinetic energy) left it 6% off in 3D --
    # chi_0's <T> is half of a 0.32 eV zero-point energy. No energy here.
    sr = float(np.trapezoid(sig_v[0] / E, E) / (4 * np.pi ** 2 * mu ** 2 / (3 * C_AU)))
    print(f"\nsum rule, v0: Int (sigma/E) dE / (4 pi^2 mu^2 / 3c) = {sr:.4f}"
          + ("" if abs(sr - 1) < 0.02 else "   !! the absolute scale is wrong"))
    sig_v = np.clip(sig_v * BOHR2_TO_CM2, 0.0, None)
    E_ev = E * HARTREE2EV

    print(f"\n  {'T/K':>6}{'pops (v0, ...)':>30}{'peak/nm':>9}{'peak sigma':>12}"
          f"{'FWHM/eV':>9}{'red HWHM':>10}")
    out = {}
    for T in args.temperatures:
        w = np.exp(-(energies - energies[0]) / (KB_AU * T))
        w /= w.sum()
        s = (w[:, None] * sig_v).sum(axis=0)
        out[T] = s
        pk, sm, fw, rh = band_stats(E_ev, s)
        print(f"  {T:6.0f}{', '.join(f'{x:.2%}' for x in w):>30}{NM_EV / pk:9.1f}"
              f"{sm:12.2e}{fw:9.3f}{rh:10.3f}")
    s298 = out[298.0] if 298.0 in out else out[max(out)]
    pk, sm, fw, rh = band_stats(E_ev, s298)
    print(f"\n  scale to match the measured band's integrated f: "
          + ", ".join(f"{k} x{v / F_CALC:.1f}" for k, v in F_OBS.items()))

    # ---- against 03's 1D band and the two measurements
    print("\n  the band, 298 K:")
    print(f"    {'':22}{'peak/nm':>9}{'FWHM/eV':>9}{'red HWHM':>10}")
    print(f"    {'3D (this run)':22}{NM_EV / pk:9.1f}{fw:9.3f}{rh:10.3f}")
    try:
        rows = [r for r in csv.DictReader(open(DATA / "hobr_band_1d.csv"))
                if r["basis"] == "cc-pvtz-dk"]
        E1 = np.array([float(r["E_eV"]) for r in rows])
        s1 = np.array([float(r["sigma_298K"]) for r in rows])
        p1, _, f1, r1 = band_stats(E1, s1)
        print(f"    {'1D (03)':22}{NM_EV / p1:9.1f}{f1:9.3f}{r1:10.3f}")
    except FileNotFoundError:
        pass
    print(f"    {'Barnes 1996':22}{437.0:9.1f}{0.548:9.3f}{0.282:10.3f}")
    print(f"    {'Ingham 1998 (= JPL)':22}{457.0:9.1f}{0.281:9.3f}{0.152:10.3f}")

    if 220.0 in out and 298.0 in out:
        print("\n  sigma(298)/sigma(220):")
        print(f"    {'nm':>6}{'as computed':>13}{'Ingham-aligned':>16}{'sigma/peak':>12}")
        shift = NM_EV / 457.0 - pk
        for lam in (400, 420, 440, 457, 480, 500, 520, 550):
            e = NM_EV / lam
            a = np.interp(e, E_ev, out[298.0]) / np.interp(e, E_ev, out[220.0])
            ea = e - shift
            b = np.interp(ea, E_ev, out[298.0]) / np.interp(ea, E_ev, out[220.0])
            print(f"    {lam:6.0f}{a:13.3f}{b:16.3f}"
                  f"{np.interp(e, E_ev, out[298.0]) / sm:12.2f}")
        print("  'as computed' is the read if Barnes's band position is right "
              "(the 3D band is where it\n  is); 'Ingham-aligned' shifts it "
              "rigidly onto 457 nm first.")

    np.savez_compressed(args.out, E_eV=E_ev, sigma_v=sig_v, energies=energies,
                        labels=np.array(labels), t_au=t, S=S_all,
                        **{f"sigma_{T:.0f}K": s for T, s in out.items()})
    print(f"\n  wrote {args.out}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
        for v in range(n):
            ax[0].plot(t * AU_TIME_FS, np.abs(S_all[v]) / abs(S_all[v][0]),
                       label=labels[v])
        ax[0].set_xlim(0, 60)
        ax[0].set_xlabel("t / fs")
        ax[0].set_ylabel("|S(t)|/|S(0)|")
        ax[0].legend(fontsize=8)
        lam = NM_EV / E_ev
        for T in args.temperatures:
            ax[1].plot(lam, out[T], label=f"{T:.0f} K")
        g = lambda l, c, w, a: a * np.exp(-4 * np.log(2) * ((NM_EV / l - NM_EV / c) / w) ** 2)  # noqa: E731
        ax[1].plot(lam, g(lam, 437, 0.548, sm), "k--", lw=1, label="Barnes 1996 shape")
        ax[1].plot(lam, g(lam, 457, 0.281, sm), "k:", lw=1, label="Ingham 1998 shape")
        ax[1].set_xlim(330, 650)
        ax[1].set_xlabel("wavelength / nm")
        ax[1].set_ylabel("sigma / cm2 (computed f)")
        ax[1].legend(fontsize=7)
        base = out[min(args.temperatures)]
        for T in args.temperatures:
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = np.where(base > 1e-3 * base.max(), out[T] / base, np.nan)
            ax[2].plot(lam, rr, label=f"{T:.0f} K")
        ax[2].axvspan(440, 500, color="C2", alpha=0.1)
        ax[2].set_xlim(330, 650)
        ax[2].set_ylim(0.8, 2.0)
        ax[2].set_xlabel("wavelength / nm")
        ax[2].set_ylabel(f"sigma(T)/sigma({min(args.temperatures):.0f} K)")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.png, dpi=140)
        print(f"  wrote {args.png}")
    except Exception as exc:
        print(f"  plot skipped: {exc}")


if __name__ == "__main__":
    main()
