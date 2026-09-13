#!/usr/bin/env python3
"""
Tier 3d: how much does the cross section depend on the transition dipole?

Deferred from tier 2a, where SA-CASSCF and EE-ADC(2) disagreed on mu by a
factor of ~1.7 in magnitude and ~5 in the logarithmic slope across the
Franck-Condon window, with no way to decide between them. The cross section
decides it, because only mu enters the observable.

Three dipole surfaces, same everything else:

  ab-initio   mu(R,r,gamma) as computed (SA-CASSCF)
  condon      mu frozen at its value at the equilibrium geometry
  scaled      mu multiplied by a constant to match the observed peak

The comparison separates two questions that are easy to conflate:
  - does the SHAPE of mu across the FC window matter?  (ab-initio vs condon)
  - is the MAGNITUDE simply wrong?                     (ab-initio vs scaled)

If ab-initio and condon give the same band shape, the dipole's variation is
irrelevant and any remaining discrepancy is a pure scale factor, which
cancels in isotope RATIOS even though it matters for absolute sigma.

Usage:
    python 10_mu_sensitivity.py --dt 2.0 --nsteps 1000 \
        --R-abs 8.0 --eta 0.15 --r-abs 4.0
"""
import argparse
import importlib.util
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HARTREE2EV = 27.211386245988
C_AU = 137.035999084
BOHR2_TO_CM2 = 2.8002852e-17
KB_AU = 3.166811563e-6


def load_prop_module(path="09_propagate.py"):
    spec = importlib.util.spec_from_file_location("prop09", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_one(P, d, sd, mu, args, label):
    prop = P.RTProp(d, args.R_abs, args.eta,
                    r_abs=args.r_abs, eta_r=args.eta_r)
    energies = sd["energies"][:args.nstates]
    t = np.arange(args.nsteps + 1) * args.dt

    win = np.ones_like(t)
    t0 = 0.85 * t[-1]
    m = t > t0
    win[m] = np.cos(0.5 * np.pi * (t[m] - t0) / (t[-1] - t0))**2

    E = np.linspace(4.0, 12.0, 1600) / HARTREE2EV
    sig = np.zeros((len(energies), E.size))

    for v in range(len(energies)):
        chi = P.embed(sd["states"][v], sd, prop.R, prop.r)
        phi0 = (mu * chi).astype(complex)
        psi = phi0.copy()
        S = np.empty(args.nsteps + 1, dtype=complex)
        S[0] = prop.dot(phi0, psi)
        for it in range(1, args.nsteps + 1):
            psi = prop.step(psi, args.dt)
            S[it] = prop.dot(phi0, psi)
        phase = np.exp(1j * (energies[v] + E[:, None]) * t[None, :])
        integ = np.trapezoid(phase * (S * win)[None, :], t, axis=1)
        sig[v] = (4.0 * np.pi * E / (3.0 * C_AU)) * 2.0 * np.real(integ)
        print(f"    {label} v={v} done", flush=True)

    sig = np.clip(sig * BOHR2_TO_CM2, 0.0, None)
    w = np.exp(-(energies - energies[0]) / (KB_AU * args.temperature))
    w /= w.sum()
    return E, (w[:, None] * sig).sum(axis=0)


def summarize(E, s):
    i = int(np.argmax(s))
    above = E[s >= s.max() / 2] * HARTREE2EV
    nu = E * HARTREE2EV / 1.239841984e-4
    o = np.argsort(nu)
    f = 1.1296e12 * np.trapezoid(s[o], nu[o])
    return (1239.84 / (E[i] * HARTREE2EV), s.max(),
            above.max() - above.min(), f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jacobi", default="jacobi_surface.npz")
    p.add_argument("--states", default="vib_states.npz")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--nsteps", type=int, default=1000)
    p.add_argument("--eta", type=float, default=0.15)
    p.add_argument("--eta-r", type=float, default=None)
    p.add_argument("--R-abs", type=float, default=8.0)
    p.add_argument("--r-abs", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=200.0)
    p.add_argument("--target-sigma", type=float, default=6.0e-18,
                   help="observed peak cross section, cm^2")
    p.add_argument("--out", default="mu_sensitivity.npz")
    p.add_argument("--png", default="mu_sensitivity.png")
    args = p.parse_args()

    P = load_prop_module()
    d = np.load(args.jacobi)
    sd = np.load(args.states)
    mu = d["mu"]

    # mu at the equilibrium geometry, for the Condon variant
    V = d["V_gs"]
    i, j, k = np.unravel_index(np.argmin(V), V.shape)
    mu_eq = float(mu[i, j, k])
    print(f"mu at equilibrium (R={d['R'][i]:.3f}, r={d['r'][j]:.3f} bohr) "
          f"= {mu_eq:.4f} a.u.")
    print(f"mu over the grid: min {mu.min():.4f}, max {mu.max():.4f}\n")

    runs = {}
    print("running ab-initio mu")
    E, s_ab = run_one(P, d, sd, mu, args, "ab-initio")
    runs["ab_initio"] = s_ab

    print("running Condon (mu frozen at equilibrium)")
    mu_c = np.where(mu > 0, mu_eq, 0.0)
    _, s_cd = run_one(P, d, sd, mu_c, args, "condon")
    runs["condon"] = s_cd

    scale = np.sqrt(args.target_sigma / s_ab.max())
    print(f"running scaled mu (x{scale:.3f} to match "
          f"{args.target_sigma:.2e} cm2)")
    _, s_sc = run_one(P, d, sd, mu * scale, args, "scaled")
    runs["scaled"] = s_sc

    print(f"\n{'variant':>12}{'peak/nm':>10}{'sigma/cm2':>14}"
          f"{'FWHM/eV':>10}{'f':>9}")
    for name, s in runs.items():
        pk, sm, fw, f = summarize(E, s)
        print(f"{name:>12}{pk:10.1f}{sm:14.3e}{fw:10.3f}{f:9.4f}")
    print(f"{'observed':>12}{165.0:10.1f}{args.target_sigma:14.3e}"
          f"{1.0:10.3f}{0.045:9.4f}")

    # the question this script exists to answer
    a, c = runs["ab_initio"], runs["condon"]
    na, nc = a / a.max(), c / c.max()
    print(f"\nshape difference, ab-initio vs Condon (both normalised): "
          f"max {np.abs(na - nc).max():.4f}")
    print("If that is small, the VARIATION of mu across the FC window does "
          "not\nmatter and the whole discrepancy is a scale factor, which "
          "cancels in\nisotope ratios.")

    np.savez_compressed(args.out, E_eV=E * HARTREE2EV,
                        wavelength_nm=1239.84 / (E * HARTREE2EV),
                        mu_eq=mu_eq, scale=scale, **runs)
    print(f"\nwrote {args.out}")

    lam = 1239.84 / (E * HARTREE2EV)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, s in runs.items():
        ax[0].plot(lam, s, label=name)
        ax[1].plot(lam, s / s.max(), label=name)
    for a_ in ax:
        a_.set_xlim(130, 210)
        a_.set_xlabel("wavelength / nm")
        a_.grid(alpha=0.3)
        a_.legend(fontsize=8)
    ax[0].set_ylabel(r"$\sigma$ / cm$^2$")
    ax[0].set_title("absolute")
    ax[1].set_ylabel("normalised")
    ax[1].set_title("shape only")
    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
