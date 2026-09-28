#!/usr/bin/env python3
"""
HOBr step 12: vibrational states of the ground surface, by imaginary time.

The machinery is water's: this imports water/08_relax.py's Propagator -- the
split-operator step in Jacobi coordinates, FFT in R and r, Legendre DVR in
gamma, the variational energy <psi|H|psi> -- rather than copying it. What is
HOBr's own is the relaxation loop, whose starting guess in water's version is
hard-wired to water's geometry, and the assignment and checks below.

Which states, and why. sigma(lambda, T) is a Boltzmann sum over them. At
298 K the O-Br stretch nu3 (620 cm-1 observed) holds 4.7%, the bend nu2
(1163) 0.3%, 2nu3 ~0.2%; everything else is negligible. So four states:
v0, nu3, nu2, 2nu3. nu2 and 2nu3 are only ~70 cm-1 apart, and imaginary time
separates states at a rate set by their gap, so those two are slow -- that is
why the step limit is high.

ASSIGNMENT is not trusted to energy order. Each state's spread along R, r
and gamma is compared with v0's: a nu3 state is ~3x as wide in R, a bend ~3x
in gamma. The energies are then checked against 01's harmonic frequencies
and the observed fundamentals.

Usage:
    python 12_relax.py 2>&1 | tee ../logs/relax.log
    python 12_relax.py --jacobi ../data/jacobi_coarse.npz --nstates 2 --out /tmp/v.npz
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DATA = HERE.parents[1] / "data"
WATER = HERE.parents[3] / "water"
HARTREE2CM = 219474.6313702
HARTREE2EV = 27.211386245988

_spec = importlib.util.spec_from_file_location("water08", WATER / "08_relax.py")
water08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(water08)
Propagator = water08.Propagator

OBS = {"nu3 (O-Br stretch)": 620.23, "nu2 (bend)": 1162.57,
       "2 nu3": None}                               # observed fundamentals
HARM = {"nu3 (O-Br stretch)": 632.4, "nu2 (bend)": 1199.4}   # 01_geometry.py


def spreads(prop, psi):
    """<(q - <q>)^2> along R, r and gamma, from |psi|^2 with the grid measure."""
    rho = np.abs(psi) ** 2 * prop.wvol
    rho /= rho.sum()
    g = np.arccos(prop.x_gl)
    out = []
    for q in (prop.R[:, None, None], prop.r[None, :, None], g[None, None, :]):
        m = float((rho * q).sum())
        out.append(float((rho * (q - m) ** 2).sum()))
    return np.array(out)


def relax(prop, dtau, states, max_steps, tol, seed, label, check_every=200):
    """One state: relax from a noisy Gaussian at the minimum, projecting out
    the states already converged."""
    rng = np.random.default_rng(seed)
    i, j, k = np.unravel_index(int(np.argmin(prop.V)), prop.shape)
    R0, r0, g0 = prop.R[i], prop.r[j], np.arccos(prop.x_gl)[k]
    g = np.arccos(prop.x_gl)
    env = np.exp(-((prop.R[:, None, None] - R0) ** 2) * 8.0
                 - ((prop.r[None, :, None] - r0) ** 2) * 8.0
                 - ((g[None, None, :] - g0) ** 2) * 6.0)
    psi = (env * (1.0 + 0.5 * rng.standard_normal(prop.shape))).astype(complex)
    psi[prop.V > 0.3] = 0.0

    def project(p):
        for s in states:
            p = p - s * prop.dot(s, p)
        return p

    psi = project(psi)
    psi /= prop.norm(psi)
    e_old = prop.energy(psi)
    t0 = time.perf_counter()
    for n in range(1, max_steps + 1):
        psi = project(prop.step_imag(psi, dtau))
        nrm = prop.norm(psi)
        if nrm < 1e-14:
            raise SystemExit(f"{label}: norm collapsed; reduce --dtau")
        psi /= nrm
        if n % check_every == 0:
            e = prop.energy(psi)
            de = abs(e - e_old)
            if n % (10 * check_every) == 0:
                print(f"    {label} step {n:6d}  E = {e * HARTREE2CM:10.2f} cm-1"
                      f"  dE = {de * HARTREE2CM:9.2e} cm-1", flush=True)
            if de < tol:
                print(f"    {label} converged in {n} steps, "
                      f"{time.perf_counter() - t0:.0f} s", flush=True)
                return psi, e, True
            e_old = e
    print(f"    {label} hit --max-steps ({max_steps}) without converging")
    return psi, prop.energy(psi), False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jacobi", default=str(DATA / "jacobi_hobr.npz"))
    p.add_argument("--nstates", type=int, default=4)
    p.add_argument("--dtau", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=80000)
    p.add_argument("--tol", type=float, default=1e-10, help="hartree")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(DATA / "vib_states_hobr.npz"))
    args = p.parse_args()

    d = np.load(args.jacobi)
    prop = Propagator(d, "V_gs")
    print(f"grid {prop.shape} (R, r, gamma); V_gs min {prop.V.min() * HARTREE2EV:.4f}"
          f" eV; dtau {args.dtau} au")

    states, energies, conv = [], [], []
    for n in range(args.nstates):
        psi, e, ok = relax(prop, args.dtau, states, args.max_steps, args.tol,
                           args.seed + n, f"v={n}")
        states.append(psi)
        energies.append(e)
        conv.append(ok)
    energies = np.array(energies)

    s0 = spreads(prop, states[0])
    print(f"\n  ZPE above the surface minimum: "
          f"{(energies[0] - prop.V.min()) * HARTREE2CM:.0f} cm-1")
    print(f"  {'state':>6}{'E-E0/cm-1':>11}{'<dR2>/v0':>10}{'<dr2>/v0':>10}"
          f"{'<dg2>/v0':>10}   assignment")
    labels = []
    for n, (psi, e) in enumerate(zip(states, energies)):
        ratio = spreads(prop, psi) / s0
        if n == 0:
            lab = "v0"
        elif ratio[0] > 2.0 and ratio[2] < 1.5:
            lab = "nu3 (O-Br stretch)" if ratio[0] < 4.0 else "2 nu3"
        elif ratio[2] > 2.0 and ratio[0] < 1.5:
            lab = "nu2 (bend)"
        elif ratio[1] > 2.0:
            lab = "nu1 (O-H stretch)"
        else:
            lab = "mixed / unassigned"
        labels.append(lab)
        print(f"  {n:>6}{(e - energies[0]) * HARTREE2CM:11.1f}{ratio[0]:10.2f}"
              f"{ratio[1]:10.2f}{ratio[2]:10.2f}   {lab}"
              + ("" if conv[n] else "   (NOT converged)"))

    print("\n  against 01's harmonic frequencies and the observed fundamentals:")
    for n, lab in enumerate(labels):
        if lab in OBS and OBS[lab]:
            de = (energies[n] - energies[0]) * HARTREE2CM
            print(f"    {lab:20s} {de:8.1f} cm-1   harmonic {HARM[lab]:.1f}, "
                  f"observed {OBS[lab]:.1f}  (calc/obs {de / OBS[lab]:.3f})")
    print("  An anharmonic fundamental from the full surface should sit BELOW "
          "the harmonic value and\n  near the observed one; calc/obs well away "
          "from 1 means the surface or the grid is off.")

    np.savez_compressed(args.out, energies=energies,
                        states=np.array([s.real for s in states]),
                        labels=np.array(labels), converged=np.array(conv),
                        R=prop.R, r=prop.r, x_gl=prop.x_gl, w_gl=prop.w_gl)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
