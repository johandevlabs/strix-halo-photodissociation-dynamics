#!/usr/bin/env python3
"""
Tier 3b: vibrational eigenstates of the ground surface by imaginary-time
relaxation in Jacobi coordinates.

exp(-H tau) damps every eigenstate at a rate set by its energy, so any trial
packet relaxes to the ground state. Excited states follow by projecting the
converged lower states out of the packet at every step, which keeps it in the
orthogonal complement and lets the next-lowest state emerge.

The propagator is the same split-operator machinery the real-time run needs,
so 09_propagate.py reuses it rather than reimplementing:

    exp(-V dt/2) exp(-Tang dt/2) exp(-Trad dt) exp(-Tang dt/2) exp(-V dt/2)

Trad is diagonal in the (p_R, p_r) FFT representation; Tang is diagonal in
the Legendre FBR with eigenvalue j(j+1), weighted by the R- and r-dependent
factor 1/(2 mu_R R^2) + 1/(2 mu_r r^2), which is diagonal in position. So the
two pieces are each diagonal somewhere, just not in the same place.

Energies are reported as <psi|H|psi> rather than from the norm decay: the
expectation value is variational and free of Trotter error, so it is far more
trustworthy than -ln||psi||/dtau at finite step size.

Reference (H2O, cm-1 above ZPE): bend 1594.7, 2*bend 3151.6,
sym stretch 3657.1, antisym stretch 3755.9.

NOTE: Jacobi coordinates break the r1 <-> r2 permutation symmetry, so the
symmetric and antisymmetric stretches are not symmetry-adapted here. Water's
stretches are strongly local-mode (99 cm-1 apart out of 3700) so they should
still converge, but expect them to be harder than the bend.

Usage:
    python 08_relax.py --jacobi jacobi_surface.npz --nstates 5
"""
import argparse
import os
import time

import numpy as np
import scipy.fft as sfft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKERS = int(os.environ.get('QC_FFT_WORKERS', os.cpu_count() or 1))

HARTREE2CM = 219474.6313702
HARTREE2EV = 27.211386245988


class Propagator:
    """Split-operator propagator on the (R, r, gamma) Jacobi grid."""

    def __init__(self, d, surface="V_gs"):
        self.R = d["R"]
        self.r = d["r"]
        self.x_gl = d["x_gl"]
        self.w_gl = d["w_gl"]
        self.V = d[surface]
        self.mu_R = float(d["mu_R"])
        self.mu_r = float(d["mu_r"])

        nR, nr, ng = self.V.shape
        self.shape = (nR, nr, ng)
        self.dR = self.R[1] - self.R[0]
        self.dr = self.r[1] - self.r[0]

        # radial kinetic energy in the FFT representation
        kR = 2.0 * np.pi * np.fft.fftfreq(nR, d=self.dR)
        kr = 2.0 * np.pi * np.fft.fftfreq(nr, d=self.dr)
        self.T_rad = (kR[:, None]**2 / (2.0 * self.mu_R)
                      + kr[None, :]**2 / (2.0 * self.mu_r))[:, :, None]

        # angular: j(j+1) in the Legendre FBR, times a position-diagonal factor
        self.jj = np.arange(ng) * (np.arange(ng) + 1.0)
        self.Bfac = (1.0 / (2.0 * self.mu_R * self.R[:, None]**2)
                     + 1.0 / (2.0 * self.mu_r * self.r[None, :]**2))[:, :, None]

        # Gauss-Legendre DVR <-> FBR transforms
        P = np.array([np.polynomial.legendre.legval(
            self.x_gl, np.eye(ng)[j]) for j in range(ng)])       # P[j, k]
        self.A = P * self.w_gl[None, :] * ((2 * np.arange(ng) + 1) / 2.0)[:, None]
        self.B = P.T                                             # B[k, j]

        # integration measure: dR dr d(cos gamma), the last via GL weights
        self.wvol = self.dR * self.dr * self.w_gl[None, None, :]

    # --- inner products ---------------------------------------------------
    @staticmethod
    def _mm(psi, M):
        """
        Angular transform as a single BLAS GEMM.

        Computes psi @ M.T with (R, r) collapsed into one axis.

        einsum('jk,irk->irj', M, psi) is the same contraction but does not
        always dispatch to gemm; matmul guarantees it, and gemm is threaded.

        BOTH directions take M.T, so both calls pass the matrix itself:
            forward   einsum('jk,irk->irj', A, psi) -> _mm(psi, A)
            backward  einsum('kj,irj->irk', B, c)   -> _mm(c,   B)
        Passing B.T to the backward call silently breaks unitarity and the
        norm diverges over a few thousand steps.
        """
        sh = psi.shape
        return (psi.reshape(-1, sh[2]) @ M.T).reshape(sh)

    def dot(self, a, b):
        return float(np.real(np.sum(np.conj(a) * b * self.wvol)))

    def norm(self, a):
        return np.sqrt(self.dot(a, a))

    # --- operator pieces --------------------------------------------------
    def _cache(self, dtau):
        """Same caching as the real-time propagator: the exponentials depend
        only on dtau and recomputing them each step dominates the cost."""
        if getattr(self, "_dtau_cached", None) == dtau:
            return
        self.eV = np.exp(-0.5 * dtau * self.V)
        self.eA = np.exp(-0.5 * dtau * self.jj[None, None, :] * self.Bfac)
        self.eT = np.exp(-dtau * self.T_rad)
        self._dtau_cached = dtau

    def _ang_c(self, psi):
        c = self._mm(psi, self.A)
        c *= self.eA
        return self._mm(c, self.B)

    def step_imag(self, psi, dtau):
        self._cache(dtau)
        psi = psi * self.eV
        psi = self._ang_c(psi)
        p = sfft.fftn(psi, axes=(0, 1), workers=WORKERS)
        p *= self.eT
        psi = sfft.ifftn(p, axes=(0, 1), workers=WORKERS)
        psi = self._ang_c(psi)
        psi = psi * self.eV
        # H is real, so the imaginary part is pure FFT roundoff. Dropping it
        # matters when a projected excited state is small: normalising a
        # near-zero vector otherwise amplifies that noise into the answer.
        return psi.real.astype(complex)

    def apply_H(self, psi):
        """Full H for the variational energy, no Trotter error."""
        c = self._mm(psi, self.A)
        c *= self.jj[None, None, :] * self.Bfac
        ang = self._mm(c, self.B)
        rad = sfft.ifftn(sfft.fftn(psi, axes=(0, 1), workers=WORKERS)
                         * self.T_rad, axes=(0, 1), workers=WORKERS)
        return rad + ang + self.V * psi

    def energy(self, psi):
        return self.dot(psi, self.apply_H(psi)) / self.dot(psi, psi)


def relax(prop, dtau, states, max_steps, tol, seed, label):
    """One state: relax, projecting out everything already converged."""
    rng = np.random.default_rng(seed)
    # Gaussian in the well, times mild noise so excited states have something
    # to grow from. A purely random start wastes many steps damping junk.
    R0, r0 = 1.84, 1.81
    env = np.exp(-((prop.R[:, None, None] - R0)**2) * 2.0
                 - ((prop.r[None, :, None] - r0)**2) * 4.0)
    psi = (env * (1.0 + 0.5 * rng.standard_normal(prop.shape))).astype(complex)
    psi[prop.V > 0.5] = 0.0

    def project(p):
        for s in states:
            p = p - s * prop.dot(s, p)
        return p

    psi = project(psi)
    psi /= prop.norm(psi)

    e_old = prop.energy(psi)
    npts = int(np.prod(prop.shape))
    t0 = time.perf_counter()
    for n in range(1, max_steps + 1):
        psi = prop.step_imag(psi, dtau)
        psi = project(psi)
        nrm = prop.norm(psi)
        if nrm < 1e-14:
            raise SystemExit(f"{label}: norm collapsed, reduce --dtau")
        psi /= nrm
        if n % 200 == 0:
            e = prop.energy(psi)
            de = abs(e - e_old)
            print(f"    {label} step {n:6d}  E = {e * HARTREE2EV:10.6f} eV"
                  f"  dE = {de * HARTREE2CM:9.3e} cm-1", flush=True)
            if de < tol:
                # Participation ratio: how many grid points the state
                # actually occupies. A collapse to a handful of points means
                # dtau is too large and the potential term has overwhelmed
                # the kinetic one, giving a converged-looking but meaningless
                # energy pinned near min(V).
                d2 = np.abs(psi)**2
                pr = (d2.sum())**2 / (d2**2).sum()
                if pr < 20:
                    print(f"    {label} WARNING: participation ratio {pr:.1f}"
                          f" -- the state has collapsed onto a few grid "
                          f"points. Reduce --dtau (try 1.0 or less).")
                print(f"    {label} converged in {n} steps, "
                      f"{time.perf_counter() - t0:.1f} s, "
                      f"occupying ~{pr:.0f} of {npts} grid points")
                return psi, e
            e_old = e
    print(f"    {label} hit max_steps ({max_steps}) without reaching tol")
    return psi, prop.energy(psi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jacobi", default="jacobi_surface.npz")
    p.add_argument("--nstates", type=int, default=5)
    p.add_argument("--dtau", type=float, default=1.0,
                   help="atomic units. Must satisfy dtau * (V range) << 1: "
                        "at dtau=20 the potential factor spans exp(-15) and "
                        "the state collapses onto min(V).")
    p.add_argument("--max-steps", type=int, default=60000)
    p.add_argument("--tol", type=float, default=1e-9,
                   help="hartree; energy change between checks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="vib_states.npz")
    p.add_argument("--png", default="vib_states.png")
    args = p.parse_args()

    d = np.load(args.jacobi)
    prop = Propagator(d, "V_gs")
    print(f"grid     {prop.shape[0]} x {prop.shape[1]} x {prop.shape[2]}"
          f"  (R, r, gamma)")
    print(f"         V_gs min {prop.V.min() * HARTREE2EV:.4f} eV, "
          f"dtau = {args.dtau} a.u.\n")

    states, energies = [], []
    for n in range(args.nstates):
        psi, e = relax(prop, args.dtau, states, args.max_steps,
                       args.tol, args.seed + n, f"v={n}")
        states.append(psi)
        energies.append(e)

    energies = np.array(energies)
    print(f"\n{'state':>6}{'E / eV':>12}{'E - E0 / cm-1':>16}")
    for n, e in enumerate(energies):
        print(f"{n:>6}{e * HARTREE2EV:12.6f}"
              f"{(e - energies[0]) * HARTREE2CM:16.1f}")

    print("\nH2O observed (cm-1): bend 1594.7, 2*bend 3151.6, "
          "sym str 3657.1, antisym str 3755.9")
    zpe = energies[0] - prop.V.min()
    print(f"ZPE above the surface minimum: {zpe * HARTREE2CM:.1f} cm-1 "
          f"(H2O observed ~4638)")

    np.savez_compressed(args.out, energies=energies,
                        states=np.array([s.real for s in states]),
                        R=prop.R, r=prop.r, x_gl=prop.x_gl, w_gl=prop.w_gl)
    print(f"\nwrote {args.out}")

    ng = prop.shape[2]
    kg = int(np.argmin(np.abs(np.degrees(np.arccos(prop.x_gl)) - 108.0)))
    fig, ax = plt.subplots(1, args.nstates, figsize=(3.1 * args.nstates, 3.4))
    for n, s in enumerate(states):
        a = ax[n] if args.nstates > 1 else ax
        z = np.real(s[:, :, kg]).T
        lim = np.abs(z).max()
        a.pcolormesh(prop.R, prop.r, z, cmap="RdBu_r",
                     vmin=-lim, vmax=lim, shading="auto")
        a.set_xlim(1.2, 4.5)
        a.set_ylim(1.2, 3.0)
        a.set_xlabel("R / bohr")
        if n == 0:
            a.set_ylabel("r / bohr")
        a.set_title(f"v={n}, {(energies[n]-energies[0])*HARTREE2CM:.0f} cm$^{{-1}}$")
    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
