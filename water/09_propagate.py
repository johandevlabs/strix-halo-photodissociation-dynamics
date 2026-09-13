#!/usr/bin/env python3
"""
Tier 3c: real-time propagation and the UV absorption cross section.

For each vibrational state v of the ground surface:

    Phi_v(0) = mu(R,r,gamma) * chi_v        prepared on the EXCITED surface
    S_v(t)   = <Phi_v(0)|Phi_v(t)>          autocorrelation
    sigma_v(E) ~ E * Re Int dt exp(i(E_v + E) t) S_v(t)

and the temperature-dependent cross section is the Boltzmann average over v.
The dipole enters only at t = 0, which is why mu(R) was only ever needed
across the Franck-Condon window.

Absorbing boundary: a negative imaginary potential near R_max removes the
dissociating flux before it wraps around the periodic FFT grid. Without it
the packet reappears at small R and puts spurious structure in the spectrum.

The Ã band of water is a direct, repulsive dissociation, so S(t) decays in
roughly 10 fs and the band is broad and structureless. A long propagation is
not needed; resolution only has to match the width.

Validation targets for H2O: peak near 165 nm (7.5 eV), FWHM ~1 eV,
peak cross section ~6e-18 cm^2.

Usage:
    python 09_propagate.py --jacobi jacobi_surface.npz --states vib_states.npz
    python 09_propagate.py --temperatures 200 250 300
"""
import argparse
import os

import numpy as np
import scipy.fft as sfft

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator

HARTREE2EV = 27.211386245988
WORKERS = int(os.environ.get('QC_FFT_WORKERS', os.cpu_count() or 1))

HARTREE2CM = 219474.6313702
AU_TIME_FS = 0.02418884326
C_AU = 137.035999084
BOHR2_TO_CM2 = 2.8002852e-17
KB_AU = 3.166811563e-6          # hartree per kelvin


class RTProp:
    """Real-time split-operator propagator with an absorbing boundary."""

    def __init__(self, d, R_abs=None, eta=0.02, power=3,
                 r_abs=None, eta_r=None):
        self.R = d["R"]
        self.r = d["r"]
        self.x_gl = d["x_gl"]
        self.w_gl = d["w_gl"]
        self.V = d["V_ex"].astype(complex)
        self.mu_R = float(d["mu_R"])
        self.mu_r = float(d["mu_r"])

        nR, nr, ng = self.V.shape
        self.shape = (nR, nr, ng)
        self.dR = self.R[1] - self.R[0]
        self.dr = self.r[1] - self.r[0]

        kR = 2.0 * np.pi * np.fft.fftfreq(nR, d=self.dR)
        kr = 2.0 * np.pi * np.fft.fftfreq(nr, d=self.dr)
        self.T_rad = (kR[:, None]**2 / (2.0 * self.mu_R)
                      + kr[None, :]**2 / (2.0 * self.mu_r))[:, :, None]

        self.jj = np.arange(ng) * (np.arange(ng) + 1.0)
        self.Bfac = (1.0 / (2.0 * self.mu_R * self.R[:, None]**2)
                     + 1.0 / (2.0 * self.mu_r * self.r[None, :]**2))[:, :, None]

        P = np.array([np.polynomial.legendre.legval(
            self.x_gl, np.eye(ng)[j]) for j in range(ng)])
        self.A = P * self.w_gl[None, :] * ((2 * np.arange(ng) + 1) / 2.0)[:, None]
        self.B = P.T
        self.wvol = self.dR * self.dr * self.w_gl[None, None, :]

        # Absorbing potentials in BOTH radial coordinates.
        #
        # An absorber in R alone is not enough for H2O. The two OH bonds are
        # equivalent, so roughly half the flux leaves by breaking the OTHER
        # bond, which in this Jacobi set appears as r -> large. With no
        # absorber there it wraps around the periodic FFT grid and returns to
        # the Franck-Condon region, producing a spurious recurrence and a
        # comb of fake structure in the spectrum. Molecules with an
        # inequivalent spectator bond (N2O, OCS) do not have this channel.
        if R_abs is None:
            R_abs = self.R[0] + 0.75 * (self.R[-1] - self.R[0])
        if r_abs is None:
            r_abs = self.r[0] + 0.70 * (self.r[-1] - self.r[0])
        if eta_r is None:
            eta_r = eta
        self.R_abs, self.r_abs = R_abs, r_abs

        wR = np.zeros_like(self.R)
        m = self.R > R_abs
        wR[m] = eta * ((self.R[m] - R_abs) / (self.R[-1] - R_abs))**power

        wr = np.zeros_like(self.r)
        m = self.r > r_abs
        wr[m] = eta_r * ((self.r[m] - r_abs) / (self.r[-1] - r_abs))**power

        # kept SEPARATE so the flux into each channel can be integrated.
        # In the Jacobi set, R -> large is the departing atom leaving the
        # spectator OH/OD intact; r -> large is the OTHER bond breaking.
        # For HDO with H departing: R-channel = H + OD, r-channel = D + OH.
        self.capR = wR[:, None, None] * np.ones_like(wr)[None, :, None]
        self.capr = wr[None, :, None] * np.ones_like(wR)[:, None, None]
        self.cap = self.capR + self.capr
        self.V = self.V - 1j * self.cap

    def _mm(self, psi, M, out=None):
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
        if out is None:
            return (psi.reshape(-1, sh[2]) @ M.T).reshape(sh)
        np.matmul(psi.reshape(-1, sh[2]), M.T, out=out.reshape(-1, sh[2]))
        return out

    def dot(self, a, b):
        return complex(np.sum(np.conj(a) * b * self.wvol))

    def _cache(self, dt):
        """
        Precompute the propagator factors.

        These depend only on dt, which is constant, but the naive version
        rebuilt them inside every step: four np.exp calls over the full
        1.18M-point grid, ~138 ms/step, single-threaded. That dominated
        everything else and is why adding FFT workers made no difference.
        Caching turns each into an array multiply, ~11x cheaper.
        """
        if getattr(self, "_dt_cached", None) == dt:
            return
        self.eV = np.exp(-0.5j * dt * self.V)
        self.eA = np.exp(-0.5j * dt * self.jj[None, None, :] * self.Bfac)
        self.eT = np.exp(-1j * dt * self.T_rad)
        self._dt_cached = dt

    def step(self, psi, dt):
        """
        In-place throughout, with two preallocated buffers.

        The naive form allocates ~8 fresh 9.4 MB arrays per step, about
        75 MB of traffic, which on this machine costs more than the FFT.
        overwrite_x lets scipy transform in place. NOTE: the input psi is
        NOT modified; the first multiply writes into a buffer.
        """
        self._cache(dt)
        if getattr(self, "_w1", None) is None or self._w1.shape != psi.shape:
            self._w1 = np.empty(psi.shape, dtype=complex)
            self._w2 = np.empty(psi.shape, dtype=complex)
        a, b = self._w1, self._w2

        np.multiply(psi, self.eV, out=a)
        self._mm(a, self.A, out=b)
        b *= self.eA
        self._mm(b, self.B, out=a)
        a = sfft.fftn(a, axes=(0, 1), workers=WORKERS, overwrite_x=True)
        a *= self.eT
        a = sfft.ifftn(a, axes=(0, 1), workers=WORKERS, overwrite_x=True)
        self._mm(a, self.A, out=b)
        b *= self.eA
        self._mm(b, self.B, out=a)
        a *= self.eV
        return a.copy()


def embed(chi, src, dst_R, dst_r):
    """
    Put a bound-grid state onto the propagation grid.

    The two grids generally differ in extent and spacing (the bound run uses
    a small box for speed), so interpolate rather than slice. chi is smooth
    and localised well inside both grids, so this is benign; anything outside
    the source box is zero.
    """
    f = RegularGridInterpolator((src["R"], src["r"], np.arange(chi.shape[2])),
                                chi, bounds_error=False, fill_value=0.0)
    RR, rr, gg = np.meshgrid(dst_R, dst_r, np.arange(chi.shape[2]),
                             indexing="ij")
    return f(np.stack([RR, rr, gg], axis=-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jacobi", default="jacobi_surface.npz")
    p.add_argument("--states", default="vib_states.npz")
    p.add_argument("--nstates", type=int, default=None)
    p.add_argument("--dt", type=float, default=1.0, help="atomic units")
    p.add_argument("--nsteps", type=int, default=6000)
    p.add_argument("--eta", type=float, default=0.02,
                   help="absorbing potential strength, hartree")
    p.add_argument("--R-abs", type=float, default=None,
                   help="where absorption starts, bohr")
    p.add_argument("--r-abs", type=float, default=None,
                   help="where absorption starts in r, bohr. Needed for H2O: "
                        "half the flux leaves via the other OH bond.")
    p.add_argument("--eta-r", type=float, default=None,
                   help="absorbing strength in r; defaults to --eta")
    p.add_argument("--temperatures", type=float, nargs="+",
                   default=[200.0, 250.0, 298.0])
    p.add_argument("--out", default="cross_section.npz")
    p.add_argument("--png", default="cross_section.png")
    args = p.parse_args()

    d = np.load(args.jacobi)
    sd = np.load(args.states)
    prop = RTProp(d, args.R_abs, args.eta,
                  r_abs=args.r_abs, eta_r=args.eta_r)
    mu = d["mu"]

    energies = sd["energies"]
    chis = sd["states"]
    n = min(args.nstates or len(energies), len(energies))
    # truncate the level list too: the Boltzmann weights must be built from
    # the SAME set of states that were propagated
    energies = energies[:n]

    print(f"grid     {prop.shape}  dt = {args.dt} a.u., "
          f"{args.nsteps} steps = {args.nsteps * args.dt * AU_TIME_FS:.1f} fs")
    print(f"absorb   R from {prop.R_abs:.2f} bohr, r from {prop.r_abs:.2f} "
          f"bohr, eta = {args.eta} Ha")
    print(f"states   {n} vibrational levels\n")

    T = args.nsteps * args.dt
    print(f"energy resolution 2pi/T = "
          f"{2 * np.pi / T * HARTREE2EV * 1000:.1f} meV\n")

    S_all, branch = [], []
    for v in range(n):
        chi = embed(chis[v], sd, prop.R, prop.r)
        phi0 = (mu * chi).astype(complex)
        nrm = np.sqrt(prop.dot(phi0, phi0).real)
        print(f"  v={v}  |mu chi| = {nrm:.6e}", flush=True)

        psi = phi0.copy()
        S = np.empty(args.nsteps + 1, dtype=complex)
        S[0] = prop.dot(phi0, psi)
        # Absorbed flux per channel. For an optical potential V - iW the norm
        # decays as dN/dt = -2<psi|W|psi>, so integrating each W separately
        # partitions the dissociating flux between the two arrangement
        # channels. Summed over the band, not energy resolved.
        absR = absr = 0.0
        n0 = prop.dot(phi0, phi0).real
        for it in range(1, args.nsteps + 1):
            psi = prop.step(psi, args.dt)
            d2 = np.abs(psi)**2 * prop.wvol
            absR += 2.0 * float(np.sum(d2 * prop.capR)) * args.dt
            absr += 2.0 * float(np.sum(d2 * prop.capr)) * args.dt
            S[it] = prop.dot(phi0, psi)
            if it % 1000 == 0:
                print(f"        step {it:6d}  |S| = {abs(S[it]) / abs(S[0]):.3e}",
                      flush=True)
        tot = absR + absr
        if tot > 0:
            branch.append((absR / tot, absr / tot, tot / n0))
            print(f"        channels: R (departing atom) {100*absR/tot:5.1f}%"
                  f"   r (other bond) {100*absr/tot:5.1f}%"
                  f"   captured {100*tot/n0:5.1f}% of norm", flush=True)
        S_all.append(S)

    S_all = np.array(S_all)
    t = np.arange(args.nsteps + 1) * args.dt

    # cosine window: S has already decayed, this only removes the residual
    # step at t = T which would otherwise ring across the spectrum
    win = np.cos(0.5 * np.pi * t / t[-1])**2

    E = np.linspace(4.0, 12.0, 1600) / HARTREE2EV
    sigma_v = np.zeros((n, E.size))
    for v in range(n):
        phase = np.exp(1j * (energies[v] + E[:, None]) * t[None, :])
        integ = np.trapezoid(phase * (S_all[v] * win)[None, :], t, axis=1)
        # isotropic average (1/3), atomic units, then to cm^2
        sigma_v[v] = (4.0 * np.pi * E / (3.0 * C_AU)) * 2.0 * np.real(integ)
    sigma_v *= BOHR2_TO_CM2
    sigma_v = np.clip(sigma_v, 0.0, None)

    print(f"\n{'T/K':>7}{'peak/nm':>10}{'peak sigma/cm2':>18}{'FWHM/eV':>10}")
    out = {}
    for Temp in args.temperatures:
        w = np.exp(-(energies - energies[0]) / (KB_AU * Temp))
        w /= w.sum()
        s = (w[:, None] * sigma_v).sum(axis=0)
        i = int(np.argmax(s))
        half = s.max() / 2.0
        above = E[s >= half] * HARTREE2EV
        fwhm = above.max() - above.min() if above.size else np.nan
        print(f"{Temp:7.0f}{1239.84 / (E[i] * HARTREE2EV):10.1f}"
              f"{s[i]:18.3e}{fwhm:10.3f}")
        out[f"sigma_{Temp:.0f}K"] = s

    if branch:
        b = np.array(branch)
        print(f"\nchannel branching, band-integrated (not energy resolved):")
        print(f"{'v':>4}{'R channel':>12}{'r channel':>12}{'norm captured':>15}")
        for v, (fR, fr, cap) in enumerate(b):
            print(f"{v:>4}{100*fR:11.1f}%{100*fr:11.1f}%{100*cap:14.1f}%")
        worst = b[:, 2].min()
        if worst < 0.95:
            print(f"\nWARNING: only {100*worst:.0f}% of the norm was absorbed. "
                  f"The two CAPs sit at\ndifferent distances, so the nearer "
                  f"one captures its channel first and an\nincomplete run is "
                  f"biased toward it. Increase --nsteps until this is >95%\n"
                  f"before reading any branching ratio.")
        elif abs(b[0, 0] - 0.5) < 0.02:
            print("50/50 as required when the two bonds are equivalent.")

    np.savez_compressed(args.out, branching=np.array(branch) if branch
                        else np.zeros((0, 3)),
                        E_Ha=E, E_eV=E * HARTREE2EV,
                        wavelength_nm=1239.84 / (E * HARTREE2EV),
                        sigma_v=sigma_v, energies=energies[:n],
                        t=t, S=S_all, **out)
    print(f"\nwrote {args.out}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for v in range(n):
        ax[0].plot(t * AU_TIME_FS, np.abs(S_all[v]) / abs(S_all[v][0]),
                   label=f"v={v}")
    ax[0].set_xlabel("t / fs")
    ax[0].set_ylabel("|S(t)| / |S(0)|")
    ax[0].set_title("autocorrelation")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    lam = 1239.84 / (E * HARTREE2EV)
    for v in range(n):
        ax[1].plot(lam, sigma_v[v], lw=1, label=f"v={v}")
    ax[1].set_xlim(120, 220)
    ax[1].set_xlabel("wavelength / nm")
    ax[1].set_ylabel(r"$\sigma$ / cm$^2$")
    ax[1].set_title("per vibrational state")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for Temp in args.temperatures:
        ax[2].plot(lam, out[f"sigma_{Temp:.0f}K"], label=f"{Temp:.0f} K")
    ax[2].set_xlim(120, 220)
    ax[2].set_xlabel("wavelength / nm")
    ax[2].set_ylabel(r"$\sigma$ / cm$^2$")
    ax[2].set_title("Boltzmann averaged")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.png, dpi=140)
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
