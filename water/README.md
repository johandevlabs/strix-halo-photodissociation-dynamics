# H2O photodissociation from first principles, on a mini PC

What started as a hardware benchmark — how does a GMKtec EVO-X2 (Ryzen AI
Max+ 395, 16 cores, 96 GB unified) compare to the HPC allocations a
computational chemistry PhD used around 2011-2013 — turned into a complete
pipeline: electronic structure, potential energy surfaces, vibrational
eigenstates, wavepacket dynamics, and a UV absorption cross section compared
against measurement, for H2O, HDO and D2O.

Everything below was produced on one 120 W machine. The full chain runs in a
few hours; individual stages run in minutes.

## Background

The first absorption band of water, X~ + hv -> A~(1B1) -> H(2S) + OH(X 2Pi),
is the textbook case of fast direct dissociation on a repulsive surface, and
one of the most thoroughly studied problems in photodissociation dynamics
[1,2]. HDO is the classic demonstration of bond-selective photochemistry:
because the O-H and O-D stretches are local modes, exciting one of them
preferentially breaks that bond [3,4].

This project reproduces that physics end to end with open-source tools on a
single desktop machine. Nothing here is a new result. What is offered is a
complete, scripted, reproducible pipeline -- electronic structure, potential
energy surfaces, vibrational eigenstates, wavepacket dynamics, comparison
against measurement -- plus a characterisation of what a 2025 mini PC can do
with this class of problem, and a catalogue of the ways it can quietly go
wrong.

---

## Results

### The isotope effect

The sharpest result, and the one nothing was fitted to. All four measured
datasets come from a single apparatus and a single paper [7], so
experimental systematics largely cancel in the ratio.

| band width relative to H2O | calc | measured | error |
| --- | --- | --- | --- |
| HDO | 0.900 | 0.873 | +3.1% |
| D2O | 0.816 | 0.812 | **+0.5%** |

The D2O band is 19% narrower than H2O's, predicted to 0.5%. The effect
follows from chi_0 narrowing with mass, so it tests the CCSD(T) ground-state
surface rather than the excited one.

### The cross section

Against Chung et al. (2001), 0.2 nm resolution, 295 K [7,8]:

| | peak / nm | | FWHM / eV | | mu scale |
| --- | --- | --- | --- | --- | --- |
| | calc | meas | calc | meas | |
| H2O | 166.9 | 166.4 | 0.951 | 0.937 | 1.288 |
| HDO | 166.9 | 165.8 | 0.856 | 0.818 | 1.311 |
| HDOd | 166.9 | 165.8 | 0.856 | 0.818 | 1.311 |
| D2O | 166.8 | 166.0 | 0.775 | 0.761 | 1.254 |

Peak positions within 0.024-0.051 eV, widths within 1.4-4.6%. sigma and f are
low by the same factor in every case, the signature of a pure scale error:
the surfaces are right and the transition dipole magnitude is not. That
factor is **1.291 ± 0.023** across four independent runs — constant, as an
electronic quantity must be. Measured oscillator strengths vary 3.8% across
isotopologues, computed ones 5.0%.

Two honest limits. The measured peaks are *non-monotonic* in deuteration
(H2O 166.4, HDO 165.8, D2O 166.0 nm) but the whole spread is 0.027 eV,
smaller than the calculation's own peak errors, so this cannot resolve it.
And HDO's width error (+4.6%) is worse than either symmetric case, the same
in both coordinate sets, possibly because a mixed local-mode ground state is
more sensitive to the residual softness in the surface.

### Vibrational levels

Ground-state surface at CCSD(T)/aug-cc-pVTZ, levels from imaginary-time
relaxation. Assignments confirmed from wavefunction node patterns, not
energy order.

| | mode | calc | obs | err |
| --- | --- | --- | --- | --- |
| H2O | bend | 1587.3 | 1594.7 | −0.46% |
| | 2*bend | 3138.3 | 3151.6 | −0.42% |
| | sym str | 3628.4 | 3657.1 | −0.78% |
| | antisym str | 3723.8 | 3755.9 | −0.85% |
| HDO | bend | 1397.9 | 1403.5 | −0.40% |
| | OD str | 2699.5 | 2723.7 | −0.89% |
| | OH str | 3677.7 | 3707.5 | −0.80% |
| D2O | bend | 1173.3 | 1178.4 | −0.43% |
| | sym str | 2650.2 | 2671.6 | −0.80% |
| | antisym str | 2762.0 | 2788.0 | −0.93% |

Ten comparisons, mean −0.68%, spread 0.53%, every one low. A uniform
softness in the surface, essentially identical across isotopologues, so it
largely cancels in isotope ratios.

ZPE: 4690.5 cm-1 vs 4638 observed. The earlier CAS(8,6)/NEVPT2 ground
surface gave 4811.8, so switching to CCSD(T) cut the error 3.3x. It also
fixed the stretch splitting, 153.1 -> 95.4 cm-1 against 98.8 observed.

### Isotopologues

The isotope effect is a **band narrowing**, 18% from H2O to D2O, not a
shift. The peak barely moves because two effects cancel: deuteration lowers
the ZPE (raising hv = E_ex − E_v) but also narrows chi_0 so it samples less
of the repulsive wall (lowering <V_ex>). Both scale as mu^(-1/2). The width
follows the reflection principle, FWHM ~ |dV_ex/dr| * sigma with
sigma ~ mu^(-1/4): predicted D2O/H2O ratio 0.853, observed 0.815.

f varies by only 2.3% and 5.0% across isotopologues, as it must, since f
depends on |mu|^2 and is nearly mass-independent. Three separate
relaxations and propagations on different grids agreeing to that level is
the strongest internal consistency check in the set.

HDO was run twice in *different Jacobi coordinate sets* (H departing vs D
departing). Peak, sigma and f agree to 0.0%, and the wavefunctions show the
same physical modes with node orientations rotated, since R and r swap
meaning. Coordinate-system independence, demonstrated numerically.

### Product channel branching

Integrating the flux absorbed by each CAP separately partitions the
dissociating flux between the two arrangement channels. Band-integrated, not
energy resolved. All runs reached 100% norm capture.

H2O and D2O are the calibration: the two bonds are equivalent, so any
deviation from 50/50 is the coordinate system, not physics. Observed
deviation 0.2-0.8% (H2O) and 0.0-0.5% (D2O). That is the systematic floor.

HDO, with modes assigned from the wavefunction node patterns:

| | mode | H+OD | D+OH | ratio |
| --- | --- | --- | --- | --- |
| v=0 | ZPE | 74.6% | 25.4% | 2.94 |
| v=1 | bend 1398 | 76.1% | 23.9% | 3.18 |
| v=2 | **OD stretch 2699** | 58.5% | 41.5% | **1.41** |
| v=3 | 2*bend 2771 | 76.3% | 23.7% | 3.22 |
| v=4 | **OH stretch 3678** | 84.0% | 16.0% | **5.25** |

Bond-selective photodissociation. Exciting the OD stretch raises the O-D
channel from 25.4% to 41.5%; exciting the OH stretch raises O-H from 74.6%
to 84.0%. Between the two the branching ratio changes by a factor of 3.7.

The bend is a clean control: v=1 and v=3 give 3.18 and 3.22 against the ZPE
2.94, so ~1400 and ~2800 cm-1 deposited in the bend buys essentially no
selectivity while 2700 cm-1 in the OD stretch flips it. The selectivity is
bond-specific, not energy-driven -- which follows directly from the
local-mode character visible in the HDO wavefunctions.

Cross-check: the same quantity from the two independent coordinate sets.

| | via HDO | via HDOd | diff |
| --- | --- | --- | --- |
| v=0 | 74.6% | 74.8% | −0.2 |
| v=1 | 76.1% | 76.2% | −0.1 |
| v=2 | 58.5% | 59.2% | −0.7 |
| v=3 | 76.3% | 76.4% | −0.1 |
| v=4 | 84.0% | 84.3% | −0.3 |

Different grids, different absorber geometry, independent relaxations and
propagations, agreeing to 0.7% worst case.

### Other validations along the way

- CCSD(T)/cc-pVDZ and /cc-pVTZ on water reproduce reference energies to all
  10 digits.
- OH diatomic: De 4.628 eV (NEVPT2) and 4.631 (UCCSD(T)) vs 4.62 observed;
  two unrelated methods agreeing to 3 meV.
- Splice size-consistency: the offset between the raster asymptote and
  E_H + V_OH(r) is constant to 13 meV across the probe range, as SC-NEVPT2
  requires.
- Vertical excitation at equilibrium, aug-cc-pVQZ: NEVPT2 7.699 eV,
  EOM-CCSD 7.688 eV. Two unrelated methods within 11 meV.

---

## Hardware findings (tier 0)

Measure, don't reason. Several confident predictions were wrong.

| test | result |
| --- | --- |
| DGEMM, 16 threads | 1573-1623 GFLOP/s, 77% parallel efficiency |
| DGEMM, 1 thread | 127.9 GFLOP/s |
| memory bandwidth | ~115 GB/s (triad), 144 GB/s (copy), saturates at 8 threads |
| FFT 256^3, scipy 16 threads | 13.6 GFLOP/s |
| FFT 256^3, pyfftw 16 threads | 65 GFLOP/s |

**Strix Halo has a full 512-bit FP64 datapath.** 127.9 GFLOP/s on one core
would need 7.8 GHz at 16 FLOP/cycle, which is impossible; at 32 FLOP/cycle
it implies ~3.9 GHz. A widely quoted 1.31 TFLOP/s "peak" for this part
assumes the narrower datapath and is wrong by 2x.

**Don't force the BLAS kernel.** Auto-detection beat every explicit choice:
auto 1623, COOPERLAKE 1469, SKYLAKEX 1446, ZEN (AVX2) 973, HASWELL 903. The
67% gap between auto and the AVX2 path is the value of AVX-512 here.

**Compute scales, bandwidth does not.** DGEMM holds 77% efficiency to 16
threads; pure bandwidth saturates at 8 threads and 2.5x. A single core
already reaches 40% of achievable bandwidth. This splits the workload
cleanly: electronic structure scales, wavepacket propagation is closer to
bandwidth-bound.

**96 GB buys less than expected.** CCSD(T)/aug-cc-pV5Z (287 functions) took
53.4 s fully in-core against 84.1 s restricted to 4 GB — only 1.57x, with
peak RSS dropping from 31.9 GB to 3.7 GB. Fast NVMe has absorbed most of
what large RAM used to buy; in 2012 that same spill went to spinning disk
and the penalty was 5-10x.

**The old job-array pattern still wins.** The PES raster runs one
single-threaded process per geometry, 30 processes, ~5000 points/h. 3289
points of SA-CASSCF+NEVPT2/aug-cc-pVTZ in 41 minutes.

---

## Pipeline

Shared inputs live in the project root; per-isotopologue outputs in
`runs/<LABEL>/`.

| script | what it does |
| --- | --- |
| `01_calibrate.py` | DGEMM, memory bandwidth, 3D FFT vs thread count |
| `02_h2o_ladder.py` | CCSD(T) basis-set ladder; in-core vs constrained memory |
| `03_tdm_check.py` | 1D FC-region scan: CASSCF vs ADC(2) energies and mu |
| `04_pes_grid.py` | 3D SA-CASSCF/NEVPT2 raster, r1>=r2 triangle, resumable |
| `05_oh_diatomic.py` | OH(X 2Pi) curve + E_H for the asymptotic splice |
| `06_splice.py` | mirror, fill, blend raster into E_H + V_OH(r2); writes surfaces |
| `07_jacobi.py` | valence (r1,r2,theta) -> Jacobi (R,r,gamma) |
| `08_relax.py` | imaginary-time relaxation, excited states by projection |
| `09_propagate.py` | real-time propagation, autocorrelation, cross section, per-channel flux |
| `10_mu_sensitivity.py` | ab-initio vs Condon vs scaled transition dipole |
| `11_compare_obs.py` | fetch MPI-Mainz data, compare shape and scale |
| `12_vertical.py` | vertical excitation vs basis, active space, method |
| `13_gs_well.py` | CCSD(T) ground-state surface over the bound region |
| `15_summary.py` | collect all isotopologue runs into one table |
| `16_compare_all.py` | all isotopologues vs measurement, incl. the narrowing |
| `run_isotopologue.sh` | full chain for one isotopologue into `runs/<LABEL>/` |

### Method choices

- **Excited surface**: SA-CASSCF + SC-NEVPT2, aug-cc-pVTZ, AVAS-defined
  CAS(8,6). NEVPT2 is size consistent (verified: the A'/A" splitting goes to
  0.0000 eV at 10 A) and intruder-free, which matters for a dissociation
  surface. It costs ~5% on top of the CASSCF that has to run anyway.
- **Ground surface**: CCSD(T)/aug-cc-pVTZ over the bound region only. Near
  equilibrium water is a closed shell where CCSD(T) is far better than
  CAS(8,6); the vibrational levels depend only on V_gs, so the two can be
  improved independently.
- **Transition dipole**: from the SA-CASSCF transition density. EOM-CCSD has
  no transition moments in PySCF; EE-ADC(2) does but overestimates mu by
  ~1.7x. The measurement says the truth lies between them.
- **Coordinates**: Cs enforced everywhere (forced subgroup, not detected),
  Jacobi for the dynamics because the valence kinetic operator does not
  separate.

---

## Gotchas

Most of these produced plausible-looking wrong answers rather than an error
message; the last two at least failed loudly, if unhelpfully.

**Thread pinning kills process parallelism.** `OMP_PLACES=cores` with
`OMP_PROC_BIND=close` is right for one process over many cores and
catastrophic for many single-threaded processes: every worker binds to place
0 and 16 processes pile onto one physical core. Symptom: two logical CPUs at
100%, everything else idle, each worker at 1/16 of a core — while the load
average still reads 16, because 16 processes are runnable. The raster scripts
clear both variables before numpy loads and assign affinity per worker.

**numpy's FFT is single-threaded.** So is `np.exp`. The first calibration
script measured single-core bandwidth and reported 20 GB/s against a 256 GB/s
bus. Use `scipy.fft` with `workers=`, and `numexpr` for elementwise work.

**Recomputed exponentials dominated the propagator.** `exp(-i dt V)` over a
1.18M-point grid, four times per step, single-threaded: 119 ms/step. They
depend only on `dt`. Caching them plus in-place buffers: 19 ms/step, 6.2x.
Adding FFT workers before this changed nothing, which is what exposed it.

**H2O needs an absorbing boundary in r, not just R.** The two OH bonds are
equivalent, so roughly half the flux dissociates via the *other* bond, which
in this Jacobi set appears as r -> large. With no absorber there it wrapped
around the periodic FFT grid and returned to the FC region, producing a
recurrence at 22 fs and a comb of fake structure spaced 0.19 eV. Molecules
with an inequivalent spectator bond (N2O, OCS) do not have this channel.
Note the mass ratio makes this channel nearly vertical in (R, r): the OH
centre of mass sits 5.9% of the bond length from O, so R barely moves as the
spectator bond breaks.

**Constrain the spin in CASSCF.** Without `fci.addons.fix_spin_` the solver
happily returns the lowest *triplet* as an excited root, and every
singlet-triplet transition dipole is rigorously zero. Symptom: mu = 0.0000
at every geometry.

**Don't seed the active space from the previous geometry.** In a diffuse
basis the aug functions lie near sigma*, CASSCF rotates them in, and the
damage compounds along the scan. Symptoms: negative excitation energies,
66 eV roots, mu collapsing to 1e-4. AVAS rebuilt at every geometry is
deterministic and has nothing to propagate.

**AVAS silently ignores orbitals its reference basis lacks.** The default
`minao` has no O 3s, so requesting a Rydberg-augmented active space returns
the same valence CAS(8,6) with no warning. `minao='ano'` gives (8,7) and
(8,10) as intended.

**Fill holes symmetrically.** Interpolating missing raster points along r1
only destroyed V(r1,r2,th) = V(r2,r1,th) and produced a 1.6 eV asymmetry.
Interpolate along both axes, average, then symmetrize explicitly.

**Asymptote beats wall.** In the Jacobi transform, gamma -> 0 sends theta ->
0 at *any* R, including large R where the geometry is just H approaching OH
from the hydrogen end. Walling those put a 1.5 Ha cliff in the middle of the
dissociation channel. The r1-beyond-the-box test must take precedence.

**Force the point group, don't detect it.** At r1 = r2 exactly, PySCF finds
C2v and the A'/A" labels vanish mid-raster. `symmetry='Cs'` keeps one
consistent labelling and lets you compute the symmetric geometry exactly,
which is better than the traditional trick of offsetting it slightly.

**pip wheels ship two OpenBLAS copies.** numpy's and pyscf's, each with its
own thread pool, and pyscf's dispatched to a *Prescott* (2004, SSE2) kernel
on Zen 5. Conda-forge links one shared `libblas`. Always check:
`OPENBLAS_VERBOSE=2 python ... 2>&1 | grep Core`.

**`tee` a Python file over a shell script name** and bash will execute
Python line by line, where `import` invokes ImageMagick and complains about
X servers. Check `head -1 *.py *.sh` when the errors make no sense.

---

## Environment

Ubuntu 24.04 headless, Miniforge:

```bash
conda create -n qc -c conda-forge python=3.12 pyscf numpy scipy matplotlib \
    h5py numexpr pyfftw
conda install -c conda-forge "libopenblas=*=*openmp*"   # not the pthreads build
```

The pthreads OpenBLAS warns `Detect OpenMP Loop and this application may
hang` when PySCF calls BLAS from inside its own OpenMP regions, and
serializes those calls rather than hanging.

---

## Relation to prior work

This reproduces established results. The canonical treatments:

- **Engel and Schinke (1988)** computed exactly this: isotope effects in the
  fragmentation of HOD in the first absorption band [3].
- **Engel, Staemmler, Vander Wal, Crim, Sension, Hudson, Andresen, Hennig,
  Weide and Schinke (1992)** is the prototype paper for the whole band [1].
- **Staemmler and Palma (1985)** provided the A~-state surface those
  calculations used, from CEPA calculations for open-shell systems [5].
- **van Harrevelt and van Hemert (2001)** revisited the band with two newer
  MRCI surfaces (Dobbyn-Knowles and Leiden) and benchmarked all three [6].
- **Vander Wal, Scott and Crim** measured the bond selectivity: HOD prepared
  in the OD-stretch second overtone gives OH/OD = 2.6 ± 0.5 [4].

### Where this surface sits

Larger basis than the 1985 work (aug-cc-pVTZ over 3289 points), but not a
better correlation treatment: CAS(8,6) is small, the vertical excitation is
0.11 eV short of the basis-set limit, and the transition dipole is 29% low.
The 2001 MRCI surfaces are better than this one.

Worth noting that van Harrevelt and van Hemert applied small empirical
corrections to their MRCI surface to improve agreement with the measured
H2O cross section [6]. Against that, an uncorrected surface reproducing the
peak to 0.5 nm and the width to 1.5% is a reasonable showing.

Comparable numbers: the computed v=0 branching of 2.94 (H+OD / D+OH) is
consistent with the established picture of O-H cleavage dominating. The
bond-selectivity result is not directly comparable to the Crim measurement,
which used the OD-stretch *second overtone* while this uses the
*fundamental*; the overtone is expected to be considerably more selective,
consistent with the v=2 -> v=4 trend here.

### What is actually new here

Nothing about the chemistry. The contribution, such as it is:

1. A complete open-source pipeline (PySCF throughout) from integrals to
   measured cross section, in readable scripts, reproducible on commodity
   hardware in hours. The literature above used MOLPRO and in-house Fortran.
2. A characterisation of AMD Strix Halo for quantum chemistry, including the
   512-bit FP64 datapath finding, which corrects a widely repeated figure.
3. The pitfall catalogue above -- twelve distinct failure modes, most of which
   produced plausible wrong answers rather than error messages.

---

## References

[1] V. Engel, V. Staemmler, R. L. Vander Wal, F. F. Crim, R. J. Sension,
B. Hudson, P. Andresen, S. Hennig, K. Weide, R. Schinke, "Photodissociation
of water in the first absorption band: a prototype for dissociation on a
repulsive potential energy surface", *J. Phys. Chem.* **96** (1992).

[2] R. Schinke, *Photodissociation Dynamics*, Cambridge University Press
(1993).

[3] V. Engel, R. Schinke, "Isotope effects in the fragmentation of water:
the photodissociation of HOD in the first absorption band", *J. Chem. Phys.*
(1988).

[4] R. L. Vander Wal, J. L. Scott, F. F. Crim, "Photodissociation of
HOD(nu_OD=3): demonstration of preferential O-D bond breaking",
*J. Chem. Phys.* **102**, 3612 (1995).

[5] V. Staemmler, A. Palma, "CEPA calculations of potential energy surfaces
for open-shell systems. IV. Photodissociation of H2O in the A~ 1B1 state",
*Chem. Phys.* **93**, 63 (1985), doi:10.1016/0301-0104(85)85049-7.

[6] R. van Harrevelt, M. C. van Hemert, "Photodissociation of water in the
A~ band revisited with new potential energy surfaces", *J. Chem. Phys.*
**114**, 9453 (2001), doi:10.1063/1.1370946.

[7] C.-Y. Chung, E. P. Chew, B.-M. Cheng, M. Bahou, Y.-P. Lee, "Temperature
dependence of absorption cross-section of H2O, HDO, and D2O in the spectral
region 140-193 nm", *Nucl. Instr. Meth. Phys. Res. A* **467-468**, 1572
(2001), doi:10.1016/S0168-9002(01)00762-8.

[8] H. Keller-Rudek, G. K. Moortgat, R. Sander, R. Sörensen, "The MPI-Mainz
UV/VIS spectral atlas of gaseous molecules of atmospheric interest",
*Earth Syst. Sci. Data* **5**, 365 (2013), doi:10.5194/essd-5-365-2013.

[9] Q. Sun et al., "Recent developments in the PySCF program package",
*J. Chem. Phys.* **153**, 024109 (2020).

Volume and page numbers for [1] and [3] were reconstructed from memory and
should be checked before being cited anywhere that matters.

---

## Open items

- **Transition dipole is 1.29x too small.** A CASSCF limitation in the
  transition density; the *shape* of mu(R) is right, verified by the Condon
  test (freezing mu moves the peak 4 nm and narrows the band 15%). Fixing it
  needs a better method for mu, not a bigger basis.
- **Ground surface ~0.7% too soft.** Uniform across isotopologues. Core-
  valence correlation and a larger basis would stiffen it, raising the
  frequencies and narrowing the band toward the measured 0.937 eV.
- **Vertical excitation not converged at aug-cc-pVTZ.** aug-cc-pVQZ is
  +0.064 eV; a full raster there is ~6 h on 16 cores.
- **Energy-resolved branching not implemented.** The channel ratios above
  are band-integrated. Bond selectivity is known to be strongly
  wavelength-dependent, and experimental values are quoted at single
  wavelengths, so the two are not directly comparable. Energy-resolved
  branching needs the reactive-flux formalism: store psi on a dividing
  surface each timestep, Fourier transform in t, then evaluate the flux
  operator. Roughly 80 lines and a ~74 MB buffer.

---

## Data

Measured cross sections from the MPI-Mainz UV/VIS Spectral Atlas:
Keller-Rudek, Moortgat, Sander, Sörensen, *Earth Syst. Sci. Data* **5**,
365-373 (2013), doi:10.5194/essd-5-365-2013,
https://www.uv-vis-spectral-atlas-mainz.org

H2O/HDO/D2O datasets: Chung, Chew, Cheng, Bahou, Lee, *Nucl. Instr. Meth.
Phys. Res. A* **467-468**, 1572-1576 (2001),
doi:10.1016/S0168-9002(01)00762-8

Observed vibrational fundamentals quoted above are standard gas-phase values
taken from memory during the work and should be checked against a
spectroscopic source before being quoted anywhere else.
