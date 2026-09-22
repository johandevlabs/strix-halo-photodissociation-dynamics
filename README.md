# Photodissociation dynamics on commodity hardware

First-principles molecular photodissociation — electronic structure through
potential energy surfaces, vibrational eigenstates, wavepacket dynamics, to
absorption cross sections compared against measurement — run entirely on a
single desktop machine with open-source tools.

The machine is a GMKtec EVO-X2 (AMD Ryzen AI Max+ 395 "Strix Halo", 16 Zen 5
cores, 96 GB unified memory, ~120 W). The stack is PySCF, NumPy/SciPy, and
about 2000 lines of Python. No cluster, no commercial quantum chemistry code.

The project started as a hardware benchmark: how does this compare to the
HPC allocations a computational chemistry PhD used around 2011-2013? It grew
into a working pipeline, and then into a search for somewhere that pipeline
could say something new.

## Folders

| folder | subject | status |
| --- | --- | --- |
| [`water/`](water/) | H2O / HDO / D2O A-band photodissociation | **complete and validated** |
| [`HOX/`](HOX/) | HOCl / HOBr / HOI triplet-band photodissociation | **in progress** — SOC toolchain validated, the HOCl a 3A" surface built; propagation and HOBr still to come |

Each folder has its own README describing what is there and how far it got.
Completeness varies by design; this repo grows as work happens.

## What `water/` established

A complete pipeline validated against measurement at several independent
points:

- CCSD(T) reference energies reproduced to 10 digits
- OH diatomic De 4.628 eV (NEVPT2) and 4.631 (UCCSD(T)) vs 4.62 observed
- Ten vibrational fundamentals across three isotopologues within 0.4-0.9%
- A-band peak within 0.5 nm and FWHM within 1.5% of Chung et al. (2001)
- **The D2O/H2O band-narrowing ratio predicted to 0.5%** — the isotope
  effect, with nothing fitted
- Bond-selective photodissociation in HDO reproduced, with the bend as a
  clean control
- Every result cross-checked in two independent Jacobi coordinate systems

The one quantity the calculation gets wrong beyond its own uncertainty is the
transition dipole magnitude, low by a stable factor of 1.291 ± 0.023 across
four independent runs.

None of this is new physics. Engel and Schinke did the HOD calculation in
1988. See `water/README.md` for a full "Relation to prior work" section.

## What is actually offered here

1. **An open pipeline.** The canonical literature used MOLPRO and in-house
   Fortran. This is PySCF and readable scripts, reproducible on hardware
   anyone can buy.
2. **Hardware characterisation.** Strix Halo for quantum chemistry is
   undocumented. Includes a correction to a widely repeated FP64 figure: the
   part has a full 512-bit datapath, not the double-pumped 256-bit one that
   a commonly quoted 1.31 TFLOP/s peak assumes.
3. **A catalogue of ways this goes quietly wrong.** Twelve distinct failure
   modes, most of which produced plausible wrong answers rather than error
   messages.
   Several are generic to scientific Python on many cores and have nothing
   to do with chemistry — OpenMP pinning destroying process parallelism,
   single-threaded NumPy FFT, exponentials recomputed inside a propagator
   loop. See the Gotchas section of `water/README.md`.

## Reproducing

Ubuntu 24.04 headless, Miniforge:

```bash
conda create -n qc -c conda-forge python=3.12 pyscf numpy scipy matplotlib \
    h5py numexpr pyfftw
conda install -c conda-forge "libopenblas=*=*openmp*"   # not the pthreads build
```

Then see `water/README.md`. The two expensive inputs (`pes_grid.csv`,
`gs_well.csv`) are committed, so everything downstream rebuilds in minutes.

## Honesty notes

- Some reference volume/page numbers were reconstructed from memory and are
  flagged in place as needing verification against DOIs.
- The `HOX/` literature review concludes a specific research gap is unfilled.
  That conclusion partly rests on absence of evidence from paywalled
  abstracts and is explicitly provisional. See `HOX/TASKS.md`.
- Observed vibrational fundamentals quoted in `water/README.md` are standard
  gas-phase values taken from memory during the work. Check them against a
  spectroscopic source before citing.

## Authorship

Built by Johan in extended collaboration with Claude (Anthropic). The code,
much of the debugging, and the structure of the arguments came out of that
conversation; so did a fair number of the errors, which were caught by
running the calculations and checking against measurement. The Gotchas
section is a record of real debugging, not a curated list.

Note that most journals and the ICMJE bar AI systems as authors, on the
grounds that authorship requires accountability. If any of this goes beyond
GitHub and arXiv, the conventional form is a methods or acknowledgements
note describing the tool's role.

## Data

Measured cross sections from the MPI-Mainz UV/VIS Spectral Atlas
(Keller-Rudek et al., *Earth Syst. Sci. Data* **5**, 365, 2013,
doi:10.5194/essd-5-365-2013), primarily Chung et al., *Nucl. Instr. Meth.
Phys. Res. A* **467-468**, 1572 (2001), doi:10.1016/S0168-9002(01)00762-8,
who measured H2O, HDO and D2O in one apparatus.
