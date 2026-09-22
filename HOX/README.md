# HOX — hypohalous acid photodissociation

Target: the visible (triplet) absorption band of **HOBr** and its temperature
dependence, from 3D quantum wavepacket dynamics on an ab initio ã 3A" surface
with spin-orbit coupling. That band carries up to 50% of the HOBr photolysis
rate at high solar zenith angles, and no temperature dependence has ever been
measured or computed for any hypohalous acid band.

**Status.** Phase 0 (premise) and Phase 0.5 (toolchain) are done. HOCl — the
validation case, where a measured band exists at 380 nm — has its ã 3A"
surface built and spliced to the asymptote. Propagation and σ(λ, T) are still
to come, and HOBr has not been started.

## Layout

| | |
| --- | --- |
| [`TASKS.md`](TASKS.md) | the running record: plan, every result, every bug and why it mattered. Read this before the code. |
| [`toolchain/`](toolchain/) | the SOC stack (Prism + socutils) and its validation against atomic fine structure. Shared by every halogen. |
| [`HOCl/`](HOCl/) | the HOCl work, by approach: method selection, the 1D band model, the surfaces. |
| [`docs/deep-review-claude.md`](docs/deep-review-claude.md) | literature review with citations |

`HOBr/` will appear alongside `HOCl/` and reuse `toolchain/` unchanged.

## How this repo is run

Code is written and discussed on the laptop; **everything is run on the EVO-X2**
(30-32 cores, conda env `qc`, PySCF 2.14 from conda-forge). The loop is: commit
and push here, pull and run there, push the CSVs back. Scripts therefore assume
nothing about the working directory — data paths resolve relative to the script
file — and every long run appends to a resumable CSV.

Two exceptions run locally on NumPy/SciPy alone, with no PySCF:
`HOCl/02_band_model/10_band_1d.py` and `HOCl/03_surfaces/15_splice.py`.

## What was established

- **The SOC toolchain works and the borrowing is real.** The ã 3A" band of
  HOCl is dark by spin selection and comes out of SO-QD-NEVPT2 with
  f = 8.9e-7, vertical 3.448 eV against the measured 380 nm band (+5.7%).
- **The band is decided in the Franck-Condon region.** Flattening V_T beyond
  2.10 Å changes the 1D band by 0.1%; nothing beyond 2.3 Å matters at all.
- **Coupled cluster has the right slope**, and NEVPT2 converges to it as the
  active space grows (15% → 12% → 6% from CAS(10,6) to full valence).
- **The surface is built**: a 2730-point CCSD(T) + EOM-CCSD raster inside
  2.40 Å, a 1050-point SA-CASSCF + NEVPT2 outer shell to 3.60 Å, and the
  OH(X 2Π) + Cl asymptote, spliced with seam checks that can actually fail.

The numbers, the caveats attached to each, and the several checks that were
wrong before they were right, are all in [`TASKS.md`](TASKS.md).
