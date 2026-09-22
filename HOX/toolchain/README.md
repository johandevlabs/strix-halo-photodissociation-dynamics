# toolchain — spin-orbit coupling in PySCF

Not HOCl-specific: everything here is needed again, unchanged, for HOBr and
HOI. The question it answers is whether this stack can produce SOC matrix
elements and **SOC-borrowed transition dipoles** at all, and how accurate they
are for a given halogen and basis.

| script | what it does |
| --- | --- |
| `00_setup_soc.sh` | installs Prism + socutils into the `qc` env on the EVO |
| `01_soc_probe.py` | diagnostic, not a calculation: dumps the API surface actually present and checks the imports a real run needs |
| `02_soc_atoms.py` | halogen fine structure (2P_1/2 − 2P_3/2) against observed |

## The two plugins

- **Prism** (sokolov-group) — FIC/QD-NEVPT2 with state-interaction SOC,
  Breit-Pauli and X2C-DKH, plus a cheaper SOMF variant. Its `kernel()` returns
  **oscillator strengths from the spin-orbit-mixed states**, which is exactly
  the borrowed intensity a spin-forbidden band needs, rather than a separate
  piece of machinery to build.
- **socutils** (xubwa) — the SOC integrals (X2CAMF, optional Gaunt and Breit).
  Prism requires it.

Note Prism's NEVPT2 is *not* PySCF's: `water/` used the native strongly
contracted `mrpt.NEVPT`, Prism is a separate fully internally contracted
implementation. This is a new dependency, not a flag.

## Building it (why `00_setup_soc.sh` is 290 lines)

socutils assumes a **pip-layout PySCF**; the EVO has conda-forge. That single
mismatch produced four separate build failures, each fixed in the script:

| failure | fix |
| --- | --- |
| BLAS not found | `-DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.so.0` |
| libcint not found | `-DPYSCF_CINT_LIB=...` |
| `cint.h` not found | `-DCMAKE_C_FLAGS=-I$CONDA_PREFIX/include` |
| no `setup.py` | install by writing a `.pth` file |

Two further traps: Prism ships no packaging metadata, so pip never resolves its
dependencies (`psutil` was missing and only surfaced at runtime) — the smoke
test therefore imports **leaf** modules, not just the top-level package. And
`prism/libsoc/general_somf.py` hardcodes a sibling `socutils` path, met with a
symlink rather than a second git submodule, which would mean a second C build
free to drift from the first.

## Validated against experiment (2026-09-13)

Cl, def2-TZVP, SA-CASSCF(5e,3o)/3 roots, DKH1-QD-NEVPT2:

| check | result |
| --- | --- |
| degeneracy pattern | [4, 2] — correct for a 2P term |
| ordering | **inverted** (4-fold at 0, 2-fold above) — correct for p^5 |
| Cl 2P_1/2 − 2P_3/2 | 844.54 vs 882.35 cm-1 observed, **−4.3%** |
| f within the term | ~1e-24, i.e. zero as required |

The inversion is the check that matters most: a normal ordering would have been
a sign error wearing a plausible magnitude.

## The Br finding, which HOBr depends on

def2-TZVP gives **−16.6%** for Br. Simply decontracting it
(`--basis unc-def2-tzvp`) gives **−7.6%** for 14.9 s instead of 9.5 s. So the
SOC treatment was never the problem: def2-TZVP is all-electron for Br but
*non-relativistically contracted*, and the SOC operator samples precisely the
near-nuclear region that contraction gets wrong.

Open, and worth settling at the atom before any HOBr raster:

- [ ] a basis built for this rather than repaired by decontraction —
      x2c-TZVPall, ANO-RCC, dyall (`pip install basis-set-exchange`)
- [ ] `--soc breit-pauli` against DKH1 on the same basis, which bounds how much
      of the residual belongs to the SOC operator rather than the basis

A caveat that does not go away: this is a free atom. It exercises neither
bonding nor the SOC *borrowing between multiplicities* that the band needs.
`HOCl/01_method/03_soc_hocl_vertical.py` is the test that does.
