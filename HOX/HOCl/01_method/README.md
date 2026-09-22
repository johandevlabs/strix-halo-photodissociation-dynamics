# 01_method — what to compute the surfaces with

Eight scripts, and the shortest honest summary of them is that the obvious
method was wrong twice and the second wrong answer was more interesting than
the first.

| script | question | verdict |
| --- | --- | --- |
| `03_soc_hocl_vertical.py` | does the SOC borrowing work on a molecule? | yes: f = 8.9e-7, vertical 3.4477 eV |
| `04_statespecific_check.py` | can the two states be had without excited-state theory? | yes: ΔCCSD(T) 3.4142 eV, 0.034 eV from the multi-state route |
| `05_triplet_scan.py` | is the state-specific triplet smooth along O-Cl? | **no** — a kink at 2.2-2.8 Å |
| `06_triplet_manifold.py` | is the kink an avoided crossing? | no: three 3A" states converging on one asymptote |
| `07_fc_active_space.py` | CAS(10,6) or CAS(12,7) in the FC window? | CAS(10,6); the 7th orbital is a spectator |
| `08_nevpt2_contraction.py` | SC or FIC contraction? | needs an external referee, not each other |
| `09_eom_triplet.py` | which method has the right slope? | EOM-CCSD, within 0.5% of UCCSD(T) |
| `11_eom_coordinates.py` | is EOM-CCSD valid across O-H and the bend too? | yes, at all 34 geometries |

`07` is also a library: `08` and `03_surfaces/13` load its `Projector`,
`select_fixed`, `canonicalize_like_avas`, `symmetrize_blocks` and
`triplet_solver` from the file itself (script names start with a digit, so a
plain import will not do), which is what makes the outer shell use exactly the
orbital selection `07` validated.

## The two states are each the lowest of their spin manifold

Johan's observation, and the way out of a cost problem. V(X 1A') and V(ã 3A")
need no excited-state theory. Only the *intensity* does, and only across the
Franck-Condon window — exactly as in water, where "the dipole enters only at
t = 0".

`03` confirms the borrowing: 3 singlets + 3 triplets in the **Ms=0**
determinant space give 12 SOC states, the count the microstates require. Prism
logs `Apply S_plus due to Ms=0` and builds the missing components itself. The
Ms=0 space with `direct_spin1` and **no** `fix_spin_` is what produces a mixed
singlet/triplet reference — the inverse of water's gotcha, where the problem
was spin contamination rather than the need for it.

## The kink, and what it turned out to be

`05` found the state-specific triplet dropping 0.096 eV at 2.60 Å, past its own
minimum — which a dissociating state cannot do. Two tests that were supposed to
settle it did not:

- **`--symmetry Cs` changed nothing, and could not have.** Energies matched to
  3e-7 Ha. In water, forcing the point group worked because state-averaged
  CASSCF *labels its roots*; a state-specific SCF with symmetry on still fills
  by aufbau. What pins a single determinant to 3A" is **`irrep_nelec`**, the
  α/β count per irrep. The water gotcha needs a different mechanism here.
- **`--pin-irrep` found no state switch** either: {A': (11,10), A": (3,2)} at
  every point from 1.4 to 4.0 Å. So whatever crowds the state is itself 3A".

Johan's reading of the plots was that a higher 3A" comes down to meet ours near
2.5 Å. `06` shows two of them do — and they **converge on a shared asymptote
rather than crossing**. The root 1 − root 0 gap closes monotonically (2.85 eV
at 1.8 Å to 0.042 eV at 3.6 Å) with no interior minimum, the roots exchange no
character, and OH(2Π) × Cl(2P) predicts exactly the three A" states seen at
3.6 Å.

Why the single determinant fails there: root 0's dominant configuration falls
from 0.82 at 1.8 Å to ~0.43 past 2.2 Å, and the T1 spike (0.041 → 0.095)
tracks it. No crossing is needed to explain the kink. With no interior
crossing, Landau-Zener does not apply and the band lives on one smooth surface.

## The slope, which decides the band width

The multireference and coupled-cluster triplets disagreed by ~15% in slope at
equilibrium, and the reflection principle turns slope into band width, so this
had to be settled rather than split.

| method | slope at 1.6891 Å | gap to UCCSD(T) |
| --- | --- | --- |
| SC-NEVPT2 CAS(10,6) | −5.954 eV/Å | 14.9% |
| SC-NEVPT2 CAS(12,7) | −6.152 | 12.1% |
| SC-NEVPT2 CAS(14,9), full valence | −6.610 | 5.6% |
| FIC-NEVPT2 CAS(14,9) | −6.339 | 9.4% |
| **EOM-CCSD on the CCSD(T) ground state** | **−6.965** | **0.5%** |
| UCCSD(T), state-specific | −6.999 | — |

Two unrelated routes to the triplet agree to 0.5%, and NEVPT2 converges towards
them as the active space grows — so the gap was active-space incompleteness,
not a coupled-cluster artefact. The (T) correction alone accounts for 4% of
it. EOM-CCSD is additionally the smoothest thing available in the window
(0.1 meV against a quintic, against UCCSD(T)'s 4.2), has no ROHF reference to
wobble and no active space to choose.

`11` then cleared the other two coordinates: across the O-H stretch, the bend
and 8 corner points, the lowest triplet is 3A" at **every** geometry, T1(S) ≤
0.010 (half the threshold), and the nearest 3A" is ≥ 2.06 eV away.

## Two checks that were wrong first

Worth knowing, because both produced confident false verdicts:

- The first **kink test** used a ratio of second differences, which flags every
  nearly straight curve, since the median is numerical noise. Replaced by
  deviation in meV from a polynomial fit.
- That fit at **degree 4** then flagged genuine Morse curvature on the O-H cut
  (9.5 meV against a 10 meV threshold). Degree 5 gives 1.4 meV. Calibrated on a
  synthetic Morse curve: degree 5 leaves 0.4 meV when smooth and 10.3 meV with
  a 30 meV step injected, so the check is degree 5 against 5 meV, which catches
  ~15 meV steps while accepting real curvature.

And one result that is not a check at all: `08`'s "the surface needs FIC"
compared the two contractions only with each other. `09`'s external referee
found SC the *closer* of the two to coupled cluster in full valence.
