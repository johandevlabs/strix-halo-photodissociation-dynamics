# 02_band_model — HOBr

| script | descends from | question |
| --- | --- | --- |
| `03_band_1d.py` | HOCl's `10_band_1d.py` | how much surface does the band need, what does the basis do to the band, and a first σ(298)/σ(220) |

A 1D wavepacket along O-Br on `01_method/02`'s cut, OH and the angle frozen.
Local — numpy and scipy, ~45 s — so it runs on the laptop.

## How much surface the band needs: to ~2.0 Å

| test | HOBr | HOCl |
| --- | --- | --- |
| absorber at r_eq − 0.09 (positive control) | 14% | 13% |
| V_T flat beyond r_eq + 0.09 (positive control) | 66% | 99% |
| band first responds (≥ 1%) to an absorber from | r_eq + 0.01 | r_eq + 0.01 |
| band first responds to a surface cut at | r_eq + 0.16 = 2.00 Å | r_eq + 0.16 |
| worst change, surface intact, absorber from r_eq + 0.51 | 0.05% | ~0.1% |

The absorber and cut positions are set *relative to r_eq* so the two molecules
can be compared, and they come out identical. The raster to 2.40 Å has 0.4 Å
of margin; the ³A′ crossing at 2.55 Å is product-branching scope.

## The basis, measured on the band

`01_method/02` found aug-cc-pvtz-dk lowers ω by 38 meV at the equilibrium
geometry, and I estimated from that a ~6 nm band shift. The model says
**+0.9 nm**. The fixed-geometry comparison missed that aug also moves the
ground-state minimum 0.006 Å inward, so χ₀ sits where the steep V_T is ~30 meV
higher; ⟨V_T⟩_χ₀ − E_v0, which is what the band samples, differs by 8 meV.
Width 3.9% narrower; σ(298)/σ(220) over 440-500 nm moves by 0.002. A property
of the band should be measured on the band, not inferred from a proxy.

## The f the band needs

Propagated at a nominal f and scaled to Ingham's peak σ = 2.3 × 10⁻²⁰ cm²:
**f ≥ 6.4 × 10⁻⁵**. A floor, since a 1D band is too narrow and its peak too
high for a given f. That is the target for HOBr's SOC vertical run.

## First look at σ(298 K)/σ(220 K)

O-Br stretch hot bands only; model band shifted rigidly (−126 meV) onto the
measured 457 nm peak before reading off wavelengths:

| nm | 420 | 440 | 457 | 480 | 500 | 520 | 550 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| σ(298)/σ(220) | 1.003 | 0.982 | 0.972 | 0.972 | 0.990 | 1.036 | 1.196 |
| σ / peak | 0.64 | 0.91 | 1.00 | 0.86 | 0.59 | 0.34 | 0.11 |

**Over 440-500 nm, −1 to −3%**: below the "negligible" line of the impact
threshold in `../../TASKS.md`. **The effect is in the red wing**: +20% at
550 nm. v=1 has a node, so its reflection through the repulsive wall is
broader, and hot bands move intensity from the centre to the wings.

Why the alignment: the model band sits 20 nm blue of the measured one, so at
*fixed* wavelength 500 nm lands further down its red side than it does in
reality. The first run printed only that read and reported "−3% to +6% over
440-500 nm"; the +6% was the band's position, not temperature.

Caveats, all pointing the same way on the wing: a 1D band is too narrow (the
frozen bend and OH would widen it), which exaggerates wing ratios, so +20% at
550 nm is likely an upper bound. No SOC splitting, which is 4× Cl's and will
move the band. Condon.

If the 3D model confirms it, the impact threshold's 440-500 nm window may be
the wrong place to look, and whether the wing matters becomes a J-value
question: at high solar zenith angle the actinic flux shifts red.
