# Peri/core asymmetry — definition history

The `peri_core_asymmetry` metric has been redefined twice as we refined the
diagnostic zones of the nucleoid intensity profile. This file preserves the
prior definitions so they can be reinstated cleanly if needed.

## Current definition (2026-07-16 — "peak-bracketed")

Rings, as fractions of the cell's equivalent radius `R = sqrt(area / π)`:

- **Core**: `r < 0.15 R`
- **Peri**: `0.35 R < r < 0.55 R`
- Intermediate band and rim (r ≥ 0.55 R) are ignored.

Signed asymmetry: `(mean_peri - mean_core) / (mean_peri + mean_core)`.
Bounded in [−1, 1]. Positive = donut / edge-clustered; negative = compacted /
center-clustered.

Config keys: `PERI_CORE_RINGS = (0.15, 0.35, 0.55)` in the notebook and in
`configs/*.yaml`. Function signature:
`_peri_core_asymmetry(pixels, coords, R, core_max, peri_min, peri_max)`.

### Rationale

Individual-cell radial profiles (see sweep below) show that donut-shaped
nucleoids in control cells have a real intensity peak at `r/R ≈ 0.5` and a
real dip at `r/R < 0.2`. The peri ring is placed to bracket that peak
directly.

## Historical definition 1 (2026-07-15 — "narrow-disk / wide-annulus")

- Core: `r < R/6` ≈ 0.167 R
- Peri: `R/2 < r < 5R/6` ≈ (0.5, 0.833) R

Ships with `_peri_core_asymmetry(..., core_max=0.167, peri_min=0.5, peri_max=0.833)`.

Why replaced: peri ring extended past the donut peak into the descending shoulder
and (for elongated cells) into background-contaminated rim, so it acted as a
"how much does the profile fall off toward the edge" measure rather than a
donut-vs-compact measure. On the shipped 3-run dataset only 0.9% of control
cell-frames registered as positive despite ~13% of tracks showing donut
morphology in individual profile inspection.

Restore recipe: set `PERI_CORE_RINGS = (0.167, 0.5, 0.833)` in notebook and
YAML configs. No code change required.

## Historical definition 0 (initial — "equal-area rings")

- Core: `r ≤ R/√2` ≈ 0.707 R (equal-area split)
- Peri: `r > R/√2`
- Formula unchanged.

Why replaced: inner disk was so wide it engulfed both donut peaks and central
dip together — donut, expanded, and compacted profiles all averaged to
similar mildly-negative values.

Restore recipe: `PERI_CORE_RINGS = (0.707, 0.707, 1.0)` roughly reproduces
this (up to a strict-vs-inclusive boundary at the mask edge). For exact
reproduction, revert to the pre-2026-07-15 implementation of
`_peri_core_asymmetry` in `src/cell_analysis/matching.py` at commit
`8ce96ec`'s parent (`b648ea9`).

## Parameter sweep — data behind the choice

Sweep run on the shipped 3-run dataset (`control_experiment`,
`protein_synthesis_arrested`, `run_01`) using each cell's centroid + recorded
`radius` to reconstruct radial fluorescence bins (matches the shipped CSV
values to 3 decimal places; see `/tmp/sweep_rings.py` in commit history for
the driver). Peri capped at 0.7 R to avoid rim/background contamination.

```
core  peri_in peri_out |  ctrl_med  cam_early_med  cam_late_med  contrast  cam_shift  ctrl_pos%
─────────────────────────────────────────────────────────────────────────────────────────────
0.15  0.35    0.55     |  -0.058    -0.182         -0.085        +0.124    +0.097     11.4%   ★ new
0.20  0.35    0.55     |  -0.056    -0.177         -0.083        +0.121    +0.094     10.8%
0.15  0.35    0.60     |  -0.069    -0.214         -0.103        +0.145    +0.111      8.6%
0.20  0.40    0.65     |  -0.091    -0.265         -0.135        +0.174    +0.130      6.2%
0.15  0.45    0.70     |  -0.121    -0.335         -0.180        +0.214    +0.155      4.0%
0.167 0.50    0.833    |  -0.218    -0.471         -0.299        +0.254    +0.173      0.9%   ← historical def 1
```

- `contrast` = control_late_median − cam_early_median (higher = better
  separation).
- `cam_shift` = cam_late_median − cam_early_median (higher = clearer
  "CAM spreads toward zero over time" signal).
- `ctrl_pos%` = fraction of control late-frame cells with positive asymmetry
  (higher = more donuts surfaced).

The trade with the new default: absolute-magnitude of the compaction signal
shrinks from ~−0.47 to ~−0.18 for CAM, because the old peri ring's magnitude
came partly from the natural profile falloff toward the cell rim (which is
present in every cell regardless of nucleoid organization). The new metric
strips that "edge-falloff bonus" and reports a cleaner biology signal.
All qualitative claims survive:

- CAM ≪ control (3× more negative in the new metric).
- CAM shifts toward zero over time (~2× shift, 0.18 → 0.09).
- Control ≈ 0 (finally — because control on average is not strongly compacted).
- 12× more control donut cells surfaced (11.4% vs 0.9%).

## Individual-cell profiles that motivated the peri location

Top-10 highest-asymmetry control cells (normalized so mean = 1):

```
r/R:   .03  .07  .12  .17 | .23  .28  .33  .38  .42  .47  .53  .57  .62  .68 | .72  .78  .82  .88  .93  .97
Donut  0.79 0.80 0.79 0.86| 0.88 0.95 1.03 1.11 1.20 1.27 1.32 1.32 1.30 1.23| 1.13 0.99 0.89 0.78 0.70 0.66
```

Center dip (r/R < 0.2, values ≈ 0.79–0.86) and donut peak (r/R ≈ 0.5,
values ≈ 1.27–1.32) are both clearly visible. Contrast with the population
median cell (roughly monotone decreasing) and with the most compact CAM cells
(sharp central spike, values 2.2× the cell mean at r/R = 0).
