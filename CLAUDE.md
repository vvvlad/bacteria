# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Microscopy image analysis pipeline for bacterial cell detection, tracking, and fluorescence quantification in time-lapse phase-contrast stacks. Interactive Jupyter notebook workflow, not a production service.

## Commands

```bash
uv sync                  # Install/sync all dependencies
uv sync --group dev      # Include pytest + ruff
uv run pytest            # Run tests
uv run ruff check src/   # Lint
uv run jupyter lab       # Start JupyterLab

# Diagnostic overlay (detection debugging)
uv run python scripts/diagnostic_overlay.py
uv run python scripts/diagnostic_overlay.py --frame 5 --min_contrast 1400
```

## Architecture

Data flows through four modules in `src/cell_analysis/`:

```
Phase TIFF ──► io.load_stack() ──► (T, Y, X) ndarray
    │
    ▼
segmentation.detect_cells_stack()
    │  Cellpose model (loaded once, reused per frame)
    │  Image inverted (dark cells → bright) before eval
    │  Raw masks → regionprops filter (area, circularity, contrast, edges)
    ▼
centroids list[(N,2)] + label_stack (T, Y, X) int32
    │
    ▼
tracking.labels_to_detections() ──► DataFrame
tracking.detect_bad_frames()    ──► dropped_frames.csv
    │  MAD-based Z-scores on cell count, mean area, area IQR
    │  Flags frames with |z| > threshold in any signal
    ▼
(filtered detections + zeroed label_stack for bad frames)
    │
    ▼
tracking.track_cells()           ──► DataFrame + track_id (via trackpy)
tracking.merge_fragmented_tracks() ──► reconnect broken tracks by proximity
tracking.compute_track_stats()   ──► per-track summary
    │
    ├──► matching.measure_fluorescence(fluor_stack, label_stack)
    │      Measures mean/total/max intensity per cell per frame
    │      using phase-contrast masks on the fluorescence channel
    │
    ├──► segmentation.detect_nuclei_stack(fluor_stack)
    │      Independent Cellpose segmentation of fluorescence channel
    │      No inversion (nuclei already bright on dark background)
    │
    ├──► pipeline.run_nucleus_persistence(label_stack, nucleus_label_stack)
    │      Frame-by-frame count comparison: phase cells vs fluor nuclei
    │      Dual test: endpoint loss agreement + offset CV stability
    │
    ▼
io.save_results() ──► CSV (with fluorescence columns)
```

`matching.py` also provides `match_cells_to_nuclei()` for aligning phase-contrast cell masks with fluorescence nucleus masks via overlap then nearest-centroid fallback (for future nucleus shape analysis).

## Key design details

- **Cellpose model reuse**: `detect_cells_frame` accepts `_model` param to avoid reloading per frame. `detect_cells_stack` loads once internally.
- **resample=False optimization**: Cellpose returns masks at internal resolution; code resizes back with nearest-neighbor. 58x faster than resampling inside Cellpose.
- **GPU routing**: `gpu=True` uses MPS on Apple Silicon, CUDA on NVIDIA. CPU works fine as fallback.
- **Warning suppression**: Cellpose emits logging warnings (not `warnings.warn`), suppressed via `logging.getLogger("cellpose.dynamics").setLevel(logging.ERROR)`.
- **Nucleus segmentation**: `detect_nuclei_stack` segments the fluorescence channel directly — no image inversion needed (unlike phase-contrast). Uses the same Cellpose model-reuse and resize-back patterns as `detect_cells_stack`.
- **Units**: `add_geometry(tracked, track_stats, pixel_size_um=...)` rescales `tracked["area"]` to µm² before deriving radius/volume/surface_area. With the notebook's default `PIXEL_SIZE_UM = 0.0645`, columns `area`, `radius`, `volume`, `surface_area` are in µm², µm, µm³, µm² respectively. Relative quantities (`volume_rel`, `fluor_rel`, `V_rel`, CV, nNRM) are dimensionless and unaffected by scaling. Pass `pixel_size_um=1.0` to recover raw pixel units. Frame gating (`detect_bad_frames`) runs before `add_geometry`, so it always sees raw pixel area regardless of the configured scale.
- **Relative-per-track columns**: `_relative_per_track(df, col)` in `pipeline.py` returns `df[col] / df[col]@first-frame-per-track`, with NaN where the per-track baseline is ≤ 0. Used by `add_geometry` (→ `volume_rel`) and `add_fluorescence` (→ `fluor_rel`) so §6.5b and §8.2b can pass these columns to `plot_metric_dynamics`.
- **Nucleoid distribution metrics**: `add_nucleoid_distribution` adds `mean_edge_distance_norm` and `gaussian_sigma_norm` to `tracked`. Both are dimensionless ratios — normalized against `R/3` and `R` respectively, where `R = sqrt(area_pixels / π)` is the cell's pixel-space effective radius. They do NOT shift with `PIXEL_SIZE_UM` (the normalization cancels the unit). Threshold rule: pixels > mean(cell) (fallback to > 0.5×mean if max/mean < 1.5). Fed into §8.14 fate prediction as the 4th and 5th LR features.
- **Fluorescence alignment around disappearance**: `add_fluorescence_alignment(window=...)` returns a long-form table of F(offset)/F(-window) for each `disappeared` cell, with offsets in `[-window, +window]` relative to last detection. Post-disappearance frames (offset > 0) are measured via the cell's last-known phase mask applied to the fluorescence stack — see `measure_post_disappearance_fluorescence` in `matching.py`. Replaces the older fluor-drop classifier.
- **Per-class fate metrics**: `predict_fate_from_frame0()` returns `summary["confusion_matrix"]` (dict with `TN, FP, FN, TP`, positive class = disappeared) and `summary["per_class"]` (nested dict with `precision, recall, f1, support` per class). `add_fate_prediction()` prints these alongside the existing AUC/accuracy block. The notebook §8.14 fits TWO LRs — `FATE_FEATURES_FULL` (5 features) and `FATE_FEATURES_NO_AREA` (4 features, drops `area`) — and `print_fate_comparison(summary_full, summary_no_area)` prints a side-by-side table plus a one-line heuristic ("consider permanently swapping" iff both per-class F1 deltas ≥ 0.02).

## Key design details — tracking

- **Track merging**: `merge_fragmented_tracks()` runs after trackpy linking. Uses greedy spatial matching + Union-Find to reconnect tracks broken by detection gaps longer than `memory`. Parameters: `max_distance` (px), `max_gap` (frames). Returns a merge log for auditability.
- **Dynamic frame gating**: `detect_bad_frames()` replaces the old `TRIM_FRAMES` hardcode. Computes per-frame detection statistics (count, mean area, IQR), flags frames whose deltas are MAD-based Z-score outliers (threshold 3.5). Cell count uses absolute Z-scores for all frames; area/IQR use delta Z cross-checked with absolute Z. Flagged frames are excluded from tracking and have label masks zeroed before fluorescence measurement. Audit CSV saved as `dropped_frames.csv`.

## Notebook conventions

- The notebook at `notebooks/analysis.ipynb` is the main entry point. It uses `sys.path.insert(0, "../src")` to import the package.
- `DETECT_PARAMS` dict at the top of the notebook holds all tunable detection parameters.
- `FLUOR_PATH` points to the background-subtracted fluorescence TIFF.
- `GATING_Z_THRESHOLD` controls frame quality gating sensitivity (default 3.5). Frame numbers in all output are raw frame numbers — no offset conversion needed.
- `MERGE_MAX_DISTANCE`, `MERGE_MAX_GAP` control track merging.
- `PIXEL_SIZE_UM` (default 0.0645) sets the µm-per-pixel scale applied in `add_geometry`. `BASELINE_FRAMES` (default 3) restricts the §6.3 area histogram to the first N frames before medium changes drive swelling. Both keys are also accepted by `scripts/run_experiment.py` config YAMLs.
- Output file links use `IPython.display.Markdown` with relative paths so they render as clickable links on GitHub.

## Commits

Do not add a `Co-Authored-By` trailer to commit messages.

## Detection parameter tuning

Detailed analysis in `docs/detection_tuning.md`. The diagnostic overlay script (`scripts/diagnostic_overlay.py`) visualizes accepted vs rejected cells with rejection reasons — use it to iterate on filter thresholds before updating `DETECT_PARAMS` in the notebook.
