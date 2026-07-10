# Deep-Review Findings

Review target: full pipeline in `src/cell_analysis/`, the `notebooks/analysis.ipynb` entrypoint, and the `scripts/run_experiment.py` CLI runner. Date of review: 2026-07-10.

---

## 1. Notebook vs. `run_experiment.py`

**Short answer: no divergence.** `run_experiment.py` is a papermill launcher, not a re-implementation. All analysis logic lives in the notebook + `cell_analysis` package.

| Aspect | Interactive notebook | `run_experiment.py` |
|---|---|---|
| Config injection | Hard-coded in Configuration cell | Papermill injects the YAML |
| `EXPORT_HTML` | Defaults to `True` — Jupyter save + `export_notebook_html` from disk | Injected as `False` — script exports the executed papermill tmp notebook directly |
| `CONFIG_PATH` | Empty string | Absolute path to YAML |
| Output dir | `results/<RUN_NAME>/` | Same |
| HTML source | Live-saved notebook (may embed stale outputs if Jupyter hasn't finished saving) | Papermill-executed tmp notebook — race-free |
| `docs/index.html` | Untouched | `publish_reports()` copies each report to `docs/<RUN_NAME>/report.html`, injects a back-link, rebuilds index |
| Failure | Cells stop, notebook stays open | Saves `report_failed.ipynb`; still exports whatever ran |

**Only material risk**: interactive path relies on a `Javascript` `docmanager:save` + 1-second sleep before reading the notebook file. If Jupyter hasn't saved yet, the exported HTML has stale outputs. The CLI is race-free.

---

## 2. Pipeline architecture at a glance

```
TIFF (T,Y,X uint16)
  │  load_paired_stacks(): shape check
  ▼
segmentation.detect_cells_stack()   # Cellpose, model loaded once
  │  per frame: invert(max-x) → cellpose.eval(resample=False)
  │           → nearest-neighbor upsample back to (Y,X)
  │           → regionprops filter: area, circ, intra-mask std, edges
  ▼ (label_stack int32 T,Y,X)
tracking.labels_to_detections()  → detections df
tracking.detect_bad_frames()     → MAD-Z on count/area/IQR, threshold 3.5
  │  Bad frames: zero label_stack[bf] + exclude from detections
  ▼
tracking.track_cells()   # trackpy nearest-neighbor link
tracking.merge_fragmented_tracks()   # greedy proximity + union-find
compute_track_stats() + filter_short_tracks(min_detections=4)
  ▼
add_geometry(pixel_size_um)     # area px² → µm², radius/V/S via sphere
add_fluorescence()              # mean/total/CV/nNRM/skew/kurt per cell/frame
add_nucleoid_distribution()     # edge_norm, sigma_norm, peri_core_asymmetry
add_fluorescence_concentration()# total_I / V
add_sav_ratio()                 # S/V
add_fluorescence_alignment()    # ±window around 'disappeared'; post uses last mask
add_preburst_fluorescence()     # slope in 5-frame window before last_frame
add_frame0_fate_comparison()    # Mann-Whitney per feature (5)
add_fate_prediction() × 2       # LR + LOO-CV, full (5) and no-area (4) sets
detect_nuclei_stack()           # Cellpose again on fluor channel
run_nucleus_persistence()       # endpoint + trajectory-CV test on counts
```

**Save cadence**: `save_main_outputs` is invoked after every `add_*` step (~6×), so `tracked_cells.csv` is rewritten repeatedly. Fine for small data, but a mid-pipeline crash leaves a partially populated CSV on disk with no indicator.

---

## 3. Performance findings

| ID | Location | Issue | Fix |
|---|---|---|---|
| P1 | `detect_nuclei_stack()` | Second Cellpose pass adds 5–15 min/frame × T = 40% of exploratory-run wall-time; feeds a single scalar test in §8.16 | Wire `RUN_NUCLEUS_PERSISTENCE` config knob (default `True` to preserve behavior) |
| P2 | `matching.measure_fluorescence` | `stats.kstest(pixels, norm(µ,σ).cdf)` per cell per frame; `norm(µ,σ)` builds a frozen distribution each call | Standardize `z` once, call `stats.kstest(z, "norm")`, or compute `max(abs(F_emp - Φ(z)))` directly |
| P3 | `tracking.merge_fragmented_tracks` | `grp.loc[grp["frame"].idxmax()]` per group | `groupby.agg({"frame": ["min","max"]})` after sort — micro-opt |
| P4 | `tracking.labels_to_detections` | Per-label `mask = labels==cid` + `ndimage.center_of_mass(mask)` + `mask.sum()` — quadratic-ish | `ndimage.center_of_mass(labels, labels, cell_ids)` + `np.bincount(labels.ravel())[cell_ids]` — one call each per frame |
| P5 | `pipeline.add_nucleoid_distribution` (per-cell loop) | 2×2 covariance eigen call allocates array + calls `eigvalsh` | Closed-form 2×2: `tr = a+d; det = a·d − b²; e = (tr ± √(tr²−4det))/2` |
| P6 | Notebook | `save_main_outputs` × 6, rewriting CSVs redundantly | Save once at export time, or write to tmp + atomic rename |
| P7 | `matching.measure_fluorescence` | Manual skewness/kurtosis from centered moments | `scipy.stats.skew(pixels, bias=False)` — numerically better, standard |

---

## 4. Quality / algorithm findings

| ID | Location | Issue |
|---|---|---|
| Q1 | `segmentation.detect_cells_frame` | `min_contrast` is intra-mask std, not cell-vs-background contrast — name is misleading. Direction of filter (reject low-std) is defensible (rejects smooth false positives) but doesn't do what its name says |
| Q2 | `tracking.detect_bad_frames` | Delta-Z for area/IQR, absolute-Z for count → asymmetric sensitivity. Intentional (recovery frames must not re-flag), but underdocumented |
| Q3 | `tracking.detect_bad_frames` | No absolute minimum-change floor. Very stable datasets have tiny MAD; small real changes can score high z |
| Q4 | `tracking.merge_fragmented_tracks` | Merges purely on centroid proximity; no area-similarity gate. A dying cell + nearby new cell can be merged |
| Q5 | `matching.predict_fate_from_frame0` | LOO-CV; no CI on AUC. 0.665 without confidence bounds is uninterpretable |
| Q6 | `matching._get_frame0_with_fate` | Cohort = `first_frame == 0`. If frame 0 is bad-gated, the cohort collapses |
| Q7 | `matching.measure_fluorescence` | `total_intensity` uses upsampled (`order=0`) mask, biased by ~5–15% depending on internal downsample factor |
| Q8 | `segmentation.detect_cells_frame` | `resample=False` + nearest-neighbor upsample: same as Q7 but for `area`, EDT edge distances, and every geometry column downstream |
| Q9 | `segmentation.detect_nuclei_stack` | Uses same `model_type=None` (cpsam) as phase. Cellpose has a dedicated `"nuclei"` model that's smaller + calibrated to fluorescent nuclei |
| Q10 | `matching.compute_preburst_fluorescence` | Silently returns NaN for disappeared tracks with < 2 frames in window. No cohort-loss reporting |
| Q11 | `pipeline.run_nucleus_persistence` | Compares object COUNTS only, not identities. Cannot detect per-cell nucleus persistence |
| Q12 | `matching.measure_fluorescence` | Manual `np.mean(z**3)` for skewness — biased population estimator, noisy for small cells |
| Q13 | `pipeline._relative_per_track` | Anchors on each track's first-observed frame. For tracks that start at frame > 0, `fluor_rel` means F(t)/F(first), not F(t)/F(0). §8.2b implicitly compares these as if apples-to-apples |

---

## 5. Data quality findings

| ID | Issue |
|---|---|
| D1 | No sanity check that fluor stack is actually background-subtracted. Median of `fluor_stack[label_stack == 0]` should be ≈0; nothing warns if not |
| D2 | `tracked["area"]` unit changes mid-pipeline (px² → µm² after `add_geometry`). No column suffix marks the transition |
| D3 | `min_contrast=1250` is uint16-scale. A uint8 input would silently pass all cells. No dtype check + rescale |
| D4 | `label` (per-frame Cellpose ID) vs. `track_id` (tracked identity) — documented but easy to conflate; some CSVs surface `label` without qualification |
| D5 | Bad frames get `label_stack[bf] = 0` — fluorescence measurement still runs but returns nothing. Consistent, not wasteful, worth documenting |
| D6 | `tracked_cells.csv` is rewritten by every intermediate `save_main_outputs` call. Mid-pipeline crash leaves the file with only some `add_*` columns populated and no indicator |

---

## 6. Detection findings

| ID | Issue | Fix |
|---|---|---|
| Det1 | `model_type=None` → cpsam (Cellpose v4 SAM backbone). Docs note `"cyto3"` is 5–10× faster on CPU | Expose a `FAST_MODE` config knob or set `MODEL_TYPE="cyto3"` for exploratory iteration |
| Det2 | `diameter=32` hard-coded. Ideal value varies per dataset | Use Cellpose auto-detect (`diameter=None`) on frame 0, log the estimate, override manually only when needed |
| Det3 | `exclude_edges=True` silently drops edge-touching detections. A cell touching edge for frames 0–1 and moving inward loses those frames + may miss the frame-0 cohort | Keep detection, add a `touches_edge` boolean flag; let downstream filter |
| Det4 | Circularity measured on the raw Cellpose mask. Straight seams between touching cells lower circularity for well-split cells | Not urgent — flag |
| Det5 | Track centroids are integer-pixel from `regionprops` | Sub-pixel refinement would improve crowded-field tracking |
| Det6 | `merge_fragmented_tracks` uses proximity only | Overlap with Q4 above |

---

## 7. New information / different perspectives

| ID | Idea |
|---|---|
| New1 | **Local density** — per (frame, cell), count of neighbors within R px. Adds a "crowding" axis to fate analysis |
| New2 | **Instantaneous velocity** — frame-to-frame centroid delta. Reveals mode-switching (motile → lysing) |
| New3 | **Shape features** — `eccentricity`, `orientation`, `solidity` from Cellpose masks. Free from `regionprops` |
| New4 | **Focus quality per frame** — `laplacian(frame).var()`. Complements the count-based gate |
| New5 | **Autofluorescence baseline over time** — median of `fluor_stack[t][label_stack[t]==0]`. Should be ≈0 if BG file is honest; drift is diagnostic |
| New6 | **Track quality score** — fraction of frames actually detected vs. imputed by trackpy memory, max displacement, area CV |
| New7 | **Phase-space trajectories** — 2D (V/V₀, F/F₀) scatter colored by fate or time. Reveals population structure the separate time-series plots miss |
| New8 | **Kaplan-Meier survival curves** stratified by frame-0 feature quartile. More informative than Mann-Whitney medians for a lifetime-continuous outcome |
| New9 | **Feature interactions** — partial-dependence plot on the LR for the top-2 features. Exposes the decision boundary |
| New10 | **Cumulative-disappearance derivative** dN/dt vs. frame. Peaks in the lysis rate |
| New11 | **Peri/core asymmetry dynamics plot** — the metric is in the CSV but nowhere in the report |
| New12 | **Fate stratification by early-swelling** — split by V/V₀ > 1.3 at frame 5, compare outcomes. Adds a temporal-early feature the LR currently misses |
| New13 | **Bootstrap CI on AUC** — makes the 0.665 number robust or reveals it's chance-level |
| New14 | **Nucleus persistence per track** — instead of ensemble count comparison, track nucleus label overlap with each cell's phase mask over time |

---

## 8. Report additions

| ID | Idea |
|---|---|
| R1 | Executive-summary cell at top: N cells, N tracked, N disappeared (%), median lifetime, AUC ± CI, key p-values |
| R2 | Data-quality dashboard: bad-frames, cells rejected per criterion, short-tracks removed, preburst-cohort size |
| R3 | Detection montage: frames 0 / T/2 / T-1 with track IDs overlaid |
| R4 | Population heatmap: rows = tracks (sorted by lifetime), cols = frame, cells = V(t)/V(0) |
| R5 | Sedimentation indicator: mean centroid Δ per frame — stage drift is otherwise invisible |
| R6 | Cumulative-disappearance derivative plot |

---

## 9. Priority list (from the review)

1. **RUN_NUCLEUS_PERSISTENCE opt-out** — biggest wall-time win; single-scalar-yield step
2. **Vectorize `labels_to_detections`** — hot per-frame loop
3. **`FAST_MODE` / `MODEL_TYPE="cyto3"`** — 5–10× faster iteration
4. **Peri/core dynamics plot** — added metric with no report surface
5. **`min_contrast` semantics fix** — misleading name; add cell-vs-BG contrast
6. **Bootstrap AUC CI + K-fold alongside LOO** — makes the number interpretable
7. **Executive-summary cell** — biggest ergonomic win

---

## 10. Implementation status (this session)

| Priority | Status | Notes |
|---|---|---|
| #1 RUN_NUCLEUS_PERSISTENCE | Explained | Deferred — needs config-schema change + notebook-cell guard; user decision on default |
| #2 Vectorize `labels_to_detections` | Applied | `tracking.py` uses `ndimage.center_of_mass(labels, labels, ids)` + `np.bincount` |
| #3 FAST_MODE / MODEL_TYPE=cyto3 | Explained | Deferred — user decision on default model + config-schema change |
| #4 Peri/core plot | Applied | New §8.5c cell added to `notebooks/analysis.ipynb` |
| #5 min_contrast semantics | Applied | Docstring clarified; new optional `min_bg_contrast` param filters on \|cell_mean − background_median\| |
| #6 Bootstrap AUC CI + K-fold | Explained | Deferred — needs design decision on whether to replace LOO or add alongside |
| #7 Executive-summary cell | Applied | New §9.0 "Summary" section added to notebook |

**Deferred priorities #1, #3, #6** — explanations in the chat message accompanying this document. They involve config-schema or evaluation-methodology decisions the user should sign off on before implementation.
