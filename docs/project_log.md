# Project Log: Bacterial Cell Detection, Tracking, and Fluorescence Analysis

Microscopy image analysis pipeline for bacterial cell detection, tracking, and fluorescence quantification in time-lapse phase-contrast stacks.

**Dataset:** `Gradient-0011.zvi` — 25 frames, 1040x1388, uint16.
- Phase-contrast channel: `Ch0.tif`
- Fluorescence channel (background-subtracted): `Ch1-BG.tif`

---

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Detection](#detection)
- [Tracking](#tracking)
- [Fluorescence Analysis](#fluorescence-analysis)
- [Output Artifacts](#output-artifacts)
- [Key Results](#key-results)
- [Open Questions and Limitations](#open-questions-and-limitations)
- [Future Work](#future-work)
- [Decision Log](#decision-log)
- [Progress Log](#progress-log)

---

## Pipeline Architecture

### End-to-end flow

```mermaid
flowchart TD
    TIFF["Phase TIFF + Fluorescence TIFF"]

    subgraph LOAD["I/O"]
        load["io.load_stack()"]
    end

    subgraph SEG["Segmentation"]
        cellpose["segmentation.detect_cells_stack()"]
        filter["regionprops filter\n(area, circularity, contrast, edges)"]
        cellpose --> filter
    end

    subgraph GATE["Frame Quality Gating"]
        l2d["tracking.labels_to_detections()"]
        dbf["tracking.detect_bad_frames()"]
        drop["Remove flagged rows\nZero label_stack for bad frames"]
        l2d --> dbf --> drop
    end

    subgraph TRACK["Tracking"]
        tp["tracking.track_cells()\n(trackpy linking)"]
        merge["tracking.merge_fragmented_tracks()\n(Union-Find)"]
        stats["tracking.compute_track_stats()"]
        tp --> merge --> stats
    end

    subgraph FLUOR["Fluorescence"]
        mf["matching.measure_fluorescence()"]
    end

    subgraph OUT["Output"]
        save["io.save_results()"]
    end

    TIFF --> load
    load -->|"(T,Y,X) ndarray"| cellpose
    filter -->|"centroids + label_stack"| l2d
    drop -->|"filtered detections"| tp
    drop -->|"cleaned label_stack"| mf
    load -->|"fluor_stack"| mf
    stats --> save
    mf -->|"intensity metrics"| save
    dbf -.->|"dropped_frames.csv"| OUT
```

### Detection detail

```mermaid
flowchart LR
    raw["Raw frame\n(uint16)"] --> inv["Invert image\n(dark cells -> bright)"]
    inv --> cp["Cellpose eval\n(model loaded once,\nresample=False)"]
    cp --> masks["Raw masks\n(~423/frame)"]
    masks --> f1["min_area=300"]
    f1 --> f2["min_circularity=0.7"]
    f2 --> f3["min_contrast=1250"]
    f3 --> f4["exclude_edges=True"]
    f4 --> accepted["Accepted\n(~371/frame)"]
    f1 -.->|"rejected"| rej["~52 rejected/frame"]
    f2 -.->|"rejected"| rej
    f3 -.->|"rejected"| rej
    f4 -.->|"rejected"| rej
```

### Frame gating logic

```mermaid
flowchart TD
    det["All detections\n(all frames)"] --> grp["Group by frame"]
    grp --> s1["Cell count"]
    grp --> s2["Mean area"]
    grp --> s3["Area IQR"]

    s1 --> z1["Absolute MAD Z-score\n(all frames)"]
    s2 --> z2["Delta MAD Z-score\n(frames 1+)\ncross-checked with absolute Z"]
    s3 --> z3["Delta MAD Z-score\n(frames 1+)\ncross-checked with absolute Z"]

    z1 --> flag{"Any |z| > 3.5?"}
    z2 --> flag
    z3 --> flag

    flag -->|yes| bad["Flag frame as bad"]
    flag -->|no| ok["Frame passes"]

    bad --> rm["Remove from detections\nZero label_stack"]
    bad --> csv["Save to dropped_frames.csv"]
```

### Tracking and merging

```mermaid
flowchart TD
    det["Filtered detections"] --> tp["trackpy.link()\nsearch_range=30, memory=3"]
    tp --> tracks["Tracks with IDs"]
    tracks --> find["Find track pairs:\nend/start within\nmax_distance=15 px,\nmax_gap=18 frames"]
    find --> uf["Union-Find merge"]
    uf --> merged["Merged tracks\n(404 -> 388)"]
    uf --> log["Merge log\n(auditable)"]
```

### Fluorescence measurement

```mermaid
flowchart LR
    subgraph Inputs
        ls["label_stack\n(cleaned)"]
        fs["fluor_stack\n(background-subtracted)"]
    end

    subgraph PerCell["Per cell per frame"]
        mask["Extract pixels\nwhere label == cell_id"]
        mask --> mean["Mean intensity"]
        mask --> total["Total intensity"]
        mask --> minmax["Min / Max"]
        mask --> cv["CV = std/mean"]
        mask --> nnrm["nNRM\n(KS vs Gaussian)"]
        mask --> skew["Skewness"]
        mask --> kurt["Kurtosis"]
    end

    subgraph Derived["Per track"]
        rel["F(t)/F(0)\n(relative decline)"]
        dis["Disappearance detection\n(>30% single-frame drop)"]
        lnorm["Lifespan-normalized\nCV and nNRM"]
    end

    ls --> mask
    fs --> mask
    mean --> rel
    total --> dis
    cv --> lnorm
    nnrm --> lnorm
```

### Per-track growth analysis

```mermaid
flowchart LR
    tracked["tracked DataFrame\n(area per cell per frame)"] --> grp["Group by track_id"]
    grp --> a0["area_initial\n(first observation)"]
    grp --> amax["area_max\n(peak area)"]
    grp --> rel["area_rel_max\n= area_max / area_initial"]
    grp --> fit["Linear fit\narea vs frame"]
    fit --> rate["growth_rate_px_per_frame"]
    amax --> fmax["frame_of_max_area"]
```

### Advanced statistics

```mermaid
flowchart TD
    tracked["tracked DataFrame\n(with area, volume, surface_area, fluorescence)"]

    subgraph CONC["Fluorescence Concentration"]
        fc["total_intensity / volume\n(dilution-corrected signal)"]
    end

    subgraph MIG["Migration"]
        spd["Per-frame centroid displacement\n→ speed column"]
        mig["Per-track: mean/max speed,\ntotal/net displacement"]
    end

    subgraph SAV["SA:V Ratio"]
        sav["surface_area / volume\n(membrane stress indicator)"]
    end

    subgraph CLUST["Death Clustering"]
        last["Last observed position\nper disappeared track"]
        nn["Nearest-neighbor distances\namong deaths"]
        perm["Permutation test\n(1000 iterations)\nvs random subset"]
    end

    subgraph PRE["Pre-burst Fluorescence"]
        win["n-frame window before\nlast detection"]
        slope["Linear fit of\nmean_intensity in window"]
        spike["Spike detection:\nslope > 0 AND\nmax > baseline"]
    end

    subgraph PHASE["Growth Phases"]
        seg["2-segment piecewise\nlinear fit of area"]
        cp["Optimal split point\n(min RSS)"]
        ratio["slope_after / slope_before"]
    end

    tracked --> fc
    tracked --> spd --> mig
    tracked --> sav
    tracked --> last --> nn --> perm
    tracked --> win --> slope --> spike
    tracked --> seg --> cp --> ratio
```

Modules in `src/cell_analysis/`: `io.py`, `segmentation.py`, `tracking.py`, `matching.py`, `pipeline.py`, `plotting.py`.

Notebook entry point: `notebooks/analysis.ipynb`.

---

## Detection

### Method: Cellpose

Cellpose was selected over classical segmentation (threshold + watershed) after a head-to-head comparison.

| Metric | Classical | Cellpose |
|---|---|---|
| Cells found | 340 | ~420 raw |
| False positives | Many (halo regions, background) | Very few |
| Centroid accuracy | Often shifted | Precise |
| Adjacent cells | Struggles to separate | Handles well |
| Speed (per frame) | Seconds | CPU 17s, MPS 3s, MPS+no_resample 0.3s |

Classical method kept as fallback (`detect_cells_frame_classical()`).

### Post-detection Filters

Applied after Cellpose, before tracking. Tuned on frame 0 (~423 raw detections -> ~371 accepted, ~52 rejected).

| Parameter | Value | Rationale |
|---|---|---|
| `diameter` | 32 | Auto-detected median cell diameter. Explicit to ensure consistency across frames. |
| `min_area` | 300 | Removes debris (<200 px) and faded blobs (~270 px) without losing real cells. Median cell area ~824 px. |
| `min_circularity` | 0.7 | Separates round cells (median 0.93) from elongated artifacts (rods, merged blobs). |
| `min_contrast` | 1250 | Intensity std-dev within mask. Lowered from original 1550 after discovering ~22 cells/frame flickering around that threshold, causing track fragmentation. |
| `exclude_edges` | True | Rejects cells whose bounding box touches the frame border (~15-20 partial cells/frame). |

Full tuning details and diagnostic images: `docs/detection_tuning.md`.

### Known Limitation: Cellpose Misses

~5-6 round cells per frame have no Cellpose mask at all (typically squeezed between neighbors in dense clusters). Cannot be recovered by post-filter tuning. A potential hybrid fallback (classical second-pass for dark round blobs without masks) has been identified but not implemented.

### Performance

Benchmarked on Apple Silicon M3 Max, 200x200 crop:

| Config | Time (200x200 crop) | Speedup |
|---|---|---|
| CPU | 17.4s | 1x |
| MPS (Metal) | 3.1s | 5.6x |
| MPS + resample=False | 0.3s | 58x |

`resample=False` returns masks at Cellpose internal resolution; code resizes with nearest-neighbor. Minimal quality loss for centroid/area.

**Suppressed warnings:** Cellpose emits harmless logging warnings (not `warnings.warn`): `"Resizing is deprecated in v4.0.1+"` from `cellpose.dynamics` logger, and `"Sparse invariant checks"` PyTorch UserWarning. Both suppressed in `segmentation.py` via `logging.getLogger("cellpose.dynamics").setLevel(logging.ERROR)` and `warnings.filterwarnings`.

#### Hardware tuning guide

The pipeline adapts to available hardware via `DETECT_PARAMS` in the notebook. The two relevant settings are `gpu` and `resample`.

| Hardware | `gpu` | `resample` | Notes |
|---|---|---|---|
| Apple Silicon (M-series) | `True` | `False` | Cellpose auto-detects MPS (Metal). Best option on Mac. |
| NVIDIA GPU (CUDA) | `True` | `False` | Requires PyTorch with CUDA. Fastest option overall. |
| CPU only (any OS) | `False` | `False` | Slower but produces identical results. `resample=False` is the main speedup here — skipping it costs ~58x. |
| Low memory (<4 GB free) | `False` | `False` | GPU off avoids loading the model onto the accelerator. The stack itself is ~69 MB (25 frames x 1040x1388 x uint16) and the label stack ~138 MB (int32). Cellpose processes one frame at a time internally, so per-frame memory footprint stays small. |

**Key points:**

- `gpu=True` is a request, not a requirement. If no compatible GPU is found, Cellpose falls back to CPU silently. Setting `gpu=False` explicitly avoids the detection attempt and any related warnings.
- `resample=False` is hardware-independent and always recommended. It provides the largest single speedup (~58x) with negligible quality impact for centroid and area measurements. There is no reason to set it to `True` for this pipeline.
- All downstream modules (tracking, fluorescence measurement, frame gating) are pure numpy/pandas and run on CPU regardless. The hardware choice only affects the Cellpose segmentation step.

---

## Tracking

### Core Approach

1. **trackpy linking**: `track_cells()` links centroids between consecutive frames using `trackpy.link()` with `search_range=30.0` and `memory=3`.
2. **Track merging**: `merge_fragmented_tracks()` reconnects tracks broken by detection gaps longer than `memory`. Uses greedy spatial matching + Union-Find. Parameters: `max_distance=15.0 px`, `max_gap=18 frames`. Returns an auditable merge log.
3. **Track statistics**: `compute_track_stats()` produces per-track summary (lifetime, disappearance, mean area/volume/surface area, fluorescence metrics).

### Dynamic Frame Quality Gating

`detect_bad_frames()` replaces the old hardcoded `TRIM_FRAMES = 2` approach. Automatically detects anomalous frames anywhere in the time series.

**Signals used:**

| Signal | What it catches |
|---|---|
| Cell count per frame | Focus loss (count drops), segmentation artifacts (count spikes) |
| Mean cell area per frame | Defocus (blurred cells appear larger), partial field (smaller) |
| Area IQR per frame | Heterogeneous detection quality (some cells detected, some missed) |

**Method:** MAD-based modified Z-scores (robust to outlier contamination). Threshold: 3.5 (Iglewicz & Hoaglin recommendation).

**Per-signal strategy:**
- Cell count: absolute Z-scores (avoids the recovery-frame false positive from delta Z).
- Mean area / area IQR: delta Z-scores (handles biological trend of swelling), cross-checked with absolute Z (guards against statistical instability on short series).
- Frame 0: absolute Z-scores for all signals (no prior frame for deltas).

**What happens to flagged frames:**
- Removed from detections DataFrame before tracking.
- Frame numbers are NOT renumbered (trackpy bridges via `memory`; larger gaps handled by `merge_fragmented_tracks()`).
- Label stack zeroed for flagged frames (prevents fluorescence measurement).
- `dropped_frames.csv` saved for audit.

### Track Identity Fragmentation (Review 1)

**Problem:** Cells temporarily lost to detection (e.g., from an out-of-focus frame) get new track IDs when they reappear, fragmenting their history. 55 tracks started after frame 0; 16 were confirmed fragments of earlier tracks (<15 px, gaps 5-18 frames). Some form chains across multiple gaps (e.g., track 98 -> 388 -> 399: same cell across frames 0, 9-11, 17-23). The 28 tracks starting at frame 1 with >100 px distance from any prior track endpoint were confirmed as genuinely new, not fragments.

**Solutions implemented:**
1. Dynamic frame gating (replaced the earlier frame trimming approach).
2. Post-hoc track merging via `merge_fragmented_tracks()`.

**Result:** 404 -> 388 tracks (16 fragments absorbed), all 28 genuinely new frame-1 tracks untouched.

---

## Fluorescence Analysis

### Measurement Approach

Fluorescence intensity is measured within phase-contrast cell masks (not via independent segmentation of the fluorescence channel).

**Key assumptions:**

- Cell boundaries come from phase-contrast Cellpose segmentation. If phase masks are slightly too large or too small relative to the true fluorescence boundary, intensity values will be biased.
- No independent nucleus segmentation (yet). Whole-cell mask includes nucleus + cytoplasm. For a nuclear stain, most signal comes from the nucleus, but cytoplasmic background is included.
- Background subtraction is pre-applied (`Ch1-BG` input). ~7% of pixels are zero after subtraction. No additional background correction is applied.
- Phase and fluorescence channels are assumed spatially registered (same microscope, same objective, simultaneous acquisition). No registration or alignment step is performed.

### Metrics Computed

**Intensity:**
- Mean, total, min, max intensity per cell per frame.

**Distribution (nucleoid heterogeneity):**

- **CV** (coefficient of variation = std/mean): measures concentration vs. dispersal. High CV (~0.9) = bright nucleoid spots against dark cytoplasm (intact, concentrated). Low CV (~0.4-0.6) = more uniform distribution (dispersed nucleoid). Scale-invariant and directly interpretable.
- **nNRM** (Non-Normality Index): Kolmogorov-Smirnov statistic comparing pixel intensity distribution against a Gaussian with the same mean and std. Range [0,1]: 0 = perfectly Gaussian, 1 = maximally non-Gaussian. Complements CV — CV measures spread, nNRM measures distribution shape. A high nNRM indicates distinct subpopulations within the mask (bright nucleoid vs dark cytoplasm). Implementation: `scipy.stats.kstest(pixels, 'norm', args=(mean, std))`. ~0.9ms/cell. Reference: Gough et al. (2014), PLOS ONE, DOI: 10.1371/journal.pone.0102678.
- **Skewness**: asymmetry of pixel distribution. Positive skew = long right tail (bright outliers). Supplementary.
- **Kurtosis**: peakedness/tail heaviness (excess kurtosis, Fisher definition). Supplementary.

### Relative Fluorescence F(t)/F(0)

Computed for the frame-0 cohort only (tracks present at frame 0 with known baseline). F(0) is each cell's mean intensity at its first detection. Cells with F(0) = 0 are excluded to avoid division by zero.

### Fluorescence Disappearance Detection

Per-track detection of large single-frame drops in total intensity.

- Uses relative change (percentage drop) rather than absolute intensity because cells vary widely in baseline fluorescence. A 30% drop is biologically meaningful regardless of starting intensity.
- Formula: `delta = (I[t] - I[t-1]) / I[t-1]`. The frame with the largest single-frame drop exceeding the threshold is flagged.
- Threshold: -30% (tuned empirically; median max drop is -13%, -30% sits at ~8th percentile).

| Threshold | Tracks flagged | Specificity |
|---|---|---|
| -50% | 8/386 (2.1%) | 100% |
| -35% | 15/386 (4.0%) | 80% |
| **-30%** | **30/386 (7.8%)** | **83%** |
| -25% | 65/386 (17.2%) | - |

### Lifespan-Normalized Dynamics

For population-averaged CV and nNRM plots, each track's frame indices are mapped to relative lifespan [0, 1], then binned into 20 equal bins and averaged across all tracks. Aligns cells with different lifetimes to reveal the trajectory from "start of life" to "end of life" regardless of absolute timing. Tracks with fewer than 2 detections are excluded.

---

## Output Artifacts

### Pipeline outputs (notebook run)

All saved to `results/<RUN_NAME>/` (default `results/run_01/`).

```mermaid
flowchart TD
    subgraph NB["notebooks/analysis.ipynb"]
        run["Full pipeline run"]
    end

    subgraph CSV["CSV files"]
        tc["tracked_cells.csv\nPer-cell per-frame data"]
        ts["track_statistics.csv\nPer-track summary"]
        diag["frame_diagnostics.csv\nPer-frame detection statistics"]
        df["dropped_frames.csv\nFlagged frame audit log"]
        ml["merge_log.csv\nTrack merging audit"]
        fp["fate_predictions.csv\nPer-cell death predictions"]
        sg["spatial_gradient.csv\nPer-cell gradient quartiles"]
        np2["nucleus_persistence.csv\nPer-frame phase vs nucleus counts"]
    end

    subgraph SUMM["Summary CSVs (single-row, flattened)"]
        cs["clustering_summary.csv\nSpatial clustering test"]
        fps["fate_prediction_summary.csv\nAUC, feature importance"]
        sgs["spatial_gradient_summary.csv\nPer-axis stats, quartile rates"]
        nps["nucleus_persistence_summary.csv\nLoss counts, conclusion"]
    end

    subgraph PLOTS["Inline notebook plots"]
        p1["Cell count per frame\n(with flagged frames)"]
        p2["Lifetime distribution"]
        p3["Disappearance per frame"]
        p4["Area histogram\n(with volume axis)"]
        p5["V(t)/V(0) swelling curves"]
        p6["S(t)/S(0) swelling curves"]
        p7["F(t)/F(0) fluorescence curves"]
        p8["CV and nNRM temporal trends"]
        p9["Fluor vs volume scatter"]
        p10["Swelling vs initial size"]
        p11["Swelling vs DNA content"]
        p12["Growth before burst\n(aligned curves, rate & max size distributions)"]
        p13["Fluorescence concentration\n(time series, outcome split, vs volume)"]
        p14["Migration speed\n(time series, outcome split, distribution)"]
        p15["SA:V ratio\n(time series, outcome split, histogram)"]
        p16["Death clustering\n(spatial map, null distribution, temporal)"]
        p17["Pre-burst fluorescence\n(aligned curves, slope distribution, classification)"]
        p18["Growth phases\n(changepoint histogram, slope scatter, examples)"]
        p19["Fate prediction\n(ROC curve, feature importance, probability distribution)"]
        p20["Spatial gradient\n(scatter, quartile rates, density, correlation)"]
    end

    run --> tc & ts & diag & df & ml & fp & sg & np2
    run --> cs & fps & sgs & nps
    run --> p1 & p2 & p3 & p4 & p5 & p6 & p7 & p8 & p9 & p10 & p11 & p12 & p13 & p14 & p15 & p16 & p17 & p18 & p19 & p20
```

#### `tracked_cells.csv`

One row per cell per frame. Columns:

| Column | Description |
| --- | --- |
| `frame` | Raw frame index (0-based) |
| `label` | Cell label within that frame's mask |
| `centroid_y`, `centroid_x` | Cell centroid in pixels |
| `area` | Cell mask area in pixels |
| `track_id` | Unique track ID (after merging) |
| `radius` | Equivalent radius: sqrt(area / pi) |
| `volume` | Spherical volume: (4/3) pi r^3 |
| `surface_area` | Spherical surface area: 4 pi r^2 |
| `mean_intensity` | Mean fluorescence within cell mask |
| `total_intensity` | Sum of fluorescence within cell mask |
| `min_intensity` | Min fluorescence pixel value |
| `max_intensity` | Max fluorescence pixel value |
| `std_intensity` | Std dev of fluorescence within mask |
| `cv` | Coefficient of variation (std/mean) |
| `skewness` | Pixel distribution skewness |
| `kurtosis` | Pixel distribution excess kurtosis |
| `nnrm` | Non-Normality Index (KS statistic) |
| `speed` | Centroid displacement from previous frame (px) |
| `fluor_concentration` | total_intensity / volume (dilution-corrected) |
| `sav_ratio` | surface_area / volume (membrane stress indicator) |

#### `track_statistics.csv`

One row per track. Columns:

| Column | Description |
| --- | --- |
| `track_id` | Unique track ID |
| `first_frame`, `last_frame` | First and last frame of detection |
| `num_detections` | Number of frames where cell was detected |
| `lifetime` | last_frame - first_frame + 1 |
| `disappeared` | True if track ends before the final frame |
| `mean_area` | Time-averaged cell area |
| `mean_volume` | Time-averaged spherical volume |
| `mean_surface_area` | Time-averaged spherical surface area |
| `mean_fluor_intensity` | Time-averaged mean fluorescence |
| `mean_fluor_total` | Time-averaged total fluorescence |
| `mean_cv` | Time-averaged CV |
| `mean_nnrm` | Time-averaged nNRM |
| `area_initial` | Cell area (pixels) at first detection |
| `area_max` | Peak cell area (pixels) over track lifetime |
| `area_rel_max` | Peak area / initial area (growth factor) |
| `frame_of_max_area` | Frame at which cell reached peak size |
| `growth_rate_px_per_frame` | Linear growth rate (slope of area vs frame) |
| `fluor_disappearance_frame` | Frame of largest fluorescence drop (if > threshold) |
| `max_drop` | Largest single-frame relative drop in total intensity |
| `mean_speed` | Time-averaged centroid speed (px/frame) |
| `max_speed` | Maximum single-frame speed (px/frame) |
| `speed_std` | Standard deviation of per-frame speed |
| `total_displacement` | Sum of all step distances (px) |
| `net_displacement` | Straight-line distance from first to last position (px) |
| `preburst_slope` | Linear slope of mean_intensity in pre-burst window |
| `preburst_spike` | True if fluorescence spikes before disappearance |
| `changepoint_frame` | Frame of detected growth phase transition |
| `slope_before` | Area growth rate before changepoint (px/frame) |
| `slope_after` | Area growth rate after changepoint (px/frame) |
| `slope_ratio` | slope_after / slope_before |

#### `dropped_frames.csv`

One row per flagged frame (only created if frames are flagged). Columns: `frame`, `cell_count`, `mean_area`, `iqr_area`, `z_count`, `z_area`, `z_iqr`, `flagged`, `reasons`.

### Diagnostic overlay (manual)

Generated by `scripts/diagnostic_overlay.py`, saved to `results/`:

| File | Description |
| --- | --- |
| `diagnostic_full.png` | Full frame with accepted (red X) and rejected (cyan O) detections |
| `diagnostic_crops.png` | Six zoomed crops with rejection reason labels per cell |

---

## Key Results

### Detection and Tracking

| Metric | Value |
|---|---|
| Raw detections per frame | ~423 (Cellpose) |
| Accepted after filtering | ~371 per frame |
| Total tracks (after merging) | 388 |
| Tracks starting after frame 0 | 55 (28 genuine, 16 merged, rest accounted for) |

### Morphology and Swelling

| Metric | Value |
|---|---|
| Final V(t)/V(0), population mean | 1.690 +/- 0.029 |

Volume and surface area derived from area assuming spherical geometry: V = (4/3)pi*r^3, S = 4*pi*r^2, where r = sqrt(area/pi).

### Fluorescence

| Metric | Value |
|---|---|
| Fluorescence measurements | 7046 rows |
| Match rate (phase masks to fluor) | 100% |
| Median fluorescence per track | 2257 |
| Final F(t)/F(0), population | 0.652 +/- 0.013 (37% decline) |
| Total fluor vs volume (r) | 0.465 |
| Mean fluor vs volume (r) | -0.238 |
| CV (frame 0 -> 22) | 0.544 -> 0.466 |
| nNRM (frame 0 -> 22) | 0.113 -> 0.098 |
| Fluor drop before phase loss | 9/25 disappeared cells |
| Fluor drop at same frame | 16/25 disappeared cells |

### Key Biological Observations

1. **Population-level fluorescence decline**: mean intensity decreases monotonically (0.63x over 23 frames). Could be photobleaching, biological DNA loss, or dilution from swelling. Cannot distinguish without unstressed controls.

2. **Disappeared vs. survived cells**: cells that disappear lose fluorescence faster, visible from ~frame 8 onward. Suggests membrane integrity loss before visible lysis.

3. **Fluorescence-volume relationship**: total fluorescence positively correlated with volume (r=0.47); mean fluorescence weakly negatively correlated (r=-0.24). Consistent with dilution model — fluorophore content doesn't scale linearly with size.

4. **Nucleoid dispersal**: CV and nNRM both decline over time, indicating fluorescence becomes more uniformly distributed. Disappeared cells start with slightly higher CV but decline faster. Consistent with nucleoid decondensation preceding lysis.

5. **Fluorescence as leading indicator**: 9/25 disappeared cells showed fluorescence drop before phase disappearance, suggesting gradual membrane failure rather than instantaneous rupture.

6. **DNA does not persist after cell lysis**: Independent Cellpose segmentation of the fluorescence channel (diameter=25, ~422 nuclei/frame) shows nucleus counts decline in parallel with phase-contrast cell counts. Phase lost 171 cells, fluorescence lost 169 nuclei — near-identical rates. The constant ~50 offset (fluorescence detects more objects) is stable across all 25 frames (std ~3.5). This means fluorescent DNA disperses immediately upon membrane rupture rather than remaining as a discrete object.

7. **Death timing is accelerating, not constant-rate**: Deaths ramp from 1-7/frame (frames 0-10) to 12-20/frame (frames 14-20), then drop to 6-8/frame (frames 21-23). The late drop reflects population depletion, not reduced stress. Suggests cumulative/threshold-based damage rather than constant-rate killing.

8. **Nucleoid dispersal precedes lysis in 90% of dying cells**: CV drops from 0.632 (first frame) to 0.456 (last frame) in disappeared cells. Only 10% show CV increase before death — possibly a different death mechanism (rapid rupture without dispersal phase).

9. **Cell fate is partially predictable from frame-0 features**: Mann-Whitney tests on the frame-0 cohort (`compare_frame0_features_by_fate()` in `src/cell_analysis/matching.py`, reported in notebook §8.14 and `results/<run>/frame0_fate_comparison.csv`) show cells that will die are already different at the start — smaller area (p≈0.0008), higher CV (p<0.0001), higher nNRM (p<0.0001) on the initial run_01 dataset. Initial nucleoid heterogeneity is the strongest early predictor of cell fate. Continuous companion: `plot_initial_features_vs_lifespan()` (notebook §8.17) reports Spearman rho against lifetime for the same three features.

10. **Cells that die later are larger at death** (r=0.30, early death median area 859 px vs late death 1196 px). Supports a "swell until critical membrane threshold" model where cells accumulate osmotic stress until the membrane can no longer compensate.

11. **Fluorescence drops 3.5 frames before phase disappearance on average** (in the 35 tracks with detectable drops, mean delta = -3.5 frames). This early warning window is substantial and could potentially be used for real-time death prediction.

---

## Open Questions and Limitations

1. **Photobleaching vs. biology**: without an unstressed control time-lapse, photobleaching cannot be separated from actual fluorescence loss. A fixed-cell or untreated control with the same imaging conditions would enable calibration.

2. **Dilution correction**: total fluorescence partially accounts for dilution by swelling, but a more rigorous approach (comparing F_total(t)/F_total(0) vs. F_mean(t)/F_mean(0)) is needed to estimate the dilution component.

3. **Cellpose segmentation misses**: ~5-6 cells/frame have no mask. Parameter sweeps (diameter, cellprob_threshold, flow_threshold) did not recover them without breaking existing detections.

4. **Nucleus shape analysis**: Independent fluorescence segmentation now exists (`detect_nuclei_stack()`) and confirms nuclei don't persist after lysis. Nucleus-to-cell area ratio and condensation/fragmentation analysis via `match_cells_to_nuclei()` remain future work.

5. **Single dataset**: all tuning and validation performed on one 25-frame time-lapse. Generalization to other datasets, imaging conditions, or cell types is untested.

6. **Speed calculation assumed 1-frame gaps (fixed)**: `compute_migration_stats` originally divided displacement by 1, regardless of actual frame gaps. ~0.5% of consecutive detections had gaps > 1 frame (from dropped frames or detection gaps bridged by trackpy memory). Fixed by dividing displacement by actual frame difference. `total_displacement` (sum of raw distances) is unaffected; `mean_speed`, `max_speed`, `speed_std`, and per-frame `speed` column are now correct px/frame values.

7. **Nucleus persistence comparison fragile to bad frames (fixed)**: `run_nucleus_persistence` compared phase label counts (zeroed for bad frames by gating) against nucleus label counts (not zeroed). This would show a spurious offset spike on any bad frame. Fixed by skipping frames where phase labels are zeroed but nucleus labels are not.

8. **Ultra-short track artifacts**: 16 tracks with ≤3 detections are all classified as "disappeared." These may be transient segmentation artifacts (debris detected across a few frames) rather than real cells that died. A minimum-lifetime filter could reduce noise in disappearance statistics. Not yet implemented — requires choosing a threshold that doesn't discard genuine brief tracks.

---

## Future Work

- [ ] **Hybrid fallback for Cellpose misses**: classical second-pass for dark round blobs without masks (~5 cells/frame recovery).
- [ ] **Nucleus shape analysis**: segment nuclei in fluorescence channel independently, measure nucleus-to-cell area ratio, condensation/fragmentation. Use existing `match_cells_to_nuclei()` infrastructure.
- [ ] **Photobleaching calibration**: acquire or identify an unstressed control time-lapse for correction.
- [ ] **Dilution correction**: F_total(t)/F_total(0) vs. F_mean(t)/F_mean(0) comparison.
- [ ] **Multi-dataset validation**: test pipeline on additional datasets to check generalization of parameters.
- [x] **Minimum-lifetime filter**: `filter_short_tracks()` with configurable `MIN_TRACK_DETECTIONS` (default 4). Removes ultra-short tracks that are likely segmentation artifacts.
- [x] **Frame-0 fate prediction model**: logistic regression on area, CV, nNRM with LOO cross-validation. ROC curve, feature importance, and probability distribution plots.
- [ ] **Fluorescence early-warning system**: fluorescence drops ~3.5 frames before phase disappearance on average. Could be developed into a real-time predictor of impending cell death.
- [x] **Spatial gradient analysis**: `analyze_spatial_gradient()` with per-axis Mann-Whitney, point-biserial correlation, logistic regression AUC, and quartile death rate stratification.
- [ ] **Survival analysis (Kaplan-Meier / Cox PH)**: time-to-event framework instead of binary died/survived. Kaplan-Meier curves stratified by initial features (area quartiles, CV quartiles, gradient position). Cox proportional hazards model for multivariate survival time prediction.
- [ ] **Dynamic trajectory features for early warning**: rate of change in first 3-5 frames (area growth rate, CV slope, fluorescence decline rate) as predictors. Could enable "this cell will die within N frames" classifier.
- [ ] **Critical membrane threshold testing**: test whether area_rel_max at death clusters around a value (mechanical rupture threshold). Plot SA:V ratio at death — if membrane stress is the driver, should converge to a critical value regardless of initial size.
- [ ] **Neighborhood effects beyond gradient**: after accounting for gradient position, test whether cells near other dying cells die sooner. Local density at frame 0 vs fate could reveal crowding effects independent of drug gradient.

---

## Decision Log

Decisions, design choices, and their rationale. Most recent first.

### Dynamic frame gating replaces hardcoded TRIM_FRAMES

**Date:** 2026-04-16
**Context:** Pipeline originally dropped the first 2 frames via `TRIM_FRAMES = 2` because frame 1 was out of focus. This was brittle (only caught initial frames, fixed count, required manual tuning per dataset).
**Decision:** Replace with `detect_bad_frames()` using MAD-based Z-scores on per-frame detection statistics.
**Rationale:** Automatically detects anomalous frames anywhere in the series. Cell count uses absolute Z (avoids recovery-frame false positive); area/IQR use delta Z cross-checked with absolute Z (handles biological trend and short-series instability). Threshold 3.5 per Iglewicz & Hoaglin.
**Result on current dataset:** Zero frames flagged — correct. With the current `min_contrast=1250`, cell counts decline smoothly from 376 (frame 1) to 200 (frame 24) with no jumps or spikes. The original `TRIM_FRAMES=2` existed because "frame 1 was out of focus," but the real cause was the stricter `min_contrast=1550`: borderline cells in slightly defocused early frames fell below the contrast threshold, making frames 0-1 appear anomalous (fewer detections). Lowering `min_contrast` to 1250 (done to fix track fragmentation from ~22 cells/frame flickering around the old threshold) resolved the root cause, so the symptom (anomalous early frames) disappeared. The gating system is validated by unit tests to catch genuinely bad frames when they exist.

### Fluorescence disappearance: relative threshold, not absolute

**Context:** Needed to detect single-frame fluorescence loss events per track.
**Decision:** Use relative change (percentage drop) rather than absolute intensity change; threshold at -30%.
**Alternatives considered:** Absolute intensity drop threshold.
**Rationale:** Cells vary widely in baseline fluorescence. A 30% drop is biologically meaningful regardless of starting intensity. Median max drop is -13% (normal variation). -30% sits at ~8th percentile: captures 25/179 disappeared cells with only 5/200 false positives among survivors. -50% too strict, -25% too permissive.

### Per-cell per-frame data over time-averaged summaries

**Context:** Reviewer asked for time-averaged area. Cells actively swell over time as external osmotic pressure decreases.
**Decision:** Keep per-cell per-frame area in the data; derive volume/surface area per frame. Time-averaged area is computed for the track summary but is not the primary analysis metric.
**Rationale:** Averaging across time smears out the signal of interest (swelling dynamics). V(t)/V(0) and S(t)/S(0) relative curves preserve the temporal structure.

### nNRM over composite skewness/kurtosis metric

**Context:** Needed a scalar metric for pixel distribution non-normality within cell masks.
**Decision:** KS-based nNRM (Gough et al. 2014).
**Alternatives considered:** sqrt(skewness^2 + kurtosis^2).
**Rationale:** Published reference in high-content screening literature, captures all forms of non-normality in bounded [0,1] scalar, no weight-choosing between skewness and kurtosis.

### CV over Shannon entropy for heterogeneity

**Context:** Needed a metric for intra-cell fluorescence heterogeneity.
**Decision:** Coefficient of variation (std/mean).
**Alternatives considered:** Shannon entropy of pixel histogram.
**Rationale:** Simpler biological interpretation, no binning decisions required, explicitly requested by reviewer.

### min_contrast lowered from 1550 to 1250

**Context:** Tracking analysis revealed ~22 cells/frame flickering around the 1550 threshold, causing track fragmentation (new track IDs 359-377).
**Decision:** Lower to 1250.
**Rationale:** Recovers borderline cells (371 accepted at 1250 vs 349 at 1550), still rejects truly faded cells. Diminishing returns below 1250.

### Cellpose over classical segmentation

**Context:** Initial pipeline used threshold + watershed on phase-contrast images.
**Decision:** Switch to Cellpose with `resample=False` + MPS acceleration.
**Rationale:** Far fewer false positives, better centroid accuracy, handles adjacent cells. 58x speedup with resample=False makes it practical for full stacks. Classical method retained as fallback.

### Post-hoc track merging via Union-Find

**Context:** Cells temporarily lost to detection get new track IDs, fragmenting their history.
**Decision:** `merge_fragmented_tracks()` with greedy spatial matching + Union-Find.
**Rationale:** 16/55 late-starting tracks confirmed as fragments. Merging with max_distance=15 px, max_gap=18 frames reconnects them. Returns auditable merge log.

### Spherical geometry assumption for volume/surface area

**Context:** Reviewer requested volume and surface area analysis for swelling dynamics.
**Decision:** Assume spherical geometry: V = (4/3)pi*r^3, S = 4*pi*r^2, r = sqrt(area/pi).
**Rationale:** Reasonable for round bacteria. Volume changes more strongly during swelling than area; surface area enables estimation of critical membrane elastic stretch.

---

## Progress Log

Chronological record of completed work.

### Phase 1: Detection Pipeline

- [x] Cellpose integration with MPS acceleration and resample=False optimization
- [x] Post-detection filtering (area, circularity, contrast, edge exclusion)
- [x] Parameter tuning on frame 0 (documented in `docs/detection_tuning.md`)
- [x] Diagnostic overlay script (`scripts/diagnostic_overlay.py`)
- [x] Contrast threshold lowered from 1550 to 1250 after tracking analysis

### Phase 2: Tracking

- [x] trackpy-based centroid linking (search_range=30, memory=3)
- [x] Post-hoc track merging via Union-Find (max_distance=15, max_gap=18)
- [x] Track identity fragmentation analysis (404 -> 388 tracks)
- [x] Dynamic frame quality gating (`detect_bad_frames()`)
- [x] Replaced hardcoded TRIM_FRAMES with automatic detection

### Phase 3: Morphology Analysis (Review 1 Requests)

- [x] Cell count per frame, 50% disappearance frame, fraction disappeared
- [x] Lifetime distribution histogram, disappearance count per frame
- [x] Volume and surface area columns (spherical geometry)
- [x] V(t)/V(0) and S(t)/S(0) population-averaged swelling curves with SEM bands
- [x] Swelling extent vs. initial cell size (scatter + linear fit)
- [x] Swelling dynamics: disappeared vs. surviving cells

### Phase 4: Fluorescence Integration

- [x] Fluorescence measurement through phase-contrast masks (mean, total, min, max intensity)
- [x] Distribution metrics: CV, nNRM, skewness, kurtosis
- [x] Relative fluorescence F(t)/F(0) for frame-0 cohort
- [x] Fluorescence disappearance detection (per-track, -30% threshold)
- [x] Timing analysis: fluorescence drop vs. phase disappearance
- [x] Fluorescence vs. volume correlation
- [x] CV and nNRM temporal trends, disappeared vs. survived comparison
- [x] Lifespan-normalized dynamics
- [x] Swelling rate/extent dependence on DNA content

### Phase 5: Growth Analysis

- [x] Per-track growth metrics: initial area, peak area, relative max size, growth rate (linear fit)
- [x] `compute_growth_stats()` in tracking module, `add_growth()` pipeline function
- [x] Growth-before-burst visualization: area curves aligned to burst frame, growth rate and max size distributions (disappeared vs survived)

### Phase 6: Advanced Statistics

- [x] **Fluorescence concentration** (`fluor_concentration = total_intensity / volume`): dilution-corrected fluorescence signal, distinguishing true fluorescence loss from dilution by swelling
- [x] **Cell migration speed**: per-frame centroid displacement, per-track mean/max/std speed, total and net displacement
- [x] **SA:V ratio dynamics** (`surface_area / volume`): membrane stress indicator, tracks how surface-to-volume ratio changes as cells swell
- [x] **Spatial clustering of cell death**: nearest-neighbor distance analysis among disappeared cells, permutation test (1000 iterations) comparing observed clustering to random subsets
- [x] **Pre-burst fluorescence behavior**: linear fit of mean_intensity in n-frame window before disappearance, spike detection (positive slope + max exceeds baseline)
- [x] **Growth phase detection**: 2-segment piecewise linear fit minimizing RSS, identifies changepoint frame and slope ratio (acceleration/deceleration)
- [x] Pipeline wiring: `add_fluorescence_concentration()`, `add_migration()`, `add_sav_ratio()`, `add_death_clustering()`, `add_preburst_fluorescence()`, `add_growth_phases()`
- [x] Visualization: 6 new 3-panel plot functions in `plotting.py`
- [x] Exports updated in `__init__.py`
- [x] 40 tests passing across all new features

### Phase 7: Nucleus Persistence Analysis

- [x] Independent fluorescence channel segmentation via `detect_nuclei_stack()` (Cellpose, diameter=25, no inversion needed)
- [x] `run_nucleus_persistence()` pipeline function: frame-by-frame count comparison with automated conclusion
- [x] `plot_nucleus_persistence()` visualization: count overlay + offset stability chart
- [x] **Finding**: DNA does not persist as discrete object after lysis — phase and fluorescence counts decline in parallel (171 vs 169 lost)

### Phase 7 post-review fixes

- [x] Removed unused `fluor_stack` parameter from `run_nucleus_persistence()` — frame count is derived from `label_stack`
- [x] Added shape validation assertion between `label_stack` and `nucleus_label_stack`
- [x] Strengthened conclusion logic: now requires **both** an endpoint test (total loss agreement within tolerance) **and** a trajectory test (offset CV below threshold) to conclude "parallel". Previously only checked endpoints, which could miss divergent mid-trajectory behavior
- [x] Added inline comment in `detect_nuclei_stack()` explaining why no image inversion is needed (fluorescence nuclei already bright on dark, unlike phase-contrast)
- [x] Removed explicit `channels=[0, 0]` from `detect_nuclei_stack()` for consistency with `detect_cells_frame()` (Cellpose defaults to `[0, 0]` for grayscale)
- [x] Moved `detect_nuclei_stack`, `run_nucleus_persistence`, `plot_nucleus_persistence` imports to top-level notebook import cell

### Phase 8: Data Quality Fixes

- [x] **Speed calculation bug fix**: `compute_migration_stats()` now divides displacement by actual frame gap instead of assuming 1-frame intervals. Affects ~0.5% of steps where trackpy memory bridged multi-frame gaps. New test `test_frame_gap_normalizes_speed` validates the fix.
- [x] **Nucleus persistence bad-frame fix**: `run_nucleus_persistence()` now skips frames where phase `label_stack` was zeroed by frame gating but `nucleus_label_stack` was not, preventing spurious offset spikes.
- [x] Documented 3 methodological issues in Open Questions (speed bug, bad-frame fragility, ultra-short tracks)
- [x] Added 3 new Future Work items (minimum-lifetime filter, frame-0 fate prediction, fluorescence early-warning)
- [x] 41 tests passing

### Phase 9: Minimum-Lifetime Filter and Notebook Integration

- [x] **Minimum-lifetime filter**: `filter_short_tracks()` pipeline function removes tracks with <N detections. Default `MIN_TRACK_DETECTIONS=4` in notebook config. Explained in notebook markdown.
- [x] **Notebook wiring for all advanced statistics**: growth analysis (`add_growth`, `plot_growth_before_burst`), growth phases, fluorescence concentration, migration speed, SA:V ratio, death clustering, pre-burst fluorescence — all now called from the notebook with explanatory markdown cells
- [x] Updated notebook imports, config cell, and export section description
- [x] Updated Future Work: minimum-lifetime filter marked as done

### Phase 10: Cell Fate Prediction

- [x] `predict_fate_from_frame0()` in `matching.py`: logistic regression with LOO cross-validation on frame-0 features (area, CV, nNRM). Returns per-cell predictions and summary (AUC, accuracy, feature importance).
- [x] `add_fate_prediction()` pipeline wrapper in `pipeline.py`
- [x] `plot_fate_prediction()` 3-panel visualization: ROC curve, feature importance bar chart, probability distribution by outcome
- [x] Notebook cells with explanatory markdown
- [x] 4 unit tests passing (output structure, AUC > random, custom features, count consistency)
- [x] 45 tests total passing

### Phase 11: Spatial Gradient Analysis

- [x] `analyze_spatial_gradient()` in `matching.py`: Mann-Whitney U test, point-biserial correlation, and logistic regression AUC for each axis (centroid_x, centroid_y). Auto-detects dominant gradient axis, bins into spatial quartiles with per-quartile death rates.
- [x] `add_spatial_gradient()` pipeline wrapper in `pipeline.py`
- [x] `plot_spatial_gradient()` 4-panel visualization: spatial scatter by fate, death rate by quartile, position distribution along gradient axis, per-axis correlation bar chart
- [x] Notebook cells with explanatory markdown (between fate prediction and nucleus persistence)
- [x] 5 unit tests passing (output structure, detects X gradient, quartile rates increase, AUC > random, counts sum)
- [x] 50 tests total passing
- [x] Documented 5 future research directions: spatial gradient analysis, survival analysis (Kaplan-Meier/Cox PH), dynamic trajectory features, critical membrane threshold testing, neighborhood effects beyond gradient

### Reviewer feedback implementation (2026-04-18)

Addressed reviewer comments on phase-contrast and fluorescence analysis:

**Phase Contrast:**
- [x] PC-1: `plot_lifetime_distribution()` now prints median/mean±SD lifetime for disappeared cells only
- [x] PC-2: `plot_swelling_vs_survival()` now prints mean initial volume ± SD split by survived vs disappeared
- PC-3 (swelling dynamics discrepancy): was already documented in notebook heading 6.5 — no action needed

**Fluorescence:**
- FL-1a (Fl per cell vs frame): already implemented — `plot_fluorescence_per_frame` uses per-cell `mean_intensity`, not total population sum
- [x] FL-1b: `plot_nucleus_persistence()` expanded from 2-panel to 3-panel — added scatter of phase cell count vs fluorescence nucleus count with Pearson r and 1:1 reference line
- [x] FL-2: `plot_metric_dynamics()` center panel (lifespan-normalized) now splits by survived vs disappeared instead of showing all tracks combined. Both cohorts shown with independent SEM bands
- [x] FL-3: `detect_fluorescence_disappearance()` now accepts `drop_window` parameter (default 1). With `drop_window=2`, measures cumulative relative drop over 2 consecutive frames instead of single-frame drops. Captures slower fluorescence loss where DNA exits membrane pores more slowly than proteins. Pipeline wrapper `add_fluorescence_disappearance()` passes through. Notebook config sets `FLUOR_DROP_WINDOW = 2`
- [x] FL-4: New `plot_initial_features_vs_lifespan()` — 3-panel scatter of frame-0 fluorescence intensity, CV, and nNRM vs track lifetime, colored by fate (survived/disappeared). Reports Spearman rho + p-value for each. Added as notebook section 8.17

**Files changed:**
- `src/cell_analysis/plotting.py` — PC-1, PC-2, FL-1b, FL-2, FL-4
- `src/cell_analysis/matching.py` — FL-3 (`drop_window` parameter)
- `src/cell_analysis/pipeline.py` — FL-3 (passthrough)
- `src/cell_analysis/__init__.py` — FL-4 (export)
- `notebooks/analysis.ipynb` — all: imports, config, cells, TOC

### Run provenance in HTML reports (2026-05-26)

- [x] Notebook parameters cell now displays a rendered Markdown summary of all input file paths and configuration values as cell output, ensuring the HTML report always contains this provenance info regardless of code cell visibility
- [x] Added `CONFIG_PATH` parameter (defaults to empty string) — when the papermill runner executes, it injects the resolved config YAML path so the HTML shows which config file was used
- [x] Updated `scripts/run_experiment.py` to inject `CONFIG_PATH` into papermill parameters, added to `ALLOWED_KEYS` and `TYPE_RULES`

### Per-plot source CSV attribution (2026-05-30)

Made the notebook and HTML reports clearer by showing which CSV backs every plot.

- [x] `src/cell_analysis/plotting.py` — added `PLOT_SOURCES` registry (plot function name → list of CSV filenames) and `show_with_source(plot_func, *args, **kwargs)` helper that calls the plot then renders a markdown line linking to each source CSV (`_Source: [tracked_cells.csv](tracked_cells.csv)_`). Multi-source plots get `_Sources: ..., ...`. Plots not in the registry (raw-image previews) render no line. Links use bare filenames so they resolve from the HTML report co-located with the CSVs in the results dir.
- [x] `src/cell_analysis/pipeline.py` — added `save_dataframe`, `save_summary_dict`, and `save_main_outputs` thin helpers around `io.save_results` / `io.save_summary`. `save_main_outputs(tracked, track_stats, results_dir)` saves both cumulative DataFrames in one call.
- [x] `notebooks/analysis.ipynb` — every plot cell now uses `show_with_source(plot_X, ...)` instead of calling the plot directly. CSV saves moved out of the end-of-notebook bulk export and into the cell that produces each dataframe: `save_main_outputs` after each `add_*` step that mutates tracked/track_stats; `save_dataframe`/`save_summary_dict` for one-shot outputs (`fate_predictions.csv`, `spatial_gradient.csv`, `nucleus_persistence.csv`, `clustering_summary.csv`, etc.) right when computed. End-of-notebook section now globs `results/<RUN_NAME>/*.csv` and prints the on-disk file list instead of calling `export_all_results`.
- [x] `src/cell_analysis/__init__.py` — exports `PLOT_SOURCES`, `show_with_source`, `save_dataframe`, `save_summary_dict`, `save_main_outputs`.

### Plot data-scope fixes (2026-05-30)

Reviewer flagged that several plots silently hid or subsampled data. Fixed the substantive cases and labelled the rest so the scope is visible on the figure.

- [x] **Pearson r on full data, not subsample.** `plot_fluorescence_vs_volume` (§8.3) and `plot_fluorescence_concentration` right panel (§8.9) used to compute `r` on the 3000-point random scatter subsample. The 3000-point subsample is now kept only for rendering (overplotting workaround); `r` is computed on the full filtered dataset and annotated as `r = X.XXX (n=...)`. Titles now show `scatter: 3000 of N (random)` so the subsampling is visible.
- [x] **`_survival_split` decoupled from frame-0 cohort.** Was silently restricting survival splits to tracks present at frame 0. Now takes `cohort="all"` (default) or `cohort="frame0"`. `plot_relative_fluorescence` (§8.2) passes `"frame0"` because F(t)/F(0) needs a frame-0 baseline; all other split panels (`plot_metric_dynamics` for CV §8.4 and nNRM §8.5, `plot_migration_speed` §8.10, `plot_sav_ratio` §8.11, `plot_fluorescence_concentration` middle §8.9) now correctly include tracks that joined after frame 0.
- [x] **Scope annotated in titles.** Every survival-split panel now appends `(frame-0 cohort)` or `(all tracks)` to its title so the audience can see which population is plotted.
- [x] **Notebook TOC §6.3 fixed.** Said "Area histograms at first and last frame" but `plot_area_distribution` pools across all frames; updated to "Area histogram pooled across all frames".
- [ ] **Not changed (already labelled or intentional):** the 20-trace background curves in `plot_swelling_dynamics` and `plot_relative_fluorescence` and the 8 example tracks in `plot_growth_phases` — these are cosmetic; the headline mean ± SEM uses the full cohort, and the count is already in the legend/title.

### Frame-0 fate Mann-Whitney is now reproducible from the notebook (2026-05-30)

Section 8.14's markdown cited Mann-Whitney p-values (area p=0.0008, CV p<0.0001, nNRM p<0.0001) that were not computed by any cell in the report — they were a hand-written reference to a one-off scratch analysis. Reader couldn't trace where the numbers came from.

- [x] `src/cell_analysis/matching.py` — added `compare_frame0_features_by_fate(tracked, track_stats, features=None)`. Mann-Whitney U (two-sided) on each feature, survived vs disappeared, on the frame-0 cohort. Returns a DataFrame with `feature, n_survived, n_died, median_survived, median_died, U, p_value`.
- [x] `src/cell_analysis/pipeline.py` — added `add_frame0_fate_comparison()` wrapper that prints a small table and returns the DataFrame.
- [x] `src/cell_analysis/__init__.py` — exports the wrapper.
- [x] `notebooks/analysis.ipynb` — new code cell between the §8.14 markdown and the fate-prediction code cell. Saves `frame0_fate_comparison.csv`. Replaced the §8.14 markdown: removed the orphan hard-coded p-values; added explicit links to the new code cell, to `predict_fate_from_frame0()`/`plot_fate_prediction()` in the source tree, and to §8.17 (Spearman) / §6.5 (initial volume) / §8.15 (position) as related views. Back-link added from §8.17 to §8.14. TOC row and Export Results table updated.
- [x] `tests/test_fate_prediction.py` — 3 new tests (output structure, cohort counts match, custom features). All 7 tests pass.

### Notebook STACK_PATH / FLUOR_PATH fix (2026-05-30)

`notebooks/analysis.ipynb` pointed at `data/gradient_0011/phase.tif` / `fluorescence.tif`, which don't exist — the actual files are `Gradient-0011.zvi  Ch0.tif` / `Ch1-BG.tif` (as `configs/run_01.yaml` already uses). Notebook raised `FileNotFoundError` on the first load cell.

- [x] `notebooks/analysis.ipynb` — updated `STACK_PATH` and `FLUOR_PATH` in the Configuration cell to the actual filenames. Verified `load_experiment()` succeeds: `(25, 1040, 1388)` uint16 for both stacks.

### Force CPU on Intel Macs to avoid MPS bfloat16 crash (2026-05-30)

Cellpose 4.x's SAM backbone runs ops in bfloat16. On 2019 Intel MacBook Pros with a discrete AMD GPU, MPS is detected as available but lacks bfloat16 support, so `CellposeModel(gpu=True)` aborts with `RuntimeError: BFloat16 is not supported on MPS`. CPU works fine; Apple-Silicon MPS supports bfloat16 in `torch>=2.5` and stays on the GPU path.

- [x] `src/cell_analysis/segmentation.py` — added `_resolve_gpu(gpu)` helper that returns `False` on Darwin x86_64 (with a `warnings.warn`), otherwise passes the requested value through. Routed the three `CellposeModel(gpu=...)` constructors in `detect_cells_frame`, `detect_cells_stack`, and `detect_nuclei_stack` through it.

### Single-source HTML export shared by notebook and CLI (2026-06-05)

Notebook runs in JupyterLab no longer require a separate `scripts/run_experiment.py` invocation to produce a `report.html`. Interactive runs and the CLI now share one exporter so the two outputs are byte-identical for the same executed notebook.

- [x] `src/cell_analysis/io.py` — added `export_notebook_html(notebook_path, output_path)`. Uses `HTMLExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)` — same config that was inline in the CLI runner.
- [x] `src/cell_analysis/__init__.py` — exports the helper.
- [x] `scripts/run_experiment.py` — replaced the inline `HTMLExporter` block with a call to `export_notebook_html(tmp_path, results_dir / "report.html")`. Also injects `EXPORT_HTML=False` into the papermill parameter overrides so the notebook's in-cell export skips during CLI runs (CLI does the export itself after papermill completes, against the executed tmp notebook).
- [x] `notebooks/analysis.ipynb` — added `EXPORT_HTML = True` to the parameters cell (default for interactive use). Added a new §10 "Export HTML Report" markdown cell + code cell that triggers a best-effort JupyterLab save via JS, sleeps ~2 s, calls `export_notebook_html("analysis.ipynb", RESULTS_DIR / "report.html")`, and displays a markdown link to the output.

### MODEL_TYPE parameter and per-frame profiling (2026-06-06)

Made it possible to swap the Cellpose model (e.g. SAM-based `cpsam` ↔ classic `cyto3`) without code edits, and added a one-frame profiling cell so users on CPU-only hardware (Intel Mac) can estimate full-stack runtime before committing to it. Motivated by reports of multi-hour runs on a 2019 MacBook Pro.

- [x] `src/cell_analysis/segmentation.py` — added `_make_cellpose_model(gpu, model_type)` helper that funnels every `CellposeModel(...)` construction through one place (`pretrained_model=model_type` when set). Threaded a new `model_type: str | None = None` kwarg through `detect_cells_frame`, `detect_cells_stack`, and `detect_nuclei_stack`. Added `profile_detection(stack, model_type=None, nucleus_pass=True, **detect_params)` that loads the model once (excluded from the timing), runs one inference on frame 0, and prints `sec_per_frame` + projected minutes for the phase-only and phase+nucleus passes.
- [x] `src/cell_analysis/__init__.py` — exports `profile_detection`.
- [x] `scripts/run_experiment.py` — `MODEL_TYPE` added to `ALLOWED_KEYS` and `TYPE_RULES` so YAML configs can pin the model per-run.
- [x] `notebooks/analysis.ipynb` — new `MODEL_TYPE = None` parameter (default keeps current behavior). Run-info table now shows it. `detect_cells_frame` (§2), `detect_cells_stack` (§4), and `detect_nuclei_stack` (§8.16) all now forward `model_type=MODEL_TYPE`. The analysis notebook itself stays free of profiling code.
- [x] `notebooks/perf_check.ipynb` — new standalone notebook with seven sections: (1) system info (platform, RAM, CPU/GPU resolution, cellpose version) so users see immediately whether their `gpu=True` got downgraded to CPU; (2) phase detection timing per `MODELS_TO_TEST` entry via `profile_detection(..., nucleus_pass=False)`; (3) nucleus detection timing via the new `profile_nucleus_detection(...)` helper; (4) process RSS vs total RAM with a swap-thrash warning above 75% utilization; (5) frame-0 cell-count + overlay comparison; (6) summary table with phase s/f, nucleus s/f, projected total minutes, cell count, and delta vs baseline; (7) shareable markdown report saved to `results/perf_check.md` and rendered inline (host, date, platform, CPU/RAM, cellpose version, stack shape, memory footprint, model comparison table) so the user can email it or paste it without re-running the notebook. Use once per machine / cellpose version bump.
- [x] `src/cell_analysis/segmentation.py` — added `profile_nucleus_detection(fluor_stack, model_type=None, diameter=25, gpu=True, resample=False)` that mirrors `profile_detection` but for the fluorescence channel (no inversion). Confirms or refutes the "nucleus pass ≈ phase pass" assumption that `profile_detection` uses for its projection.
- [x] `src/cell_analysis/__init__.py` — exports `profile_nucleus_detection`.
- [x] `profile_detection` / `profile_nucleus_detection` — added flushed progress prints (`Loading model...`, `Running inference (cpsam on CPU is 5-15 min/frame at full res)...`) so users don't think the cell is hung. Cellpose's own logging is suppressed in `segmentation.py`, which was hiding the ~2 GB cpsam download.
- [x] `notebooks/perf_check.ipynb` — `MODELS_TO_TEST` defaults to `["cyto3"]` only. Added a section-2 note that cpsam on CPU is 5-15 min/frame plus a one-time ~2 GB download, so first-run users on Intel Macs aren't surprised by a stalled-looking cell.

### Analysis pipeline Pass A: µm scaling, baseline window, normalized-lifespan plots, drop unused analyses (2026-06-08)

Reviewer feedback split into two passes; Pass A covers the mechanical changes (items 1–4 and 6–10 of the 12-item list). Items 5, 11, 12 are deferred to Pass B. Spec: [`docs/superpowers/specs/2026-06-07-analysis-pass-a-design.md`](superpowers/specs/2026-06-07-analysis-pass-a-design.md). Plan: [`docs/superpowers/plans/2026-06-07-analysis-pass-a.md`](superpowers/plans/2026-06-07-analysis-pass-a.md).

- [x] **A1 — Uniform µm scaling.** `add_geometry(tracked, track_stats, pixel_size_um=1.0)` now multiplies `tracked["area"]` by `pixel_size_um**2` before deriving radius/volume/surface_area. Default 1.0 preserves raw-pixel behavior; the notebook passes `PIXEL_SIZE_UM = 0.0645`. Frame gating (`detect_bad_frames`) is unaffected because it runs before `add_geometry` and operates on raw pixel area. Fate prediction uses `StandardScaler` on its three features, so the unit change is absorbed and AUC/coefficients are invariant. Smoke test confirms median area is now ~4 µm² (was ~875 px²).
- [x] **A1 — Axis labels.** Updated all µm-bearing plots: `plot_area_distribution` (Area µm²), `plot_swelling_vs_survival` (V(0) µm³ + print precision bumped to `.2f`), `plot_fluorescence_vs_volume` (Volume µm³), `plot_sav_ratio` (SA/V µm⁻¹), `plot_fluorescence_concentration` (F_total / Volume in a.u./µm³ on three panels + xaxis Volume µm³).
- [x] **A2 — Baseline area window.** `plot_area_distribution(tracked, baseline_frames=3)` filters to the first N frames. Title updates dynamically ("first 1 frame pooled" / "first 3 frames pooled"). Notebook surfaces `BASELINE_FRAMES = 3` as a top-level config knob; set to 1 for frame-0-only.
- [x] **A3 — Relative-per-track columns.** New `_relative_per_track(df, col)` helper in `pipeline.py`. `add_geometry` populates `volume_rel = V(t)/V(0)`; `add_fluorescence` populates `fluor_rel = mean_intensity(t)/mean_intensity(0)`. NaN where the per-track baseline ≤ 0. Both feed two new notebook sub-steps using the existing `plot_metric_dynamics` three-panel layout:
  - §6.5b "Relative swelling V(t)/V(0) vs. normalized lifespan, by fate"
  - §8.2b "Relative fluorescence F(t)/F(0) vs. normalized lifespan, by fate"
- [x] **A4 — Full deletion of unused analyses.** Removed source code, exports, tests, notebook cells, and (where applicable) TOC and CSV-registry entries for:
  - §8.7 Growth analysis (`add_growth`, `compute_growth_stats`, `plot_growth_before_burst`)
  - §8.8 Growth phase detection (`add_growth_phases`, `detect_growth_phases`, `plot_growth_phases`)
  - §8.10 Cell migration speed (`add_migration`, `compute_migration_stats`, `plot_migration_speed`)
  - §8.12 Spatial clustering of cell death (`add_death_clustering`, `compute_death_clustering`, `plot_death_clustering`)
  - Spatial gradient (orphaned helper not wired into the current notebook): `add_spatial_gradient`, `analyze_spatial_gradient`, `plot_spatial_gradient`
- [x] **Notebook surgery.** Added `PIXEL_SIZE_UM = 0.0645` and `BASELINE_FRAMES = 3` to the Configuration cell; surfaced both in the run-info markdown table. Inserted §6.5b and §8.2b cells. Removed the four deleted sections (markdown header + code cell each). Updated TOC and §9 CSV registry table to match. Cell count went from 74 → 70. Also fixed a pre-existing Python-3.11 incompatibility in the Configuration cell's f-string (nested `"default"` inside a `"`-delimited f-string broke under PEP 701-prior parsing; extracted `_model_label = MODEL_TYPE or 'default'`).
- [x] **CLI.** `scripts/run_experiment.py` accepts `PIXEL_SIZE_UM: (int, float)` and `BASELINE_FRAMES: int` in YAML configs via `ALLOWED_KEYS` + `TYPE_RULES`.
- [x] **Tests.** Added `tests/test_geometry_scaling.py` (4 tests: pixel scaling + volume_rel) and `tests/test_fluor_rel.py` (2 tests on the helper directly). Removed `test_growth_phases.py`, `test_migration_speed.py`, `test_spatial_clustering.py`, `test_spatial_gradient.py`. All remaining tests pass (43 total).
- [x] **End-to-end smoke test.** Ran `scripts/run_experiment.py configs/run_01.yaml` to completion (71 cells, ~6 min on CPU). Verified: `tracked_cells.csv` now has `volume_rel` and `fluor_rel`, no `speed`; `track_statistics.csv` no longer has growth/migration/clustering/gradient columns; `report.html` renders.

**Deferred to Pass B:**
- Item 5: ±3-frame fluorescence-disappearance window aligned to phase disappearance (replaces §8.6's single-frame detection).
- Item 11: per-class fate-prediction diagnostics (the user reported a 0.77 vs 0.51 asymmetry that suggests they computed per-class precision/recall rather than the symmetric LR AUC — needs clarification before implementation).
- Item 12: nucleoid edge-distance / Gaussian-sigma metric for nucleoid spatial distribution; potential additional fate-prediction feature.

### Analysis pipeline Pass B: ±window fluorescence alignment + nucleoid spatial distribution (2026-06-08)

Items 5 and 12 of the reviewer feedback. Item 11 still deferred (asymmetric 0.77/0.51 numbers need user clarification). Spec: [`docs/superpowers/specs/2026-06-08-analysis-pass-b-design.md`](superpowers/specs/2026-06-08-analysis-pass-b-design.md). Plan: [`docs/superpowers/plans/2026-06-08-analysis-pass-b.md`](superpowers/plans/2026-06-08-analysis-pass-b.md).

- [x] **Item 5 — §8.6 replaced.** Old single-frame fluor-drop classifier (`add_fluorescence_disappearance`, `plot_fluorescence_disappearance`, `detect_fluorescence_disappearance`) deleted from `pipeline.py` / `plotting.py` / `matching.py`. New helpers: `measure_post_disappearance_fluorescence(track_stats, last_labels, fluor_stack, label_stack, window)` in `matching.py` integrates fluor within the cell's last-known phase mask at frames `last_frame + 1..window`. `add_fluorescence_alignment(tracked, track_stats, fluor_stack, label_stack, window=3)` in `pipeline.py` builds a long-form rectangular table per disappeared cell with columns `track_id, offset, mean_intensity, F_norm` where offsets cover `[-window, +window]` and `F_norm = mean_intensity(offset) / mean_intensity(-window)`. Tracks lacking the offset=-window baseline are dropped (no normalization possible). `plot_fluorescence_alignment` renders per-cell traces + population mean ± SEM + dashed vline at offset=0 ("phase mask vanishes"). New CSV `fluorescence_alignment.csv` in the results dir.
- [x] **Item 12 — Nucleoid spatial distribution.** `add_nucleoid_distribution(tracked, track_stats, fluor_stack, label_stack, flat_threshold_ratio=1.5)` in `pipeline.py` adds two per-cell-per-frame columns to `tracked`:
  - `mean_edge_distance_norm` = mean distance of suprathreshold pixels to the cell-mask edge (via `scipy.ndimage.distance_transform_edt`) divided by R/3 where `R = sqrt(area_pixels/π)`. The expected value for points uniformly distributed inside a circle of radius R is R/3, so this ratio is 1.0 for uniform fluorescence, < 1 for "horseshoe" peripheral distributions (control spheroplasts), > 1 for compacted central clusters (CAM cells).
  - `gaussian_sigma_norm` = σ of the 2D intensity-weighted spatial distribution (mean of eigenvalues of the weighted covariance via `np.linalg.eigvalsh`) divided by R. Larger σ → more dispersed.
  - Threshold rule: pixels > mean(cell), with fallback to > 0.5×mean if max/mean < 1.5 (distribution already flat). Two new notebook sub-steps `§8.4c` (edge-distance dynamics) and `§8.5b` (σ dynamics) reuse the existing `plot_metric_dynamics` three-panel layout.
- [x] **§8.14 fate prediction expanded to 5 features.** `compare_frame0_features_by_fate` and `predict_fate_from_frame0` defaults now include `mean_edge_distance_norm` and `gaussian_sigma_norm` alongside `area`, `cv`, `nnrm`. Mann-Whitney table grows to 5 rows; LR reports 5 z-scored coefficients. On `run_01`: AUC was 0.672 with 3 features, 0.665 with 5 (the new features carry small but real signal — sigma coefficient -0.27, edge-distance -0.07; nnrm still dominates at +0.61). Updated `tests/test_fate_prediction.py` fixture to include the new columns and the `test_compare_output_structure` assertion to expect 5 features.
- [x] **Configuration.** Removed `FLUOR_DROP_THRESHOLD` and `FLUOR_DROP_WINDOW`; added `FLUOR_ALIGN_WINDOW = 3` (notebook config + `scripts/run_experiment.py` `ALLOWED_KEYS`/`TYPE_RULES` + `tests/test_run_experiment.py` fixture + all three `configs/*.yaml` files).
- [x] **Notebook surgery.** Cell count went 70 → 75 (+5 new cells for `add_nucleoid_distribution` call + §8.4c md/code + §8.5b md/code; §8.6 markdown + code cells replaced in place). TOC updated for §8.4c, §8.5b, §8.6 (new description), §8.14 (lists 5 features). §9 CSV registry adds `fluorescence_alignment.csv`. Imports cell swaps `add_fluorescence_disappearance` / `plot_fluorescence_disappearance` for `add_fluorescence_alignment` / `plot_fluorescence_alignment` / `add_nucleoid_distribution`.
- [x] **Tests.** Added `tests/test_post_disappearance_fluorescence.py` (2 tests), `tests/test_fluorescence_alignment.py` (3 tests), `tests/test_nucleoid_distribution.py` (4 tests). 6 net new tests; full suite (excluding parallel-zone files) passes.
- [x] **Smoke test.** Papermill ran `configs/run_01.yaml` end-to-end (~12 minutes, 76 cells executed). Verified: `tracked_cells.csv` now has `mean_edge_distance_norm` and `gaussian_sigma_norm`; `track_statistics.csv` has `mean_edge_distance_norm` and `mean_gaussian_sigma_norm` (and no longer has `fluor_disappearance_frame` or `max_drop`); `fluorescence_alignment.csv` exists with correct columns + F_norm = 1.0 at offset=-3 by construction; §8.14 fate-prediction summary reports 5 features.
- [x] **Pre-existing Python 3.11 f-string** in Configuration cell (carryover from Pass A's fix using `_model_label = MODEL_TYPE or 'default'`) continues to work.

**Still deferred (Item 11):** Per-class fate-prediction diagnostics. The user reported AUC asymmetry (0.77 for survival vs 0.51 for disappearance) that's incompatible with the LR's symmetric AUC. Likely they computed per-class precision/recall or used a different evaluation scheme. Needs user clarification on what produced those numbers before implementing.

### Analysis pipeline Pass C: per-class fate diagnostics + area-exclusion comparison (2026-06-08)

Item 11 of the reviewer feedback. Reviewer clarified the asymmetry came from manual per-class recall: 150/195 = 0.77 for "predicted survived among true survivors", 84/165 = 0.51 for "predicted disappeared among true disappeared". They also suggested trying the LR without `area` ("obviously a weak criterion and is the same for control and CAM cells"). Spec: [`docs/superpowers/specs/2026-06-08-analysis-pass-c-design.md`](superpowers/specs/2026-06-08-analysis-pass-c-design.md). Plan: [`docs/superpowers/plans/2026-06-08-analysis-pass-c.md`](superpowers/plans/2026-06-08-analysis-pass-c.md).

- [x] **Per-class diagnostics.** `predict_fate_from_frame0()` summary now includes `confusion_matrix` (TN/FP/FN/TP, positive class = disappeared) and `per_class` (precision/recall/F1/support for each label) — computed via `sklearn.metrics.confusion_matrix` and `sklearn.metrics.precision_recall_fscore_support`. `add_fate_prediction()` prints these alongside the existing AUC/accuracy block.
- [x] **Feature-set comparison.** Notebook Configuration adds `FATE_FEATURES_FULL` (5 features) and `FATE_FEATURES_NO_AREA` (4 features, drops `area`). §8.14 calls `add_fate_prediction()` twice; results saved to `fate_predictions{,_no_area}.csv` and `fate_prediction_summary{,_no_area}.csv`. New helper `print_fate_comparison(summary_full, summary_no_area, f1_threshold=0.02)` prints a side-by-side table of AUC, accuracy, per-class recall, and per-class F1, plus a one-line conclusion (both per-class F1 deltas ≥ 0.02 ⇒ "consider permanently swapping"; otherwise "differences small").
- [x] **Tests.** Added 3 tests to `tests/test_fate_prediction.py`: `test_summary_includes_per_class_metrics`, `test_confusion_matrix_totals_match_cohort`, `test_print_fate_comparison_uses_both_summaries`. Total: 56 → 59 tests, all passing.
- [x] **Smoke test on `run_01`.** Full LR per-class recall reproduces the reviewer's 0.77 / 0.51 numbers (recall_survived = 151/195 = 0.774; recall_disappeared = 85/165 = 0.515 — off by one cell from the reviewer's 84/165 = 0.509, attributable to threshold ties or a slightly different cohort cut). No-area LR result: AUC 0.658 (vs 0.665 full), recall(survived) 0.800 (+2.6pp), recall(disappeared) 0.497 (−1.8pp). Mixed outcome — the heuristic correctly prints "differences small; full feature set remains the default" since neither F1 delta crosses the 0.02 threshold.
- [x] **Reviewer's 12-item feedback list now fully implemented.** Pass A covered items 1–4 and 6–10; Pass B covered items 5 and 12; Pass C covered item 11. The per-class metrics + comparison surface the asymmetry the reviewer flagged in every future run, and the no-area variant is one Configuration-cell line swap away if a future dataset warrants it.


### Peri/core asymmetry — narrower rings for real intensity profiles (2026-07-15)

The original `peri_core_asymmetry` split the mask into equal-area rings at `r = R/√2` (≈ 0.707·R). On real cell intensity profiles (donut, expanded, compacted) the inner disk that wide swallows both donut peaks and the central dip together — the three functional states end up with similar (mildly-negative) values.

- **New rings.** Narrow inner disk `r < R/6`, peripheral annulus `R/2 < r < 5R/6`. The intermediate band (R/6 ≤ r ≤ R/2) and the rim (r ≥ 5R/6) are ignored so the two regions sample the diagnostic zones of the profile. Areas are unequal; mean rather than total intensity is used (the code already averaged per ring, so only the ring definitions changed).
- **Files.** `_peri_core_asymmetry` in `src/cell_analysis/matching.py`; docstring for `peri_core_asymmetry` in `src/cell_analysis/pipeline.py::add_nucleoid_distribution`; notebook §8.5c markdown.
- **Tests.** `tests/test_nucleoid_distribution.py::test_peri_core_asymmetry_peripheral_positive` fixture updated: bright band moved from `r > 22` to `r > 15` so it overlaps the new peri ring (R/2..5R/6 = 12.5..20.83 for R=25). Uniform-is-zero and central-cluster-negative pass unchanged (the narrower core is almost entirely inside the r<4 bright disk, giving a stronger negative signal).
- **Column name preserved.** `peri_core_asymmetry` / `mean_peri_core_asymmetry` — no downstream CSV or plot schema changes; values on a rerun will differ.
- **Not changed.** Sign convention (positive = edge-clustered), NaN guards on empty rings and non-positive total, geometric-centroid basis, plot code in §8.5c.


### Peri/core asymmetry — parametrized + retuned to bracket donut peak (2026-07-16)

The `(0.167, 0.5, 0.833)` rings from 2026-07-15 still put the peri annulus on the
descending shoulder of the average radial profile, not on the donut peak. On the
shipped 3-run dataset only 0.9% of control cell-frames registered positive
asymmetry despite ~13% of tracks showing donut morphology in per-cell profile
inspection.

- **Data-driven retune.** Per-cell radial fluorescence profiles (extracted via
  centroid + `radius` from `tracked_cells.csv`, so no re-segmentation was needed)
  showed donut cells have a center dip at r/R < 0.2 and a peak at r/R ≈ 0.5. A
  sweep over 25 (core_max, peri_min, peri_max) triples identified
  `(0.15, 0.35, 0.55)` as the best balance: control positive fraction goes 0.9%
  → 11.4%, control median asymmetry goes −0.22 → −0.06, CAM stays clearly
  compacted (−0.18) and still shifts ~2× toward zero over time (−0.18 → −0.09).
- **Parametrized.** `_peri_core_asymmetry` now takes a
  `rings=(core_max, peri_min, peri_max)` triple; `measure_nucleoid_metrics` and
  `add_nucleoid_distribution` accept `peri_core_rings=`; notebook Configuration
  cell exposes `PERI_CORE_RINGS = (0.15, 0.35, 0.55)`; `scripts/run_experiment.py`
  `ALLOWED_KEYS` + `TYPE_RULES` accept `PERI_CORE_RINGS` in the YAML configs.
  Switching back to the historical rings is a single-line config change.
- **Documented.** [`docs/peri_core_asymmetry_definition_history.md`](peri_core_asymmetry_definition_history.md)
  records both prior definitions (equal-area and narrow-disk/wide-annulus), the
  sweep behind the new default, and one-line restore recipes.
- **Column name preserved.** `peri_core_asymmetry` / `mean_peri_core_asymmetry`
  keep their names; a rerun will overwrite the CSVs with the new values. Users
  who want the wide/narrow score alongside the new default can call
  `add_nucleoid_distribution(..., peri_core_rings=(0.167, 0.5, 0.833))` and rename
  the returned column.
- **Tests.** `test_peri_core_asymmetry_peripheral_positive` fixture adapted to
  new peri ring (bright band at r > 8 for R=25). New
  `test_peri_core_asymmetry_respects_ring_parameter` verifies the kwarg is
  actually plumbed through — using historical rings on a fixture where only the
  outer annulus is bright reproduces a strongly positive value that goes to 0
  under the new default. Suite: 62 → 63 tests, all passing.
- **Not yet.** Rerun of the 3 configs to regenerate CSVs + reports with new
  values.


### Peri/core asymmetry — alignment to disappearance (§8.5d) (2026-07-16)

Absolute-frame medians smear the signal: cells caught at frame N are at
different stages of their own life cycle, so averaging across them mixes
donut-stage and pre-burst-stage cells. Added a per-cell alignment view
analogous to §8.6 fluorescence alignment, but for `peri_core_asymmetry`.

- **New function.** `pipeline.add_peri_core_alignment(tracked, track_stats,
  window)` returns long-form DataFrame with `track_id, offset, peri_core_asymmetry,
  delta_from_baseline` for each disappeared cell, where offset =
  `frame - last_frame` ∈ [−window, 0]. Skips tracks whose offset=-window
  observation is missing (short lifespan, no baseline).
- **Plot.** `plotting.plot_peri_core_alignment` — two-panel: (a) per-cell
  trajectories with median + IQR band, offset 0 marked; (b) baseline
  (offset −window) vs. last-frame (offset 0) scatter, one dot per cell,
  diagonal reference. Reveals convergence/divergence of initial phenotypes.
- **Notebook.** New §8.5d cell after §8.5c; `PLOT_SOURCES` registry updated
  (`peri_core_alignment.csv`); §9 CSV registry row added; imports cell
  brings in `add_peri_core_alignment` / `plot_peri_core_alignment`.
- **Tests.** New `tests/test_peri_core_alignment.py` (3 tests): only
  disappeared tracks with baseline appear, offset grid + delta correctness,
  empty-when-no-disappeared. Suite: 63 → 66 tests, all passing.
- **CSV.** `peri_core_alignment.csv` shipped per-run.
- **Reruns.** All 3 configs regenerated with the new §8.5d panel.

Per-cell aligned trajectory (median across all disappeared tracks with
`window=3` baseline):

| condition                    | n   | offset −3 | offset −2 | offset −1 | offset 0 | net Δ |
|------------------------------|-----|-----------|-----------|-----------|----------|-------|
| control_experiment           | 196 | −0.064    | −0.066    | −0.072    | −0.069   | −0.009 |
| protein_synthesis_arrested   | 282 | −0.142    | −0.141    | −0.131    | −0.108   | **+0.032** |
| run_01                       | 166 | −0.077    | −0.076    | −0.077    | −0.071   | +0.005 |

CAM shows a clear per-cell "compact → sparse before blow" trajectory of
+0.032 in the last 3 frames — a signal that was drowned by absolute-frame
averaging in §8.5c. Control is flat on average, but 20 → 8 baseline-positive
donut cells convert away from donut before disappearance (10% → 4%),
showing a real subpopulation transformation despite the overall median
being stable.

### Extraction/analysis notebook split (2026-07-29)

Monolithic `analysis.ipynb` split into two papermill notebooks to enable fast
iteration on visualizations and statistics without re-running expensive
Cellpose segmentation and tracking.

- **Two-notebook architecture.** `notebooks/extract.ipynb` (cells 1–15, ~12
  min) runs Cellpose on phase and fluorescence channels, tracks cells, detects
  bad frames, and emits extraction artifacts to `results/<RUN>/extraction/`:
  `provenance.json` (extraction params + file hashes), `label_stack.npz`,
  `nucleus_label_stack.npz`, `tracked_cells.csv`, `track_statistics.csv`,
  `frame_diagnostics.csv`, `merge_log.csv`, `dropped_frames.csv`.
  `notebooks/analysis.ipynb` (cells 1–75, ~6 min) loads the extraction
  artifacts, computes all derived metrics (geometry, fluorescence, nucleoid
  distribution, fate prediction), generates all plots, and saves to
  `results/<RUN>/analysis/` (15 CSVs + `report.html`).
- **Extraction reuse.** Runner (`scripts/run_experiment.py`) checks
  `provenance.json` hash (sha256 of extraction params + phase/fluor file
  sha256s) before extracting. If params and sources match, skip extraction and
  reuse cached artifacts — ~12 min saved. New CLI flags: `--force-extract`
  (re-segment/re-track regardless), `--skip-analysis` (extract only),
  `--analysis-only` (load extraction, skip extract step).
- **Config schema.** All three YAML configs (`control_experiment.yaml`,
  `protein_synthesis_arrested.yaml`, `run_01.yaml`) migrated to two-section
  schema with top-level `extraction:` and `analysis:` keys; runner enforces
  parameter names per section (no namespace pollution). `RESULTS_ROOT` can be
  overridden in CLI (`--results-root PATH`) or YAML (`results_root: /absolute/path`
  in top level); CLI > YAML > default.
- **Code reuse.** New `src/cell_analysis/io.py` module: `save_extraction`
  (write 8 artifacts), `load_extraction` (read + validate + return
  `ExtractionBundle` dataclass), `compute_provenance` (sha256 of params + file
  hashes), `provenance_matches` (returns bool, list of drifted field names).
  `ExtractionBundle` fields: `label_stack`, `nucleus_label_stack`,
  `tracked_cells`, `track_statistics`, `frame_diagnostics`, `merge_log`,
  `dropped_frames`, `provenance`. All existing test suites (`test_geometry_scaling`,
  `test_fate_prediction`, `test_nucleoid_distribution`, etc.) remain green—they
  exercise `pipeline.py` directly without hitting `io.py`.
- **Files.** New: `src/cell_analysis/io.py`, `notebooks/extract.ipynb`,
  updated `notebooks/analysis.ipynb`, `scripts/run_experiment.py` (new flags
  + runner logic), three YAML configs, `tests/test_extraction_io.py`,
  `tests/test_provenance.py`, `tests/test_run_experiment.py` (new tests),
  `tests/test_runner_reuse.py` (new test fixture).
- **Legacy folder.** Renamed any pre-split `results/` folder to
  `results_legacy_20260729` to preserve old-layout artifacts for user review
  without cluttering the new `results/<RUN>/{extraction,analysis}/` structure.
- **Design / plan.** [`docs/superpowers/specs/2026-07-28-extract-analysis-split-design.md`](superpowers/specs/2026-07-28-extract-analysis-split-design.md),
  [`docs/superpowers/plans/2026-07-28-extract-analysis-split.md`](superpowers/plans/2026-07-28-extract-analysis-split.md).

### Cross-machine extract/analyze workflow (2026-07-30)

Enable running extraction on one machine (workstation with GPU-backed
Cellpose) and analysis on another (laptop) without either machine
needing the other's filesystem layout.

- **`--fluor-root PATH` CLI flag** (`scripts/run_experiment.py`,
  repeatable). Fallback directories for raw TIFFs when the path
  recorded in `provenance.json` doesn't resolve on the current machine.
  Also honors env var `EXPERIMENTS_IMAGE_FLUOR_ROOTS` (`os.pathsep`-
  separated). Passed to the analysis notebook as papermill parameter
  `FLUOR_ROOTS`.
- **Resolution order in `load_extraction_with_stacks`
  (`src/cell_analysis/io.py`).** (1) Recorded path, resolved against
  `repo_root` when relative. (2) `{root}/{basename}` for each entry in
  the merged `fluor_roots` list. (3) Fail on total miss for
  `fluor.tif` with an error listing every path tried. For
  `phase.tif`, a total miss emits a `RuntimeWarning` and returns
  `None` — analysis proceeds and only skips `plot_channels_preview`.
- **`--analysis-only` no longer crashes when raw stacks aren't
  locally reachable.** Drift check catches `FileNotFoundError` from
  `_sha256_file` and prints a note ("raw stacks not locally reachable;
  skipping extraction-drift check") instead. Machine 2 can run analysis
  without a copy of the raw stack.
- **Version-drift warning.** Analysis notebook's load cell compares
  the local `cell_analysis` version against
  `provenance["cell_analysis_version"]` and prints a warning if they
  differ.
- **Files.** Updated: `src/cell_analysis/io.py`,
  `scripts/run_experiment.py`, `notebooks/analysis.ipynb`. New:
  `tests/test_cross_machine_paths.py` (7 tests covering recorded-path
  happy path, basename fallback, env-var equivalence, multi-entry env,
  arg-list priority, error message contents, phase-missing warning).
- **Design.** `docs/superpowers/specs/2026-07-28-extract-analysis-split-design.md`
  §"Cross-machine workflow".
