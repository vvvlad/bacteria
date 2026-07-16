"""High-level pipeline orchestration for the cell analysis workflow.

Each function chains lower-level modules into a single call, printing
summary statistics so the notebook stays minimal.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from .io import save_results, save_summary


def save_dataframe(df, results_dir, filename):
    """Save *df* as CSV at ``results_dir/filename``."""
    save_results(df, Path(results_dir) / filename)


def save_summary_dict(data, results_dir, filename):
    """Save flattened *data* dict as single-row CSV at ``results_dir/filename``."""
    save_summary(data, Path(results_dir) / filename)


def save_main_outputs(tracked, track_stats, results_dir):
    """Save the cumulative *tracked* and *track_stats* DataFrames.

    Called repeatedly through the notebook to keep the on-disk CSVs in
    sync with in-memory columns added by each ``add_*`` step.
    """
    save_dataframe(tracked, results_dir, "tracked_cells.csv")
    save_dataframe(track_stats, results_dir, "track_statistics.csv")


def load_experiment(phase_path, fluor_path):
    """Load paired phase-contrast and fluorescence stacks.

    Returns (phase_stack, fluor_stack) as (T, Y, X) arrays.
    Validates that both stacks have matching dimensions.
    """
    from .io import load_paired_stacks

    phase, fluor = load_paired_stacks(phase_path, fluor_path)
    T, H, W = phase.shape
    print(f"Phase stack: {phase.shape} "
          f"(frames={T}, size={W}x{H}, dtype={phase.dtype})")
    print(f"Fluorescence stack: {fluor.shape}, dtype={fluor.dtype}")
    return phase, fluor


def run_frame_gating(label_stack, z_threshold=3.5, results_dir=None):
    """Run frame quality gating on detection results.

    Flags anomalous frames using MAD-based Z-scores, removes them from
    detections, and zeroes their label masks (preventing fluorescence
    measurement on bad frames).

    Modifies *label_stack* in-place.

    Returns (detections, bad_frames, diagnostics).
    """
    from .tracking import labels_to_detections, detect_bad_frames

    detections_raw = labels_to_detections(label_stack)
    bad_frames, diagnostics = detect_bad_frames(
        detections_raw, z_threshold=z_threshold,
    )

    detections = detections_raw[~detections_raw["frame"].isin(bad_frames)].copy()

    for bf in bad_frames:
        label_stack[bf] = 0

    if results_dir is not None and len(bad_frames) > 0:
        dropped = diagnostics[diagnostics["flagged"]]
        save_results(dropped, Path(results_dir) / "dropped_frames.csv")

    return detections, bad_frames, diagnostics


def run_tracking(detections, search_range=30.0, memory=3,
                 merge_max_distance=15.0, merge_max_gap=18):
    """Link detections into tracks, merge fragments, compute statistics.

    Returns (tracked, track_stats, merge_log).
    """
    from .tracking import track_cells, merge_fragmented_tracks, compute_track_stats

    print(f"Total detections: {len(detections)}")

    tracked = track_cells(
        detections, search_range=search_range, memory=memory,
    )
    print(f"Unique tracks (before merging): {tracked['track_id'].nunique()}")

    tracked, merge_log = merge_fragmented_tracks(
        tracked,
        max_distance=merge_max_distance,
        max_gap=merge_max_gap,
    )
    print(f"Unique tracks (after merging): {tracked['track_id'].nunique()}")

    track_stats = compute_track_stats(tracked)
    print(f"Cells that disappeared before last frame: "
          f"{track_stats['disappeared'].sum()}")

    if not merge_log.empty:
        print(f"\nMerge log ({len(merge_log)} merges):")
        from IPython.display import display
        display(merge_log)

    return tracked, track_stats, merge_log


def filter_short_tracks(tracked, track_stats, min_detections=4):
    """Remove tracks with fewer than *min_detections* observations.

    Returns new (tracked, track_stats) DataFrames with short tracks removed.
    """
    is_short = track_stats["num_detections"] < min_detections
    n_short = is_short.sum()
    n_short_disappeared = track_stats.loc[is_short, "disappeared"].sum()
    short_ids = track_stats.loc[is_short, "track_id"]

    tracked = tracked[~tracked["track_id"].isin(short_ids)].copy()
    track_stats = track_stats[~is_short].copy()

    print(f"Removed {n_short} tracks with <{min_detections} detections "
          f"({n_short_disappeared} were classified as disappeared)")
    print(f"Remaining: {len(track_stats)} tracks, "
          f"{track_stats['disappeared'].sum()} disappeared")

    return tracked, track_stats


def _relative_per_track(df, col):
    """Return df[col] divided by its first non-null value per track.

    NaN where the per-track baseline is <= 0. If the very first frame's
    value is NaN, ``groupby.first()`` skips it and uses the next non-NaN
    frame as the baseline.
    """
    sorted_df = df.sort_values(["track_id", "frame"])
    first = (
        sorted_df.groupby("track_id")[col].first()
        .rename("_first_value")
    )
    merged = df.merge(first, left_on="track_id", right_index=True, how="left")
    rel = np.where(
        merged["_first_value"] > 0,
        merged[col] / merged["_first_value"],
        np.nan,
    )
    return rel


def add_geometry(tracked, track_stats, pixel_size_um=1.0):
    """Add radius, volume, surface_area, and volume_rel columns.

    Areas are converted from pixels² to (pixel_size_um · px)² before
    radius/volume/surface_area are derived under a spherical assumption.
    Pass ``pixel_size_um=1.0`` to keep raw pixel units.
    """
    tracked["area"] = tracked["area"] * pixel_size_um ** 2
    tracked["radius"] = np.sqrt(tracked["area"] / np.pi)
    tracked["volume"] = (4 / 3) * np.pi * tracked["radius"] ** 3
    tracked["surface_area"] = 4 * np.pi * tracked["radius"] ** 2
    tracked["volume_rel"] = _relative_per_track(tracked, "volume")

    track_stats = track_stats.merge(
        tracked.groupby("track_id").agg(
            mean_volume=("volume", "mean"),
            mean_surface_area=("surface_area", "mean"),
        ),
        on="track_id",
    )

    return tracked, track_stats


def add_fluorescence(tracked, track_stats, fluor_stack, label_stack):
    """Measure fluorescence per cell and merge into tracking data.

    Returns new (tracked, track_stats) DataFrames with fluorescence
    columns added.
    """
    from .matching import measure_fluorescence

    fluor_measurements = measure_fluorescence(fluor_stack, label_stack)
    print(f"Fluorescence measurements: {len(fluor_measurements)} rows")

    tracked = tracked.merge(
        fluor_measurements.rename(columns={"cell_id": "label"}),
        on=["frame", "label"],
        how="left",
    )
    tracked["fluor_rel"] = _relative_per_track(tracked, "mean_intensity")

    matched = tracked["mean_intensity"].notna().sum()
    total = len(tracked)
    print(f"Matched {matched}/{total} detections "
          f"({matched / total:.1%}) with fluorescence")

    track_stats = track_stats.merge(
        tracked.groupby("track_id").agg(
            mean_fluor_intensity=("mean_intensity", "mean"),
            mean_fluor_total=("total_intensity", "mean"),
            mean_cv=("cv", "mean"),
            mean_nnrm=("nnrm", "mean"),
        ),
        on="track_id",
    )

    print(f"Median fluorescence intensity per track: "
          f"{track_stats['mean_fluor_intensity'].median():.0f}")
    print(f"Median CV per track: {track_stats['mean_cv'].median():.3f}")
    print(f"Median nNRM per track: {track_stats['mean_nnrm'].median():.3f}")

    return tracked, track_stats


def add_nucleoid_distribution(tracked, track_stats, fluor_stack, label_stack,
                              flat_threshold_ratio=1.5,
                              peri_core_rings=None):
    """Compute per-cell-per-frame nucleoid spatial-distribution metrics.

    Adds three columns to *tracked*:

      mean_edge_distance_norm
          Mean distance from suprathreshold pixels to the cell-mask edge,
          divided by R/3 where R = sqrt(area_pixels / π). Values > 1
          indicate intensity clustered toward the cell center; < 1 toward
          the edge.
      gaussian_sigma_norm
          σ of the 2D intensity-weighted spatial distribution of
          suprathreshold pixels, divided by R. σ = sqrt of the mean of
          eigenvalues of the weighted covariance matrix.
      peri_core_asymmetry
          Signed asymmetry index (I_peri - I_core) / (I_peri + I_core)
          where I_core is the mean raw fluorescence inside an inner disk
          and I_peri is the mean in a peripheral annulus, both defined by
          *peri_core_rings*: a triple (core_max, peri_min, peri_max) of
          fractions of R that give the boundaries. Default is
          ``(0.15, 0.35, 0.55)`` — narrow core sampling the donut center
          dip, peri bracketed to the donut peak zone identified from real
          radial profiles (see
          ``docs/peri_core_asymmetry_definition_history.md``). Areas are
          unequal, so mean rather than total intensity is used. Bounded in
          [-1, 1]: positive = edge-clustered (expanded/donut), negative =
          center-clustered (compacted), 0 = radially uniform. Uses raw
          intensities (no thresholding).

    Threshold rule (applies only to mean_edge_distance_norm and
    gaussian_sigma_norm): pixels with intensity > mean(cell). If the
    cell's max/mean ratio < flat_threshold_ratio (distribution already
    flat), fall back to intensity > 0.5 * mean.

    Returns (tracked, track_stats) with the new per-cell columns in *tracked*
    and per-track mean aggregates ``mean_edge_distance_norm``,
    ``mean_gaussian_sigma_norm``, and ``mean_peri_core_asymmetry`` in
    *track_stats*.
    """
    from scipy.ndimage import find_objects
    from .matching import (measure_nucleoid_metrics, PERI_CORE_RINGS_DEFAULT)

    if peri_core_rings is None:
        peri_core_rings = PERI_CORE_RINGS_DEFAULT

    edge_vals = np.full(len(tracked), np.nan)
    sigma_vals = np.full(len(tracked), np.nan)
    asym_vals = np.full(len(tracked), np.nan)
    labels_arr = tracked["label"].to_numpy()

    by_frame = tracked.groupby("frame").indices
    for t, idx_arr in by_frame.items():
        frame_labels = label_stack[int(t)]
        fluor_frame = fluor_stack[int(t)]
        slices = find_objects(frame_labels)
        for row_idx in idx_arr:
            label = int(labels_arr[row_idx])
            sl = slices[label - 1] if 0 < label <= len(slices) else None
            if sl is None:
                continue
            crop_mask = frame_labels[sl] == label
            if not crop_mask.any():
                continue
            pixels = fluor_frame[sl][crop_mask].astype(np.float64)
            if pixels.size == 0 or pixels.mean() <= 0:
                continue
            edge_vals[row_idx], sigma_vals[row_idx], asym_vals[row_idx] = (
                measure_nucleoid_metrics(pixels, crop_mask, flat_threshold_ratio,
                                         peri_core_rings=peri_core_rings)
            )

    tracked = tracked.copy()
    tracked["mean_edge_distance_norm"] = edge_vals
    tracked["gaussian_sigma_norm"] = sigma_vals
    tracked["peri_core_asymmetry"] = asym_vals

    track_stats = track_stats.merge(
        tracked.groupby("track_id").agg(
            mean_edge_distance_norm=("mean_edge_distance_norm", "mean"),
            mean_gaussian_sigma_norm=("gaussian_sigma_norm", "mean"),
            mean_peri_core_asymmetry=("peri_core_asymmetry", "mean"),
        ),
        on="track_id", how="left",
    )

    n_valid = int(np.isfinite(edge_vals).sum())
    print(f"Nucleoid distribution: {n_valid}/{len(tracked)} "
          f"cell-frames measured")

    return tracked, track_stats


def add_fluorescence_concentration(tracked):
    """Add fluor_concentration column (total_intensity / volume).

    Returns *tracked* with the new column added in-place.
    """
    tracked["fluor_concentration"] = np.where(
        tracked["volume"] > 0,
        tracked["total_intensity"] / tracked["volume"],
        np.nan,
    )
    valid = tracked["fluor_concentration"].notna().sum()
    print(f"Fluorescence concentration: {valid}/{len(tracked)} cells computed")
    return tracked


def add_fluorescence_alignment(tracked, track_stats, fluor_stack, label_stack,
                               window=3):
    """Build a long-form DataFrame of windowed fluorescence around each
    disappeared cell's last detection.

    Returns DataFrame with columns ``track_id, offset, mean_intensity, F_norm``
    where offset ∈ [-window, +window] (frame relative to last_frame) and
    F_norm = mean_intensity(offset) / mean_intensity(-window). Survived tracks
    are excluded. Tracks whose offset=-window observation is missing are
    dropped (no baseline to normalize against).
    """
    from .matching import measure_post_disappearance_fluorescence

    disappeared = track_stats[track_stats["disappeared"]][
        ["track_id", "last_frame"]
    ]

    last_labels = (
        tracked.merge(
            disappeared.rename(columns={"last_frame": "_lf"}),
            on="track_id",
        )
        .query("frame == _lf")
        [["track_id", "label"]]
        .rename(columns={"label": "last_frame_label"})
    )

    post = measure_post_disappearance_fluorescence(
        track_stats, last_labels, fluor_stack, label_stack, window=window,
    )

    pre_and_at = (
        tracked.merge(disappeared, on="track_id")
        [["track_id", "frame", "mean_intensity", "last_frame"]]
    )
    pre_and_at = pre_and_at[
        (pre_and_at["frame"] >= pre_and_at["last_frame"] - window)
        & (pre_and_at["frame"] <= pre_and_at["last_frame"])
    ]
    post = post.merge(disappeared, on="track_id")

    full = pd.concat([pre_and_at, post], ignore_index=True)
    full["offset"] = full["frame"] - full["last_frame"]
    full = full[["track_id", "offset", "mean_intensity"]]

    out = []
    for tid, grp in full.groupby("track_id"):
        grp = grp.set_index("offset").reindex(range(-window, window + 1))
        if pd.isna(grp.loc[-window, "mean_intensity"]):
            continue
        baseline = grp.loc[-window, "mean_intensity"]
        grp["F_norm"] = grp["mean_intensity"] / baseline
        grp["track_id"] = tid
        out.append(grp.reset_index())

    if not out:
        return pd.DataFrame(
            columns=["track_id", "offset", "mean_intensity", "F_norm"]
        )

    return pd.concat(out, ignore_index=True)[
        ["track_id", "offset", "mean_intensity", "F_norm"]
    ]


def add_peri_core_alignment(tracked, track_stats, window=3):
    """Long-form DataFrame of peri/core asymmetry aligned to disappearance.

    For each disappeared track, extracts ``peri_core_asymmetry`` at offsets
    in ``[-window, 0]`` where offset = frame - last_frame. Returns columns:

    - ``track_id``
    - ``offset``  (integer, negative = before last detection, 0 = last)
    - ``peri_core_asymmetry``
    - ``delta_from_baseline`` — asymmetry(offset) − asymmetry(-window)

    Survived tracks are excluded. Tracks whose offset=-window observation
    is missing (short lifespan) are dropped — no baseline to compare
    against. Asymmetry cannot be measured post-disappearance because the
    metric needs a well-defined mask; use §8.6 fluorescence alignment for
    the post-burst window.
    """
    disappeared = track_stats[track_stats["disappeared"]][
        ["track_id", "last_frame"]
    ]
    if disappeared.empty:
        return pd.DataFrame(columns=[
            "track_id", "offset", "peri_core_asymmetry", "delta_from_baseline",
        ])

    merged = tracked.merge(disappeared, on="track_id")
    merged["offset"] = merged["frame"] - merged["last_frame"]
    merged = merged[(merged["offset"] >= -window) & (merged["offset"] <= 0)]
    merged = merged[["track_id", "offset", "peri_core_asymmetry"]]

    out = []
    for tid, grp in merged.groupby("track_id"):
        grp = grp.set_index("offset").reindex(range(-window, 1))
        if pd.isna(grp.loc[-window, "peri_core_asymmetry"]):
            continue
        baseline = grp.loc[-window, "peri_core_asymmetry"]
        grp["delta_from_baseline"] = grp["peri_core_asymmetry"] - baseline
        grp["track_id"] = tid
        out.append(grp.reset_index())

    if not out:
        return pd.DataFrame(columns=[
            "track_id", "offset", "peri_core_asymmetry", "delta_from_baseline",
        ])
    return pd.concat(out, ignore_index=True)[[
        "track_id", "offset", "peri_core_asymmetry", "delta_from_baseline",
    ]]



def add_sav_ratio(tracked):
    """Add surface-area-to-volume ratio column.

    Returns *tracked* with 'sav_ratio' column added in-place.
    """
    tracked["sav_ratio"] = np.where(
        tracked["volume"] > 0,
        tracked["surface_area"] / tracked["volume"],
        np.nan,
    )
    valid = tracked["sav_ratio"].notna().sum()
    print(f"SA:V ratio: {valid}/{len(tracked)} cells computed")
    print(f"Median SA:V at frame 0: "
          f"{tracked.loc[tracked['frame'] == 0, 'sav_ratio'].median():.4f}")
    return tracked


def add_preburst_fluorescence(tracked, track_stats, n_frames=5):
    """Analyze pre-burst fluorescence behavior for disappeared tracks.

    Returns new *track_stats* with preburst_slope and preburst_spike columns.
    """
    from .matching import compute_preburst_fluorescence

    preburst = compute_preburst_fluorescence(tracked, track_stats, n_frames=n_frames)
    track_stats = track_stats.merge(preburst, on="track_id", how="left")

    spikes = track_stats["preburst_spike"].fillna(False)
    n_spikes = spikes.sum()
    n_dis = track_stats["disappeared"].sum()
    print(f"Pre-burst fluorescence ({n_frames}-frame window):")
    if n_dis > 0:
        print(f"  Tracks with pre-burst spike: {n_spikes}/{n_dis} "
              f"({n_spikes / n_dis:.0%})")
    else:
        print("  No disappeared tracks")

    return track_stats


def add_frame0_fate_comparison(tracked, track_stats, features=None):
    """Mann-Whitney U test on frame-0 features, survived vs disappeared.

    Univariate companion to :func:`add_fate_prediction` (logistic regression).
    Returns the comparison DataFrame.
    """
    from .matching import compare_frame0_features_by_fate

    result_df = compare_frame0_features_by_fate(
        tracked, track_stats, features=features,
    )

    print("Frame-0 features by fate (Mann-Whitney U, two-sided):")
    print(f"  Cohort: {int(result_df['n_survived'].iloc[0])} survived, "
          f"{int(result_df['n_died'].iloc[0])} died")
    for _, row in result_df.iterrows():
        direction = "↑ died" if row["median_died"] > row["median_survived"] else "↓ died"
        print(f"  {row['feature']:>5}: "
              f"median {row['median_survived']:.3f} (surv) vs "
              f"{row['median_died']:.3f} (died) [{direction}], "
              f"U={row['U']:.0f}, p={row['p_value']:.2e}")

    return result_df


def add_fate_prediction(tracked, track_stats, features=None):
    """Predict cell fate from frame-0 features using logistic regression.

    Returns (prediction_df, prediction_summary).
    """
    from .matching import predict_fate_from_frame0

    result_df, summary = predict_fate_from_frame0(
        tracked, track_stats, features=features,
    )

    print(f"Fate prediction (LOO cross-validation, n={summary['n_cells']}):")
    print(f"  AUC: {summary['auc']:.3f}")
    print(f"  Accuracy: {summary['accuracy']:.1%}")
    print(f"  Died: {summary['n_died']}, Survived: {summary['n_survived']}")
    print("  Feature importance (z-scored coefficients):")
    for feat, coef in summary["feature_importance"].items():
        direction = "↑ death" if coef > 0 else "↓ death"
        print(f"    {feat}: {coef:+.3f} ({direction})")

    pc = summary["per_class"]
    print("  Per-class performance (probability threshold 0.5):")
    for label in ("survived", "disappeared"):
        m = pc[label]
        print(f"    {label:12s} (n={m['support']:3d}): "
              f"precision={m['precision']:.3f}, "
              f"recall={m['recall']:.3f}, "
              f"f1={m['f1']:.3f}")

    cm = summary["confusion_matrix"]
    print("  Confusion matrix (rows = actual, cols = predicted):")
    print("                              pred: survived  pred: disappeared")
    print(f"    actual: survived          {cm['TN']:7d}       {cm['FP']:7d}")
    print(f"    actual: disappeared       {cm['FN']:7d}       {cm['TP']:7d}")

    return result_df, summary


def print_fate_comparison(summary_full, summary_no_area, f1_threshold=0.02):
    """Print a side-by-side table comparing two fate-prediction summaries.

    Conventionally invoked from §8.14 with the 5-feature (full) summary and
    the 4-feature (no-area) summary. Prints AUC, accuracy, per-class recall,
    and per-class F1 for both. If the no-area variant improves BOTH per-class
    F1 by ≥ ``f1_threshold`` absolute, prints a "consider swapping" hint.
    """
    rows = [
        ("AUC",
         summary_full["auc"], summary_no_area["auc"]),
        ("Accuracy",
         summary_full["accuracy"], summary_no_area["accuracy"]),
        ("Recall (survived)",
         summary_full["per_class"]["survived"]["recall"],
         summary_no_area["per_class"]["survived"]["recall"]),
        ("Recall (disappeared)",
         summary_full["per_class"]["disappeared"]["recall"],
         summary_no_area["per_class"]["disappeared"]["recall"]),
        ("F1 (survived)",
         summary_full["per_class"]["survived"]["f1"],
         summary_no_area["per_class"]["survived"]["f1"]),
        ("F1 (disappeared)",
         summary_full["per_class"]["disappeared"]["f1"],
         summary_no_area["per_class"]["disappeared"]["f1"]),
    ]

    n_full = len(summary_full["feature_names"])
    n_red = len(summary_no_area["feature_names"])
    print("\nFeature-set comparison:")
    print(f"                        full ({n_full})   no-area ({n_red})")
    for label, a, b in rows:
        print(f"  {label:21s} {a:.3f}      {b:.3f}")

    surv_f1_delta = (
        summary_no_area["per_class"]["survived"]["f1"]
        - summary_full["per_class"]["survived"]["f1"]
    )
    dis_f1_delta = (
        summary_no_area["per_class"]["disappeared"]["f1"]
        - summary_full["per_class"]["disappeared"]["f1"]
    )
    if surv_f1_delta >= f1_threshold and dis_f1_delta >= f1_threshold:
        print("  (no-area variant improves both per-class F1 by "
              f"≥ {f1_threshold:.2f}; consider permanently swapping "
              "FATE_FEATURES_FULL → FATE_FEATURES_NO_AREA in the notebook)")
    else:
        print("  (differences small; full feature set remains the default)")


def run_nucleus_persistence(label_stack, nucleus_label_stack,
                            loss_tolerance=0.1, offset_cv_threshold=0.3):
    """Compare phase-cell and fluorescence-nucleus counts per frame.

    Quantifies whether fluorescent nuclei persist after cells disappear
    from phase-contrast (burst).  Two independent tests must both pass
    for a "parallel" conclusion:

    1. **Endpoint test** — total loss counts (first-to-last frame) for
       phase and fluorescence must agree within *loss_tolerance* of the
       larger value.
    2. **Trajectory test** — the coefficient of variation of the
       per-frame offset (fluor − phase) must be below
       *offset_cv_threshold*, ensuring the offset is stable across all
       frames, not just at the endpoints.

    A constant offset with parallel decline indicates nuclei do NOT
    persist; a growing or erratic offset would indicate nuclei outlive
    their cells.

    Returns (DataFrame with per-frame counts, summary dict).
    """
    assert label_stack.shape == nucleus_label_stack.shape, (
        f"Shape mismatch: label_stack {label_stack.shape} vs "
        f"nucleus_label_stack {nucleus_label_stack.shape}"
    )

    T = label_stack.shape[0]
    records = []
    for t in range(T):
        if label_stack[t].max() == 0:
            continue
        phase_n = len(np.unique(label_stack[t])) - 1
        fluor_n = len(np.unique(nucleus_label_stack[t])) - 1
        records.append({
            "frame": t,
            "phase_cells": phase_n,
            "fluor_nuclei": fluor_n,
            "difference": fluor_n - phase_n,
        })

    df = pd.DataFrame(records)

    phase_lost = df["phase_cells"].iloc[0] - df["phase_cells"].iloc[-1]
    fluor_lost = df["fluor_nuclei"].iloc[0] - df["fluor_nuclei"].iloc[-1]
    offset_std = df["difference"].std()
    mean_offset = df["difference"].mean()
    offset_cv = offset_std / abs(mean_offset) if mean_offset != 0 else float("inf")

    print(f"Phase cells lost (first to last frame): {phase_lost}")
    print(f"Fluor nuclei lost (first to last frame): {fluor_lost}")
    print(f"Difference std across frames: {offset_std:.1f}")

    endpoint_ok = (
        abs(phase_lost - fluor_lost)
        <= loss_tolerance * max(phase_lost, fluor_lost, 1)
    )
    trajectory_ok = offset_cv < offset_cv_threshold

    if endpoint_ok and trajectory_ok:
        conclusion = "parallel"
        print(
            "\nConclusion: Phase cells and fluorescent nuclei disappear "
            "at the same rate. DNA signal does NOT persist as a discrete "
            "object after cell lysis — the fluorescent material disperses "
            "upon membrane rupture."
        )
    else:
        conclusion = "divergent"
        reasons = []
        if not endpoint_ok:
            reasons.append(
                f"total loss diverged ({phase_lost} vs {fluor_lost})"
            )
        if not trajectory_ok:
            reasons.append(
                f"offset CV too high ({offset_cv:.2f} >= {offset_cv_threshold})"
            )
        print(
            f"\nConclusion: Divergent decline — nuclei may persist after "
            f"lysis ({'; '.join(reasons)})"
        )

    summary = {
        "phase_lost": int(phase_lost),
        "fluor_lost": int(fluor_lost),
        "mean_offset": float(mean_offset),
        "offset_std": float(offset_std),
        "offset_cv": float(offset_cv),
        "conclusion": conclusion,
    }

    return df, summary


def export_all_results(
    results_dir, *,
    tracked, track_stats, diagnostics,
    merge_log, prediction_df, prediction_summary,
    comparison_df, persistence_summary,
):
    """Save all analysis outputs to *results_dir* as CSV files.

    DataFrames are saved directly; summary dicts are flattened to
    single-row CSVs.
    """
    results_dir = Path(results_dir)
    saved = []

    for df, name in [
        (tracked, "tracked_cells.csv"),
        (track_stats, "track_statistics.csv"),
        (diagnostics, "frame_diagnostics.csv"),
        (prediction_df, "fate_predictions.csv"),
        (comparison_df, "nucleus_persistence.csv"),
    ]:
        save_results(df, results_dir / name)
        saved.append(name)

    if not merge_log.empty:
        save_results(merge_log, results_dir / "merge_log.csv")
        saved.append("merge_log.csv")

    for summary, name in [
        (prediction_summary, "fate_prediction_summary.csv"),
        (persistence_summary, "nucleus_persistence_summary.csv"),
    ]:
        save_summary(summary, results_dir / name)
        saved.append(name)

    print(f"Saved {len(saved)} files to {results_dir}/")
    for name in saved:
        print(f"  {name}")

    return saved
