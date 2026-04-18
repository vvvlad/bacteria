"""High-level pipeline orchestration for the cell analysis workflow.

Each function chains lower-level modules into a single call, printing
summary statistics so the notebook stays minimal.
"""

import numpy as np
from pathlib import Path


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
    from .io import save_results

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


def add_geometry(tracked, track_stats):
    """Add radius, volume, and surface_area columns (spherical assumption).

    Adds columns to *tracked* in-place. Returns a new *track_stats*
    with mean_volume and mean_surface_area columns merged in.
    """
    tracked["radius"] = np.sqrt(tracked["area"] / np.pi)
    tracked["volume"] = (4 / 3) * np.pi * tracked["radius"] ** 3
    tracked["surface_area"] = 4 * np.pi * tracked["radius"] ** 2

    track_stats = track_stats.merge(
        tracked.groupby("track_id").agg(
            mean_volume=("volume", "mean"),
            mean_surface_area=("surface_area", "mean"),
        ),
        on="track_id",
    )

    return tracked, track_stats


def add_growth(tracked, track_stats):
    """Compute per-track growth metrics and merge into track_stats.

    Returns a new *track_stats* DataFrame with growth columns added.
    """
    from .tracking import compute_growth_stats

    growth = compute_growth_stats(tracked)
    track_stats = track_stats.merge(growth, on="track_id", how="left")

    grew = track_stats["area_rel_max"].dropna()
    print(f"Growth stats computed for {len(grew)} tracks")
    print(f"Median max relative size: {grew.median():.2f}x")
    print(f"Median growth rate: "
          f"{track_stats['growth_rate_px_per_frame'].median():.1f} px/frame")

    return track_stats


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


def add_fluorescence_disappearance(tracked, track_stats, threshold=-0.3,
                                    drop_window=1):
    """Detect per-track fluorescence disappearance and merge into track_stats.

    Returns a new *track_stats* DataFrame with fluor_disappearance_frame
    and max_drop columns added.
    """
    from .matching import detect_fluorescence_disappearance

    fluor_disapp = detect_fluorescence_disappearance(
        tracked, threshold=threshold, drop_window=drop_window,
    )
    n_detected = fluor_disapp["fluor_disappearance_frame"].notna().sum()
    window_label = "single-frame" if drop_window == 1 else f"{drop_window}-frame"
    print(f"Fluorescence disappearance detected: "
          f"{n_detected}/{len(fluor_disapp)} tracks "
          f"(threshold: {threshold:.0%} {window_label} drop)")

    track_stats = track_stats.merge(fluor_disapp, on="track_id", how="left")
    return track_stats


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


def add_migration(tracked, track_stats):
    """Compute per-frame speed and per-track migration statistics.

    Adds 'speed' column to *tracked* in-place. Returns new *track_stats*
    with migration columns merged in.
    """
    from .tracking import compute_migration_stats

    migration = compute_migration_stats(tracked, per_frame=tracked)
    track_stats = track_stats.merge(migration, on="track_id", how="left")

    valid = track_stats["mean_speed"].notna()
    print(f"Migration stats: {valid.sum()} tracks")
    print(f"Median mean speed: {track_stats.loc[valid, 'mean_speed'].median():.1f} px/frame")
    print(f"Median net displacement: "
          f"{track_stats.loc[valid, 'net_displacement'].median():.1f} px")

    return tracked, track_stats


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


def add_death_clustering(tracked, track_stats, n_permutations=1000):
    """Compute spatial clustering of cell death and print results.

    Returns (clustering_result dict, track_stats with last_y/last_x).
    """
    from .matching import compute_death_clustering

    last_obs = (
        tracked.sort_values("frame")
        .groupby("track_id")
        .agg(last_y=("centroid_y", "last"), last_x=("centroid_x", "last"))
    )
    ts = track_stats.merge(last_obs, on="track_id", how="left")

    result = compute_death_clustering(ts, n_permutations=n_permutations)

    if not np.isnan(result["clustering_ratio"]):
        print(f"Death clustering analysis ({n_permutations} permutations):")
        print(f"  Mean NN distance (deaths): {result['mean_nn_distance_deaths']:.1f} px")
        print(f"  Mean NN distance (random): {result['mean_nn_distance_random']:.1f} px")
        print(f"  Clustering ratio: {result['clustering_ratio']:.3f} "
              f"({'clustered' if result['clustering_ratio'] < 0.8 else 'not clustered'})")
        print(f"  p-value: {result['p_value']:.4f}")
    else:
        print("Death clustering: too few deaths for analysis")

    return result, ts


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
    print(f"  Feature importance (z-scored coefficients):")
    for feat, coef in summary["feature_importance"].items():
        direction = "↑ death" if coef > 0 else "↓ death"
        print(f"    {feat}: {coef:+.3f} ({direction})")

    return result_df, summary


def add_spatial_gradient(tracked, track_stats):
    """Analyze spatial gradient in cell fate across the field of view.

    Returns (gradient_df, gradient_summary).
    """
    from .matching import analyze_spatial_gradient

    gradient_df, summary = analyze_spatial_gradient(tracked, track_stats)

    axis_label = summary["gradient_axis"].replace("centroid_", "")
    ax_stats = summary["axes_results"][summary["gradient_axis"]]

    print(f"Spatial gradient analysis (dominant axis: {axis_label}):")
    print(f"  Mann-Whitney p = {ax_stats['mann_whitney_p']:.2e}")
    print(f"  Point-biserial r = {ax_stats['point_biserial_r']:+.3f} "
          f"(p = {ax_stats['point_biserial_p']:.2e})")
    print(f"  AUC (position only): {summary['auc_position_only']:.3f}")
    print(f"  Death rate by quartile along {axis_label}:")
    for q, rates in summary["quartile_death_rates"].items():
        print(f"    Q{q}: {rates['death_rate']:.1%} "
              f"({rates['n_died']}/{rates['n_cells']})")

    return gradient_df, summary


def add_growth_phases(tracked, track_stats, min_points=4):
    """Detect growth phase transitions and merge into track_stats.

    Returns new *track_stats* with changepoint_frame, slope_before,
    slope_after, and slope_ratio columns.
    """
    from .tracking import detect_growth_phases

    phases = detect_growth_phases(tracked, min_points=min_points)
    track_stats = track_stats.merge(phases, on="track_id", how="left")

    valid = track_stats["changepoint_frame"].notna()
    print(f"Growth phases detected: {valid.sum()} tracks")
    if valid.any():
        print(f"Median changepoint frame: "
              f"{track_stats.loc[valid, 'changepoint_frame'].median():.0f}")
        ratios = track_stats.loc[valid, "slope_ratio"].dropna()
        if len(ratios) > 0:
            print(f"Median slope ratio (after/before): {ratios.median():.2f}")

    return track_stats


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
    import pandas as pd

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
    gradient_df, gradient_summary,
    clustering_result, comparison_df, persistence_summary,
):
    """Save all analysis outputs to *results_dir* as CSV files.

    DataFrames are saved directly; summary dicts are flattened to
    single-row CSVs.
    """
    from .io import save_results, save_summary
    from pathlib import Path

    results_dir = Path(results_dir)
    saved = []

    for df, name in [
        (tracked, "tracked_cells.csv"),
        (track_stats, "track_statistics.csv"),
        (diagnostics, "frame_diagnostics.csv"),
        (prediction_df, "fate_predictions.csv"),
        (gradient_df, "spatial_gradient.csv"),
        (comparison_df, "nucleus_persistence.csv"),
    ]:
        save_results(df, results_dir / name)
        saved.append(name)

    if not merge_log.empty:
        save_results(merge_log, results_dir / "merge_log.csv")
        saved.append("merge_log.csv")

    for summary, name in [
        (clustering_result, "clustering_summary.csv"),
        (prediction_summary, "fate_prediction_summary.csv"),
        (gradient_summary, "spatial_gradient_summary.csv"),
        (persistence_summary, "nucleus_persistence_summary.csv"),
    ]:
        save_summary(summary, results_dir / name)
        saved.append(name)

    print(f"Saved {len(saved)} files to {results_dir}/")
    for name in saved:
        print(f"  {name}")

    return saved
