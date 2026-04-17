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


def add_fluorescence_disappearance(tracked, track_stats, threshold=-0.3):
    """Detect per-track fluorescence disappearance and merge into track_stats.

    Returns a new *track_stats* DataFrame with fluor_disappearance_frame
    and max_drop columns added.
    """
    from .matching import detect_fluorescence_disappearance

    fluor_disapp = detect_fluorescence_disappearance(
        tracked, threshold=threshold,
    )
    n_detected = fluor_disapp["fluor_disappearance_frame"].notna().sum()
    print(f"Fluorescence disappearance detected: "
          f"{n_detected}/{len(fluor_disapp)} tracks "
          f"(threshold: {threshold:.0%} single-frame drop)")

    track_stats = track_stats.merge(fluor_disapp, on="track_id", how="left")
    return track_stats
