"""Cell tracking over time-lapse stacks."""

import numpy as np
import pandas as pd
from scipy import ndimage


def labels_to_detections(labels: np.ndarray) -> pd.DataFrame:
    """Convert a label stack to a detections table.

    Parameters
    ----------
    labels : np.ndarray
        Label stack with shape (T, Y, X).

    Returns
    -------
    pd.DataFrame
        Columns: frame, label, centroid_y, centroid_x, area
    """
    records = []
    for t in range(labels.shape[0]):
        frame_labels = labels[t]
        cell_ids = np.unique(frame_labels)
        cell_ids = cell_ids[cell_ids != 0]  # skip background

        for cell_id in cell_ids:
            mask = frame_labels == cell_id
            cy, cx = ndimage.center_of_mass(mask)
            area = mask.sum()
            records.append({
                "frame": t,
                "label": int(cell_id),
                "centroid_y": cy,
                "centroid_x": cx,
                "area": area,
            })

    return pd.DataFrame(records)


def track_cells(detections: pd.DataFrame, search_range: float = 30.0, memory: int = 3) -> pd.DataFrame:
    """Link detections across frames into tracks using trackpy.

    Parameters
    ----------
    detections : pd.DataFrame
        Must have columns: frame, centroid_y, centroid_x.
    search_range : float
        Max distance (pixels) a cell can move between frames.
    memory : int
        Number of frames a cell can "disappear" and still be linked.

    Returns
    -------
    pd.DataFrame
        Same as input with an added 'track_id' column.
    """
    import trackpy as tp

    # trackpy expects columns named 'x', 'y', 'frame'
    tp_input = detections.rename(columns={"centroid_x": "x", "centroid_y": "y"})
    linked = tp.link(tp_input, search_range=search_range, memory=memory)
    linked = linked.rename(columns={"x": "centroid_x", "y": "centroid_y", "particle": "track_id"})
    return linked


def merge_fragmented_tracks(
    tracked: pd.DataFrame,
    max_distance: float = 15.0,
    max_gap: int = 18,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge track fragments that are likely the same physical cell.

    When a cell goes undetected for longer than trackpy's ``memory``
    parameter, it reappears with a new track ID.  This function finds
    tracks whose start position is spatially close to another track's
    end position and merges them into a single track ID.

    Parameters
    ----------
    tracked : pd.DataFrame
        Output of :func:`track_cells` with columns
        ``frame, label, centroid_y, centroid_x, area, track_id``.
    max_distance : float
        Maximum Euclidean distance (pixels) between the last position
        of an ended track and the first position of a starting track
        for them to be considered the same cell.
    max_gap : int
        Maximum frame gap between the end of one track and the start
        of another for merge eligibility.

    Returns
    -------
    merged : pd.DataFrame
        Copy of *tracked* with ``track_id`` updated for merged fragments.
    merge_log : pd.DataFrame
        One row per merge: ``absorbed_track, into_track, distance,
        gap_frames, end_frame, start_frame``.
    """
    if tracked.empty:
        return tracked.copy(), pd.DataFrame(
            columns=["absorbed_track", "into_track", "distance",
                     "gap_frames", "end_frame", "start_frame"])

    # --- Phase A: build candidate pairs --------------------------------
    grouped = tracked.groupby("track_id")

    # Last detection per track
    endpoints = {}
    for tid, grp in grouped:
        row = grp.loc[grp["frame"].idxmax()]
        endpoints[tid] = (int(row["frame"]), row["centroid_y"], row["centroid_x"])

    # First detection per track
    startpoints = {}
    for tid, grp in grouped:
        row = grp.loc[grp["frame"].idxmin()]
        startpoints[tid] = (int(row["frame"]), row["centroid_y"], row["centroid_x"])

    # Index start points by frame for fast lookup
    from collections import defaultdict
    starts_by_frame: dict[int, list[tuple]] = defaultdict(list)
    for tid, (sf, sy, sx) in startpoints.items():
        starts_by_frame[sf].append((tid, sy, sx))

    candidates = []
    for tid_a, (ef, ey, ex) in endpoints.items():
        for gap_frame in range(ef + 1, ef + max_gap + 1):
            for tid_b, sy, sx in starts_by_frame.get(gap_frame, []):
                if tid_a == tid_b:
                    continue
                dist = np.sqrt((ey - sy) ** 2 + (ex - sx) ** 2)
                if dist <= max_distance:
                    candidates.append((tid_a, tid_b, dist, gap_frame - ef))

    # Greedy best-match: sort by distance, accept if neither side used
    candidates.sort(key=lambda c: c[2])
    used_ends: set[int] = set()
    used_starts: set[int] = set()
    accepted = []
    for tid_a, tid_b, dist, gap in candidates:
        if tid_a not in used_ends and tid_b not in used_starts:
            accepted.append((tid_a, tid_b, dist, gap))
            used_ends.add(tid_a)
            used_starts.add(tid_b)

    if not accepted:
        return tracked.copy(), pd.DataFrame(
            columns=["absorbed_track", "into_track", "distance",
                     "gap_frames", "end_frame", "start_frame"])

    # --- Phase B: resolve chains via Union-Find ------------------------
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Earlier track ID becomes the root
            if ra > rb:
                ra, rb = rb, ra
            parent[rb] = ra

    for tid_a, tid_b, _dist, _gap in accepted:
        union(tid_a, tid_b)

    # --- Phase C: remap IDs --------------------------------------------
    all_tids = tracked["track_id"].unique()
    remap = {tid: find(tid) for tid in all_tids}

    merged = tracked.copy()
    merged["track_id"] = merged["track_id"].map(remap)

    # Build merge log
    log_rows = []
    for tid_a, tid_b, dist, gap in accepted:
        log_rows.append({
            "absorbed_track": tid_b,
            "into_track": find(tid_b),
            "distance": round(dist, 1),
            "gap_frames": gap,
            "end_frame": endpoints[tid_a][0],
            "start_frame": startpoints[tid_b][0],
        })
    merge_log = pd.DataFrame(log_rows)

    n_before = tracked["track_id"].nunique()
    n_after = merged["track_id"].nunique()
    print(f"Track merging: {n_before} → {n_after} tracks "
          f"({len(accepted)} merges, {n_before - n_after} fragments absorbed)")

    return merged, merge_log


def compute_track_stats(tracked: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track statistics.

    Returns
    -------
    pd.DataFrame
        Columns: track_id, first_frame, last_frame, lifetime,
                 disappeared (True if track ends before the last frame),
                 mean_area.
    """
    last_frame = tracked["frame"].max()

    stats = tracked.groupby("track_id").agg(
        first_frame=("frame", "min"),
        last_frame=("frame", "max"),
        mean_area=("area", "mean"),
        num_detections=("frame", "count"),
    ).reset_index()

    stats["lifetime"] = stats["last_frame"] - stats["first_frame"] + 1
    stats["disappeared"] = stats["last_frame"] < last_frame

    return stats


def compute_growth_stats(tracked: pd.DataFrame) -> pd.DataFrame:
    """Compute per-track growth metrics from area time series.

    For each track, measures how much the cell grew relative to its first
    observation, the frame at which it was largest, and a linear growth
    rate (pixels per frame) fitted over the track's lifespan.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain columns: track_id, frame, area.

    Returns
    -------
    pd.DataFrame
        One row per track with columns: track_id, area_initial, area_max,
        area_rel_max (area_max / area_initial), frame_of_max_area,
        growth_rate_px_per_frame (slope of linear fit of area vs frame).
    """
    records = []
    for tid, grp in tracked.groupby("track_id"):
        ts = grp.sort_values("frame")
        areas = ts["area"].values.astype(np.float64)
        frames = ts["frame"].values.astype(np.float64)
        a0 = areas[0]

        max_idx = np.argmax(areas)
        a_max = areas[max_idx]

        if len(frames) >= 2:
            slope = np.polyfit(frames, areas, 1)[0]
        else:
            slope = np.nan

        records.append({
            "track_id": tid,
            "area_initial": float(a0),
            "area_max": float(a_max),
            "area_rel_max": float(a_max / a0) if a0 > 0 else np.nan,
            "frame_of_max_area": int(ts["frame"].iloc[max_idx]),
            "growth_rate_px_per_frame": float(slope),
        })

    return pd.DataFrame(records)


def compute_migration_stats(tracked, per_frame=None):
    """Compute per-track migration speed statistics.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain columns: track_id, frame, centroid_y, centroid_x.
    per_frame : pd.DataFrame or None
        If provided, a 'speed' column is added to this DataFrame in-place
        (NaN for the first frame of each track).

    Returns
    -------
    pd.DataFrame
        One row per track: track_id, mean_speed, max_speed, speed_std,
        total_displacement, net_displacement (px).
    """
    speeds_per_frame = []
    records = []

    for tid, grp in tracked.groupby("track_id"):
        ts = grp.sort_values("frame")
        y = ts["centroid_y"].values
        x = ts["centroid_x"].values

        if len(ts) < 2:
            records.append({
                "track_id": tid, "mean_speed": np.nan, "max_speed": np.nan,
                "speed_std": np.nan, "total_displacement": np.nan,
                "net_displacement": np.nan,
            })
            if per_frame is not None:
                for idx in ts.index:
                    speeds_per_frame.append((idx, np.nan))
            continue

        dy = np.diff(y)
        dx = np.diff(x)
        step_dists = np.sqrt(dy**2 + dx**2)
        frame_gaps = np.diff(ts["frame"].values)
        # Speed = displacement / frames elapsed (handles gaps from dropped frames)
        step_speeds = step_dists / frame_gaps

        net = np.sqrt((y[-1] - y[0])**2 + (x[-1] - x[0])**2)

        records.append({
            "track_id": tid,
            "mean_speed": float(step_speeds.mean()),
            "max_speed": float(step_speeds.max()),
            "speed_std": float(step_speeds.std()),
            "total_displacement": float(step_dists.sum()),
            "net_displacement": float(net),
        })

        if per_frame is not None:
            indices = ts.index.tolist()
            speeds_per_frame.append((indices[0], np.nan))
            for i, idx in enumerate(indices[1:]):
                speeds_per_frame.append((idx, float(step_speeds[i])))

    if per_frame is not None:
        speed_series = pd.Series(np.nan, index=per_frame.index, dtype=float)
        for idx, val in speeds_per_frame:
            speed_series[idx] = val
        per_frame["speed"] = speed_series

    return pd.DataFrame(records)


def detect_growth_phases(tracked, min_points=4):
    """Detect growth phase transition via optimal 2-segment piecewise linear fit.

    For each track, tries every possible split point and picks the one
    that minimizes total residual sum of squares.

    Parameters
    ----------
    tracked : pd.DataFrame
        Must contain: track_id, frame, area.
    min_points : int
        Minimum detections per track for phase detection.

    Returns
    -------
    pd.DataFrame
        One row per track: track_id, changepoint_frame, slope_before,
        slope_after, slope_ratio (after/before).
    """
    records = []
    for tid, grp in tracked.groupby("track_id"):
        ts = grp.sort_values("frame")
        frames = ts["frame"].values.astype(np.float64)
        areas = ts["area"].values.astype(np.float64)
        n = len(frames)

        if n < min_points:
            records.append({
                "track_id": tid, "changepoint_frame": np.nan,
                "slope_before": np.nan, "slope_after": np.nan,
                "slope_ratio": np.nan,
            })
            continue

        best_rss = np.inf
        best_k = None
        best_slopes = (0.0, 0.0)

        for k in range(2, n - 1):
            f1, a1 = frames[:k], areas[:k]
            f2, a2 = frames[k:], areas[k:]

            if len(f1) < 2 or len(f2) < 2:
                continue

            c1 = np.polyfit(f1, a1, 1)
            c2 = np.polyfit(f2, a2, 1)
            r1 = a1 - np.polyval(c1, f1)
            r2 = a2 - np.polyval(c2, f2)
            rss = float(np.sum(r1**2) + np.sum(r2**2))

            if rss < best_rss:
                best_rss = rss
                best_k = k
                best_slopes = (float(c1[0]), float(c2[0]))

        if best_k is not None:
            cp_frame = int(frames[best_k])
            s_before, s_after = best_slopes
            ratio = s_after / s_before if abs(s_before) > 1e-6 else np.nan
        else:
            cp_frame = np.nan
            s_before = s_after = ratio = np.nan

        records.append({
            "track_id": tid,
            "changepoint_frame": cp_frame,
            "slope_before": s_before,
            "slope_after": s_after,
            "slope_ratio": ratio,
        })

    return pd.DataFrame(records)


def detect_bad_frames(
    detections: pd.DataFrame,
    z_threshold: float = 3.5,
) -> tuple[list[int], pd.DataFrame]:
    """Flag frames with anomalous detection statistics.

    Computes per-frame cell count, mean area, and area IQR, then flags
    frames whose frame-to-frame deltas are outliers using MAD-based
    modified Z-scores.  Frame 0 is evaluated using absolute Z-scores
    against the full population (no delta available).

    Parameters
    ----------
    detections : pd.DataFrame
        Output of :func:`labels_to_detections`.  Must have columns
        ``frame`` and ``area``.
    z_threshold : float
        Modified Z-score threshold for flagging.  Default 3.5 follows
        the Iglewicz & Hoaglin recommendation for MAD-based outlier
        detection.

    Returns
    -------
    bad_frames : list[int]
        Frame indices flagged as anomalous.
    diagnostics : pd.DataFrame
        One row per frame with columns: ``frame``, ``cell_count``,
        ``mean_area``, ``iqr_area``, ``z_count``, ``z_area``,
        ``z_iqr``, ``flagged``, ``reasons``.
    """
    if detections.empty:
        return [], pd.DataFrame(
            columns=["frame", "cell_count", "mean_area", "iqr_area",
                     "z_count", "z_area", "z_iqr", "flagged", "reasons"])

    # --- Per-frame statistics ---
    grouped = detections.groupby("frame")
    stats = pd.DataFrame({
        "frame": sorted(detections["frame"].unique()),
    })
    counts = grouped.size().rename("cell_count")
    means = grouped["area"].mean().rename("mean_area")
    q1 = grouped["area"].quantile(0.25)
    q3 = grouped["area"].quantile(0.75)
    iqrs = (q3 - q1).rename("iqr_area")

    stats = stats.merge(counts, left_on="frame", right_index=True)
    stats = stats.merge(means, left_on="frame", right_index=True)
    stats = stats.merge(iqrs, left_on="frame", right_index=True)

    n_frames = len(stats)

    # --- Modified Z-scores on frame-to-frame deltas ---
    def _mad_z(series):
        """Compute MAD-based modified Z-scores. Returns array of same length."""
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        if mad == 0:
            return np.zeros_like(series, dtype=float)
        return 0.6745 * (series - median) / mad

    # Deltas (frame-to-frame changes) — undefined for frame 0
    d_area = np.diff(stats["mean_area"].values)
    d_iqr = np.diff(stats["iqr_area"].values)

    # Z-scores for deltas (frames 1..N-1)
    z_area_delta = _mad_z(d_area) if len(d_area) > 0 else np.array([])
    z_iqr_delta = _mad_z(d_iqr) if len(d_iqr) > 0 else np.array([])

    # Cell count uses absolute Z-scores for ALL frames — this naturally
    # handles "recovery" frames (a frame returning to normal after a bad
    # frame has a normal absolute count, so it won't be flagged).
    z_count = _mad_z(stats["cell_count"].values.astype(float))

    # Area and IQR: absolute Z for frame 0, delta Z for frames 1+
    z_area = np.zeros(n_frames)
    z_iqr = np.zeros(n_frames)

    z_area_abs = _mad_z(stats["mean_area"].values)
    z_iqr_abs = _mad_z(stats["iqr_area"].values)
    z_area[0] = z_area_abs[0]
    z_iqr[0] = z_iqr_abs[0]

    if len(z_area_delta) > 0:
        z_area[1:] = z_area_delta
        z_iqr[1:] = z_iqr_delta

    stats["z_count"] = z_count
    stats["z_area"] = z_area
    stats["z_iqr"] = z_iqr

    # --- Flag frames where ANY signal exceeds threshold ---
    # For area/IQR on frames 1+, require both delta Z AND absolute Z to
    # exceed thresholds — delta Z alone is too sensitive to sampling noise.
    flagged = np.zeros(n_frames, dtype=bool)
    reasons_list = []
    for i in range(n_frames):
        r = []
        if abs(z_count[i]) > z_threshold:
            r.append(f"count(z={z_count[i]:.1f})")
        if abs(z_area[i]) > z_threshold:
            if i == 0 or abs(z_area_abs[i]) > z_threshold / 2:
                r.append(f"area(z={z_area[i]:.1f})")
        if abs(z_iqr[i]) > z_threshold:
            if i == 0 or abs(z_iqr_abs[i]) > z_threshold / 2:
                r.append(f"iqr(z={z_iqr[i]:.1f})")
        if r:
            flagged[i] = True
        reasons_list.append(", ".join(r))

    stats["flagged"] = flagged
    stats["reasons"] = reasons_list

    bad_frames = stats.loc[stats["flagged"], "frame"].tolist()

    if bad_frames:
        print(f"Frame gating: {len(bad_frames)} frame(s) flagged: {bad_frames}")
        for _, row in stats[stats["flagged"]].iterrows():
            print(f"  Frame {int(row['frame'])}: {row['reasons']}")
    else:
        print("Frame gating: no anomalous frames detected")

    return bad_frames, stats
