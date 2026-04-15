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
