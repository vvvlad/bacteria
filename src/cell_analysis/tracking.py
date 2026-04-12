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
