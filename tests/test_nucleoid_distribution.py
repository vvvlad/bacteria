import numpy as np
import pandas as pd

from cell_analysis.pipeline import add_nucleoid_distribution


def _disk_mask(size, cy, cx, r):
    yy, xx = np.indices((size, size))
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def test_uniform_intensity_gives_ratio_near_one():
    """Uniform-intensity disk: mean_edge_distance / (R/3) ≈ 1."""
    size = 80
    mask = _disk_mask(size, 40, 40, 25)
    label_stack = np.zeros((1, size, size), dtype=np.int32)
    label_stack[0][mask] = 1
    fluor_stack = np.zeros((1, size, size), dtype=np.float32)
    fluor_stack[0][mask] = 100.0  # uniform inside

    tracked = pd.DataFrame({
        "track_id": [1], "frame": [0], "label": [1],
        "area": [int(mask.sum())],
    })
    track_stats = pd.DataFrame({"track_id": [1]})

    out_tracked, _ = add_nucleoid_distribution(
        tracked, track_stats, fluor_stack, label_stack,
    )
    ratio = out_tracked["mean_edge_distance_norm"].iloc[0]
    assert 0.85 < ratio < 1.15


def test_peripheral_cluster_gives_ratio_below_one():
    """Intensity only in a thin annulus near the edge ⇒ ratio < 1."""
    size = 80
    mask = _disk_mask(size, 40, 40, 25)
    annulus = mask & ~_disk_mask(size, 40, 40, 22)
    label_stack = np.zeros((1, size, size), dtype=np.int32)
    label_stack[0][mask] = 1
    fluor_stack = np.zeros((1, size, size), dtype=np.float32)
    fluor_stack[0][annulus] = 100.0
    fluor_stack[0][mask & ~annulus] = 10.0

    tracked = pd.DataFrame({
        "track_id": [1], "frame": [0], "label": [1],
        "area": [int(mask.sum())],
    })
    track_stats = pd.DataFrame({"track_id": [1]})
    out_tracked, _ = add_nucleoid_distribution(
        tracked, track_stats, fluor_stack, label_stack,
    )
    assert out_tracked["mean_edge_distance_norm"].iloc[0] < 0.5


def test_central_cluster_gives_ratio_above_one():
    """Intensity only at center ⇒ ratio > 1."""
    size = 80
    mask = _disk_mask(size, 40, 40, 25)
    center = _disk_mask(size, 40, 40, 4)
    label_stack = np.zeros((1, size, size), dtype=np.int32)
    label_stack[0][mask] = 1
    fluor_stack = np.zeros((1, size, size), dtype=np.float32)
    fluor_stack[0][center] = 100.0
    fluor_stack[0][mask & ~center] = 10.0

    tracked = pd.DataFrame({
        "track_id": [1], "frame": [0], "label": [1],
        "area": [int(mask.sum())],
    })
    track_stats = pd.DataFrame({"track_id": [1]})
    out_tracked, _ = add_nucleoid_distribution(
        tracked, track_stats, fluor_stack, label_stack,
    )
    assert out_tracked["mean_edge_distance_norm"].iloc[0] > 1.5


def test_gaussian_sigma_grows_with_dispersion():
    """Two cells: one with a tight central blob, one with broad intensity."""
    size = 80
    cell1_mask = _disk_mask(size, 40, 20, 25)
    cell2_mask = _disk_mask(size, 40, 60, 25)
    tight = _disk_mask(size, 40, 20, 3)
    label_stack = np.zeros((1, size, size), dtype=np.int32)
    label_stack[0][cell1_mask] = 1
    label_stack[0][cell2_mask] = 2
    fluor_stack = np.zeros((1, size, size), dtype=np.float32)
    fluor_stack[0][tight] = 100.0
    fluor_stack[0][cell1_mask & ~tight] = 5.0
    fluor_stack[0][cell2_mask] = 100.0

    tracked = pd.DataFrame({
        "track_id": [1, 2], "frame": [0, 0], "label": [1, 2],
        "area": [int(cell1_mask.sum()), int(cell2_mask.sum())],
    })
    track_stats = pd.DataFrame({"track_id": [1, 2]})
    out_tracked, _ = add_nucleoid_distribution(
        tracked, track_stats, fluor_stack, label_stack,
    )
    sigma1 = out_tracked.loc[out_tracked["track_id"] == 1,
                             "gaussian_sigma_norm"].iloc[0]
    sigma2 = out_tracked.loc[out_tracked["track_id"] == 2,
                             "gaussian_sigma_norm"].iloc[0]
    assert sigma2 > sigma1
