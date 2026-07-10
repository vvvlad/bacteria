import numpy as np
import pandas as pd

from cell_analysis.pipeline import add_nucleoid_distribution


def _disk_mask(size, cy, cx, r):
    yy, xx = np.indices((size, size))
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _run_single_disk(fluor_2d, size=80):
    """Run add_nucleoid_distribution on a single centered disk with the given
    2D fluorescence image (same shape as the mask)."""
    mask = _disk_mask(size, 40, 40, 25)
    label_stack = np.zeros((1, size, size), dtype=np.int32)
    label_stack[0][mask] = 1
    fluor_stack = fluor_2d[None, :, :].astype(np.float32)
    tracked = pd.DataFrame({
        "track_id": [1], "frame": [0], "label": [1],
        "area": [int(mask.sum())],
    })
    track_stats = pd.DataFrame({"track_id": [1]})
    return add_nucleoid_distribution(
        tracked, track_stats, fluor_stack, label_stack,
    )


def test_uniform_intensity_gives_ratio_near_one():
    """Uniform-intensity disk: mean_edge_distance / (R/3) ≈ 1."""
    mask = _disk_mask(80, 40, 40, 25)
    fluor = mask * 100.0
    out_tracked, _ = _run_single_disk(fluor)
    assert 0.85 < out_tracked["mean_edge_distance_norm"].iloc[0] < 1.15


def test_peripheral_cluster_gives_ratio_below_one():
    """Intensity only in a thin annulus near the edge ⇒ ratio < 1."""
    mask = _disk_mask(80, 40, 40, 25)
    fluor = mask * 10.0
    fluor[mask & ~_disk_mask(80, 40, 40, 22)] = 100.0
    out_tracked, _ = _run_single_disk(fluor)
    assert out_tracked["mean_edge_distance_norm"].iloc[0] < 0.5


def test_central_cluster_gives_ratio_above_one():
    """Intensity only at center ⇒ ratio > 1."""
    mask = _disk_mask(80, 40, 40, 25)
    fluor = mask * 10.0
    fluor[_disk_mask(80, 40, 40, 4)] = 100.0
    out_tracked, _ = _run_single_disk(fluor)
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


def test_peri_core_asymmetry_uniform_is_zero():
    """Uniform-intensity disk: peri_core_asymmetry ≈ 0."""
    mask = _disk_mask(80, 40, 40, 25)
    fluor = mask * 100.0
    out_tracked, out_stats = _run_single_disk(fluor)
    assert abs(out_tracked["peri_core_asymmetry"].iloc[0]) < 0.01
    assert "mean_peri_core_asymmetry" in out_stats.columns


def test_peri_core_asymmetry_central_cluster_negative():
    """Bright core, dim edge ⇒ peri_core_asymmetry < 0."""
    mask = _disk_mask(80, 40, 40, 25)
    fluor = mask * 10.0
    fluor[_disk_mask(80, 40, 40, 4)] = 100.0
    out_tracked, _ = _run_single_disk(fluor)
    assert out_tracked["peri_core_asymmetry"].iloc[0] < -0.05


def test_peri_core_asymmetry_peripheral_positive():
    """Bright edge annulus, dim core ⇒ peri_core_asymmetry > 0."""
    mask = _disk_mask(80, 40, 40, 25)
    fluor = mask * 10.0
    fluor[mask & ~_disk_mask(80, 40, 40, 22)] = 100.0
    out_tracked, _ = _run_single_disk(fluor)
    assert out_tracked["peri_core_asymmetry"].iloc[0] > 0.3
