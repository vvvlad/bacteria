import numpy as np
import pandas as pd
import pytest

from cell_analysis.tracking import compute_migration_stats


def _make_tracked():
    """Two tracks: one stationary, one moving."""
    return pd.DataFrame({
        "frame": [0, 1, 2, 3, 0, 1, 2, 3],
        "track_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "label": [1, 1, 1, 1, 2, 2, 2, 2],
        "centroid_y": [100.0, 100.0, 100.0, 100.0, 100.0, 110.0, 120.0, 130.0],
        "centroid_x": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "area": [800] * 8,
    })


def test_output_columns():
    tracked = _make_tracked()
    result = compute_migration_stats(tracked)
    expected_cols = {"track_id", "mean_speed", "max_speed", "total_displacement",
                     "speed_std", "net_displacement"}
    assert expected_cols.issubset(set(result.columns))


def test_stationary_cell_zero_speed():
    tracked = _make_tracked()
    result = compute_migration_stats(tracked)
    t1 = result[result["track_id"] == 1].iloc[0]
    assert t1["mean_speed"] == 0.0
    assert t1["max_speed"] == 0.0
    assert t1["total_displacement"] == 0.0


def test_moving_cell_correct_speed():
    tracked = _make_tracked()
    result = compute_migration_stats(tracked)
    t2 = result[result["track_id"] == 2].iloc[0]
    assert t2["mean_speed"] == pytest.approx(10.0)
    assert t2["max_speed"] == pytest.approx(10.0)
    assert t2["total_displacement"] == pytest.approx(30.0)
    assert t2["net_displacement"] == pytest.approx(30.0)


def test_single_detection_nan():
    tracked = pd.DataFrame({
        "frame": [0], "track_id": [1], "label": [1],
        "centroid_y": [100.0], "centroid_x": [100.0], "area": [800],
    })
    result = compute_migration_stats(tracked)
    assert np.isnan(result.iloc[0]["mean_speed"])


def test_diagonal_movement():
    tracked = pd.DataFrame({
        "frame": [0, 1], "track_id": [1, 1], "label": [1, 1],
        "centroid_y": [0.0, 3.0], "centroid_x": [0.0, 4.0], "area": [800, 800],
    })
    result = compute_migration_stats(tracked)
    assert result.iloc[0]["mean_speed"] == pytest.approx(5.0)


def test_per_frame_speed_column():
    tracked = _make_tracked()
    result_tracked = tracked.copy()
    compute_migration_stats(tracked, per_frame=result_tracked)
    assert "speed" in result_tracked.columns
    t1_f0 = result_tracked[(result_tracked["track_id"] == 1) & (result_tracked["frame"] == 0)]
    assert np.isnan(t1_f0["speed"].iloc[0])
