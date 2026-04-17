import numpy as np
import pandas as pd
import pytest

from cell_analysis.matching import compute_preburst_fluorescence


def _make_tracked():
    return pd.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5],
        "track_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "label": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        "mean_intensity": [
            100.0, 100.0, 100.0, 120.0, 150.0, 50.0,
            100.0, 90.0, 80.0, 70.0, 60.0, 50.0,
        ],
        "total_intensity": [
            5000, 5000, 5000, 6000, 7500, 2500,
            5000, 4500, 4000, 3500, 3000, 2500,
        ],
    })


def _make_track_stats():
    return pd.DataFrame({
        "track_id": [1, 2],
        "disappeared": [True, True],
        "last_frame": [5, 5],
    })


def test_output_columns():
    tracked = _make_tracked()
    ts = _make_track_stats()
    result = compute_preburst_fluorescence(tracked, ts, n_frames=3)
    expected_cols = {"track_id", "preburst_slope", "preburst_spike"}
    assert expected_cols.issubset(set(result.columns))


def test_spike_detected():
    tracked = _make_tracked()
    ts = _make_track_stats()
    result = compute_preburst_fluorescence(tracked, ts, n_frames=4)
    t1 = result[result["track_id"] == 1].iloc[0]
    assert t1["preburst_spike"], "Track 1 should have a pre-burst spike"


def test_decline_no_spike():
    tracked = _make_tracked()
    ts = _make_track_stats()
    result = compute_preburst_fluorescence(tracked, ts, n_frames=4)
    t2 = result[result["track_id"] == 2].iloc[0]
    assert not t2["preburst_spike"], "Track 2 (steady decline) should not show spike"


def test_positive_slope_for_spike():
    tracked = _make_tracked()
    ts = _make_track_stats()
    result = compute_preburst_fluorescence(tracked, ts, n_frames=4)
    t1 = result[result["track_id"] == 1].iloc[0]
    t2 = result[result["track_id"] == 2].iloc[0]
    assert t1["preburst_slope"] > t2["preburst_slope"]


def test_survived_tracks_excluded():
    tracked = _make_tracked()
    ts = _make_track_stats()
    ts.loc[1, "disappeared"] = False
    result = compute_preburst_fluorescence(tracked, ts, n_frames=3)
    assert len(result) == 1
    assert result.iloc[0]["track_id"] == 1


def test_short_track_nan():
    tracked = pd.DataFrame({
        "frame": [4, 5], "track_id": [3, 3], "label": [3, 3],
        "mean_intensity": [100.0, 50.0], "total_intensity": [5000, 2500],
    })
    ts = pd.DataFrame({
        "track_id": [3], "disappeared": [True], "last_frame": [5],
    })
    result = compute_preburst_fluorescence(tracked, ts, n_frames=3)
    assert np.isnan(result.iloc[0]["preburst_slope"])
