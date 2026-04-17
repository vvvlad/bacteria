import numpy as np
import pandas as pd
import pytest

from cell_analysis.tracking import detect_growth_phases


def _make_tracked_two_phase():
    frames = list(range(10))
    areas = [800, 800, 800, 800, 850, 950, 1100, 1300, 1500, 1700]
    return pd.DataFrame({
        "frame": frames,
        "track_id": [1] * 10,
        "label": [1] * 10,
        "area": [float(a) for a in areas],
    })


def _make_tracked_constant():
    frames = list(range(8))
    return pd.DataFrame({
        "frame": frames,
        "track_id": [1] * 8,
        "label": [1] * 8,
        "area": [800.0] * 8,
    })


def test_output_columns():
    tracked = _make_tracked_two_phase()
    result = detect_growth_phases(tracked)
    expected = {"track_id", "changepoint_frame", "slope_before", "slope_after",
                "slope_ratio"}
    assert expected.issubset(set(result.columns))


def test_changepoint_location():
    tracked = _make_tracked_two_phase()
    result = detect_growth_phases(tracked)
    cp = result.iloc[0]["changepoint_frame"]
    assert 3 <= cp <= 5, f"Changepoint should be near transition, got {cp}"


def test_slope_increase_at_changepoint():
    tracked = _make_tracked_two_phase()
    result = detect_growth_phases(tracked)
    r = result.iloc[0]
    assert r["slope_after"] > r["slope_before"], "Slope should increase at changepoint"


def test_constant_track():
    tracked = _make_tracked_constant()
    result = detect_growth_phases(tracked)
    r = result.iloc[0]
    assert abs(r["slope_before"]) < 5.0
    assert abs(r["slope_after"]) < 5.0


def test_short_track_nan():
    tracked = pd.DataFrame({
        "frame": [0, 1], "track_id": [1, 1], "label": [1, 1],
        "area": [800.0, 900.0],
    })
    result = detect_growth_phases(tracked)
    assert np.isnan(result.iloc[0]["changepoint_frame"])


def test_multiple_tracks():
    t1 = _make_tracked_two_phase()
    t2 = _make_tracked_constant()
    t2 = t2.assign(track_id=2, label=2)
    tracked = pd.concat([t1, t2], ignore_index=True)
    result = detect_growth_phases(tracked)
    assert len(result) == 2
    assert set(result["track_id"]) == {1, 2}
