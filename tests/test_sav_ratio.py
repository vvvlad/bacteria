import numpy as np
import pandas as pd
import pytest

from cell_analysis.pipeline import add_sav_ratio


def _make_tracked():
    return pd.DataFrame({
        "frame": [0, 1, 2, 0, 1, 2],
        "track_id": [1, 1, 1, 2, 2, 2],
        "label": [1, 1, 1, 2, 2, 2],
        "area": [800, 1200, 1600, 800, 800, 800],
        "volume": [1000.0, 2000.0, 4000.0, 1000.0, 1000.0, 1000.0],
        "surface_area": [500.0, 700.0, 900.0, 500.0, 500.0, 500.0],
    })


def test_sav_column_added():
    tracked = _make_tracked()
    result = add_sav_ratio(tracked)
    assert "sav_ratio" in result.columns


def test_sav_values():
    tracked = _make_tracked()
    result = add_sav_ratio(tracked)
    t1 = result[result["track_id"] == 1].sort_values("frame")
    expected = [500.0 / 1000.0, 700.0 / 2000.0, 900.0 / 4000.0]
    np.testing.assert_allclose(t1["sav_ratio"].values, expected)


def test_sav_decreases_with_swelling():
    tracked = _make_tracked()
    result = add_sav_ratio(tracked)
    t1 = result[result["track_id"] == 1].sort_values("frame")
    sav = t1["sav_ratio"].values
    assert sav[0] > sav[1] > sav[2], "SA:V should decrease as cell swells"


def test_sav_zero_volume_gives_nan():
    tracked = _make_tracked()
    tracked.loc[0, "volume"] = 0.0
    result = add_sav_ratio(tracked)
    assert np.isnan(result.loc[0, "sav_ratio"])
