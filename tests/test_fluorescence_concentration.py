import numpy as np
import pandas as pd
import pytest

from cell_analysis.pipeline import add_fluorescence_concentration


def _make_tracked():
    """Tracked DataFrame with volume and total_intensity columns."""
    return pd.DataFrame({
        "frame": [0, 1, 2, 0, 1, 2],
        "track_id": [1, 1, 1, 2, 2, 2],
        "label": [1, 1, 1, 2, 2, 2],
        "area": [800, 900, 1000, 800, 800, 800],
        "volume": [1000.0, 1500.0, 2000.0, 1000.0, 1000.0, 1000.0],
        "total_intensity": [5000.0, 5000.0, 5000.0, 5000.0, 3000.0, 1000.0],
    })


def test_concentration_column_added():
    tracked = _make_tracked()
    result = add_fluorescence_concentration(tracked)
    assert "fluor_concentration" in result.columns


def test_concentration_values():
    tracked = _make_tracked()
    result = add_fluorescence_concentration(tracked)
    t1 = result[result["track_id"] == 1].sort_values("frame")
    conc = t1["fluor_concentration"].values
    assert conc[0] > conc[2], "Concentration should drop as cell swells with constant F"

    t2 = result[result["track_id"] == 2].sort_values("frame")
    conc2 = t2["fluor_concentration"].values
    assert conc2[0] > conc2[2], "Concentration should drop as F decreases"


def test_concentration_exact_values():
    tracked = _make_tracked()
    result = add_fluorescence_concentration(tracked)
    t1 = result[result["track_id"] == 1].sort_values("frame")
    expected = [5000.0 / 1000.0, 5000.0 / 1500.0, 5000.0 / 2000.0]
    np.testing.assert_allclose(t1["fluor_concentration"].values, expected)


def test_missing_volume_gives_nan():
    tracked = _make_tracked()
    tracked.loc[0, "volume"] = 0.0
    result = add_fluorescence_concentration(tracked)
    assert np.isnan(result.loc[0, "fluor_concentration"])


def test_missing_intensity_gives_nan():
    tracked = _make_tracked()
    tracked.loc[0, "total_intensity"] = np.nan
    result = add_fluorescence_concentration(tracked)
    assert np.isnan(result.loc[0, "fluor_concentration"])
