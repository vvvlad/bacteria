import numpy as np
import pandas as pd

from cell_analysis.pipeline import _relative_per_track


def test_fluor_rel_helper_first_frame_one():
    df = pd.DataFrame({
        "track_id": [1, 1, 2, 2],
        "frame":    [0, 1, 0, 1],
        "mean_intensity": [100.0, 50.0, 200.0, 400.0],
    })
    df["fluor_rel"] = _relative_per_track(df, "mean_intensity")
    track1 = df[df["track_id"] == 1].sort_values("frame")
    np.testing.assert_allclose(track1["fluor_rel"].values, [1.0, 0.5])
    track2 = df[df["track_id"] == 2].sort_values("frame")
    np.testing.assert_allclose(track2["fluor_rel"].values, [1.0, 2.0])


def test_fluor_rel_nan_when_initial_is_zero():
    df = pd.DataFrame({
        "track_id": [1, 1],
        "frame":    [0, 1],
        "mean_intensity": [0.0, 5.0],
    })
    df["fluor_rel"] = _relative_per_track(df, "mean_intensity")
    assert df["fluor_rel"].isna().all()
