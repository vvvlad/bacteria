import numpy as np
import pandas as pd

from cell_analysis.pipeline import add_peri_core_alignment


def _make_tracked_ts():
    """Track 1 disappears at frame 4; asymm 0.20, 0.05, -0.10 at frames 2-4.
    Track 2 disappears at frame 3; asymm -0.40, -0.20 at frames 1-2. Missing
    frame 3's asymm (NaN) — should still surface because baseline (-window)
    is not NaN.
    Track 3 survives (excluded).
    Track 4 disappears but has no baseline (first detection too late — dropped).
    """
    tracked = pd.DataFrame({
        "track_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4],
        "frame":    [2, 3, 4, 1, 2, 3, 0, 1, 2, 3, 4],
        "peri_core_asymmetry": [
            0.20, 0.05, -0.10,     # track 1
            -0.40, -0.20, np.nan,  # track 2 (missing at offset 0)
             0.10,  0.11, 0.12,    # track 3 (survives)
            -0.15, -0.05,          # track 4 (first detection at frame 3 → no offset -2)
        ],
    })
    track_stats = pd.DataFrame({
        "track_id":    [1, 2, 3, 4],
        "last_frame":  [4, 3, 2, 4],
        "disappeared": [True, True, False, True],
    })
    return tracked, track_stats


def test_only_disappeared_tracks_with_baseline():
    tracked, track_stats = _make_tracked_ts()
    out = add_peri_core_alignment(tracked, track_stats, window=2)
    # Track 3 (survived) and Track 4 (no baseline) both excluded.
    assert set(out["track_id"].unique()) == {1, 2}


def test_offset_grid_and_delta():
    tracked, track_stats = _make_tracked_ts()
    out = add_peri_core_alignment(tracked, track_stats, window=2)
    # 2 tracks * 3 offsets = 6 rows
    assert len(out) == 6
    for tid in (1, 2):
        sub = out[out["track_id"] == tid].sort_values("offset")
        assert sub["offset"].tolist() == [-2, -1, 0]
        # delta_from_baseline at offset -window is always 0
        assert sub.iloc[0]["delta_from_baseline"] == 0.0
    # Track 1: baseline=0.20, delta at offset 0 = -0.10 - 0.20 = -0.30
    t1 = out[out["track_id"] == 1].set_index("offset")
    assert abs(t1.loc[0, "delta_from_baseline"] - (-0.30)) < 1e-9
    # Track 2: baseline=-0.40; offset 0 asymm is NaN → delta is NaN
    t2 = out[out["track_id"] == 2].set_index("offset")
    assert np.isnan(t2.loc[0, "peri_core_asymmetry"])
    assert np.isnan(t2.loc[0, "delta_from_baseline"])


def test_empty_when_no_disappeared():
    tracked = pd.DataFrame({
        "track_id": [1, 1, 1],
        "frame":    [0, 1, 2],
        "peri_core_asymmetry": [0.1, 0.2, 0.3],
    })
    track_stats = pd.DataFrame({
        "track_id": [1], "last_frame": [2], "disappeared": [False],
    })
    out = add_peri_core_alignment(tracked, track_stats, window=2)
    assert out.empty
    assert list(out.columns) == [
        "track_id", "offset", "peri_core_asymmetry", "delta_from_baseline",
    ]
