import numpy as np
import pandas as pd

from cell_analysis.pipeline import add_fluorescence_alignment


def _make_inputs():
    """Two disappeared tracks, one survived. window=2 → 5 offsets each.

    Track 1 disappears at frame 4, pre-disappearance intensities [10, 20, 30].
    Track 2 disappears at frame 3, pre-disappearance intensities [50, 60].
    Track 3 survives (excluded from output).
    """
    tracked = pd.DataFrame({
        "track_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "frame":    [2, 3, 4, 1, 2, 3, 0, 1, 2],
        "label":    [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "mean_intensity": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 5.0, 6.0, 7.0],
    })
    track_stats = pd.DataFrame({
        "track_id": [1, 2, 3],
        "last_frame": [4, 3, 2],
        "disappeared": [True, True, False],
    })
    # 6-frame stack: 0..5. Plenty of post-disappearance room.
    label_stack = np.zeros((6, 4, 4), dtype=np.int32)
    fluor_stack = np.zeros((6, 4, 4), dtype=np.float32)
    # Track 1: label-1 in top-left at frames 0..4; intensity = 30 in mask
    label_stack[0:5, 0:2, 0:2] = 1
    fluor_stack[0:5, 0:2, 0:2] = 30.0  # post-disappearance value also 30
    # Track 2: label-2 in bottom-right at frames 0..3; intensity = 70 in mask
    label_stack[0:4, 2:4, 2:4] = 2
    fluor_stack[0:4, 2:4, 2:4] = 70.0
    # Last-mask post-disappearance frames keep the same hot region:
    fluor_stack[4, 2:4, 2:4] = 70.0  # frame 4 = track 2's offset +1
    fluor_stack[5, 2:4, 2:4] = 70.0  # frame 5 = track 2's offset +2
    return tracked, track_stats, label_stack, fluor_stack


def test_only_disappeared_tracks_appear():
    tracked, track_stats, label_stack, fluor_stack = _make_inputs()
    out = add_fluorescence_alignment(
        tracked, track_stats, fluor_stack, label_stack, window=2,
    )
    assert set(out["track_id"].unique()) == {1, 2}


def test_offset_grid_and_normalization():
    tracked, track_stats, label_stack, fluor_stack = _make_inputs()
    out = add_fluorescence_alignment(
        tracked, track_stats, fluor_stack, label_stack, window=2,
    )
    # 2 cells * 5 offsets = 10 rows
    assert len(out) == 10
    # Offsets cover -2..+2 for each cell
    for tid in (1, 2):
        sub = out[out["track_id"] == tid].sort_values("offset")
        assert sub["offset"].tolist() == [-2, -1, 0, 1, 2]
        # F_norm at offset=-2 is 1.0 by construction
        assert sub.iloc[0]["F_norm"] == 1.0


def test_drops_track_when_window_start_is_missing():
    """Track whose first detection is too late to provide the offset=-window
    baseline should be dropped entirely, not normalized against NaN."""
    tracked = pd.DataFrame({
        "track_id": [1, 1],
        "frame":    [3, 4],   # first detection at frame 3
        "label":    [1, 1],
        "mean_intensity": [100.0, 50.0],
    })
    track_stats = pd.DataFrame({
        "track_id": [1], "last_frame": [4], "disappeared": [True],
    })
    label_stack = np.zeros((5, 2, 2), dtype=np.int32)
    label_stack[3:5, 0, 0] = 1
    fluor_stack = np.zeros((5, 2, 2), dtype=np.float32)
    fluor_stack[3:5, 0, 0] = 100.0

    out = add_fluorescence_alignment(
        tracked, track_stats, fluor_stack, label_stack, window=2,
    )
    # window=2 wants offset -2..+2, i.e., frames 2..6. Track first appears
    # at frame 3 → offset=-2 (frame 2) is missing → drop the whole track.
    assert out.empty
