import numpy as np
import pandas as pd

from cell_analysis.matching import measure_post_disappearance_fluorescence


def test_uses_last_frame_mask_on_subsequent_fluor_frames():
    # 4 frames, 4x4 images. Cell at label=1 occupies the top-left 2x2 block,
    # but only in frames 0 and 1 (then "disappears" — mask is zero).
    label_stack = np.zeros((4, 4, 4), dtype=np.int32)
    label_stack[0, 0:2, 0:2] = 1
    label_stack[1, 0:2, 0:2] = 1
    # frames 2 and 3 have no label-1 region.

    # Fluorescence: constant 100 in the top-left 2x2 across all 4 frames.
    fluor_stack = np.zeros((4, 4, 4), dtype=np.float32)
    fluor_stack[:, 0:2, 0:2] = 100.0

    track_stats = pd.DataFrame({
        "track_id": [1],
        "last_frame": [1],
        "disappeared": [True],
    })

    last_labels = pd.DataFrame({
        "track_id": [1],
        "last_frame_label": [1],
    })

    out = measure_post_disappearance_fluorescence(
        track_stats, last_labels, fluor_stack, label_stack, window=2,
    )

    assert set(out.columns) == {"track_id", "frame", "mean_intensity"}
    assert len(out) == 2
    np.testing.assert_allclose(out["mean_intensity"].values, [100.0, 100.0])
    assert sorted(out["frame"].tolist()) == [2, 3]


def test_skips_frames_past_end_of_stack():
    # 3-frame stack; disappearance at frame 1; window=3 ⇒ would ask for
    # frames 2, 3, 4. Only frame 2 exists.
    label_stack = np.zeros((3, 4, 4), dtype=np.int32)
    label_stack[1, 0:2, 0:2] = 1
    fluor_stack = np.zeros((3, 4, 4), dtype=np.float32)
    fluor_stack[:, 0:2, 0:2] = 50.0

    track_stats = pd.DataFrame({
        "track_id": [1], "last_frame": [1], "disappeared": [True],
    })
    last_labels = pd.DataFrame({"track_id": [1], "last_frame_label": [1]})

    out = measure_post_disappearance_fluorescence(
        track_stats, last_labels, fluor_stack, label_stack, window=3,
    )

    assert len(out) == 1
    assert out["frame"].iloc[0] == 2
    np.testing.assert_allclose(out["mean_intensity"].values, [50.0])
