# tests/test_frame_gating.py
import numpy as np
import pandas as pd
import pytest

from cell_analysis.tracking import detect_bad_frames


def _make_detections(counts_per_frame, areas_per_frame=None):
    """Helper: build a detections DataFrame from per-frame cell counts and areas."""
    records = []
    for t, n in enumerate(counts_per_frame):
        if areas_per_frame is not None:
            areas = areas_per_frame[t]
        else:
            areas = np.random.default_rng(t).normal(900, 100, size=n).clip(300, 1500)
        for i in range(n):
            records.append({
                "frame": t,
                "label": i + 1,
                "centroid_y": float(i * 10),
                "centroid_x": float(i * 10),
                "area": float(areas[i]) if areas_per_frame else float(areas[i]),
            })
    return pd.DataFrame(records)


def test_clean_data_no_flags():
    """Smoothly declining cell count, stable area — nothing flagged."""
    counts = [370, 365, 360, 355, 350, 345, 340, 335, 330, 325]
    det = _make_detections(counts)
    bad_frames, diag = detect_bad_frames(det)
    assert bad_frames == []
    assert len(diag) == len(counts)
    assert "flagged" in diag.columns
    assert not diag["flagged"].any()


def test_single_bad_frame_count_drop():
    """One frame has a massive cell count drop — should be flagged."""
    counts = [370, 365, 360, 100, 355, 350, 345, 340, 335, 330]
    #                         ^^^ frame 3: catastrophic drop
    det = _make_detections(counts)
    bad_frames, diag = detect_bad_frames(det)
    assert 3 in bad_frames
    assert bool(diag.loc[diag["frame"] == 3, "flagged"].iloc[0]) is True


def test_bad_frame_area_spike():
    """One frame has much larger mean area (defocus) — should be flagged."""
    rng = np.random.default_rng(42)
    counts = [300] * 10
    areas = []
    for t in range(10):
        if t == 5:
            # Frame 5: defocused, cells appear 3x larger
            areas.append(rng.normal(2700, 200, size=300).clip(1500, 4000))
        else:
            areas.append(rng.normal(900, 100, size=300).clip(300, 1500))
    det = _make_detections(counts, areas)
    bad_frames, diag = detect_bad_frames(det)
    assert 5 in bad_frames


def test_bad_first_frame():
    """Frame 0 is an outlier in absolute metrics — should be flagged."""
    counts = [50, 365, 360, 355, 350, 345, 340, 335, 330, 325]
    #         ^^ frame 0: far fewer cells than the rest
    det = _make_detections(counts)
    bad_frames, diag = detect_bad_frames(det)
    assert 0 in bad_frames


def test_multiple_bad_frames():
    """Frames 0 and 1 both bad (the original TRIM_FRAMES=2 scenario)."""
    counts = [50, 80, 365, 360, 355, 350, 345, 340, 335, 330]
    det = _make_detections(counts)
    bad_frames, diag = detect_bad_frames(det)
    assert 0 in bad_frames
    assert 1 in bad_frames
    # Good frames should NOT be flagged
    for f in [2, 3, 4, 5, 6, 7, 8, 9]:
        assert f not in bad_frames


def test_custom_threshold():
    """Higher threshold = fewer flags; lower = more flags."""
    counts = [370, 365, 360, 320, 355, 350, 345, 340, 335, 330]
    #                         ^^^ moderate drop at frame 3
    det = _make_detections(counts)

    # Strict threshold: might not flag the moderate drop
    bad_strict, _ = detect_bad_frames(det, z_threshold=5.0)

    # Loose threshold: should flag it
    bad_loose, _ = detect_bad_frames(det, z_threshold=2.0)

    assert len(bad_loose) >= len(bad_strict)


def test_diagnostics_structure():
    """Diagnostics DataFrame has expected columns and one row per frame."""
    counts = [370, 365, 100, 355, 350]
    det = _make_detections(counts)
    bad_frames, diag = detect_bad_frames(det)

    assert list(diag["frame"]) == [0, 1, 2, 3, 4]
    expected_cols = {
        "frame", "cell_count", "mean_area", "iqr_area",
        "z_count", "z_area", "z_iqr", "flagged", "reasons",
    }
    assert expected_cols.issubset(set(diag.columns))
    # reasons should be a string (possibly empty)
    assert all(isinstance(r, str) for r in diag["reasons"])


def test_too_few_frames_no_crash():
    """With very few frames, function should return without crashing."""
    counts = [370, 365]
    det = _make_detections(counts)
    bad_frames, diag = detect_bad_frames(det)
    assert isinstance(bad_frames, list)
    assert len(diag) == 2


def test_empty_detections():
    """Empty input should return empty results."""
    det = pd.DataFrame(columns=["frame", "label", "centroid_y", "centroid_x", "area"])
    bad_frames, diag = detect_bad_frames(det)
    assert bad_frames == []
    assert len(diag) == 0
