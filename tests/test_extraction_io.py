
import numpy as np
import pandas as pd
import pytest

from cell_analysis.io import (
    ExtractionBundle, compute_provenance, save_extraction, load_extraction,
)


@pytest.fixture
def bundle(tmp_path):
    stack_path = tmp_path / "a.tif"
    fluor_path = tmp_path / "b.tif"
    stack_path.write_bytes(b"phase")
    fluor_path.write_bytes(b"fluor")
    prov = compute_provenance(
        {"GATING_Z_THRESHOLD": 3.5}, stack_path, fluor_path,
    )
    label_stack = np.zeros((3, 4, 5), dtype=np.int32)
    label_stack[0, 1, 1] = 1
    nucleus_label_stack = np.zeros((3, 4, 5), dtype=np.int32)
    tracked = pd.DataFrame({
        "frame": [0, 1], "track_id": [1, 1], "label": [1, 1],
        "centroid_y": [1.0, 1.0], "centroid_x": [1.0, 1.0],
        "area": [10, 11],
    })
    track_stats = pd.DataFrame({
        "track_id": [1], "first_frame": [0], "last_frame": [1],
        "mean_area": [10.5], "num_detections": [2],
        "lifetime": [2], "disappeared": [False],
    })
    diagnostics = pd.DataFrame({"frame": [0, 1, 2], "flagged": [False]*3})
    merge_log = pd.DataFrame({"from_id": [], "to_id": []})
    dropped_frames = pd.DataFrame({"frame": [], "reason": []})
    return dict(
        label_stack=label_stack,
        nucleus_label_stack=nucleus_label_stack,
        tracked=tracked, track_stats=track_stats,
        diagnostics=diagnostics, merge_log=merge_log,
        dropped_frames=dropped_frames, provenance=prov,
    )


def test_save_creates_all_files(tmp_path, bundle):
    results_dir = tmp_path / "run" / "extraction"
    save_extraction(results_dir, **bundle)
    for name in ("provenance.json", "label_stack.npz",
                 "nucleus_label_stack.npz", "tracked_cells.csv",
                 "track_statistics.csv", "frame_diagnostics.csv",
                 "merge_log.csv", "dropped_frames.csv"):
        assert (results_dir / name).exists(), f"missing {name}"


def test_roundtrip_preserves_arrays(tmp_path, bundle):
    save_extraction(tmp_path / "run" / "extraction", **bundle)
    loaded = load_extraction(tmp_path, "run")
    assert isinstance(loaded, ExtractionBundle)
    np.testing.assert_array_equal(loaded.label_stack, bundle["label_stack"])
    assert loaded.label_stack.dtype == np.int32
    np.testing.assert_array_equal(
        loaded.nucleus_label_stack, bundle["nucleus_label_stack"])


def test_roundtrip_preserves_dataframes(tmp_path, bundle):
    save_extraction(tmp_path / "run" / "extraction", **bundle)
    loaded = load_extraction(tmp_path, "run")
    pd.testing.assert_frame_equal(
        loaded.tracked.reset_index(drop=True),
        bundle["tracked"].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        loaded.track_stats.reset_index(drop=True),
        bundle["track_stats"].reset_index(drop=True))


def test_roundtrip_preserves_provenance(tmp_path, bundle):
    save_extraction(tmp_path / "run" / "extraction", **bundle)
    loaded = load_extraction(tmp_path, "run")
    assert loaded.provenance["extract_hash"] == bundle["provenance"]["extract_hash"]


def test_missing_provenance_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_extraction(tmp_path, "does_not_exist")
