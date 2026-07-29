
import numpy as np
import pandas as pd
import pytest

from cell_analysis.io import (
    EXTRACTION_PARAM_NAMES,
    ExtractionBundle,
    compute_provenance,
    finalize_extraction_run,
    load_extraction,
    load_extraction_with_stacks,
    save_extraction,
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



# ---------------------------------------------------------------------------
# Notebook-facing helpers: finalize_extraction_run, load_extraction_with_stacks
# ---------------------------------------------------------------------------

# (helpers imported at top of file)


def _tiny_bundle_inputs(stack_path, fluor_path):
    """Common in-memory pipeline outputs for finalize_extraction_run tests."""
    label_stack = np.zeros((3, 4, 5), dtype=np.int32)
    label_stack[0, 1, 1] = 1
    nucleus_label_stack = np.zeros((3, 4, 5), dtype=np.int32)
    tracked = pd.DataFrame({
        "frame": [0], "track_id": [1], "label": [1],
        "centroid_y": [1.0], "centroid_x": [1.0], "area": [10],
    })
    track_stats = pd.DataFrame({
        "track_id": [1], "first_frame": [0], "last_frame": [0],
        "mean_area": [10.0], "num_detections": [1],
        "lifetime": [1], "disappeared": [False],
    })
    merge_log = pd.DataFrame({"from_id": [], "to_id": []})
    return {
        "label_stack": label_stack,
        "nucleus_label_stack": nucleus_label_stack,
        "tracked": tracked,
        "track_stats": track_stats,
        "merge_log": merge_log,
        "params": {
            "STACK_PATH": str(stack_path), "FLUOR_PATH": str(fluor_path),
            "GATING_Z_THRESHOLD": 3.5,
        },
        "stack_path": stack_path,
        "fluor_path": fluor_path,
    }


def test_extraction_param_names_is_a_tuple_of_strings():
    """Guards against accidental type change — the runner treats it as a
    membership set, and callers slice globals() by it."""
    assert isinstance(EXTRACTION_PARAM_NAMES, tuple)
    assert all(isinstance(k, str) for k in EXTRACTION_PARAM_NAMES)
    # Sanity check that required keys are present.
    assert "STACK_PATH" in EXTRACTION_PARAM_NAMES
    assert "FLUOR_PATH" in EXTRACTION_PARAM_NAMES


def test_finalize_extraction_run_writes_bundle(tmp_path):
    stack = tmp_path / "phase.tif"
    fluor = tmp_path / "fluor.tif"
    stack.write_bytes(b"phase-bytes")
    fluor.write_bytes(b"fluor-bytes")
    inputs = _tiny_bundle_inputs(stack, fluor)

    diagnostics = pd.DataFrame({
        "frame": [0, 1, 2], "flagged": [False, False, False],
    })
    prov = finalize_extraction_run(
        tmp_path / "run" / "extraction",
        diagnostics=diagnostics,
        **inputs,
    )
    # All 8 artifacts on disk.
    for name in ("provenance.json", "label_stack.npz",
                 "nucleus_label_stack.npz", "tracked_cells.csv",
                 "track_statistics.csv", "frame_diagnostics.csv",
                 "merge_log.csv", "dropped_frames.csv"):
        assert (tmp_path / "run" / "extraction" / name).exists()
    # Provenance dict was returned and looks right.
    assert "extract_hash" in prov
    assert prov["params"]["GATING_Z_THRESHOLD"] == 3.5


@pytest.mark.parametrize("flagged, expected_frames", [
    ([False, True, True], [1, 2]),
    ([False, False, False], []),
])
def test_finalize_extraction_run_derives_dropped_from_diagnostics(
    tmp_path, flagged, expected_frames,
):
    """dropped_frames.csv is derived from diagnostics[flagged] — no disk
    round-trip. Bundle-writing itself is covered by
    test_finalize_extraction_run_writes_bundle above."""
    stack = tmp_path / "phase.tif"
    fluor = tmp_path / "fluor.tif"
    stack.write_bytes(b"a")
    fluor.write_bytes(b"b")
    inputs = _tiny_bundle_inputs(stack, fluor)

    diagnostics = pd.DataFrame({
        "frame": list(range(len(flagged))), "flagged": flagged,
    })
    finalize_extraction_run(
        tmp_path / "run" / "extraction",
        diagnostics=diagnostics,
        **inputs,
    )
    dropped = pd.read_csv(tmp_path / "run" / "extraction" / "dropped_frames.csv")
    assert list(dropped["frame"]) == expected_frames


def test_load_extraction_with_stacks_roundtrip(tmp_path):
    """finalize + load_extraction_with_stacks together — the full
    notebook path in miniature."""
    import tifffile
    stack_path = tmp_path / "phase.tif"
    fluor_path = tmp_path / "fluor.tif"
    # Write real (T, Y, X) TIFFs so load_stack works.
    tifffile.imwrite(stack_path, np.zeros((3, 4, 5), dtype=np.uint16))
    tifffile.imwrite(fluor_path, np.ones((3, 4, 5), dtype=np.uint16))

    inputs = _tiny_bundle_inputs(stack_path, fluor_path)
    diagnostics = pd.DataFrame({"frame": [0], "flagged": [False]})
    finalize_extraction_run(
        tmp_path / "myrun" / "extraction",
        diagnostics=diagnostics,
        repo_root=tmp_path,
        **inputs,
    )

    bundle, phase, fluor = load_extraction_with_stacks(
        tmp_path, "myrun", repo_root=tmp_path,
    )
    assert phase.shape == (3, 4, 5)
    assert fluor.shape == (3, 4, 5)
    assert bundle.label_stack.shape == (3, 4, 5)


def test_load_extraction_with_stacks_squeezes_multichannel(tmp_path):
    """Multi-channel (T, C, Y, X) TIFFs get first channel picked."""
    import tifffile
    stack_path = tmp_path / "phase.tif"
    fluor_path = tmp_path / "fluor.tif"
    tifffile.imwrite(stack_path, np.zeros((3, 2, 4, 5), dtype=np.uint16))
    tifffile.imwrite(fluor_path, np.ones((3, 2, 4, 5), dtype=np.uint16))

    inputs = _tiny_bundle_inputs(stack_path, fluor_path)
    diagnostics = pd.DataFrame({"frame": [0], "flagged": [False]})
    finalize_extraction_run(
        tmp_path / "myrun" / "extraction",
        diagnostics=diagnostics,
        repo_root=tmp_path,
        **inputs,
    )

    _, phase, fluor = load_extraction_with_stacks(
        tmp_path, "myrun", repo_root=tmp_path,
    )
    assert phase.ndim == 3
    assert fluor.ndim == 3
