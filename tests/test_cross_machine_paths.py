"""Cross-machine path resolution for load_extraction_with_stacks.

Covers the machine1→machine2 workflow: extraction is done on one host,
analysis on another whose raw TIFFs may live at a different path.
"""
import os

import numpy as np
import pandas as pd
import pytest
import tifffile

from cell_analysis.io import (
    FLUOR_ROOT_ENV,
    finalize_extraction_run,
    load_extraction_with_stacks,
)


def _write_tiny_tiff(path, value=0):
    tifffile.imwrite(path, np.full((2, 3, 4), value, dtype=np.uint16))


def _minimal_inputs(stack_path, fluor_path):
    return {
        "label_stack": np.zeros((2, 3, 4), dtype=np.int32),
        "nucleus_label_stack": np.zeros((2, 3, 4), dtype=np.int32),
        "tracked": pd.DataFrame({
            "frame": [0], "track_id": [1], "label": [1],
            "centroid_y": [1.0], "centroid_x": [1.0], "area": [10],
        }),
        "track_stats": pd.DataFrame({
            "track_id": [1], "first_frame": [0], "last_frame": [0],
            "mean_area": [10.0], "num_detections": [1],
            "lifetime": [1], "disappeared": [False],
        }),
        "merge_log": pd.DataFrame({"from_id": [], "to_id": []}),
        "params": {
            "STACK_PATH": str(stack_path), "FLUOR_PATH": str(fluor_path),
            "GATING_Z_THRESHOLD": 3.5,
        },
        "stack_path": stack_path,
        "fluor_path": fluor_path,
    }


@pytest.fixture
def extracted_run(tmp_path):
    """Machine1: write raw TIFFs + finalize an extraction bundle."""
    stack = tmp_path / "machine1" / "phase.tif"
    fluor = tmp_path / "machine1" / "fluor.tif"
    stack.parent.mkdir(parents=True)
    _write_tiny_tiff(stack, value=0)
    _write_tiny_tiff(fluor, value=7)

    finalize_extraction_run(
        tmp_path / "results" / "run" / "extraction",
        diagnostics=pd.DataFrame({"frame": [0], "flagged": [False]}),
        repo_root=tmp_path,
        **_minimal_inputs(stack, fluor),
    )
    return tmp_path, stack, fluor


def test_recorded_path_resolves_normally(extracted_run):
    tmp_path = extracted_run[0]
    _, phase, fluor = load_extraction_with_stacks(
        tmp_path / "results", "run", repo_root=tmp_path,
    )
    assert phase is not None
    assert phase.shape == (2, 3, 4)
    assert fluor.shape == (2, 3, 4)
    # Fluor value should be 7 (from the fixture).
    assert fluor.max() == 7


def test_fluor_root_fallback_by_basename(extracted_run, tmp_path):
    """Machine2: the recorded path doesn't exist locally, but a
    directory in --fluor-root contains a file with the same basename."""
    _tmp1, stack, fluor = extracted_run
    machine2_data = tmp_path / "machine2_data"
    machine2_data.mkdir()
    # Move both raw TIFFs to machine2's location.
    stack.rename(machine2_data / stack.name)
    fluor.rename(machine2_data / fluor.name)

    _, phase, fluor_loaded = load_extraction_with_stacks(
        _tmp1 / "results", "run",
        repo_root=_tmp1,
        fluor_roots=[machine2_data],
    )
    assert phase is not None
    assert phase.shape == (2, 3, 4)
    assert fluor_loaded.max() == 7


def test_env_var_equivalent_to_flag(extracted_run, tmp_path, monkeypatch):
    _tmp1, stack, fluor = extracted_run
    machine2_data = tmp_path / "machine2_via_env"
    machine2_data.mkdir()
    stack.rename(machine2_data / stack.name)
    fluor.rename(machine2_data / fluor.name)

    monkeypatch.setenv(FLUOR_ROOT_ENV, str(machine2_data))
    _, _phase, fluor_loaded = load_extraction_with_stacks(
        _tmp1 / "results", "run", repo_root=_tmp1,
    )
    assert fluor_loaded.max() == 7


def test_env_var_multi_entry(extracted_run, tmp_path, monkeypatch):
    """Multiple roots separated by os.pathsep; only one contains the file."""
    _tmp1, stack, fluor = extracted_run
    good = tmp_path / "has_it"
    empty = tmp_path / "empty"
    good.mkdir()
    empty.mkdir()
    stack.rename(good / stack.name)
    fluor.rename(good / fluor.name)

    monkeypatch.setenv(
        FLUOR_ROOT_ENV, f"{empty}{os.pathsep}{good}")
    _, _phase, fluor_loaded = load_extraction_with_stacks(
        _tmp1 / "results", "run", repo_root=_tmp1,
    )
    assert fluor_loaded.max() == 7


def test_missing_fluor_raises_with_paths_tried(extracted_run, tmp_path):
    """When no root has the fluor file, the error lists every path tried."""
    _tmp1, stack, fluor = extracted_run
    # Remove both raw TIFFs entirely.
    stack.unlink()
    fluor.unlink()
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_extraction_with_stacks(
            _tmp1 / "results", "run",
            repo_root=_tmp1,
            fluor_roots=[empty_root],
        )
    msg = str(exc_info.value)
    # Recorded path AND the fallback root path both appear.
    assert "fluor.tif" in msg
    assert str(empty_root) in msg
    # Actionable hint mentions the CLI flag.
    assert "--fluor-root" in msg


def test_missing_phase_only_warns(extracted_run, tmp_path):
    """Fluor available, phase missing → warn, return None for phase."""
    _tmp1, stack, fluor = extracted_run
    stack.unlink()  # phase gone, fluor still present at recorded path

    with pytest.warns(RuntimeWarning, match="Phase stack not found"):
        bundle, phase, fluor_loaded = load_extraction_with_stacks(
            _tmp1 / "results", "run", repo_root=_tmp1,
        )
    assert phase is None
    assert fluor_loaded.shape == (2, 3, 4)
    # Bundle itself is unaffected.
    assert bundle.tracked.iloc[0]["track_id"] == 1


def test_flag_takes_precedence_over_env(extracted_run, tmp_path, monkeypatch):
    """When both --fluor-root and env var are set and only one has the
    file, either resolving is fine; verify the arg-list order is
    honored (arg roots probed first)."""
    _tmp1, stack, fluor = extracted_run
    arg_root = tmp_path / "arg_root"
    env_root = tmp_path / "env_root"
    arg_root.mkdir()
    env_root.mkdir()
    # Put the file in arg_root; leave env_root empty.
    stack.rename(arg_root / stack.name)
    fluor.rename(arg_root / fluor.name)

    monkeypatch.setenv(FLUOR_ROOT_ENV, str(env_root))
    _, _phase, fluor_loaded = load_extraction_with_stacks(
        _tmp1 / "results", "run",
        repo_root=_tmp1,
        fluor_roots=[arg_root],
    )
    assert fluor_loaded.max() == 7
