import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_experiment import (
    run_single_config, resolve_results_root, extraction_is_stale,
)
from cell_analysis.io import compute_provenance, save_extraction


def _write_config(tmp_path, stack_path, fluor_path, run_name="run_x"):
    cfg = {
        "RUN_NAME": run_name,
        "RESULTS_ROOT": str(tmp_path / "results"),
        "extraction": {
            "STACK_PATH": str(stack_path),
            "FLUOR_PATH": str(fluor_path),
            "GATING_Z_THRESHOLD": 3.5,
        },
        "analysis": {"PIXEL_SIZE_UM": 0.0645},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p, cfg


def _prime_extraction(tmp_path, cfg):
    """Populate a synthetic extraction directory that provenance-matches cfg."""
    stack = Path(cfg["extraction"]["STACK_PATH"])
    fluor = Path(cfg["extraction"]["FLUOR_PATH"])
    prov = compute_provenance(cfg["extraction"], stack, fluor)
    results_dir = Path(cfg["RESULTS_ROOT"]) / cfg["RUN_NAME"] / "extraction"
    empty = pd.DataFrame()
    save_extraction(
        results_dir,
        label_stack=np.zeros((1, 2, 2), np.int32),
        nucleus_label_stack=np.zeros((1, 2, 2), np.int32),
        tracked=pd.DataFrame({"frame": [], "track_id": [], "y": [],
                              "x": [], "area_pixels": []}),
        track_stats=pd.DataFrame({"track_id": [], "first_frame": [],
                                  "last_frame": [], "lifetime": [],
                                  "num_detections": [], "disappeared": []}),
        diagnostics=empty, merge_log=empty, dropped_frames=empty,
        provenance=prov,
    )


@pytest.fixture
def stacks(tmp_path):
    s = tmp_path / "phase.tif"
    f = tmp_path / "fluor.tif"
    s.write_bytes(b"phase-bytes")
    f.write_bytes(b"fluor-bytes")
    return s, f


def test_resolve_results_root_cli_wins(tmp_path):
    cfg = {"RESULTS_ROOT": "/from/yaml"}
    got = resolve_results_root(cfg, str(tmp_path / "cli"))
    assert got == (tmp_path / "cli")


def test_resolve_results_root_yaml_used(tmp_path):
    cfg = {"RESULTS_ROOT": str(tmp_path / "yaml")}
    got = resolve_results_root(cfg, None)
    assert got == (tmp_path / "yaml")


def test_resolve_results_root_default_is_repo_results():
    got = resolve_results_root({}, None)
    assert got.name == "results"


def test_extraction_is_stale_missing(tmp_path, stacks):
    s, f = stacks
    stale, reason = extraction_is_stale(tmp_path, {"GATING_Z_THRESHOLD": 3.5}, s, f)
    assert stale is True
    assert "missing" in reason


def test_extraction_fresh_when_provenance_matches(tmp_path, stacks):
    s, f = stacks
    prov = compute_provenance({"GATING_Z_THRESHOLD": 3.5}, s, f)
    empty = pd.DataFrame()
    save_extraction(
        tmp_path,
        label_stack=np.zeros((1, 2, 2), np.int32),
        nucleus_label_stack=np.zeros((1, 2, 2), np.int32),
        tracked=empty, track_stats=empty,
        diagnostics=empty, merge_log=empty, dropped_frames=empty,
        provenance=prov,
    )
    stale, reason = extraction_is_stale(tmp_path, {"GATING_Z_THRESHOLD": 3.5}, s, f)
    assert stale is False


def test_extraction_stale_on_param_drift(tmp_path, stacks):
    s, f = stacks
    prov = compute_provenance({"GATING_Z_THRESHOLD": 3.5}, s, f)
    empty = pd.DataFrame()
    save_extraction(
        tmp_path,
        label_stack=np.zeros((1, 2, 2), np.int32),
        nucleus_label_stack=np.zeros((1, 2, 2), np.int32),
        tracked=empty, track_stats=empty,
        diagnostics=empty, merge_log=empty, dropped_frames=empty,
        provenance=prov,
    )
    stale, reason = extraction_is_stale(tmp_path, {"GATING_Z_THRESHOLD": 4.0}, s, f)
    assert stale is True
    assert "GATING_Z_THRESHOLD" in reason


def test_reuse_skips_extract_notebook(tmp_path, stacks):
    s, f = stacks
    cfg_path, cfg = _write_config(tmp_path, s, f)
    _prime_extraction(tmp_path, cfg)
    with patch("run_experiment.pm.execute_notebook") as m_exec, \
         patch("run_experiment.export_notebook_html"), \
         patch("run_experiment.read_kernel_name", return_value="python3"):
        run_single_config(cfg_path)
        called_paths = [call.args[0] for call in m_exec.call_args_list]
        assert not any("extract.ipynb" in p for p in called_paths)
        assert any("analysis.ipynb" in p for p in called_paths)


def test_force_extract_reruns_extract(tmp_path, stacks):
    s, f = stacks
    cfg_path, cfg = _write_config(tmp_path, s, f)
    _prime_extraction(tmp_path, cfg)
    with patch("run_experiment.pm.execute_notebook") as m_exec, \
         patch("run_experiment.export_notebook_html"), \
         patch("run_experiment.read_kernel_name", return_value="python3"):
        run_single_config(cfg_path, force_extract=True)
        called_paths = [call.args[0] for call in m_exec.call_args_list]
        assert any("extract.ipynb" in p for p in called_paths)


def test_analysis_only_errors_without_extraction(tmp_path, stacks):
    s, f = stacks
    cfg_path, _ = _write_config(tmp_path, s, f)
    with patch("run_experiment.pm.execute_notebook"):
        with pytest.raises(FileNotFoundError):
            run_single_config(cfg_path, analysis_only=True)


def test_skip_analysis_runs_only_extract(tmp_path, stacks):
    s, f = stacks
    cfg_path, _ = _write_config(tmp_path, s, f)
    with patch("run_experiment.pm.execute_notebook") as m_exec, \
         patch("run_experiment.export_notebook_html"), \
         patch("run_experiment.read_kernel_name", return_value="python3"):
        run_single_config(cfg_path, skip_analysis=True)
        called_paths = [call.args[0] for call in m_exec.call_args_list]
        assert any("extract.ipynb" in p for p in called_paths)
        assert not any("analysis.ipynb" in p for p in called_paths)
