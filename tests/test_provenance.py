from pathlib import Path

import pytest

from cell_analysis.io import compute_provenance, provenance_matches


@pytest.fixture
def stacks(tmp_path):
    a = tmp_path / "a.tif"
    b = tmp_path / "b.tif"
    a.write_bytes(b"phase-bytes")
    b.write_bytes(b"fluor-bytes")
    return a, b


PARAMS = {
    "STACK_PATH": "a.tif",
    "FLUOR_PATH": "b.tif",
    "DETECT_PARAMS": {"diameter": 32, "min_area": 300},
    "GATING_Z_THRESHOLD": 3.5,
}


def test_extract_hash_is_deterministic(stacks):
    a, b = stacks
    p1 = compute_provenance(PARAMS, a, b, package_version="0.1.0")
    p2 = compute_provenance(PARAMS, a, b, package_version="0.1.0")
    assert p1["extract_hash"] == p2["extract_hash"]


def test_extract_hash_ignores_version(stacks):
    a, b = stacks
    p1 = compute_provenance(PARAMS, a, b, package_version="0.1.0")
    p2 = compute_provenance(PARAMS, a, b, package_version="9.9.9")
    assert p1["extract_hash"] == p2["extract_hash"]
    assert p1["cell_analysis_version"] != p2["cell_analysis_version"]


def test_extract_hash_changes_on_param(stacks):
    a, b = stacks
    p1 = compute_provenance(PARAMS, a, b)
    p2 = compute_provenance({**PARAMS, "GATING_Z_THRESHOLD": 4.0}, a, b)
    assert p1["extract_hash"] != p2["extract_hash"]


def test_extract_hash_changes_on_stack_bytes(stacks):
    a, b = stacks
    p1 = compute_provenance(PARAMS, a, b)
    a.write_bytes(b"phase-bytes-different")
    p2 = compute_provenance(PARAMS, a, b)
    assert p1["extract_hash"] != p2["extract_hash"]
    assert p1["stack_sha256"] != p2["stack_sha256"]


def test_key_order_insensitive(stacks):
    a, b = stacks
    p1 = compute_provenance(
        {"DETECT_PARAMS": {"diameter": 32, "min_area": 300}}, a, b
    )
    p2 = compute_provenance(
        {"DETECT_PARAMS": {"min_area": 300, "diameter": 32}}, a, b
    )
    assert p1["extract_hash"] == p2["extract_hash"]


def test_repo_relative_paths(stacks, tmp_path):
    a, b = stacks
    prov = compute_provenance(PARAMS, a, b, repo_root=tmp_path)
    assert prov["stack_path"] == "a.tif"
    assert prov["fluor_path"] == "b.tif"


def test_absolute_paths_when_outside_repo(stacks):
    """Test that files outside repo_root return absolute paths."""
    import tempfile
    a, b = stacks
    # Use a completely different temp directory as repo_root
    with tempfile.TemporaryDirectory() as other_dir:
        other = Path(other_dir)
        prov = compute_provenance(PARAMS, a, b, repo_root=other)
        # Files are NOT under other, so should be absolute
        assert Path(prov["stack_path"]).is_absolute()
        assert Path(prov["fluor_path"]).is_absolute()
def test_provenance_matches_true(stacks):
    a, b = stacks
    prov = compute_provenance(PARAMS, a, b)
    ok, drift = provenance_matches(prov, prov)
    assert ok is True
    assert drift == []


def test_provenance_matches_flags_stack_drift(stacks):
    a, b = stacks
    p1 = compute_provenance(PARAMS, a, b)
    a.write_bytes(b"changed")
    p2 = compute_provenance(PARAMS, a, b)
    ok, drift = provenance_matches(p1, p2)
    assert ok is False
    assert "stack_sha256" in drift


def test_provenance_matches_flags_param_drift(stacks):
    a, b = stacks
    p1 = compute_provenance(PARAMS, a, b)
    p2 = compute_provenance({**PARAMS, "GATING_Z_THRESHOLD": 4.0}, a, b)
    ok, drift = provenance_matches(p1, p2)
    assert ok is False
    assert any("GATING_Z_THRESHOLD" in f for f in drift)


def test_repo_relative_paths_in_subdirectory(tmp_path):
    subdir = tmp_path / "data" / "Control-experiment"
    subdir.mkdir(parents=True)
    a = subdir / "phase.tif"
    b = subdir / "fluor.tif"
    a.write_bytes(b"phase")
    b.write_bytes(b"fluor")
    prov = compute_provenance(PARAMS, a, b, repo_root=tmp_path)
    assert prov["stack_path"] == "data/Control-experiment/phase.tif"
    assert prov["fluor_path"] == "data/Control-experiment/fluor.tif"



def test_extract_hash_ignores_path_values(stacks):
    a, b = stacks
    # Same file bytes, different string representations of the same file.
    p_rel = compute_provenance(
        {**PARAMS, "STACK_PATH": "../data/a.tif", "FLUOR_PATH": "../data/b.tif"},
        a, b,
    )
    p_abs = compute_provenance(
        {**PARAMS, "STACK_PATH": str(a), "FLUOR_PATH": str(b)},
        a, b,
    )
    # The path *strings* differ, but the file bytes are the same →
    # extract_hash must match, so runner+notebook agree on reuse.
    assert p_rel["extract_hash"] == p_abs["extract_hash"]
    # Both provenance dicts preserve the original path string for audit.
    assert p_rel["params"]["STACK_PATH"] == "../data/a.tif"
    assert p_abs["params"]["STACK_PATH"] == str(a)