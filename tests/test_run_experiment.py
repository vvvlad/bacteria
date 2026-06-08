import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from run_experiment import validate_config


VALID_CONFIG = {
    "RUN_NAME": "run_01",
    "STACK_PATH": "../data/test_dataset/phase.tif",
    "FLUOR_PATH": "../data/test_dataset/fluorescence.tif",
    "GATING_Z_THRESHOLD": 3.5,
    "DETECT_PARAMS": {
        "diameter": 32,
        "min_area": 300,
        "min_circularity": 0.7,
        "min_contrast": 1250,
        "exclude_edges": True,
        "gpu": True,
        "resample": False,
    },
    "SEARCH_RANGE": 30.0,
    "MEMORY": 3,
    "MERGE_MAX_DISTANCE": 15.0,
    "MERGE_MAX_GAP": 18,
    "MIN_TRACK_DETECTIONS": 4,
    "FLUOR_ALIGN_WINDOW": 3,
}


def test_valid_config_passes():
    validate_config(VALID_CONFIG)


def test_minimal_config_passes():
    minimal = {
        "RUN_NAME": "run_01",
        "STACK_PATH": "../data/test_dataset/phase.tif",
        "FLUOR_PATH": "../data/test_dataset/fluorescence.tif",
    }
    validate_config(minimal)


def test_unknown_key_rejected():
    bad = {**VALID_CONFIG, "SEARCH_RNAGE": 30.0}
    with pytest.raises(ValueError, match="SEARCH_RNAGE"):
        validate_config(bad)


def test_missing_required_key():
    missing = {k: v for k, v in VALID_CONFIG.items() if k != "RUN_NAME"}
    with pytest.raises(ValueError, match="RUN_NAME"):
        validate_config(missing)


def test_non_dict_rejected():
    with pytest.raises(ValueError, match="mapping"):
        validate_config([1, 2, 3])
    with pytest.raises(ValueError, match="mapping"):
        validate_config(None)


def test_run_name_path_traversal():
    bad = {**VALID_CONFIG, "RUN_NAME": "../etc/passwd"}
    with pytest.raises(ValueError, match="path traversal"):
        validate_config(bad)
    bad2 = {**VALID_CONFIG, "RUN_NAME": "foo/bar"}
    with pytest.raises(ValueError, match="path traversal"):
        validate_config(bad2)
    bad3 = {**VALID_CONFIG, "RUN_NAME": "foo\\bar"}
    with pytest.raises(ValueError, match="path traversal"):
        validate_config(bad3)


def test_wrong_type_for_string_field():
    bad = {**VALID_CONFIG, "STACK_PATH": 123}
    with pytest.raises(ValueError, match="STACK_PATH"):
        validate_config(bad)


def test_wrong_type_for_numeric_field():
    bad = {**VALID_CONFIG, "SEARCH_RANGE": "thirty"}
    with pytest.raises(ValueError, match="SEARCH_RANGE"):
        validate_config(bad)


def test_wrong_type_for_int_field():
    bad = {**VALID_CONFIG, "MEMORY": 3.5}
    with pytest.raises(ValueError, match="MEMORY"):
        validate_config(bad)


def test_wrong_type_for_dict_field():
    bad = {**VALID_CONFIG, "DETECT_PARAMS": "not a dict"}
    with pytest.raises(ValueError, match="DETECT_PARAMS"):
        validate_config(bad)
