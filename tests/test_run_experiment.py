import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from run_experiment import validate_config


VALID_CONFIG = {
    "RUN_NAME": "run_01",
    "extraction": {
        "STACK_PATH": "../data/test/phase.tif",
        "FLUOR_PATH": "../data/test/fluor.tif",
        "MODEL_TYPE": "cyto3",
        "DETECT_PARAMS": {
            "diameter": 32, "min_area": 300, "min_circularity": 0.7,
            "min_contrast": 1250, "exclude_edges": True,
            "gpu": True, "resample": False,
        },
        "GATING_Z_THRESHOLD": 3.5,
        "SEARCH_RANGE": 30.0,
        "MEMORY": 3,
        "MERGE_MAX_DISTANCE": 15.0,
        "MERGE_MAX_GAP": 18,
        "MIN_TRACK_DETECTIONS": 4,
        "NUCLEUS_DIAMETER": 25,
        "NUCLEUS_MIN_AREA": 100,
    },
    "analysis": {
        "PIXEL_SIZE_UM": 0.0645,
        "BASELINE_FRAMES": 3,
        "PERI_CORE_RINGS": [0.7, 1.0],
        "FLUOR_ALIGN_WINDOW": 3,
        "FATE_FEATURES_FULL": ["area"],
        "FATE_FEATURES_NO_AREA": [],
    },
}


def test_valid_config_passes():
    validate_config(VALID_CONFIG)


def test_minimal_config_passes():
    minimal = {
        "RUN_NAME": "run_01",
        "extraction": {
            "STACK_PATH": "../data/x.tif",
            "FLUOR_PATH": "../data/y.tif",
        },
        "analysis": {},
    }
    validate_config(minimal)


def test_results_root_optional_string():
    ok = {**VALID_CONFIG, "RESULTS_ROOT": "/tmp/results"}
    validate_config(ok)


def test_unknown_top_level_key_rejected():
    bad = {**VALID_CONFIG, "MYSTERY": 1}
    with pytest.raises(ValueError, match="MYSTERY"):
        validate_config(bad)


def test_unknown_extraction_key_rejected():
    bad = {**VALID_CONFIG,
           "extraction": {**VALID_CONFIG["extraction"], "MYSTERY": 1}}
    with pytest.raises(ValueError, match="MYSTERY"):
        validate_config(bad)


def test_unknown_analysis_key_rejected():
    bad = {**VALID_CONFIG,
           "analysis": {**VALID_CONFIG["analysis"], "MYSTERY": 1}}
    with pytest.raises(ValueError, match="MYSTERY"):
        validate_config(bad)


def test_missing_run_name():
    bad = {k: v for k, v in VALID_CONFIG.items() if k != "RUN_NAME"}
    with pytest.raises(ValueError, match="RUN_NAME"):
        validate_config(bad)


def test_missing_extraction_section():
    bad = {"RUN_NAME": "r", "analysis": {}}
    with pytest.raises(ValueError, match="extraction"):
        validate_config(bad)


def test_missing_stack_path():
    bad = {**VALID_CONFIG,
           "extraction": {k: v for k, v in VALID_CONFIG["extraction"].items()
                          if k != "STACK_PATH"}}
    with pytest.raises(ValueError, match="STACK_PATH"):
        validate_config(bad)


def test_run_name_path_traversal():
    for evil in ("../etc", "foo/bar", "foo\\bar"):
        with pytest.raises(ValueError, match="path traversal"):
            validate_config({**VALID_CONFIG, "RUN_NAME": evil})


def test_wrong_type_string():
    bad = {**VALID_CONFIG,
           "extraction": {**VALID_CONFIG["extraction"], "STACK_PATH": 123}}
    with pytest.raises(ValueError, match="STACK_PATH"):
        validate_config(bad)


def test_wrong_type_numeric():
    bad = {**VALID_CONFIG,
           "extraction": {**VALID_CONFIG["extraction"],
                          "SEARCH_RANGE": "thirty"}}
    with pytest.raises(ValueError, match="SEARCH_RANGE"):
        validate_config(bad)


def test_wrong_type_int():
    bad = {**VALID_CONFIG,
           "extraction": {**VALID_CONFIG["extraction"], "MEMORY": 3.5}}
    with pytest.raises(ValueError, match="MEMORY"):
        validate_config(bad)


def test_wrong_type_dict():
    bad = {**VALID_CONFIG,
           "extraction": {**VALID_CONFIG["extraction"],
                          "DETECT_PARAMS": "not-a-dict"}}
    with pytest.raises(ValueError, match="DETECT_PARAMS"):
        validate_config(bad)


def test_analysis_wrong_type_list():
    bad = {**VALID_CONFIG,
           "analysis": {**VALID_CONFIG["analysis"],
                        "PERI_CORE_RINGS": "not-a-list"}}
    with pytest.raises(ValueError, match="PERI_CORE_RINGS"):
        validate_config(bad)


def test_non_dict_rejected():
    with pytest.raises(ValueError, match="mapping"):
        validate_config([1, 2, 3])
    with pytest.raises(ValueError, match="mapping"):
        validate_config(None)
