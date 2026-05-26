"""CLI runner for executing analysis notebooks via papermill."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

ALLOWED_KEYS = {
    "RUN_NAME", "STACK_PATH", "FLUOR_PATH",
    "GATING_Z_THRESHOLD",
    "DETECT_PARAMS",
    "SEARCH_RANGE", "MEMORY",
    "MERGE_MAX_DISTANCE", "MERGE_MAX_GAP",
    "MIN_TRACK_DETECTIONS",
    "FLUOR_DROP_THRESHOLD", "FLUOR_DROP_WINDOW",
}

REQUIRED_KEYS = {"RUN_NAME", "STACK_PATH", "FLUOR_PATH"}

TYPE_RULES = {
    "RUN_NAME": str,
    "STACK_PATH": str,
    "FLUOR_PATH": str,
    "GATING_Z_THRESHOLD": (int, float),
    "DETECT_PARAMS": dict,
    "SEARCH_RANGE": (int, float),
    "MEMORY": int,
    "MERGE_MAX_DISTANCE": (int, float),
    "MERGE_MAX_GAP": int,
    "MIN_TRACK_DETECTIONS": int,
    "FLUOR_DROP_THRESHOLD": (int, float),
    "FLUOR_DROP_WINDOW": int,
}


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping (dict), not "
                         f"{type(config).__name__}")

    unknown = set(config.keys()) - ALLOWED_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

    missing = REQUIRED_KEYS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required keys: {', '.join(sorted(missing))}")

    run_name = config.get("RUN_NAME", "")
    if isinstance(run_name, str) and (".." in run_name or "/" in run_name
                                      or "\\" in run_name):
        raise ValueError(f"RUN_NAME contains path traversal characters: "
                         f"{run_name!r}")

    for key, expected_type in TYPE_RULES.items():
        if key not in config:
            continue
        value = config[key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{key} must be {expected_type}, got {type(value).__name__}: "
                f"{value!r}"
            )
