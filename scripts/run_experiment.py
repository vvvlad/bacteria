"""CLI runner for executing analysis notebooks via papermill."""

import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
import papermill as pm
import yaml
from nbconvert import HTMLExporter

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


def read_kernel_name(notebook_path):
    nb = nbformat.read(str(notebook_path), as_version=4)
    return nb.metadata.get("kernelspec", {}).get("name", "python3")


def run_single_config(config_path):
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    validate_config(config)

    run_name = config["RUN_NAME"]
    notebook_path = REPO_ROOT / "notebooks" / "analysis.ipynb"
    results_dir = REPO_ROOT / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    kernel_name = read_kernel_name(notebook_path)

    tmp = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    failed = False
    try:
        pm.execute_notebook(
            str(notebook_path),
            str(tmp_path),
            parameters=config,
            cwd=str(REPO_ROOT / "notebooks"),
            kernel_name=kernel_name,
        )
    except pm.PapermillExecutionError as exc:
        print(f"  FAILED: {exc}")
        failed = True
        shutil.copy(tmp_path, results_dir / "report_failed.ipynb")

    exporter = HTMLExporter()
    nb = nbformat.read(str(tmp_path), as_version=4)
    body, _ = exporter.from_notebook_node(nb)
    (results_dir / "report.html").write_text(body, encoding="utf-8")

    shutil.copy(config_path, results_dir / "config.yaml")

    if not failed:
        tmp_path.unlink(missing_ok=True)

    return not failed


def main():
    if len(sys.argv) < 2:
        print("Usage: run_experiment.py <config.yaml> [config2.yaml ...]")
        sys.exit(1)

    config_paths = sys.argv[1:]
    results = {}

    for path in config_paths:
        print(f"Running: {path}")
        try:
            ok = run_single_config(path)
            with open(path) as f:
                run_name = yaml.safe_load(f)["RUN_NAME"]
            status = "OK" if ok else "FAILED"
            results[run_name] = (status, path)
        except (ValueError, FileNotFoundError) as exc:
            print(f"  SKIPPED: {exc}")
            results[Path(path).stem] = ("SKIPPED", path)

    if len(config_paths) > 1:
        print("\nResults:")
        for run_name, (status, path) in results.items():
            suffix = ""
            if status == "OK":
                suffix = f" → results/{run_name}/report.html"
            elif status == "FAILED":
                suffix = f" (see results/{run_name}/report.html)"
            print(f"  {run_name}: {status}{suffix}")

    any_failed = any(s != "OK" for s, _ in results.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
