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
    "RUN_NAME", "STACK_PATH", "FLUOR_PATH", "CONFIG_PATH",
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
    "CONFIG_PATH": str,
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

    config["CONFIG_PATH"] = str(config_path)

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


def publish_reports():
    docs_dir = REPO_ROOT / "docs" / "reports"
    docs_dir.mkdir(parents=True, exist_ok=True)

    results_dir = REPO_ROOT / "results"
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        report = run_dir / "report.html"
        config = run_dir / "config.yaml"
        if not report.exists():
            continue
        target = docs_dir / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(report, target / "report.html")
        run_info = {"name": run_dir.name, "date": report.stat().st_mtime}
        if config.exists():
            with open(config) as f:
                cfg = yaml.safe_load(f) or {}
            run_info["dataset"] = Path(cfg.get("STACK_PATH", "")).parent.name
        runs.append(run_info)

    from datetime import datetime
    rows = ""
    for r in runs:
        date = datetime.fromtimestamp(r["date"]).strftime("%Y-%m-%d %H:%M")
        dataset = r.get("dataset", "")
        rows += (f'      <tr><td><a href="{r["name"]}/report.html">{r["name"]}</a>'
                 f"</td><td>{dataset}</td><td>{date}</td></tr>\n")

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment Reports</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #ddd; }}
  th {{ background: #f5f5f5; }}
  a {{ color: #0366d6; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>Experiment Reports</h1>
<table>
  <thead><tr><th>Run</th><th>Dataset</th><th>Date</th></tr></thead>
  <tbody>
{rows}  </tbody>
</table>
</body>
</html>"""

    (docs_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"Published {len(runs)} reports to docs/reports/")


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

    publish_reports()

    any_failed = any(s != "OK" for s, _ in results.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
