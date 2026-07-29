"""CLI runner for executing analysis notebooks via papermill."""

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
import papermill as pm
import yaml

from cell_analysis.io import (
    compute_provenance, provenance_matches,
    export_notebook_html,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_EXTRACT = REPO_ROOT / "notebooks" / "extract.ipynb"
NOTEBOOK_ANALYSIS = REPO_ROOT / "notebooks" / "analysis.ipynb"

EXTRACTION_ARTIFACTS = (
    "provenance.json", "label_stack.npz", "nucleus_label_stack.npz",
    "tracked_cells.csv", "track_statistics.csv",
    "frame_diagnostics.csv", "merge_log.csv", "dropped_frames.csv",
)

TOP_LEVEL_ALLOWED = {"RUN_NAME", "RESULTS_ROOT", "extraction", "analysis"}
TOP_LEVEL_REQUIRED = {"RUN_NAME", "extraction"}

EXTRACTION_ALLOWED = {
    "STACK_PATH", "FLUOR_PATH", "MODEL_TYPE", "DETECT_PARAMS",
    "GATING_Z_THRESHOLD", "SEARCH_RANGE", "MEMORY",
    "MERGE_MAX_DISTANCE", "MERGE_MAX_GAP", "MIN_TRACK_DETECTIONS",
    "NUCLEUS_DIAMETER", "NUCLEUS_MIN_AREA",
}
EXTRACTION_REQUIRED = {"STACK_PATH", "FLUOR_PATH"}
EXTRACTION_TYPES = {
    "STACK_PATH": str, "FLUOR_PATH": str, "MODEL_TYPE": str,
    "DETECT_PARAMS": dict,
    "GATING_Z_THRESHOLD": (int, float),
    "SEARCH_RANGE": (int, float),
    "MEMORY": int,
    "MERGE_MAX_DISTANCE": (int, float),
    "MERGE_MAX_GAP": int,
    "MIN_TRACK_DETECTIONS": int,
    "NUCLEUS_DIAMETER": int,
    "NUCLEUS_MIN_AREA": int,
}

ANALYSIS_ALLOWED = {
    "PIXEL_SIZE_UM", "BASELINE_FRAMES", "PERI_CORE_RINGS",
    "FLUOR_ALIGN_WINDOW", "FATE_FEATURES_FULL", "FATE_FEATURES_NO_AREA",
}
ANALYSIS_TYPES = {
    "PIXEL_SIZE_UM": (int, float),
    "BASELINE_FRAMES": int,
    "PERI_CORE_RINGS": (list, tuple),
    "FLUOR_ALIGN_WINDOW": int,
    "FATE_FEATURES_FULL": (list, tuple),
    "FATE_FEATURES_NO_AREA": (list, tuple),
}

TOP_LEVEL_TYPES = {
    "RUN_NAME": str,
    "RESULTS_ROOT": str,
    "extraction": dict,
    "analysis": dict,
}


def _check_section(section_name, section, allowed, required, types):
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {section_name} keys: {', '.join(sorted(unknown))}")
    missing = required - set(section)
    if missing:
        raise ValueError(
            f"Missing required {section_name} keys: "
            f"{', '.join(sorted(missing))}")
    for k, expected in types.items():
        if k not in section:
            continue
        if not isinstance(section[k], expected):
            raise ValueError(
                f"{k} must be {expected}, got "
                f"{type(section[k]).__name__}: {section[k]!r}")


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping (dict), not "
                         f"{type(config).__name__}")

    unknown = set(config) - TOP_LEVEL_ALLOWED
    if unknown:
        raise ValueError(f"Unknown top-level keys: {', '.join(sorted(unknown))}")

    missing = TOP_LEVEL_REQUIRED - set(config)
    if missing:
        raise ValueError(
            f"Missing required top-level keys: {', '.join(sorted(missing))}")

    for k, expected in TOP_LEVEL_TYPES.items():
        if k not in config:
            continue
        if not isinstance(config[k], expected):
            raise ValueError(
                f"{k} must be {expected}, got "
                f"{type(config[k]).__name__}: {config[k]!r}")

    run_name = config["RUN_NAME"]
    if ".." in run_name or "/" in run_name or "\\" in run_name:
        raise ValueError(
            f"RUN_NAME contains path traversal characters: {run_name!r}")

    _check_section("extraction", config["extraction"],
                   EXTRACTION_ALLOWED, EXTRACTION_REQUIRED, EXTRACTION_TYPES)
    _check_section("analysis", config.get("analysis", {}),
                   ANALYSIS_ALLOWED, set(), ANALYSIS_TYPES)


def read_kernel_name(notebook_path):
    nb = nbformat.read(str(notebook_path), as_version=4)
    return nb.metadata.get("kernelspec", {}).get("name", "python3")



def resolve_results_root(cfg: dict, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    if cfg.get("RESULTS_ROOT"):
        return Path(cfg["RESULTS_ROOT"]).resolve()
    return REPO_ROOT / "results"


def extraction_is_stale(extraction_dir: Path, extraction_params: dict,
                        stack_path: Path, fluor_path: Path) -> tuple[bool, str]:
    for name in EXTRACTION_ARTIFACTS:
        if not (extraction_dir / name).exists():
            return True, f"missing artifacts ({name})"
    import json
    existing = json.loads(
        (extraction_dir / "provenance.json").read_text(encoding="utf-8"))
    candidate = compute_provenance(extraction_params, stack_path, fluor_path)
    ok, drift = provenance_matches(existing, candidate)
    if ok:
        return False, ""
    return True, f"param drift: {', '.join(drift)}"


def _run_notebook(notebook_path: Path, parameters: dict,
                  out_html: Path | None) -> None:
    """Execute *notebook_path* with papermill.

    Writes an HTML export to *out_html* when non-None. The papermill tmp
    notebook is unlinked before returning; there is no useful path to hand
    back to the caller.
    """
    kernel_name = read_kernel_name(notebook_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    pm.execute_notebook(
        str(notebook_path), str(tmp_path),
        parameters=parameters,
        cwd=str(REPO_ROOT / "notebooks"),
        kernel_name=kernel_name,
    )
    if out_html is not None:
        export_notebook_html(tmp_path, out_html)
    tmp_path.unlink(missing_ok=True)

def run_single_config(config_path, *, force_extract=False,
                      skip_analysis=False, analysis_only=False,
                      results_root_override=None) -> bool:
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(cfg)

    run_name = cfg["RUN_NAME"]
    results_root = resolve_results_root(cfg, results_root_override)
    run_dir = results_root / run_name
    extraction_dir = run_dir / "extraction"
    analysis_dir = run_dir / "analysis"
    run_dir.mkdir(parents=True, exist_ok=True)

    # STACK_PATH / FLUOR_PATH in YAML are relative to notebooks/ (papermill cwd).
    # Resolve once here; notebooks receive absolute paths and do no path
    # arithmetic.
    stack_path = (REPO_ROOT / "notebooks" /
                  cfg["extraction"]["STACK_PATH"]).resolve()
    fluor_path = (REPO_ROOT / "notebooks" /
                  cfg["extraction"]["FLUOR_PATH"]).resolve()

    extraction_absolute = {
        **cfg["extraction"],
        "STACK_PATH": str(stack_path),
        "FLUOR_PATH": str(fluor_path),
    }

    if not analysis_only:
        stale, reason = (True, "forced (--force-extract)") if force_extract \
            else extraction_is_stale(
                extraction_dir, cfg["extraction"], stack_path, fluor_path)
        if stale:
            print(f"  extracting: {reason}")
            _run_notebook(
                NOTEBOOK_EXTRACT,
                {**extraction_absolute,
                 "RUN_NAME": run_name,
                 "RESULTS_ROOT": str(results_root)},
                out_html=None,
            )
        else:
            print(f"  reusing extraction at {extraction_dir}")

    if analysis_only:
        if not (extraction_dir / "provenance.json").exists():
            raise FileNotFoundError(
                f"--analysis-only requested but no extraction at "
                f"{extraction_dir}")
        # Warn (don't block) if the extraction on disk drifted from the
        # config's current `extraction:` section.
        stale, reason = extraction_is_stale(
            extraction_dir, cfg["extraction"], stack_path, fluor_path)
        if stale:
            print(f"  WARNING: --analysis-only against stale extraction "
                  f"({reason}). Analysis will run against on-disk artifacts.")

    if not skip_analysis:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        _run_notebook(
            NOTEBOOK_ANALYSIS,
            {**cfg.get("analysis", {}),
             "RUN_NAME": run_name,
             "RESULTS_ROOT": str(results_root)},
            out_html=analysis_dir / "report.html",
        )

    shutil.copy(config_path, run_dir / "config.yaml")
    return True


BODY_RE = re.compile(r"<body\b[^>]*>", re.IGNORECASE)
BACK_LINK = (
    '\n<div style="font-family:-apple-system,system-ui,sans-serif;'
    'max-width:800px;margin:20px auto;padding:0 20px;">'
    '<a href="../index.html" style="color:#0366d6;text-decoration:none;">'
    '&larr; Back to index</a></div>'
)


def publish_reports():
    docs_dir = REPO_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    results_dir = REPO_ROOT / "results"
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        report = run_dir / "analysis" / "report.html"
        config = run_dir / "config.yaml"
        if not report.exists():
            continue
        target = docs_dir / run_dir.name
        target.mkdir(parents=True, exist_ok=True)
        html = report.read_text(encoding="utf-8")
        html = BODY_RE.sub(lambda m: m.group(0) + BACK_LINK, html, count=1)
        (target / "report.html").write_text(html, encoding="utf-8")
        run_info = {"name": run_dir.name, "date": report.stat().st_mtime}
        if config.exists():
            with open(config) as f:
                cfg = yaml.safe_load(f) or {}
            stack = cfg.get("extraction", {}).get("STACK_PATH", "")
            run_info["dataset"] = Path(stack).parent.name
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
    print(f"Published {len(runs)} reports to docs/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--results-root")
    args = parser.parse_args()

    if args.skip_analysis and args.analysis_only:
        parser.error("--skip-analysis and --analysis-only are mutually exclusive")

    results = {}
    for path in args.configs:
        print(f"Running: {path}")
        try:
            ok = run_single_config(
                path,
                force_extract=args.force_extract,
                skip_analysis=args.skip_analysis,
                analysis_only=args.analysis_only,
                results_root_override=args.results_root,
            )
            with open(path) as f:
                run_name = yaml.safe_load(f)["RUN_NAME"]
            results[run_name] = ("OK" if ok else "FAILED", path)
        except (ValueError, FileNotFoundError) as exc:
            print(f"  SKIPPED: {exc}")
            results[Path(path).stem] = ("SKIPPED", path)
        except pm.PapermillExecutionError as exc:
            print(f"  FAILED: {exc}")
            with open(path) as f:
                run_name = yaml.safe_load(f)["RUN_NAME"]
            results[run_name] = ("FAILED", path)

    if len(args.configs) > 1:
        print("\nResults:")
        for run_name, (status, path) in results.items():
            print(f"  {run_name}: {status}")

    publish_reports()

    any_failed = any(s != "OK" for s, _ in results.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
