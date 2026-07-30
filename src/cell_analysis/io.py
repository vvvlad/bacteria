"""Loading and saving image stacks and results."""

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile


if TYPE_CHECKING:
    import pandas as pd


def load_stack(path: str | Path) -> np.ndarray:
    """Load a TIFF image stack.

    Parameters
    ----------
    path : str or Path
        Path to a .tif/.tiff file. Expected shapes:
        - Single channel time-lapse: (T, Y, X)
        - Multi-channel time-lapse: (T, C, Y, X)

    Returns
    -------
    np.ndarray
        The image stack.
    """
    return tifffile.imread(str(path))


def load_paired_stacks(
    phase_path: str | Path,
    fluor_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load paired phase-contrast and fluorescence stacks.

    Validates that both stacks have matching T, Y, X dimensions.

    Returns
    -------
    tuple of (phase_stack, fluor_stack)
        Both as np.ndarray with shape (T, Y, X).
    """
    phase = load_stack(phase_path)
    fluor = load_stack(fluor_path)

    # If multi-channel, take first channel
    if phase.ndim == 4:
        phase = phase[:, 0]
    if fluor.ndim == 4:
        fluor = fluor[:, 0]

    if phase.shape != fluor.shape:
        raise ValueError(
            f"Shape mismatch: phase {phase.shape} vs fluorescence {fluor.shape}. "
            "Both stacks must have identical (T, Y, X) dimensions."
        )
    return phase, fluor


def save_results(df, path: str | Path) -> None:
    """Save a pandas DataFrame of results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def export_notebook_html(
    notebook_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Export an executed notebook to a self-contained HTML file."""
    import nbformat
    from nbconvert import HTMLExporter

    exporter = HTMLExporter(
        exclude_input=True,
        exclude_input_prompt=True,
        exclude_output_prompt=True,
    )
    nb = nbformat.read(notebook_path, as_version=4)
    body, _ = exporter.from_notebook_node(nb)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body, encoding="utf-8")
    return output_path


def save_summary(data: dict, path: str | Path) -> None:
    """Flatten a nested summary dict and save as a single-row CSV.

    Recursively flattens nested dicts into underscore-joined column names.
    Skips array-like values (e.g. null distributions) that don't fit a
    single-row tabular format.
    """
    import pandas as pd

    def _flatten(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(_flatten(v, prefix=f"{key}_"))
            elif hasattr(v, "__len__") and not isinstance(v, str):
                continue
            else:
                flat[key] = v
        return flat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_flatten(data)]).to_csv(path, index=False)


def _sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_path(path: str | Path, repo_root: Path | None) -> str:
    p = Path(path).resolve()
    if repo_root is None:
        return str(p)
    try:
        return str(p.relative_to(Path(repo_root).resolve()))
    except ValueError:
        return str(p)


def compute_provenance(
    extraction_params: dict,
    stack_path: str | Path,
    fluor_path: str | Path,
    *,
    repo_root: Path | None = None,
    package_version: str | None = None,
) -> dict:
    """Build a provenance dict for an extraction run.

    ``extract_hash`` = sha256 of (canonicalized params JSON + stack sha256
    + fluor sha256). ``cell_analysis_version`` is recorded for audit but
    is NOT part of ``extract_hash``.
    """
    stack_sha = _sha256_file(stack_path)
    fluor_sha = _sha256_file(fluor_path)

    # Path values are stored in extraction_params for audit but excluded from
    # the hash — file identity is covered by stack_sha256/fluor_sha256 below,
    # and callers may pass either relative or absolute path strings.
    params_for_hash = {k: v for k, v in extraction_params.items()
                       if k not in ("STACK_PATH", "FLUOR_PATH")}
    params_canonical = json.dumps(params_for_hash, sort_keys=True,
                                  separators=(",", ":"))
    h = hashlib.sha256()
    h.update(params_canonical.encode("utf-8"))
    h.update(stack_sha.encode("utf-8"))
    h.update(fluor_sha.encode("utf-8"))
    if package_version is None:
        try:
            from importlib.metadata import version
            package_version = version("cell-analysis")
        except Exception:
            package_version = "unknown"
    return {
        "extract_hash": h.hexdigest(),
        "stack_path": _canonical_path(stack_path, repo_root),
        "fluor_path": _canonical_path(fluor_path, repo_root),
        "stack_sha256": stack_sha,
        "fluor_sha256": fluor_sha,
        "params": extraction_params,
        "cell_analysis_version": package_version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
    }


def provenance_matches(existing: dict, candidate: dict) -> tuple[bool, list[str]]:
    """Compare two provenance dicts; return (match, drifted_fields).

    ``drifted_fields`` lists which top-level fields changed. For
    ``params``, entries look like ``params.KEY`` for each differing key.
    """
    if existing.get("extract_hash") == candidate.get("extract_hash"):
        return True, []
    drift: list[str] = []
    for field in ("stack_sha256", "fluor_sha256"):
        if existing.get(field) != candidate.get(field):
            drift.append(field)
    p_old = existing.get("params", {})
    p_new = candidate.get("params", {})
    for key in sorted(set(p_old) | set(p_new)):
        if p_old.get(key) != p_new.get(key):
            drift.append(f"params.{key}")
    return False, drift


@dataclass
class ExtractionBundle:
    tracked: "pd.DataFrame"
    track_stats: "pd.DataFrame"
    label_stack: np.ndarray
    nucleus_label_stack: np.ndarray
    diagnostics: "pd.DataFrame"
    merge_log: "pd.DataFrame"
    dropped_frames: "pd.DataFrame"
    provenance: dict


def save_extraction(
    results_dir: str | Path, *,
    label_stack: np.ndarray,
    nucleus_label_stack: np.ndarray,
    tracked, track_stats,
    diagnostics, merge_log, dropped_frames,
    provenance: dict,
) -> None:
    """Persist a full extraction bundle to ``results_dir``."""

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")

    np.savez_compressed(results_dir / "label_stack.npz",
                        label_stack=label_stack.astype(np.int32))
    np.savez_compressed(results_dir / "nucleus_label_stack.npz",
                        nucleus_label_stack=nucleus_label_stack.astype(np.int32))

    for df, name in (
        (tracked, "tracked_cells.csv"),
        (track_stats, "track_statistics.csv"),
        (diagnostics, "frame_diagnostics.csv"),
        (merge_log, "merge_log.csv"),
        (dropped_frames, "dropped_frames.csv"),
    ):
        df.to_csv(results_dir / name, index=False)


def load_extraction(results_root: str | Path, run_name: str) -> ExtractionBundle:
    """Load a previously saved extraction bundle."""
    import pandas as pd

    extraction_dir = Path(results_root) / run_name / "extraction"
    prov_path = extraction_dir / "provenance.json"
    if not prov_path.exists():
        raise FileNotFoundError(f"No provenance at {prov_path}")

    provenance = json.loads(prov_path.read_text(encoding="utf-8"))
    label_stack = np.load(extraction_dir / "label_stack.npz")["label_stack"]
    nucleus_label_stack = np.load(
        extraction_dir / "nucleus_label_stack.npz")["nucleus_label_stack"]

    def _read(name):
        return pd.read_csv(extraction_dir / name)

    return ExtractionBundle(
        tracked=_read("tracked_cells.csv"),
        track_stats=_read("track_statistics.csv"),
        label_stack=label_stack,
        nucleus_label_stack=nucleus_label_stack,
        diagnostics=_read("frame_diagnostics.csv"),
        merge_log=_read("merge_log.csv"),
        dropped_frames=_read("dropped_frames.csv"),
        provenance=provenance,
    )



# --- Notebook-facing helpers -----------------------------------------------
#
# These wrap the low-level bundle I/O so the notebooks stay minimal.
# Notebooks should call these instead of assembling provenance dicts or
# resolving `provenance.json` paths themselves.

# Canonical list of `extraction:` YAML keys. When you add a new extraction
# param, update this AND `EXTRACTION_ALLOWED`/`EXTRACTION_TYPES` in
# scripts/run_experiment.py AND the parameters cell of extract.ipynb.
EXTRACTION_PARAM_NAMES = (
    "STACK_PATH", "FLUOR_PATH", "MODEL_TYPE", "DETECT_PARAMS",
    "GATING_Z_THRESHOLD", "SEARCH_RANGE", "MEMORY",
    "MERGE_MAX_DISTANCE", "MERGE_MAX_GAP", "MIN_TRACK_DETECTIONS",
    "NUCLEUS_DIAMETER", "NUCLEUS_MIN_AREA",
)


def resolve_provenance_path(
    prov_path: str | Path, *, repo_root: Path | None = None,
) -> Path:
    """Resolve a `provenance.json` path field to an absolute Path.

    ``compute_provenance`` stores paths as repo-root-relative when the
    source lives inside ``repo_root``, absolute otherwise. This is the
    inverse: given the stored string, return the absolute path suitable
    for `load_stack`, `open()`, etc.
    """
    p = Path(prov_path)
    if p.is_absolute():
        return p
    if repo_root is None:
        return p.resolve()
    return (Path(repo_root) / prov_path).resolve()


def finalize_extraction_run(
    extraction_dir: str | Path, *,
    label_stack: np.ndarray,
    nucleus_label_stack: np.ndarray,
    tracked, track_stats,
    diagnostics, merge_log,
    params: dict,
    stack_path: str | Path,
    fluor_path: str | Path,
    repo_root: Path | None = None,
) -> dict:
    """Compute provenance, derive dropped_frames, and save the bundle.

    Called from the last cell of `notebooks/extract.ipynb`. Single source
    of truth for turning the in-memory pipeline outputs into an on-disk
    extraction bundle.

    ``params`` should be a dict of the extraction-config values that end
    up hashed into the provenance — typically built by the caller as
    ``{k: globals()[k] for k in EXTRACTION_PARAM_NAMES}``.

    ``dropped_frames`` is derived from ``diagnostics[diagnostics["flagged"]]``,
    matching ``run_frame_gating``'s own logic. An empty-schema DataFrame
    is used if the ``flagged`` column is absent.

    Returns the provenance dict that was written to disk.
    """
    import pandas as pd

    if "flagged" in diagnostics.columns:
        dropped = diagnostics[diagnostics["flagged"]].copy()
    else:
        dropped = pd.DataFrame({"frame": [], "reason": []})

    provenance = compute_provenance(
        params,
        Path(stack_path).resolve(),
        Path(fluor_path).resolve(),
        repo_root=repo_root,
    )

    save_extraction(
        extraction_dir,
        label_stack=label_stack,
        nucleus_label_stack=nucleus_label_stack,
        tracked=tracked, track_stats=track_stats,
        diagnostics=diagnostics, merge_log=merge_log,
        dropped_frames=dropped,
        provenance=provenance,
    )
    return provenance


FLUOR_ROOT_ENV = "EXPERIMENTS_IMAGE_FLUOR_ROOTS"


def _effective_fluor_roots(
    fluor_roots: list[str | Path] | tuple[str | Path, ...] | None,
) -> list[Path]:
    import os
    roots: list[Path] = []
    if fluor_roots:
        roots.extend(Path(r) for r in fluor_roots)
    env_val = os.environ.get(FLUOR_ROOT_ENV, "")
    if env_val:
        roots.extend(Path(r) for r in env_val.split(os.pathsep) if r)
    return roots


def _resolve_raw_path(
    recorded: str,
    *,
    fluor_roots: list[Path],
    repo_root: Path | None,
) -> tuple[Path | None, list[Path]]:
    tried: list[Path] = []
    primary = resolve_provenance_path(recorded, repo_root=repo_root)
    tried.append(primary)
    if primary.exists():
        return primary, tried
    basename = Path(recorded).name
    for root in fluor_roots:
        candidate = (Path(root) / basename).resolve()
        tried.append(candidate)
        if candidate.exists():
            return candidate, tried
    return None, tried


def load_extraction_with_stacks(
    results_root: str | Path, run_name: str, *,
    repo_root: Path | None = None,
    fluor_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> tuple[ExtractionBundle, np.ndarray | None, np.ndarray]:
    """Load an extraction bundle *and* its raw phase + fluor stacks.

    Resolves ``stack_path`` and ``fluor_path`` from provenance in this
    order:

    1. As recorded, resolved against ``repo_root`` when relative.
    2. ``{root}/{basename}`` for each entry in ``fluor_roots`` and the
       ``EXPERIMENTS_IMAGE_FLUOR_ROOTS`` env var (``os.pathsep``-split).
    3. Fail on total miss for fluor (analysis needs pixel values inside
       the masks). Warn (don't raise) on total miss for phase — only
       ``plot_channels_preview`` uses it, and its cell should guard
       against ``phase_stack is None``.

    Returns ``(bundle, phase_stack_or_None, fluor_stack)``.
    """
    import warnings

    bundle = load_extraction(results_root, run_name)
    roots = _effective_fluor_roots(fluor_roots)

    fluor_resolved, fluor_tried = _resolve_raw_path(
        bundle.provenance["fluor_path"],
        fluor_roots=roots, repo_root=repo_root)
    if fluor_resolved is None:
        tried_str = "\n  ".join(str(p) for p in fluor_tried)
        raise FileNotFoundError(
            f"Could not resolve fluor stack. Tried:\n  {tried_str}\n"
            f"Set --fluor-root PATH (or ${FLUOR_ROOT_ENV}) to a directory "
            f"containing '{Path(bundle.provenance['fluor_path']).name}'."
        )

    phase_resolved, phase_tried = _resolve_raw_path(
        bundle.provenance["stack_path"],
        fluor_roots=roots, repo_root=repo_root)

    fluor_stack = load_stack(fluor_resolved)
    if fluor_stack.ndim == 4:
        fluor_stack = fluor_stack[:, 0]

    if phase_resolved is None:
        tried_str = ", ".join(str(p) for p in phase_tried)
        warnings.warn(
            f"Phase stack not found (tried {tried_str}); phase-dependent "
            f"plots will be skipped.",
            RuntimeWarning, stacklevel=2,
        )
        return bundle, None, fluor_stack

    phase_stack = load_stack(phase_resolved)
    if phase_stack.ndim == 4:
        phase_stack = phase_stack[:, 0]

    if phase_stack.shape != fluor_stack.shape:
        raise ValueError(
            f"Shape mismatch: phase {phase_stack.shape} vs "
            f"fluorescence {fluor_stack.shape}. Both stacks must have "
            f"identical (T, Y, X) dimensions."
        )

    return bundle, phase_stack, fluor_stack
