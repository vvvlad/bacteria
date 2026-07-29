# Cell Analysis Pipeline

Time-lapse microscopy pipeline for bacterial cells. Given a paired
phase-contrast + fluorescence TIFF, it detects cells with
[Cellpose](https://github.com/MouseLand/cellpose), links them into
tracks with [trackpy](https://github.com/soft-matter/trackpy), measures
per-cell fluorescence and geometry, and produces a per-run HTML report
plus CSV tables you can pick up in any downstream tool.

The main entry points are two Jupyter notebooks — `notebooks/extract.ipynb`
(detection + tracking) and `notebooks/analysis.ipynb` (metrics + plots) —
driven by a small CLI wrapper (`scripts/run_experiment.py`) for batch runs.

---

## What the pipeline does

Every experiment goes through the same two phases. The split matters
because the first is slow (~10–20 min of Cellpose) and the second is
fast (~15 s of plotting), so you'll iterate on the second one all day
without paying for the first.

```
                       ┌─────────────────────────────┐
   phase.tif (T,Y,X)   │  extract.ipynb              │
   fluor.tif (T,Y,X)   │  ───────────────────────    │
        │              │  Cellpose (phase → masks)   │    ┌─────────────────────┐
        └──────────────▶ frame-quality gating        ├───▶│  extraction/        │
                       │  trackpy linking + merging  │    │    label_stack.npz  │
                       │  Cellpose (fluor → nuclei)  │    │    nucleus_labels…  │
                       │                             │    │    tracked_cells.csv│
                       └─────────────────────────────┘    │    provenance.json  │
                                                          └─────────┬───────────┘
                                                                    │
                       ┌─────────────────────────────┐               │
                       │  analysis.ipynb             │◀──────────────┘
                       │  ───────────────────────    │
                       │  fluorescence per cell      │     ┌─────────────────────┐
                       │  nucleoid metrics           │     │  analysis/          │
                       │  geometry (µm units)        │────▶│    tracked_cells.csv│
                       │  fluorescence alignment     │     │    fate_predictions.│
                       │  fate prediction (LR)       │     │    report.html      │
                       │  all plots                  │     └─────────────────────┘
                       └─────────────────────────────┘
```

The **extraction** phase produces raw per-cell identities (which pixel
belongs to which cell in which frame). The **analysis** phase computes
everything you'd actually put in a paper — fluorescence dynamics,
morphology, fate prediction, etc. — from those identities plus the raw
images.

Because extraction outputs are cached and content-hashed, the runner
only re-runs Cellpose when you change something that would affect it
(the extraction config or the input files). Editing a plot or a
calibration constant re-runs only analysis.

---

## Repository layout

```
bacteria/
├── configs/                          # One YAML config per experiment
│   ├── control_experiment.yaml
│   ├── protein_synthesis_arrested.yaml
│   └── run_01.yaml
├── data/                             # Raw TIFFs (gitignored — too big)
│   └── <dataset>/
│       ├── ...Ch0.tif                # Phase-contrast channel
│       └── ...Ch1-BG.tif             # Fluorescence (background-subtracted)
├── notebooks/
│   ├── extract.ipynb                 # Phase 1: image → identities
│   └── analysis.ipynb                # Phase 2: identities → metrics + plots
├── src/cell_analysis/                # Python package used by both notebooks
│   ├── io.py                         # TIFF loading, extraction bundle save/load
│   ├── segmentation.py               # Cellpose wrapper + filtering
│   ├── tracking.py                   # trackpy linking, gating, merging
│   ├── matching.py                   # Nucleoid metrics, fate prediction
│   ├── pipeline.py                   # High-level orchestration functions
│   └── plotting.py                   # All plots
├── scripts/
│   ├── run_experiment.py             # CLI: drives both notebooks with a config
│   ├── view_labels.py                # Open a run's masks in napari
│   └── diagnostic_overlay.py         # Tune detection interactively
├── results/                          # Per-run outputs (gitignored)
│   └── <run_name>/
│       ├── config.yaml               # Frozen copy of config used
│       ├── extraction/               # Expensive, cached artifacts
│       └── analysis/                 # Fast, rebuilt every run
├── docs/                             # Real docs + generated reports
│   ├── detection_tuning.md
│   ├── peri_core_asymmetry_definition_history.md
│   ├── project_log.md
│   ├── images/
│   └── reports/                      # Landing page + per-run report copies
├── tests/                            # pytest — mostly unit + one smoke test
├── pyproject.toml                    # uv-managed dependencies
└── uv.lock
```

---

## Setup

### Prerequisites

- **Python 3.11 or newer**
- **[uv](https://github.com/astral-sh/uv)** — fast Python package manager
- **Git**
- **VS Code** with the Python + Jupyter extensions (recommended) or JupyterLab

### Install

Once you have Python 3.11+, uv, and git installed, clone and sync:

```bash
git clone https://github.com/vvvlad/bacteria.git
cd bacteria
uv sync                    # creates .venv/, installs pinned deps
```

That's it. `uv run <command>` from here on uses the project's virtualenv
automatically — you don't need to activate it.

If you're setting up from scratch (fresh laptop, no uv), expand the
matching OS section below for step-by-step commands.

<details>
<summary><strong>macOS setup from scratch</strong></summary>

```bash
# Homebrew (if you don't have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Git (often comes with Xcode CLT)
xcode-select --install

# Python 3.11+
brew install python@3.11

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your terminal for uv to appear on PATH.

# Clone and sync
git clone https://github.com/vvvlad/bacteria.git
cd bacteria
uv sync
```

**Apple Silicon (M1/M2/M3):** Cellpose uses MPS GPU acceleration
automatically when `gpu: true` is set in the config. First run downloads
the pretrained Cellpose model (~1 GB) — needs internet.

</details>

<details>
<summary><strong>Windows setup from scratch</strong></summary>

```powershell
# Install Git from https://git-scm.com/download/win — accept default options
# but pick "Git from the command line" and "Checkout as-is, commit Unix-style".

# Install Python 3.11+ from https://www.python.org/downloads/ — check
# "Add python.exe to PATH" during install.

# Install uv:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Close and reopen PowerShell, then:
git clone https://github.com/vvvlad/bacteria.git
cd bacteria
uv sync
```

**NVIDIA GPU (optional):** With CUDA installed, set `gpu: true` in the
config for faster Cellpose. CPU mode is fine but slower.

**Long paths:** If you hit path-length errors, run
`git config --global core.longpaths true`.

</details>

### VS Code setup

Open the repo, then:

- **Cmd/Ctrl+Shift+P** → **"Python: Select Interpreter"** → pick
  `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe`
  (Windows). Jupyter and the Python extension both need this.

Recommended extensions if you don't have them:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

---

## Your first run

The repo ships with three ready-to-run configs. Extraction artifacts
for them may already be on disk (pre-baked for the maintainer's
machine) — if so, your first run will skip Cellpose entirely. If not,
expect ~10 min per config while Cellpose runs the first time.

```bash
uv run python scripts/run_experiment.py configs/control_experiment.yaml
```

Expected output (extraction cached):

```
Running: configs/control_experiment.yaml
  reusing extraction at .../results/control_experiment/extraction
Executing: ... 63/63 [00:15]
Published 1 reports to docs/reports/
```

Or, on a cold run:

```
Running: configs/control_experiment.yaml
  extracting: missing artifacts (provenance.json)
Executing: ... 16/16 [08:00]     ← Cellpose runs here
Executing: ... 63/63 [00:15]     ← analysis runs here
Published 1 reports to docs/reports/
```

Open the report:

```bash
open docs/reports/control_experiment/report.html    # macOS
start docs\reports\control_experiment\report.html   # Windows
```

If that opens with plots visible, you're set up correctly.

---

## Everyday workflows

### Iterating on plots or metrics (fast — where you'll spend most time)

Open the analysis notebook interactively:

```bash
uv run jupyter lab notebooks/analysis.ipynb
```

The first cell (tagged `parameters`) defines defaults — by default it
loads `control_experiment`'s extraction bundle. Edit that cell to
switch runs, then run all cells. Iterate on any plot, any metric, any
cell — the extraction bundle is already on disk, so re-runs take
seconds.

When your changes are ready to be captured in the HTML report for a
run, just re-run the CLI:

```bash
uv run python scripts/run_experiment.py configs/control_experiment.yaml
```

Extraction stays cached; analysis re-runs; `docs/reports/` updates.

### Iterating on detection or tracking (slower — Cellpose re-runs)

Any change to the `extraction:` section of a config (or to the source
TIFFs) invalidates the extraction cache. The next `run_experiment.py`
invocation will notice and re-extract:

```
extracting: param drift: params.GATING_Z_THRESHOLD
```

Before changing detection thresholds blindly, use the diagnostic
overlay (see [Auxiliary tools](#auxiliary-tools) below) — it visually
shows which cells were accepted vs. rejected and why. Faster feedback
than a full pipeline run.

### Adding a new dataset

1. **Drop your TIFFs into a folder under `data/`:**
   ```
   data/my_experiment/phase.tif
   data/my_experiment/fluorescence.tif
   ```
   Both stacks must have shape `(T, Y, X)` (uint8 or uint16). Multi-channel
   `(T, C, Y, X)` also works — the first channel is used.

2. **Copy a config and edit the paths + name:**
   ```bash
   # macOS / Linux
   cp configs/control_experiment.yaml configs/my_experiment.yaml
   # Windows PowerShell
   Copy-Item configs/control_experiment.yaml configs/my_experiment.yaml
   ```
   Change `RUN_NAME`, `STACK_PATH`, `FLUOR_PATH` at minimum. Detection
   defaults are tuned for the existing datasets — you may need to
   adjust `DETECT_PARAMS` for a different microscope / cell type. Use
   the diagnostic overlay to check before running the full pipeline.

3. **Run:**
   ```bash
   uv run python scripts/run_experiment.py configs/my_experiment.yaml
   ```
   First run triggers Cellpose (5–20 min depending on stack size and
   GPU); subsequent runs will reuse the extraction unless you change
   the extraction params.

### Inspecting cell/nucleus masks

The `.npz` files under `extraction/` hold the Cellpose output — one
integer per pixel, indicating which cell it belongs to. Three ways to
look at them:

```bash
# 1. napari (best for stack navigation)
uv run python scripts/view_labels.py results/control_experiment/
```

The napari viewer opens with four toggleable layers: phase, fluor
(green additive), cell masks, nucleus masks. Use the time slider to
scrub frames. Pass `--no-raw` to skip loading the raw TIFFs (faster
if they live on a slow drive).

```python
# 2. matplotlib one-liner (for a quick single-frame check)
import matplotlib.pyplot as plt, numpy as np
labels = np.load("results/control_experiment/extraction/label_stack.npz")["label_stack"]
plt.imshow(labels[0], cmap="tab20"); plt.show()
```

```bash
# 3. Fiji / ImageJ — convert to TIFF first. Writes labels.tif to the
# current directory (works on any OS):
uv run python -c "
import numpy as np, tifffile
labels = np.load('results/control_experiment/extraction/label_stack.npz')['label_stack']
tifffile.imwrite('labels.tif', labels.astype('uint16'))
"
```

Then open `labels.tif` in Fiji and set **Image → Lookup Tables →
glasbey_on_dark** for a colored view (**Image → Adjust →
Brightness/Contrast → Auto** stretches the LUT across all label ids).

---

## Configuration reference

Configs live in `configs/*.yaml`, one file per experiment. The runner
validates every file against a strict schema — unknown keys, wrong
types, or missing required fields all abort with a descriptive error.

### YAML shape

```yaml
RUN_NAME: my_experiment
# RESULTS_ROOT: /path/to/cloud/synced/folder      # optional, see below

extraction:
  STACK_PATH: "../data/my_experiment/phase.tif"
  FLUOR_PATH: "../data/my_experiment/fluor.tif"
  # ...more extraction params...

analysis:
  PIXEL_SIZE_UM: 0.0645
  # ...more analysis params...
```

**Which section owns which knob:** anything that would change the
produced `label_stack.npz` or the tracked identities goes under
`extraction:` (and triggers a Cellpose re-run when changed). Anything
that only reshapes already-extracted data — pixel-size calibration,
plot styling, fate-prediction features — goes under `analysis:` and
re-runs in seconds.

### Top-level keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `RUN_NAME` | str | *(required)* | Unique per-experiment name. Results go to `<results_root>/<RUN_NAME>/`. No `..`, `/`, or `\`. |
| `RESULTS_ROOT` | str | `<repo>/results` | Optional override — e.g. a Dropbox / iCloud folder if you want extractions out of the repo. `--results-root` on the CLI overrides both. |
| `extraction` | dict | *(required)* | Extraction params (below). |
| `analysis` | dict | `{}` | Analysis params (below). May be empty. |

### `extraction:` — Cellpose + tracking + gating

Change any of these → cached extraction is invalidated → Cellpose re-runs.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `STACK_PATH` | str | *(required)* | Path to phase-contrast TIFF, relative to `notebooks/` |
| `FLUOR_PATH` | str | *(required)* | Path to fluorescence TIFF, relative to `notebooks/` |
| `MODEL_TYPE` | str | `cyto3` | Cellpose model name |
| `DETECT_PARAMS.diameter` | int | 32 | Median cell diameter in pixels |
| `DETECT_PARAMS.min_area` | int | 300 | Minimum cell area (rejects debris) |
| `DETECT_PARAMS.min_circularity` | float | 0.7 | Minimum circularity (1.0 = perfect circle) |
| `DETECT_PARAMS.min_contrast` | int | 1250 | Minimum intensity contrast (rejects faded cells) |
| `DETECT_PARAMS.exclude_edges` | bool | true | Drop cells touching the frame border |
| `DETECT_PARAMS.gpu` | bool | true | GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA) |
| `DETECT_PARAMS.resample` | bool | false | Resample masks (slower, more precise boundaries) |
| `GATING_Z_THRESHOLD` | float | 3.5 | MAD-based Z-score for flagging anomalous frames |
| `SEARCH_RANGE` | float | 30.0 | Max cell displacement between frames (pixels) |
| `MEMORY` | int | 3 | Frames a cell can disappear before breaking the track |
| `MERGE_MAX_DISTANCE` | float | 15.0 | Max pixels between track end/start to merge fragments |
| `MERGE_MAX_GAP` | int | 18 | Max frame gap for fragment merging |
| `MIN_TRACK_DETECTIONS` | int | 4 | Minimum frames a track must span to be kept |
| `NUCLEUS_DIAMETER` | int | 25 | Cellpose diameter for the fluorescence-channel nucleus segmentation |
| `NUCLEUS_MIN_AREA` | int | 100 | Minimum nucleus area |

### `analysis:` — geometry + metrics + fate prediction

Change these freely; only the analysis phase re-runs.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `PIXEL_SIZE_UM` | float | 0.0645 | µm-per-pixel calibration. Scales `area` to µm², derives `radius`, `volume`, `surface_area`. Pass `1.0` to keep pixel units. |
| `BASELINE_FRAMES` | int | 3 | First N frames used for the area-histogram baseline (before medium changes drive swelling) |
| `PERI_CORE_RINGS` | list of 3 floats | `[0.15, 0.35, 0.55]` | `(core_max, peri_min, peri_max)` as fractions of R (equivalent radius). Defines inner disk and outer annulus for the peri/core asymmetry metric — see `docs/peri_core_asymmetry_definition_history.md`. |
| `FLUOR_ALIGN_WINDOW` | int | 5 | Frames on each side of disappearance for the fluorescence-alignment tables |
| `FATE_FEATURES_FULL` | list of str | `[area, cv, nnrm, mean_edge_distance_norm, gaussian_sigma_norm]` | Frame-0 columns fed to the logistic-regression fate predictor |
| `FATE_FEATURES_NO_AREA` | list of str | `[cv, nnrm, mean_edge_distance_norm, gaussian_sigma_norm]` | Same, without `area` — for the ablation comparison |

---

## The two notebooks

Both notebooks live in `notebooks/` and are executable both interactively
(via JupyterLab / VS Code) and programmatically (via papermill, driven by
the CLI runner).

### `extract.ipynb` — image → identities

Does the expensive image-processing:

1. Load paired phase-contrast + fluorescence TIFFs.
2. Cellpose on the phase channel → per-frame cell masks.
3. Frame-quality gating (drop frames whose stats are outliers).
4. Trackpy linking → track ids across frames.
5. Merge fragmented tracks + filter tracks shorter than
   `MIN_TRACK_DETECTIONS`.
6. Cellpose on the fluorescence channel → per-frame nucleus masks.
7. Save everything to `results/<run>/extraction/`.

The first cell is a `parameters`-tagged cell with defaults for
`control_experiment`. When run under papermill, the runner overrides
these with the `extraction:` section from the YAML config.

### `analysis.ipynb` — identities → metrics + plots

Loads the extraction bundle produced by `extract.ipynb` and does
everything else:

1. `load_extraction()` reads back `label_stack.npz`,
   `nucleus_label_stack.npz`, `tracked_cells.csv`, `track_statistics.csv`,
   and provenance.
2. Loads the raw phase + fluor TIFFs (paths recorded in provenance).
3. Measures fluorescence per cell per frame from the masks.
4. Computes nucleoid distribution metrics (`cv`, `nnrm`,
   `mean_edge_distance_norm`, `gaussian_sigma_norm`,
   `peri_core_asymmetry`).
5. Adds geometry (µm units, `radius`, `volume`, `surface_area`,
   `sav_ratio`).
6. Builds alignment tables around cell disappearance.
7. Compares phase-cell vs. fluorescence-nucleus counts (nucleus
   persistence).
8. Runs the logistic-regression fate predictor with LOO cross-validation.
9. Emits all plots and saves the enriched CSVs.

Again, the first cell is `parameters`-tagged with defaults; the CLI
runner overrides them with the `analysis:` section.

### How the CLI runner ties them together

`scripts/run_experiment.py` isn't a reimplementation — it's just a
parameterizer:

1. Reads the YAML, validates it against the two-section schema.
2. Computes a hash of the `extraction:` section + source-file sha256s;
   compares to the `provenance.json` on disk.
3. If stale (or `--force-extract`), papermill executes `extract.ipynb`
   with the `extraction:` section as parameters.
4. Papermill executes `analysis.ipynb` with the `analysis:` section as
   parameters.
5. Renders the executed analysis notebook to
   `results/<run>/analysis/report.html` and copies to
   `docs/reports/<run>/report.html`.

Consequences worth knowing:

- **New cells you add to `analysis.ipynb` automatically appear in the
  HTML report.** No wiring needed.
- **New config knobs must be registered.** Add the key to the notebook's
  `parameters` cell **and** to the appropriate allowlist
  (`EXTRACTION_ALLOWED` / `EXTRACTION_TYPES` or `ANALYSIS_ALLOWED` /
  `ANALYSIS_TYPES`) in `scripts/run_experiment.py`. Missing the
  allowlist entry means the runner rejects the config as "unknown key".
- **Cell failures abort the run.** Papermill stops at the first
  exception; the partial notebook is saved as `report_failed.ipynb`
  under the run's `analysis/` folder for debugging in JupyterLab.

---

## The CLI runner

### Basic usage

```bash
# One config
uv run python scripts/run_experiment.py configs/run_01.yaml

# Multiple configs (sequential, independent — one failure won't block others)
uv run python scripts/run_experiment.py configs/control_experiment.yaml configs/protein_synthesis_arrested.yaml

# All configs at once
uv run python scripts/run_experiment.py configs/*.yaml
```

When multiple configs are given, the script prints a summary table at
the end and exits with code 0 if everything succeeded, or 1 if any run
failed.

### Flags

| Flag | Effect |
|------|--------|
| `--force-extract` | Re-run extraction even when the provenance hash matches. Use after changing extraction *code* (the hash tracks config and inputs, not code). |
| `--skip-analysis` | Run extraction only. Useful when pre-baking extractions across many datasets. |
| `--analysis-only` | Skip extraction; error if artifacts are missing. If the provenance hash drifted, prints a warning listing the drifted fields but proceeds. |
| `--results-root PATH` | Override the results root for this invocation. Wins over YAML `RESULTS_ROOT`. Handy for redirecting to a cloud-synced folder. |

### How extraction reuse decides

On every run, before touching extraction:

- If `provenance.json` is missing → **re-extract** (nothing to reuse).
- If any of the 8 expected artifacts is missing → **re-extract**.
- If the stored `extract_hash` differs from the fresh one computed from
  the current config + source-file sha256s → **re-extract** (with a
  message listing which specific fields drifted).
- Otherwise → **reuse**: prints `reusing extraction at <path>` and
  skips directly to analysis.

The hash covers extraction config values *and* source-file bytes. If
someone replaces a TIFF on disk with a new version, the sha256 changes
and the cache invalidates automatically.

The hash does **not** cover extraction *code* — if you edit
`src/cell_analysis/segmentation.py`, you'll need `--force-extract` to
re-do the work. This is a deliberate tradeoff: hashing code would
invalidate every cached extraction on every minor edit, which is worse
than the occasional manual force.

---

## Output artifacts

Every run writes to `results/<run_name>/`. Two subfolders match the
two phases.

### `results/<run_name>/extraction/` — image → identities

Written by `extract.ipynb`. Skipped on subsequent runs when the
provenance hash matches.

| File | Format | Contents |
|---|---|---|
| `provenance.json` | JSON | Hash of extraction params + source-file sha256s + timestamps. Used to decide reuse. |
| `label_stack.npz` | `np.savez_compressed`, key `label_stack` | `(T, Y, X) int32` Cellpose masks on the phase channel. `0` = background, `N` = per-frame mask id. Frames flagged as bad by gating are zeroed. |
| `nucleus_label_stack.npz` | key `nucleus_label_stack` | Same shape; Cellpose masks on the fluor channel. |
| `tracked_cells.csv` | CSV | `frame, track_id, label, centroid_y, centroid_x, area` — post-merge `track_id` joined with the per-frame `label` (the mask id in `label_stack`). |
| `track_statistics.csv` | CSV | `track_id, first_frame, last_frame, mean_area, num_detections, lifetime, disappeared` |
| `frame_diagnostics.csv` | CSV | Per-frame gating statistics + flags |
| `merge_log.csv` | CSV | Audit trail of trackpy → merged track_id remapping |
| `dropped_frames.csv` | CSV | Bad-frame audit (empty when nothing was dropped) |

### `results/<run_name>/analysis/` — identities → metrics + plots

Written by `analysis.ipynb`. Rebuilt every run.

| File | Description |
|---|---|
| `report.html` | Self-contained HTML report with all plots and stats |
| `tracked_cells.csv` | Enriched per-cell per-frame — adds fluorescence, nucleoid metrics (`cv`, `nnrm`, `mean_edge_distance_norm`, `gaussian_sigma_norm`, `peri_core_asymmetry`), geometry (`radius`, `volume`, `surface_area`, `volume_rel`), `fluor_concentration`, `sav_ratio` |
| `track_statistics.csv` | Enriched per-track summary — adds means (`mean_fluor_intensity`, `mean_cv`, etc.) plus pre-burst columns |
| `fluorescence_alignment.csv` | Long-form F(offset)/F(-window) around each disappearing cell |
| `peri_core_alignment.csv` | Same shape but for peri/core asymmetry |
| `nucleus_persistence.csv` | Per-frame phase-cell vs. fluorescence-nucleus counts |
| `nucleus_persistence_summary.csv` | Endpoint / trajectory verdict (parallel vs. divergent) |
| `frame0_fate_comparison.csv` | Frame-0 Mann-Whitney U per feature |
| `fate_predictions.csv` | Per-cell logistic-regression predictions (LOO CV) |
| `fate_prediction_summary.csv` | AUC, accuracy, feature importances |
| `fate_predictions_no_area.csv`, `fate_prediction_summary_no_area.csv` | Same LR but with `area` dropped |

`results/<run_name>/config.yaml` is the frozen copy of the YAML config
used for the run.

Reports are also copied to `docs/reports/<run>/report.html` with a
"Back to index" link, and `docs/reports/index.html` lists all runs in
one page.

---

## Auxiliary tools

### `scripts/view_labels.py` — inspect masks in napari

```bash
uv run python scripts/view_labels.py results/control_experiment/
uv run python scripts/view_labels.py results/control_experiment/ --no-raw
```

Loads a run's `label_stack.npz`, `nucleus_label_stack.npz`, and (unless
`--no-raw`) the phase + fluor TIFFs via paths recorded in
`provenance.json`. Opens napari with four layers: phase, fluor
(green additive), cell masks, nucleus masks (hidden by default —
toggle from the layer panel).

### `scripts/diagnostic_overlay.py` — tune detection

Renders one frame with **accepted** cells (red X) and **rejected**
cells (cyan O, labeled with the reason each was rejected). This is the
fastest way to iterate on detection thresholds before touching a config.

```bash
uv run python scripts/diagnostic_overlay.py                        # defaults
uv run python scripts/diagnostic_overlay.py --min_contrast 1400    # relax contrast
uv run python scripts/diagnostic_overlay.py --frame 12             # different frame
```

Outputs `results/diagnostic_full.png` (whole frame) and
`results/diagnostic_crops.png` (6 zoomed regions), plus a
rejection-reason count table to stdout.

Rejection reason labels on the overlay:

- `edge` — cell touches the frame border (`exclude_edges` filter)
- `area=N` — area below `min_area`
- `circ=N.NN` — circularity below `min_circularity`
- `c=N` — intensity std-dev below `min_contrast`

Typical tuning loop: (1) look at the rejection reasons, (2) relax the
threshold that's rejecting real cells, (3) re-run the overlay, (4) once
happy, copy the values into your config's `DETECT_PARAMS`. See
`docs/detection_tuning.md` for a worked example with images.

Full flag list:

| Flag | Default | Description |
|------|---------|-------------|
| `--stack` | `data/gradient_0011/phase.tif` | Path to the TIFF stack |
| `--frame` | `0` | Frame index |
| `--outdir` | `results` | Output directory for PNGs |
| `--diameter` | `32` | Cellpose cell diameter |
| `--min_area` | `300` | Minimum cell area in pixels |
| `--min_circularity` | `0.7` | Minimum circularity (0-1) |
| `--min_contrast` | `1250` | Minimum intensity std-dev |
| `--exclude_edges` / `--no_exclude_edges` | `True` | Include/exclude border cells |
| `--gpu` / `--no_gpu` | `True` | Enable/disable GPU |
| `--resample` | `False` | Resample masks (slower, more precise) |
| `--crops` | auto | Custom crop regions as `Y0:Y1:X0:X1` (up to 6) |

---

## Development

Install the dev dependency group and run tests + lint:

```bash
uv sync --group dev

uv run pytest              # unit tests + end-to-end smoke test
uv run ruff check src/     # lint (uses ruff defaults)
```

The test suite includes:

- **Unit tests** per module (`test_fate_prediction.py`,
  `test_nucleoid_distribution.py`, `test_frame_gating.py`, …) that
  exercise each `add_*` / `run_*` function on synthetic data.
- **A smoke test** (`test_analysis_smoke.py`) that chains the whole
  analysis pipeline on a tiny 5-frame synthetic stack. Catches
  integration regressions (missing `__init__.py` re-exports, wrong
  return-value orders, orphan column references) that unit tests miss.
- **Runner tests** (`test_run_experiment.py`, `test_runner_reuse.py`)
  that verify config validation, extraction reuse, and CLI flags —
  they mock papermill so they run in seconds.

Where to put new code:

- **A new metric that runs on the tracked DataFrame** → new `add_foo()`
  function in `src/cell_analysis/pipeline.py`, plus a unit test.
- **A new plot** → new `plot_foo()` function in
  `src/cell_analysis/plotting.py`, plus a `show_with_source()` cell in
  `notebooks/analysis.ipynb`.
- **A new config knob** → parameters cell in the appropriate notebook +
  allowlist entry in `scripts/run_experiment.py` + defaults in each
  `configs/*.yaml`.
- **A new one-off analysis script** → `scripts/`. Keep it CLI-driven
  (argparse) so it stays reusable, not a hardcoded-paths script that
  rots into archaeology.

---

## Troubleshooting

**"ImportError: cannot import name '…' from 'cell_analysis'"** — you
added a symbol to `src/cell_analysis/<module>.py` but didn't re-export
it from `src/cell_analysis/__init__.py`. The notebooks import from the
top-level package.

**"Cell failures abort the run"** — papermill stopped at some cell in
one of the two notebooks. Look at
`results/<run>/analysis/report_failed.ipynb` (or the extraction
equivalent) in JupyterLab — the failing cell has its traceback in the
notebook itself.

**"reusing extraction at …" but I know I changed detection code** —
the hash tracks config and inputs, not source code. Use
`--force-extract` to force a re-run.

**"extracting: param drift: …"** — the runner detected a mismatch
between your current config's extraction section and the cached
provenance. The named fields have changed. This is expected behavior;
Cellpose will re-run.

**Plotly figures missing from `report.html`** — the analysis notebook
needs `plotly.io.renderers.default = "notebook_connected"` at the top
so nbconvert can capture the HTML output. This is already set in
`analysis.ipynb`; if you fork it, don't drop that line.

**Cellpose downloads models on first run** — expected, ~1 GB. Needs
internet. If behind a proxy, set `HTTPS_PROXY` before running.

**Large `.npz` files bloat your Dropbox/iCloud** — set `RESULTS_ROOT`
in the config (or `--results-root` on the CLI) to a local-only folder
if that matters.


### Windows-specific notes

**Glob patterns like `configs/*.yaml` work.** The runner expands them
itself (bash/zsh expand before the string arrives; PowerShell doesn't).
No shell workaround needed.

**Path separators.** Forward slashes work everywhere Python touches
paths — you can write `../data/foo.tif` in YAML on Windows and Python
handles it. Only worry about backslashes when composing paths in
PowerShell itself.

**No NVIDIA GPU?** Set `DETECT_PARAMS.gpu: false` in your config.
Cellpose will fall back to CPU. Expect ~10× slower on a mid-range CPU
(15–30 min per extraction instead of 2–5 min on GPU) — but everything
else works identically. Extraction reuse means you only pay this once
per dataset.

**Opening reports.** Use `start docs\reports\<run>\report.html` in
PowerShell (equivalent of macOS `open`).

**napari display issues.** If `scripts/view_labels.py` opens but the
window is blank or errors out with a Qt message, install the
`PyQt5` extras — some Windows Python installs ship without them:
```powershell
uv pip install PyQt5
```

**Long filenames with spaces or `+`.** All the shipped configs use
paths like `"Grad LB+sucr-20-0.zvi  Ch0.tif"` (spaces, `+`, double
space). YAML handles them fine when quoted, and Python's `pathlib`
handles them cross-platform. Just keep the quotes in the YAML.
