# Cell Analysis Pipeline

Napari-based pipeline for bacterial cell tracking and fluorescence analysis in time-lapse phase-contrast microscopy.

The pipeline detects individual cells using [Cellpose](https://github.com/MouseLand/cellpose), links them across frames with [trackpy](https://github.com/soft-matter/trackpy), and measures fluorescence intensity per cell. Results are exported as CSV files for downstream analysis.

## Overview

| Step | Module | Description |
|------|--------|-------------|
| 1 | `cell_analysis.io` | Load single- or multi-channel TIFF stacks |
| 2 | `cell_analysis.segmentation` | Detect cells per frame (Cellpose or classical fallback) |
| 3 | `cell_analysis.tracking` | Link detections into tracks across time |
| 4 | `cell_analysis.matching` | Match phase-contrast cells to fluorescence nuclei |
| 5 | `cell_analysis.io` | Export tracked data and statistics to CSV |

The main entry point is the Jupyter notebook at `notebooks/analysis.ipynb`.

## Prerequisites

- **Python 3.11 or newer**
- **uv** (Python package manager)
- **VS Code** with the Jupyter extension (recommended) or JupyterLab
- **Git**

---

## Environment Setup

<details>
<summary><strong>macOS</strong></summary>

### 1. Install Homebrew (if not already installed)

Open **Terminal** (Cmd+Space, type "Terminal") and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow the on-screen instructions. After installation, restart your terminal or run the commands shown at the end of the installer to add Homebrew to your PATH.

### 2. Install Git

macOS includes Git via Xcode Command Line Tools. If you don't have it:

```bash
xcode-select --install
```

Or install via Homebrew:

```bash
brew install git
```

Verify:

```bash
git --version
```

### 3. Install Python 3.11+

```bash
brew install python@3.11
```

Verify:

```bash
python3 --version
```

### 4. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then verify:

```bash
uv --version
```

### 5. Install VS Code

Download from https://code.visualstudio.com/download and drag the `.app` to your Applications folder.

To launch VS Code from the terminal, open VS Code, press **Cmd+Shift+P**, type "Shell Command: Install 'code' command in PATH", and select it. Then you can run:

```bash
code .
```

### 6. Install VS Code Extensions

Open VS Code and install the following extensions (Cmd+Shift+X to open the Extensions panel):

- **Python** (`ms-python.python`) — Python language support and interpreter selection
- **Jupyter** (`ms-toolsai.jupyter`) — Run `.ipynb` notebooks inside VS Code

Or install from the terminal:

```bash
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

### 7. Clone the Repository

```bash
cd ~/Projects  # or wherever you keep your repos
git clone https://github.com/vvvlad/bacteria.git
cd bacteria
```

### 8. Create the Virtual Environment and Install Dependencies

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv/` directory, and installs all pinned dependencies into it.

### 9. Select the Python Interpreter in VS Code

1. Open the project folder in VS Code: `code .`
2. Press **Cmd+Shift+P** and type **"Python: Select Interpreter"**
3. Choose the interpreter at `./.venv/bin/python`

This ensures both the editor and Jupyter notebooks use the project's virtual environment.

### 10. Open and Run the Notebook

1. Open `notebooks/analysis.ipynb` in VS Code
2. When prompted, select the kernel from `.venv`
3. Place your TIFF stack(s) in `data/raw/`
4. Run cells with **Shift+Enter**

### Notes

- **Apple Silicon (M1/M2/M3):** Cellpose GPU acceleration is not supported on Apple Silicon. The pipeline runs on CPU by default (`gpu=False`). First run downloads the Cellpose model (~1 GB).
- **Large files:** Raw TIFF stacks can be 50-100+ MB per stack. They are excluded from git via `.gitignore`.

</details>

<details>
<summary><strong>Windows</strong></summary>

### 1. Install Git

Download the installer from https://git-scm.com/download/win and run it.

During installation:
- Keep the default options
- When asked about adjusting your PATH, select **"Git from the command line and also from 3rd-party software"**
- When asked about line endings, select **"Checkout as-is, commit Unix-style line endings"**

Open a new **PowerShell** window and verify:

```powershell
git --version
```

### 2. Install Python 3.11+

Download the installer from https://www.python.org/downloads/ (choose 3.11 or newer).

During installation:
- **Check "Add python.exe to PATH"** at the bottom of the first screen
- Click **"Install Now"**

Open a new PowerShell window and verify:

```powershell
python --version
```

### 3. Install uv

In PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen PowerShell, then verify:

```powershell
uv --version
```

### 4. Install VS Code

Download the installer from https://code.visualstudio.com/download and run it.

During installation:
- **Check "Add to PATH"** so you can launch VS Code from the terminal
- **Check "Register Code as an editor for supported file types"** (optional)

Restart PowerShell, then verify:

```powershell
code --version
```

### 5. Install VS Code Extensions

Open VS Code and install the following extensions (Ctrl+Shift+X to open the Extensions panel):

- **Python** (`ms-python.python`) — Python language support and interpreter selection
- **Jupyter** (`ms-toolsai.jupyter`) — Run `.ipynb` notebooks inside VS Code

Or install from PowerShell:

```powershell
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

### 6. Clone the Repository

```powershell
cd ~\Projects  # or wherever you keep your repos
git clone https://github.com/vvvlad/bacteria.git
cd bacteria
```

### 7. Create the Virtual Environment and Install Dependencies

```powershell
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv\` directory, and installs all pinned dependencies into it.

### 8. Select the Python Interpreter in VS Code

1. Open the project folder in VS Code: `code .`
2. Press **Ctrl+Shift+P** and type **"Python: Select Interpreter"**
3. Choose the interpreter at `.\.venv\Scripts\python.exe`

This ensures both the editor and Jupyter notebooks use the project's virtual environment.

### 9. Open and Run the Notebook

1. Open `notebooks/analysis.ipynb` in VS Code
2. When prompted, select the kernel from `.venv`
3. Place your TIFF stack(s) in `data\raw\`
4. Run cells with **Shift+Enter**

### Notes

- **NVIDIA GPU (optional):** If you have an NVIDIA GPU with CUDA drivers, you can pass `gpu=True` to the detection functions for faster Cellpose inference. This is not required — CPU mode works fine.
- **First run:** Cellpose downloads its pretrained model (~1 GB) on first use. Make sure you have a stable internet connection.
- **Large files:** Raw TIFF stacks can be 50-100+ MB per stack. They are excluded from git via `.gitignore`.
- **Long paths:** If you encounter path-length errors, enable long paths in Windows:
  ```powershell
  git config --global core.longpaths true
  ```

</details>

---

## Project Structure

```
bacteria/
├── data/
│   ├── raw/           # Raw TIFF stacks (not tracked by git)
│   └── processed/     # Intermediate outputs (not tracked by git)
├── notebooks/
│   └── analysis.ipynb # Main analysis notebook
├── results/           # Exported CSVs (not tracked by git)
├── scripts/
│   └── diagnostic_overlay.py  # Visual debugging of detection filters
├── src/
│   └── cell_analysis/
│       ├── __init__.py
│       ├── io.py            # TIFF loading and CSV export
│       ├── segmentation.py  # Cell detection (Cellpose + classical)
│       ├── tracking.py      # Temporal linking with trackpy
│       └── matching.py      # Phase-to-fluorescence cell matching
├── .gitignore
├── pyproject.toml
└── uv.lock
```

## Input Data

Place your microscopy TIFF stacks in `data/raw/`. Expected formats:

- **Single-channel:** shape `(T, Y, X)` — e.g. 25 frames of 1040x1388 pixels
- **Multi-channel:** shape `(T, C, Y, X)` — phase-contrast + fluorescence
- **Bit depth:** uint8 or uint16

## Detection Parameters

The notebook defines tunable parameters at the top:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `diameter` | 32 | Median cell diameter in pixels |
| `min_area` | 300 | Minimum cell area (rejects debris) |
| `min_circularity` | 0.7 | Minimum circularity (1.0 = perfect circle) |
| `min_contrast` | 1550 | Minimum intensity std-dev (rejects faded cells) |
| `gpu` | True | Enable GPU acceleration (MPS on Apple Silicon, CUDA on NVIDIA) |
| `search_range` | 30.0 | Max cell displacement between frames (pixels) |
| `memory` | 3 | Frames a cell can disappear before breaking the track |
| `TRIM_FRAMES` | 2 | Initial frames to discard before detection (e.g. out-of-focus frames) |
| `MERGE_MAX_DISTANCE` | 15.0 | Max pixels between track end/start to merge fragments |
| `MERGE_MAX_GAP` | 18 | Max frame gap for fragment merging |

## Output

The pipeline exports two CSV files to `results/`:

- **`tracked_cells.csv`** — per-cell, per-frame data including track ID, centroid coordinates, and area
- **`track_statistics.csv`** — per-track summary: lifetime, mean area, disappearance flag

## Diagnostic Overlay

The `scripts/diagnostic_overlay.py` script generates visual overlays that show which cells were **accepted** and which were **rejected** by the detection filters, along with the reason each cell was rejected. This is the main tool for tuning detection parameters.

### What it produces

Two images saved to `results/`:

| File | Description |
|------|-------------|
| `diagnostic_full.png` | Full frame — accepted cells marked with **red X**, rejected cells marked with **cyan O** and labeled with the rejection reason |
| `diagnostic_crops.png` | Six zoomed crop regions for close inspection of individual cells and their rejection labels |

### Basic usage

Run with the current default parameters:

```bash
uv run python scripts/diagnostic_overlay.py
```

### Overriding detection parameters

Pass any detection parameter as a CLI flag to test different thresholds without editing code:

```bash
# Relax contrast filter to accept more faded cells
uv run python scripts/diagnostic_overlay.py --min_contrast 1400

# Lower area threshold and relax circularity
uv run python scripts/diagnostic_overlay.py --min_area 200 --min_circularity 0.5

# Combine multiple overrides
uv run python scripts/diagnostic_overlay.py --min_area 200 --min_contrast 1400 --min_circularity 0.5
```

### Inspecting a specific frame

By default the script analyses frame 0. Use `--frame` to pick another:

```bash
uv run python scripts/diagnostic_overlay.py --frame 12
```

### Custom crop regions

Zoom into specific areas of interest by passing `--crops` with `Y0:Y1:X0:X1` coordinates (up to 6 regions):

```bash
uv run python scripts/diagnostic_overlay.py --crops 150:400:250:550 400:650:700:1000
```

If omitted, six crops are auto-generated spread across the frame.

### Using a different stack

```bash
uv run python scripts/diagnostic_overlay.py --stack path/to/other_stack.tif
```

### Debugging detection with the overlay

The overlay is designed for an iterative tuning workflow:

1. **Run the script** with current defaults and open `results/diagnostic_full.png`.

2. **Look at the cyan circles.** Each rejected cell has a label explaining why it was rejected:
   - `edge` — cell touches the frame border (filtered by `--exclude_edges`)
   - `area=N` — cell area is below `--min_area`
   - `circ=N.NN` — circularity is below `--min_circularity`
   - `c=N` — intensity contrast (std-dev) is below `--min_contrast`
   - A cell can have **multiple reasons** (e.g. `area=180, circ=0.52`)

3. **Decide if rejected cells should be accepted.** Open `results/diagnostic_crops.png` for a closer look. If you see real cells being rejected, relax the corresponding threshold:
   - Too many real cells rejected for contrast? Lower `--min_contrast`
   - Small but valid cells being dropped? Lower `--min_area`
   - Slightly elongated cells being rejected? Lower `--min_circularity`

4. **Re-run with adjusted parameters** and compare the new overlay:
   ```bash
   uv run python scripts/diagnostic_overlay.py --min_contrast 1400
   ```

5. **Check for false positives.** If relaxing a threshold lets in debris or halos (red X on non-cells), tighten the threshold back.

6. **Once satisfied**, update `DETECT_PARAMS` in `notebooks/analysis.ipynb` with the tuned values and re-run the full pipeline.

The script also prints a rejection reason summary to the terminal:

```
Accepted: 349, Rejected: 72

Rejection reasons (cells can have multiple):
  edge: 38
  c: 22
  area: 15
  circ: 8
```

This tells you at a glance which filter is rejecting the most cells, helping you prioritise which threshold to adjust first.

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--stack` | `data/raw/Gradient-0011.zvi  Ch0.tif` | Path to the TIFF stack |
| `--frame` | `0` | Frame index to analyse |
| `--outdir` | `results` | Output directory for PNG files |
| `--diameter` | `32` | Cellpose cell diameter |
| `--min_area` | `300` | Minimum cell area in pixels |
| `--min_circularity` | `0.7` | Minimum circularity (0-1) |
| `--min_contrast` | `1550` | Minimum intensity std-dev |
| `--exclude_edges` / `--no_exclude_edges` | `True` | Include/exclude border cells |
| `--gpu` / `--no_gpu` | `True` | Enable/disable GPU |
| `--resample` | `False` | Resample masks (slower, more precise boundaries) |
| `--crops` | auto | Custom crop regions as `Y0:Y1:X0:X1` (up to 6) |

## Alternative: Running with JupyterLab

If you prefer JupyterLab over VS Code:

```bash
uv run jupyter lab
```

This starts a local Jupyter server and opens it in your browser. Navigate to `notebooks/analysis.ipynb`.

## Development

Install dev dependencies:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run pytest
```

Lint:

```bash
uv run ruff check src/
```
