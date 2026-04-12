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
| `min_area` | 200 | Minimum cell area (rejects debris) |
| `min_circularity` | 0.7 | Minimum circularity (1.0 = perfect circle) |
| `gpu` | False | Enable CUDA GPU acceleration |
| `search_range` | 30.0 | Max cell displacement between frames (pixels) |
| `memory` | 3 | Frames a cell can disappear before breaking the track |

## Output

The pipeline exports two CSV files to `results/`:

- **`tracked_cells.csv`** — per-cell, per-frame data including track ID, centroid coordinates, and area
- **`track_statistics.csv`** — per-track summary: lifetime, mean area, disappearance flag

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
