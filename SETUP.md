# Setup Guide

This guide will help you set up your development environment for the CVRP Challenge repository.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- For C++ solvers: CMake, g++/clang++

## Quick Start

### 1. Create a Virtual Environment

A virtual environment isolates your project dependencies from system-wide Python packages.

```bash
# Create a new virtual environment
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

**On Linux/macOS/WSL:**
```bash
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` appear at the beginning of your terminal prompt, indicating the virtual environment is active.

### 3. Install Dependencies

With the virtual environment activated, install all required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** First-time installation typically takes **3-10 minutes** depending on your system. This is normal! Large packages like `numpy`, `pandas`, and `matplotlib` need to be downloaded and sometimes compiled.

**To see progress:**
```bash
# Show progress bars
pip install -r requirements.txt --progress-bar on

# Or use verbose mode to see detailed output
pip install -r requirements.txt -v
```

### 4. Verify Installation

Test that the basic dependencies are installed:

```bash
python -c "import numpy, matplotlib, yaml; print('All core dependencies installed!')"
```

## Working with the Repository

### Daily Workflow

1. **Always activate the virtual environment** before working:
   ```bash
   source .venv/bin/activate  # Linux/macOS/WSL
   ```

2. Work on your code

3. When done, deactivate (optional):
   ```bash
   deactivate
   ```

### Installing Additional Packages

If you need to install additional packages:

```bash
# Activate venv first
source .venv/bin/activate

# Install the package
pip install package-name

# Update requirements.txt
pip freeze > requirements.txt
```

### Building C++ Solvers

For the C++ solvers (filo1, filo2, hgs), you'll need to build them separately:

```bash
# Example for filo1
cd solver/filo1
mkdir -p build && cd build
cmake ..
make
```

## PyVRP Solver

The PyVRP solver has its own build system using `meson`. To work with it:

```bash
cd solver/pyvrp

# Install in development mode (recommended)
pip install -e .

# Or build and install
pip install .
```

## Troubleshooting

### Installation Takes Too Long or Appears Frozen

**This is normal!** First-time pip installations can take 3-10 minutes because:
- Large packages (`numpy`, `pandas`, `matplotlib`, `jupyter`) are being downloaded
- Some packages may compile from source if pre-built wheels aren't available
- Dependencies are being resolved

**To see what's happening:**
```bash
# Option 1: Progress bars
pip install -r requirements.txt --progress-bar on

# Option 2: Verbose output (shows each step)
pip install -r requirements.txt -v

# Option 3: Very verbose (shows everything)
pip install -r requirements.txt -vv
```

**Speed it up:**
```bash
# Use binary wheels only (no compilation)
pip install -r requirements.txt --only-binary :all:

# Or upgrade pip first (newer pip is faster)
pip install --upgrade pip
```

### Virtual Environment Not Activating

- Ensure Python 3 is installed: `python3 --version`
- Try using `python` instead of `python3` if on Windows

### Permission Errors on Windows

If you get execution policy errors on Windows PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Import Errors

- Make sure your virtual environment is activated (you should see `(.venv)` in your prompt)
- Reinstall requirements: `pip install -r requirements.txt`

### Compilation Errors (numpy, pandas, etc.)

If packages fail to compile:
```bash
# Install build tools first
# On Ubuntu/Debian:
sudo apt-get install python3-dev build-essential

# On macOS:
xcode-select --install

# Then retry:
pip install -r requirements.txt
```

## Tools and Configuration

### Linting with Ruff

```bash
# Check code quality
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov
```

## Additional Resources

- CVRPLIB Challenge: https://vrp.atd-lab.inf.puc-rio.br/index.php/en/bks-challenge
- PyVRP Documentation: https://pyvrp.org/
- VRPLib Python Package: https://pypi.org/project/vrplib/

