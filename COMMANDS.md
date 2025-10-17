# Quick Command Reference

## Virtual Environment Management

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux/macOS/WSL)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Deactivate
deactivate

# Delete virtual environment
rm -rf .venv
```

## Package Management

```bash
# Install all dependencies (takes 3-10 mins first time)
pip install -r requirements.txt

# Install with progress bars (recommended!)
pip install -r requirements.txt --progress-bar on

# Install with verbose output (see what's happening)
pip install -r requirements.txt -v

# Install a new package
pip install package-name

# Update requirements.txt after installing new packages
pip freeze > requirements.txt

# Upgrade a package
pip install --upgrade package-name

# List installed packages
pip list

# Check for outdated packages
pip list --outdated

# Check if there are any dependency conflicts
pip check
```

## Running the Generator

```bash
# Example: Generate a test instance
cd instances
python generatorLarge.py 1000 2 1 3 4 42

# Arguments: n depotPos custPos demandType avgRouteSize randSeed
```

## Building C++ Solvers

### FILO1
```bash
cd solver/filo1
mkdir -p build && cd build
cmake ..
make
./main ../../instances/test-instances/x/X-n101-k25.vrp
```

### FILO2
```bash
cd solver/filo2
mkdir -p build && cd build
cmake ..
make
./main ../../instances/test-instances/x/X-n101-k25.vrp
```

### HGS (Hybrid Genetic Search)
```bash
cd solver/hgs
mkdir -p build && cd build
cmake ..
make
./hgs ../Instances/CVRP/X-n101-k25.vrp
```

## Working with PyVRP

```bash
# Install PyVRP in development mode
cd solver/pyvrp
pip install -e .

# Run PyVRP CLI
pyvrp --help
pyvrp instances/test-instances/x/X-n101-k25.vrp

# Run PyVRP tests
cd solver/pyvrp
pytest
```

## Code Quality

```bash
# Run linter (Ruff)
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .
```

## Testing

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov

# Run specific test file
pytest tests/test_file.py

# Run tests matching a pattern
pytest -k "test_name_pattern"
```

## Git Operations

```bash
# Clone with submodules
git clone --recursive <repo-url>

# Update submodules
git submodule update --init --recursive

# Pull latest changes including submodules
git pull --recurse-submodules
```

## Benchmarking

```bash
# Run benchmark configuration
# (Add your benchmark runner commands here)
```

## Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or Jupyter Lab
jupyter lab
```

## Checking Your Setup

```bash
# Verify Python version
python --version

# Verify pip
pip --version

# Verify CMake (for C++ solvers)
cmake --version

# Test Python imports
python -c "import numpy, matplotlib, yaml; print('✓ Core dependencies OK')"
```

## Troubleshooting

```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Reinstall all packages
pip install --force-reinstall -r requirements.txt

# Check for conflicts
pip check
```

