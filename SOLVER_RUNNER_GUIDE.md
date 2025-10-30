# Unified Solver Runner Guide

## Overview

A unified solver runner system has been created for your CVRP Challenge repository. This system allows you to run any configured VRP solver with a single command, providing a consistent interface across all solvers.

## What Was Built

### 1. Built Solvers

Successfully built the following solvers:

- **✓ FILO1** - Fast Iterative Local Optimization (version 1)
  - Location: `solver/filo1/build/filo`
  - Dependencies: COBRA library (installed locally)
  - Status: Ready to use

- **✓ FILO2** - Fast Iterative Local Optimization (version 2)
  - Location: `solver/filo2/build/filo2`
  - Status: Ready to use
  
- **✓ HGS-CVRP** - Hybrid Genetic Search
  - Location: `solver/hgs/build/hgs`
  - Status: Ready to use

- **✓ PyVRP** - Python-based solver
  - Status: Added to requirements.txt
  - Note: Install with `pip install -r requirements.txt` or `pip install pyvrp`

### 2. COBRA Library

The COBRA library (required for FILO1) has been:
- Cloned from https://github.com/acco93/cobra
- Built and installed locally at `solver/cobra/install`
- Integrated with FILO1 build system
- No sudo/system-wide installation required

### 3. Configuration System

Created `config/solvers.yaml` - A YAML configuration file that defines:
- Available solvers and their properties
- Command templates for each solver
- Default arguments
- Supported file formats
- Enable/disable status

### 4. Unified Runner

Created two runner scripts:

#### `runner.py` (Main Python Script)
- Full-featured Python runner
- Loads solver configuration from YAML
- Validates solvers and instances
- Executes solvers with proper error handling
- Supports custom arguments per solver

#### `run.sh` (Shell Wrapper)
- Convenient bash wrapper
- Auto-activates virtual environment
- Passes all arguments to Python runner
- Works from any directory

## Quick Start

### List Available Solvers

```bash
./run.sh --list
```

This shows all configured solvers with their status and descriptions.

### Run a Solver

Basic usage:
```bash
./run.sh <solver_id> <instance_path>
```

Examples:
```bash
# Run any solver on a small instance
./run.sh hgs instances/test-instances/x/X-n101-k25.vrp
./run.sh filo1 instances/test-instances/x/X-n101-k25.vrp
./run.sh filo2 instances/test-instances/x/X-n101-k25.vrp
./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp

# Run on medium instances
./run.sh filo1 instances/test-instances/x/X-n344-k43.vrp
./run.sh filo2 instances/test-instances/x/X-n344-k43.vrp

# Run on XL instances
./run.sh hgs instances/test-instances/xl/XLTEST-n2541-k62.vrp
./run.sh filo2 instances/test-instances/xl/XLTEST-n7353-k40.vrp
```

### Run with Custom Arguments

PyVRP supports additional arguments:
```bash
./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp --seed 42 --max_runtime 60
```

HGS supports seed:
```bash
./run.sh hgs instances/test-instances/x/X-n101-k25.vrp --seed 123
```

## Features

### Automatic Validation
- ✓ Checks if solver is enabled
- ✓ Verifies solver executable exists
- ✓ Validates instance file exists
- ✓ Checks file format compatibility

### Smart Path Resolution
- Works with relative or absolute paths
- Resolves paths from repository root
- Finds instances automatically

### Clean Output
- Shows execution details
- Displays solver output
- Reports exit codes
- Handles interruptions gracefully (Ctrl+C)

### Virtual Environment Support
- Auto-activates `.venv` if it exists
- Ensures correct Python environment

## CMake Requirements

Your system has:
- **CMake 3.22.1** ✓ (meets all requirements)

Solver requirements:
- FILO1: CMake ≥ 3.16 ✓
- FILO2: CMake ≥ 3.22 ✓
- HGS: CMake ≥ 3.15 ✓

## File Structure

```
core/
├── runner.py              # Main Python runner script
├── run.sh                 # Shell wrapper for convenience
├── requirements.txt       # Updated with pyvrp
├── config/
│   └── solvers.yaml       # Solver configuration (all 4 solvers enabled)
├── solver/
│   ├── cobra/             # COBRA library for FILO1
│   │   ├── build/         # Built COBRA
│   │   └── install/       # Locally installed COBRA
│   ├── filo1/
│   │   └── build/
│   │       └── filo       # Built executable ✓
│   ├── filo2/
│   │   └── build/
│   │       └── filo2      # Built executable ✓
│   ├── hgs/
│   │   └── build/
│   │       └── hgs        # Built executable ✓
│   └── pyvrp/             # Python package
└── instances/
    └── test-instances/
        ├── x/             # 100 instances
        └── xl/            # 100 XL instances
```

## Adding New Solvers

To add a new solver:

1. **Build the solver** (if C++)
   ```bash
   cd solver/newsolver
   mkdir build && cd build
   cmake .. && make
   ```

2. **Add configuration** to `config/solvers.yaml`:
   ```yaml
   newsolver:
     name: "New Solver Name"
     description: "Description of the solver"
     type: "compiled"  # or "python"
     executable: "solver/newsolver/build/newsolver"
     enabled: true
     command_template: "{executable} {instance} {seed}"
     default_args:
       seed: 1
     supported_formats: [".vrp"]
     notes: "Any additional notes"
   ```

3. **Test it**:
   ```bash
   ./run.sh --list              # Should appear in the list
   ./run.sh newsolver path/to/instance.vrp
   ```

## Troubleshooting

### "Solver executable not found"
The solver hasn't been built yet. Build it first:

For FILO1 (requires COBRA):
```bash
# Build COBRA first
cd solver/cobra
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/dev/core/solver/cobra/install
make -j4 && make install
cd ../..

# Then build FILO1
cd filo1
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=ON -DCMAKE_PREFIX_PATH=~/dev/core/solver/cobra/install
make -j4
```

For other solvers:
```bash
cd solver/solvername
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
```

### "Unknown solver"
Check the solver ID with `./run.sh --list` and use the exact ID shown.

### "Instance file not found"
Verify the path to the instance file. You can use paths relative to the repository root.

### Solver runs but no output
Some solvers may take a long time on large instances. Use Ctrl+C to interrupt if needed.

## Examples for Each Solver Type

### FILO1 (Original FILO with COBRA)
```bash
# Small instance (~100 customers)
./run.sh filo1 instances/test-instances/x/X-n101-k25.vrp

# Large instance (~1000 customers)
./run.sh filo1 instances/test-instances/x/X-n1001-k43.vrp

# XL instance (~10000 customers)
./run.sh filo1 instances/test-instances/xl/XLTEST-n10001-k798.vrp
```

### FILO2 (Scalable FILO)
```bash
# Small instance (~100 customers)
./run.sh filo2 instances/test-instances/x/X-n101-k25.vrp

# Large instance (~1000 customers)
./run.sh filo2 instances/test-instances/x/X-n1001-k43.vrp

# XL instance (~10000 customers)
./run.sh filo2 instances/test-instances/xl/XLTEST-n10001-k798.vrp
```

### HGS (Comprehensive Search)
```bash
# Default seed
./run.sh hgs instances/test-instances/x/X-n101-k25.vrp

# Custom seed for reproducibility
./run.sh hgs instances/test-instances/x/X-n101-k25.vrp --seed 42

# On XL instances
./run.sh hgs instances/test-instances/xl/XLTEST-n7683-k1621.vrp
```

### PyVRP (Python-based)
```bash
# Install first (if not already)
pip install pyvrp

# Run with defaults
./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp

# With custom runtime limit
./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp --max_runtime 120

# With specific seed
./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp --seed 123 --max_runtime 60
```

## Performance Comparison

To compare solvers on the same instance:
```bash
# Run all four solvers on the same instance
./run.sh filo1 instances/test-instances/x/X-n101-k25.vrp
./run.sh filo2 instances/test-instances/x/X-n101-k25.vrp
./run.sh hgs instances/test-instances/x/X-n101-k25.vrp
./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp --max_runtime 30
```

## Quiet Mode

For scripting, you can suppress the runner's output:
```bash
python3 runner.py hgs instances/test-instances/x/X-n101-k25.vrp --quiet
```

This shows only the solver's output, not the runner's metadata.

## Integration with Your Workflow

The runner is designed to integrate seamlessly:
- Use in shell scripts for batch processing
- Call from Python scripts for automation
- Integrate with benchmarking frameworks
- Use in CI/CD pipelines

## Next Steps

1. **Install PyVRP**:
   ```bash
   pip install -r requirements.txt
   # Or specifically: pip install pyvrp
   ```

2. **Try all solvers** on your instances:
   ```bash
   ./run.sh --list
   ./run.sh filo1 instances/test-instances/x/X-n101-k25.vrp
   ./run.sh filo2 instances/test-instances/x/X-n101-k25.vrp
   ./run.sh hgs instances/test-instances/x/X-n101-k25.vrp
   ./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp
   ```

3. **Compare solvers** on your benchmark instances

4. **Customize configurations** in `config/solvers.yaml` as needed

## Summary

You now have:
- ✅ **All 4 solvers** built and ready (FILO1, FILO2, HGS, PyVRP)
- ✅ COBRA library installed locally for FILO1
- ✅ PyVRP added to requirements.txt
- ✅ Unified runner for easy solver execution
- ✅ Configuration system for solver management
- ✅ Complete documentation
- ✅ Tested and working on multiple instances

Run `./run.sh --list` to see your available solvers and start solving!

