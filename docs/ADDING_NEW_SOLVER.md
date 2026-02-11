# Adding a New Solver (External Repo)

This guide explains how to add a new external solver similar to FILO/FILO2.

## Overview

To add a new solver, you need to:
1. Create a solver module file in `src/master/routing/`
2. Register it in the solver registry
3. Optionally update configuration files

## Step-by-Step Guide

### Step 1: Add External Repo as Git Submodule

Add your external solver repository as a git submodule:

```bash
git submodule add <repository_url> solver/<your_solver_name>
git submodule update --init --recursive solver/<your_solver_name>
```

This will:
- Clone the external repo into `solver/<your_solver_name>/`
- Add it to `.gitmodules`
- Track the specific commit

**Example:**
```bash
git submodule add https://github.com/user/solver-name.git solver/my_solver
git submodule update --init --recursive solver/my_solver
```

### Step 2: Build the Solver (if needed)

If your solver is a compiled executable (like FILO), build it:

```bash
cd solver/<your_solver_name>
# Follow the solver's build instructions (e.g., cmake, make, etc.)
# The executable should end up in a predictable location like:
# - solver/<name>/build/<exe>
# - solver/<name>/build/Release/<exe>
# - solver/<name>/bin/<exe>
```

### Step 3: Create Solver Module File

Create a new file: `src/master/routing/solver_<your_solver_name>.py`

This file should contain:
- Functions to locate the executable
- Functions to run the executable via subprocess
- Functions to parse the output
- A registered solver function that returns `SolveOutput`

### Step 4: Register in Solver Registry

Add your solver to the lazy import loader in `src/master/routing/solver.py`:

```python
def _lazy_import_backend(solver_key: str) -> None:
    if solver_key == "hexaly":
        importlib.import_module("master.routing.solver_hexaly")
    elif solver_key in {"filo1", "filo2"}:
        importlib.import_module("master.routing.solver_filo")
    elif solver_key == "your_solver_name":  # ADD THIS
        importlib.import_module("master.routing.solver_your_solver_name")
    elif solver_key == "pyvrp":
        pass
```

### Step 5: Update Documentation (Optional)

Update the docstring in `src/master/routing/solver.py` to include your solver in the list of supported backends.

### Step 6: Update Configuration (Optional)

Add your solver to `config/solvers.yaml` for documentation purposes.

### Step 7: Update Hardcoded Lists (Optional)

If you want your solver to appear in CLI choices or default lists, update:
- `src/master/challenge_runner.py` - routing_solvers parameter
- `src/master/run_drsci_probabilistic.py` - SOLVERS list
- Other files that have hardcoded solver lists

## Template Structure

Your solver module should follow this structure:

```python
from master.routing.solver import SolveOutput, register_solver
from pathlib import Path
from typing import Any, Mapping, Dict
import subprocess
import time

def _resolve_executable(solver_name: str, override: Optional[str | Path] = None) -> Path:
    """Locate the solver executable."""
    # Similar to FILO's _resolve_executable
    # Check common build locations: solver/<name>/build/<exe>
    pass

def _run_solver_executable(
    instance_vrp: Path,
    work_dir: Path,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """Run the solver and return (stdout+stderr, runtime_seconds)."""
    pass

def _parse_solution(output: str, sol_file: Optional[Path] = None) -> Tuple[List[List[int]], float]:
    """Parse routes and cost from solver output."""
    pass

@register_solver("your_solver_name")
def _solve_with_your_solver(
    instance_path: Path,
    options: Mapping[str, Any],
) -> SolveOutput:
    """Main solver function - must return SolveOutput."""
    # Handle cluster_nodes if provided (for cluster subproblems)
    # Run solver
    # Parse results
    # Return SolveOutput with routes_vrplib in metadata
    pass
```

## Key Requirements

1. **Return Format**: Your solver function must return `SolveOutput` with:
   - `metadata["routes_vrplib"]`: List[List[int]] in VRPLIB format (global node IDs)
   - Routes should be in format `[1, customer1, customer2, ..., 1]` (depot at start/end)

2. **Cluster Support**: If your solver supports cluster subproblems:
   - Check for `options.get("cluster_nodes")`
   - Create sub-instance VRP file
   - Map local IDs back to global IDs

3. **Executable Location**: Follow FILO's pattern:
   - Check `solver/<name>/build/<exe>`
   - Support override via `options.get("<name>_exe")`

4. **Options Handling**: Common options:
   - `seed`: Random seed
   - `no_improvement`: Max iterations without improvement
   - `extra_args`: Additional CLI arguments
   - `keep_tmp`: Keep temp directories for debugging

## Example: Minimal Solver

See `src/master/routing/solver_filo.py` for a complete example of an external C++ solver integration.
