# CVRP XL Challenge Workspace

This repository contains the code, instances, solver integrations, and result artifacts used for large-scale CVRP experimentation on the 2026 XL challenge set (100 instances, 1,000-10,000 customers).

Challenge page: https://vrp.atd-lab.inf.puc-rio.br/index.php/en/bks-challenge

## What is in this repo

- `src/master`: DRSCI-style solving pipeline, orchestration, benchmarking, and utilities.
- `instances`: challenge/test instances, metadata, and helper scripts.
- `solver`: integrated third-party and in-house solvers (`filo1`, `filo2`, `hgs`, `pyvrp`, `ails2`, `cobra`).
- `config/solvers.yaml`: solver registry and execution metadata.
- `results`: generated summaries, plots, and analysis scripts.
- `docs`: setup, command reference, runner/solver documentation.

## Prerequisites

- Python 3.11+
- `pip`
- C++ toolchain + CMake (for compiled solvers)
- Java (for AILS2)
- Git (submodules)
- Gurobi license (if using `gurobi_mip` / `gurobi_lp` SCP solvers)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Detailed setup guide: `docs/SETUP.md`.

## Build integrated C++ solvers

```bash
# COBRA (required by FILO1)
cd solver/cobra
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/dev/core/solver/cobra/install
make -j4 && make install
cd ../../..

# FILO1
cd solver/filo1
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=ON -DCMAKE_PREFIX_PATH=~/dev/core/solver/cobra/install
make -j4
cd ../../..

# FILO2
cd solver/filo2
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../..

# HGS
cd solver/hgs
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../..
```

More build/use commands: `docs/COMMANDS.md`.

## Run the challenge pipeline

The current benchmark entrypoint is `src/master/challenge_runner.py`.

```bash
# Run full probabilistic DRSCI challenge run
python src/master/challenge_runner.py output

# Limit worker processes
python src/master/challenge_runner.py output --max_workers 4

# Switch SCP solver(s)
python src/master/challenge_runner.py output --scp_solvers gurobi_mip gurobi_lp
```

Outputs are written under `output/challenge_<timestamp>/` and include solution files, logs, and per-instance traces.

## Results and analysis

- Consolidated metrics: `results/summary.csv`
- Current high-level summary: `results/OVERVIEW.md`
- Plotting/analysis scripts: `results/*.py`

## Configuration

- Edit solver metadata and executable paths in `config/solvers.yaml`.
- Instance BKS references are in:
  - `instances/challenge-instances/challenge-bks.json`
  - `instances/challenge-instances/initial-bks.json`

## Documentation index

- Setup: `docs/SETUP.md`
- Command reference: `docs/COMMANDS.md`
- Runner guide: `docs/SOLVER_RUNNER_GUIDE.md`
- Add a new solver: `docs/ADDING_NEW_SOLVER.md`
- Docker notes: `docs/DOCKER.md`
- AILS2 setup: `docs/AILS2_SETUP.md`