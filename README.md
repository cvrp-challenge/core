# Umbrella Repository for the CVRP Challenge

This repository is the brain of the 2026 CVRPLIB Best Known Solution Challenge --- a 30-day competition aimed at finding extremely high-quality vehicle routing solutions for a new XL benchmark set, 100 challenging CVRP instances ranging from 1,000 to 10,000 customers.

More info: https://vrp.atd-lab.inf.puc-rio.br/index.php/en/bks-challenge


## Contents

Main logic, runners, instances, wrappers, logging, links to solvers as submodules.

## Requirements

- Python 3.11+ with pip
- CMake and C++ compiler (for C++ solvers)
- Git (for submodules)

See `requirements.txt` for Python dependencies.


## Setup

### Quick Setup

Use the automated setup script:

```bash
./setup.sh
```

Or manually:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

### Building Solvers

Build the C++ solvers (WSL/Linux):

```bash
# Build COBRA library (required for FILO1)
cd solver/cobra && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/dev/core/solver/cobra/install
make -j4 && make install
cd ../../..

# Build FILO1
cd solver/filo1 && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_VERBOSE=ON -DCMAKE_PREFIX_PATH=~/dev/core/solver/cobra/install
make -j4
cd ../../..

# Build FILO2
cd solver/filo2 && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
cd ../../..

# Build HGS
cd solver/hgs && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4
cd ../../..
```

See [COMMANDS.md](COMMANDS.md) for more build options.

## Usage

### Unified Solver Runner

Run any solver on any instance with a single command:

```bash
# List available solvers
./run.sh --list

# Run any solver on an instance
./run.sh hgs instances/test-instances/x/X-n101-k25.vrp
./run.sh filo1 instances/test-instances/x/X-n101-k25.vrp
./run.sh filo2 instances/test-instances/x/X-n101-k25.vrp
./run.sh pyvrp instances/test-instances/x/X-n101-k25.vrp

# Run on XL instances
./run.sh hgs instances/test-instances/xl/XLTEST-n2541-k62.vrp
```

The runner automatically:
- Validates solver availability
- Checks instance files
- Activates virtual environment
- Provides unified interface for all solvers

See [COMMANDS.md](COMMANDS.md) for more usage examples.


## To Do

    - add forks for remaining relevant solvers (SISR, KGLS-XXL, AILS-II, potentially to some decomposition approaches)
    - server setup
    - create test instances