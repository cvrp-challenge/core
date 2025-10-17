# Umbrella Repository for the CVRP Challenge

This repository is the brain of the 2026 CVRPLIB Best Known Solution Challenge -- a 30-day competition aimed at finding extremely high-quality vehicle routing solutions for a new XL benchmark set, 100 challenging CVRP instances ranging from 1,000 to 10,000 customers.

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


## To Do

    - add forks for remaining relevant solvers (SISR, KGLS-XXL, AILS-II, potentially to some decomposition approaches)
    - server setup
    - create test instances