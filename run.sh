#!/bin/bash
# Unified solver runner wrapper script
# This provides a convenient interface to run_solver.py

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if it exists
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# Run the Python runner script with all arguments
python3 "$SCRIPT_DIR/runner.py" "$@"

