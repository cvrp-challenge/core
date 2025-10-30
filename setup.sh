#!/bin/bash
# Quick setup script for CVRP Challenge repository

set -e  # Exit on error

echo "================================"
echo "CVRP Challenge Setup Script"
echo "================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $PYTHON_VERSION"

# Create virtual environment
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists at .venv"
    read -p "Do you want to recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf .venv
        echo "Creating new virtual environment..."
        python3 -m venv .venv
    fi
else
    echo "Creating virtual environment at .venv..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo "Installing dependencies from requirements.txt..."
echo "This may take 3-10 minutes on first install, please be patient..."
pip install -r requirements.txt --progress-bar on

echo ""
echo "================================"
echo "✓ Setup complete!"
echo "================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To verify installation, run:"
echo "  python -c 'import numpy, matplotlib, yaml; print(\"All core dependencies installed!\")'"
echo ""

