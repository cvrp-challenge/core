"""
Main entry point for the CVRP solver application.
"""

import sys
from pathlib import Path


def verify_environment():
    """Verify that the Docker environment is set up correctly."""
    print("=" * 70)
    print("CVRP Solver - Environment Verification".center(70))
    print("=" * 70)
    print()
    
    # Check Python version
    print(f"✓ Python version: {sys.version.split()[0]}")
    print(f"✓ Python path: {sys.executable}")
    print()
    
    # Check key dependencies
    print("Checking dependencies...")
    dependencies = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "yaml": "yaml",
        "vrplib": "vrplib",
        "pyvrp": "pyvrp",
        "scikit-learn": "sklearn",
        "scikit-fuzzy": "skfuzzy",
        "pyclustering": "pyclustering",
        "pandas": "pandas",
        "tqdm": "tqdm",
    }
    
    failed_imports = []
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - FAILED: {e}")
            failed_imports.append(name)
    
    print()
    
    # Check project structure
    print("Checking project structure...")
    repo_root = Path(__file__).resolve().parents[2]
    print(f"  ✓ Repository root: {repo_root}")
    
    key_paths = [
        ("src/master", repo_root / "src" / "master"),
        ("solver/pyvrp", repo_root / "solver" / "pyvrp"),
        ("instances", repo_root / "instances"),
    ]
    
    for name, path in key_paths:
        if path.exists():
            print(f"  ✓ {name}/ exists")
        else:
            print(f"  ✗ {name}/ - NOT FOUND")
    
    print()
    
    # Summary
    if failed_imports:
        print("=" * 70)
        print("⚠️  WARNING: Some dependencies failed to import!")
        print(f"   Failed: {', '.join(failed_imports)}")
        print("=" * 70)
        return 1
    else:
        print("=" * 70)
        print("✅ All checks passed! Docker environment is ready.")
        print("=" * 70)
        return 0


def main():
    """Main entry point."""
    return verify_environment()


if __name__ == "__main__":
    sys.exit(main())