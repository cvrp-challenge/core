#!/usr/bin/env python3
"""Test script to capture AILS2 actual output format."""

import sys
from pathlib import Path
import tempfile
import subprocess
import shutil

sys.path.insert(0, str(Path(__file__).parent / "src"))

from master.routing.solver_ails2 import _resolve_jar, _run_ails2_executable

def main():
    print("Testing AILS2 output format...")
    
    # Use a small test instance
    test_instance = Path("instances/test-instances/x/X-n101-k25.vrp")
    if not test_instance.exists():
        print(f"Test instance not found: {test_instance}")
        return 1
    
    jar_path = _resolve_jar()
    print(f"Using JAR: {jar_path}")
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="ails2_test_"))
    try:
        # Copy and clean instance
        local_inst = tmp_dir / test_instance.name
        content = test_instance.read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        local_inst.write_text("\n".join(lines) + "\n")
        
        print(f"\nRunning AILS2 with 5 second timeout...")
        print(f"Instance: {local_inst}")
        print(f"Work dir: {tmp_dir}")
        
        out, runtime = _run_ails2_executable(
            instance_vrp=local_inst,
            work_dir=tmp_dir,
            jar_path=jar_path,
            max_runtime=5.0,
            rounded=True,
            best=0.0,
        )
        
        print(f"\n{'='*60}")
        print("STDOUT OUTPUT:")
        print(f"{'='*60}")
        print(out)
        print(f"{'='*60}\n")
        
        # Check for solution files
        sol_files = list(tmp_dir.glob("*.sol"))
        print(f"Solution files found: {len(sol_files)}")
        for sf in sol_files:
            print(f"\n  File: {sf.name}")
            print(f"  Size: {sf.stat().st_size} bytes")
            content = sf.read_text()
            print(f"  Content:\n{content[:500]}")
            if len(content) > 500:
                print(f"  ... (truncated, total {len(content)} chars)")
        
        # Test parsing
        from master.routing.solver_ails2 import _parse_routes_from_text, _parse_cost_from_text
        
        text_to_parse = out
        if sol_files:
            text_to_parse = sol_files[0].read_text()
        
        routes = _parse_routes_from_text(text_to_parse)
        cost = _parse_cost_from_text(text_to_parse)
        
        print(f"\n{'='*60}")
        print("PARSING RESULTS:")
        print(f"{'='*60}")
        print(f"Routes found: {len(routes)}")
        for i, r in enumerate(routes[:5], 1):
            print(f"  Route {i}: {r}")
        if len(routes) > 5:
            print(f"  ... and {len(routes) - 5} more routes")
        print(f"Cost: {cost}")
        print(f"{'='*60}\n")
        
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
