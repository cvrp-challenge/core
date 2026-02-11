#!/usr/bin/env python3
"""
Quick test for AILS2 solver - minimal test to verify it works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from master.routing.solver import solve

def main():
    print("Quick AILS2 Test")
    print("=" * 50)
    
    # Try to find any test instance
    test_instances = [
        "instances/test-instances/x/X-n101-k25.vrp",
        "instances/test-instances/x/X-n110-k13.vrp",
        "instances/challenge-instances/XL-n1048-k237.vrp",
    ]
    
    instance_path = None
    for inst in test_instances:
        path = Path(__file__).parent / inst
        if path.exists():
            instance_path = inst
            break
    
    if not instance_path:
        print("❌ No test instance found!")
        print("Available instances:")
        for inst in test_instances:
            print(f"  - {inst}")
        return 1
    
    print(f"Testing with: {instance_path}")
    print("Running AILS2 with 10 second timeout...")
    print()
    
    try:
        result = solve(
            instance=instance_path,
            solver="ails2",
            solver_options={
                "max_runtime": 10.0,
                "rounded": True,
            }
        )
        
        print("✅ SUCCESS!")
        print(f"   Cost: {result.cost}")
        print(f"   Runtime: {result.runtime:.2f}s")
        print(f"   Feasible: {result.feasible}")
        print(f"   Routes: {len(result.metadata.get('routes_vrplib', []))}")
        
        return 0
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
