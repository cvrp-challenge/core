#!/usr/bin/env python3
"""Simple test for AILS2 in DRSCI probabilistic"""
import sys
from pathlib import Path

CURRENT = Path(__file__).parent
SRC_ROOT = CURRENT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

print("=" * 80)
print("Testing AILS2 integration in run_drsci_probabilistic.py")
print("=" * 80)

try:
    from master.run_drsci_probabilistic import run_drsci_probabilistic
    print("✓ Import successful")
    
    print("\nRunning DRSCI probabilistic with AILS2 only...")
    print("Instance: instances/test-instances/x/X-n101-k25.vrp")
    print("Time limit: 20 seconds")
    print("Max iterations without improvement: 2")
    print()
    
    result = run_drsci_probabilistic(
        instance_name="instances/test-instances/x/X-n101-k25.vrp",
        seed=42,
        routing_solvers=["ails2"],  # Only AILS2
        routing_solver_options={
            "max_runtime": 3.0,  # 3 seconds per cluster
        },
        time_limit_total=20.0,  # Total 20 seconds
        max_no_improvement_iters=2,
        scp_every=100,  # Skip SCP
        periodic_sol_dump=False,
        enable_logging=True,
        log_to_console=True,
    )
    
    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Best cost: {result['best_cost']}")
    print(f"Number of routes: {len(result['routes'])}")
    print(f"Iterations completed: {result['iterations']}")
    print(f"Total runtime: {result['runtime']:.2f}s")
    print(f"Route pool size: {result['route_pool_size']}")
    
    if result['best_cost'] != float('inf') and len(result['routes']) > 0:
        print("\n✓ SUCCESS: AILS2 integration is working!")
        print(f"  Found solution with {len(result['routes'])} routes")
        print(f"  Cost: {result['best_cost']}")
    else:
        print("\n⚠ WARNING: No valid solution found")
        print("  This might be normal if time limit was too short")
        
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
