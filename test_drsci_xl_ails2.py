#!/usr/bin/env python3
"""
Quick test script to verify AILS2 integration with an XL instance
"""
import sys
from pathlib import Path

# Add src to path
CURRENT = Path(__file__).parent
SRC_ROOT = CURRENT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from master.run_drsci_probabilistic import run_drsci_probabilistic

def test_xl_instance_with_ails2():
    """Test AILS2 integration with an XL instance and short time limit"""
    
    # Use a smaller XL instance for testing
    instance_name = "XL-n1094-k157.vrp"
    
    print("=" * 80)
    print("Testing AILS2 integration with XL instance")
    print("=" * 80)
    print(f"Instance: {instance_name}")
    print(f"Solvers: AILS2, FILO1, FILO2, PyVRP")
    print(f"Time limit: 60 seconds")
    print("=" * 80)
    print()
    
    try:
        result = run_drsci_probabilistic(
            instance_name=instance_name,
            seed=42,
            routing_solvers=["ails2", "filo1", "filo2", "pyvrp"],  # Include AILS2 with others
            routing_solver_options={
                "max_runtime": 20.0,  # 10 seconds per cluster solve (adaptive will override)
            },
            time_limit_total=600.0,  # Total 60 seconds
            max_no_improvement_iters=3,  # Stop after 3 iterations without improvement
            scp_every=3,  # Skip SCP for this quick test
            periodic_sol_dump=False,  # Don't dump solutions
            enable_logging=True,
            log_to_console=True,
        )
        
        print()
        print("=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print(f"Best cost: {result['best_cost']}")
        print(f"Number of routes: {len(result['routes'])}")
        print(f"Iterations: {result['iterations']}")
        print(f"Runtime: {result['runtime']:.2f}s")
        print(f"Route pool size: {result['route_pool_size']}")
        print()
        
        if result['best_cost'] != float('inf') and len(result['routes']) > 0:
            print("✓ SUCCESS: Test completed successfully!")
            print(f"  - Found solution with {len(result['routes'])} routes")
            print(f"  - Cost: {result['best_cost']}")
        else:
            print("⚠ WARNING: No valid solution found")
            print("  - This might be normal if the time limit was too short")
            
        return result
        
    except Exception as e:
        print()
        print("=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_xl_instance_with_ails2()
