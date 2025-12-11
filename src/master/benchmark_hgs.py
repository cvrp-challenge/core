"""
Benchmark script for running PyVRP solver on multiple instances.

This script:
- Runs 10-20 instances in parallel using PyVRP
- Stores results in .sol files
- Calculates gap to existing solutions if available
- Accepts output path as command-line argument
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# Ensure project root packages
CURRENT = Path(__file__).parent
ROOT = CURRENT  # core/src/master
PROJECT_ROOT = ROOT.parent.parent  # core

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from master.routing.solver import solve
from master.utils.solution_helpers import (
    extract_cost_from_sol,
    find_existing_solution,
    calculate_gap,
    _write_solution,
)

# Hard-coded list of instances to benchmark (10-20 instances)
# Use filenames including the .vrp extension so they can be resolved inside the container.
INSTANCES = [
    "X-n502-k39.vrp",
    "X-n524-k153.vrp",
    "X-n561-k42.vrp",
    "X-n641-k35.vrp",
    "X-n685-k75.vrp",
    "X-n716-k35.vrp",
    "X-n749-k98.vrp",
    "X-n801-k40.vrp",
    "X-n856-k95.vrp",
    "X-n916-k207.vrp",
    "XLTEST-n1048-k138.vrp",
    "XLTEST-n1794-k408.vrp",
    "XLTEST-n2541-k62.vrp",
    "XLTEST-n3147-k210.vrp",
    "XLTEST-n4153-k259.vrp",
    "XLTEST-n6034-k1685.vrp",
    "XLTEST-n6734-k1347.vrp",
    "XLTEST-n8028-k691.vrp",
    "XLTEST-n8766-k55.vrp",
    "XLTEST-n10001-k798.vrp",
]

def _init_worker():
    """Initialize worker process - ensures imports are available in each process."""
    import sys
    from pathlib import Path
    
    CURRENT = Path(__file__).parent
    ROOT = CURRENT
    PROJECT_ROOT = ROOT.parent.parent
    
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


def solve_instance(instance_name: str, output_dir: Path, no_improvement: int = 100000) -> dict:
    """
    Solve a single instance and save the solution.
    
    Args:
        instance_name: Name of the instance to solve
        output_dir: Directory to save the .sol file
        no_improvement: Maximum number of iterations without improvement
        
    Returns:
        Dictionary with results
    """
    # Import here to ensure it's available in worker processes
    from master.routing.solver import solve
    from master.utils.solution_helpers import (
        find_existing_solution,
        calculate_gap,
        _write_solution,
    )
    
    try:
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Solve the instance (don't use solution_dir, we'll write it ourselves)
        result = solve(
            instance=instance_name,
            solver="pyvrp",
            solver_options={
                "no_improvement": no_improvement,
                "seed": 1,
                "solution_dir": None,  # We'll write the solution ourselves
            }
        )
        
        # Check for existing solution and calculate gap
        existing_sol = find_existing_solution(instance_name)
        gap = None
        reference_cost = None
        if existing_sol:
            reference_cost = existing_sol[1]
            gap = calculate_gap(result.cost, reference_cost)
        
        # Write solution file with custom format
        stopping_criteria = f"NoImprovement({no_improvement})"
        _write_solution(
            where=output_dir,
            instance_name=instance_name,
            data=result.data,
            result=result.raw_result,
            solver=result.solver,
            runtime=result.runtime,
            stopping_criteria=stopping_criteria,
            gap_percent=gap,
        )
        
        # Get instance stem for file naming (for return value)
        instance_stem = Path(instance_name).stem
        sol_path = output_dir / f"{instance_stem}.sol"
        
        return {
            "instance": instance_name,
            "cost": result.cost,
            "runtime": result.runtime,
            "feasible": result.feasible,
            "iterations": result.num_iterations,
            "gap_percent": gap,
            "reference_cost": reference_cost,
            "sol_path": str(sol_path),
            "success": True,
        }
    except Exception as e:
        return {
            "instance": instance_name,
            "error": str(e),
            "success": False,
        }


def run_benchmark(output_path: str, max_workers: Optional[int] = None, no_improvement: int = 100000):
    """
    Run benchmark on all instances in parallel.
    
    Args:
        output_path: Path to directory where .sol files will be stored
        max_workers: Maximum number of parallel workers (None = auto)
        no_improvement: Maximum number of iterations without improvement
    """
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running benchmark on {len(INSTANCES)} instances")
    print(f"Output directory: {output_dir}")
    print(f"No-Improvement stopping criterion: {no_improvement} iterations")
    print(f"Parallel workers: {max_workers or 'auto'}")
    print("-" * 80)
    
    results = []
    
    # Use ProcessPoolExecutor for parallel execution
    # Initialize worker processes to ensure proper imports
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(solve_instance, inst, output_dir, no_improvement): inst
            for inst in INSTANCES
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_instance):
            instance = future_to_instance[future]
            try:
                result = future.result()
                results.append(result)
                
                if result["success"]:
                    gap_str = ""
                    if result["gap_percent"] is not None:
                        gap_str = f" | Gap: {result['gap_percent']:+.2f}% (ref: {result['reference_cost']:.2f})"
                    print(
                        f"✓ {result['instance']}: "
                        f"Cost={result['cost']:.2f}, "
                        f"Time={result['runtime']:.2f}s, "
                        f"Feasible={result['feasible']}"
                        f"{gap_str}"
                    )
                else:
                    print(f"✗ {result['instance']}: ERROR - {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"✗ {instance}: EXCEPTION - {e}")
                results.append({
                    "instance": instance,
                    "error": str(e),
                    "success": False,
                })
    
    # Print summary
    print("-" * 80)
    print("\nSummary:")
    successful = [r for r in results if r.get("success", False)]
    failed = [r for r in results if not r.get("success", False)]
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if successful:
        avg_cost = sum(r["cost"] for r in successful) / len(successful)
        avg_runtime = sum(r["runtime"] for r in successful) / len(successful)
        print(f"Average cost: {avg_cost:.2f}")
        print(f"Average runtime: {avg_runtime:.2f}s")
        
        # Gap statistics
        gaps = [r["gap_percent"] for r in successful if r["gap_percent"] is not None]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"Average gap: {avg_gap:+.2f}% (over {len(gaps)} instances with reference solutions)")
    
    if failed:
        print("\nFailed instances:")
        for r in failed:
            print(f"  - {r['instance']}: {r.get('error', 'Unknown error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run PyVRP benchmark on multiple instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py output/solutions
  python benchmark.py output/solutions --no_improvement 5000
  python benchmark.py output/solutions --max_workers 4
        """
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to directory where .sol files will be stored"
    )
    parser.add_argument(
        "--no_improvement",
        type=int,
        default=100000,
        help="Maximum number of iterations without improvement (default: 100000)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: auto)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        no_improvement=args.no_improvement
    )


if __name__ == "__main__":
    main()
