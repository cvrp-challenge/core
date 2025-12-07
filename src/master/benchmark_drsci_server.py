"""
benchmark_drsci.py

Benchmark script for running the DRSCI solver on multiple VRP instances.

This script:
- Runs instances in parallel using DRSCI
- Saves results in .sol files
- Computes gap to reference solutions (if available)
- Mirrors the structure of the PyVRP benchmark script used on TUM servers
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
CURRENT = Path(__file__).parent            # core/src/master
PROJECT_ROOT = CURRENT.parent.parent       # core/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from master.utils.solution_helpers import (
    find_existing_solution,
    calculate_gap,
    _write_solution,
)
from master.utils.loader import load_instance


# ---------------------------------------------------------
# Instances for benchmarking
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Multiprocessing initialization
# ---------------------------------------------------------
def _init_worker():
    """Ensure project root is known inside workers."""
    import sys
    from pathlib import Path

    CURRENT = Path(__file__).parent
    PROJECT_ROOT = CURRENT.parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Solve a single instance with DRSCI
# ---------------------------------------------------------
def solve_instance_drsci(instance_name: str,
                         output_dir: Path,
                         max_iters: int,
                         seed: int = 0) -> dict:
    """
    Solve one instance using DRSCI and write its .sol file.
    """
    try:
        # Local import (important for ProcessPoolExecutor)
        from master.run_drsci import drsci_solve

        output_dir.mkdir(parents=True, exist_ok=True)

        # Run DRSCI
        result = drsci_solve(
            instance=instance_name,
            max_iters=max_iters,
            seed=seed,
        )

        # Evaluate gap vs reference
        existing_sol = find_existing_solution(instance_name)
        reference_cost = existing_sol[1] if existing_sol else None
        gap = calculate_gap(result["best_cost"], reference_cost) if reference_cost else None

        # Write .sol file
        sol_path = output_dir / f"{Path(instance_name).stem}.sol"
        inst = load_instance(instance_name)

        _write_solution(
            where=output_dir,
            instance_name=instance_name,
            data=inst,
            result=result["routes"],
            solver="DRSCI",
            runtime=result["runtime"],
            stopping_criteria=f"DRSCI(Iters={result['iterations']})",
            gap_percent=gap,
        )

        return {
            "instance": instance_name,
            "cost": result["best_cost"],
            "runtime": result["runtime"],
            "iterations": result["iterations"],
            "feasible": True,
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


# ---------------------------------------------------------
# Run benchmark on all instances
# ---------------------------------------------------------
def run_benchmark(output_path: str,
                  max_workers: Optional[int],
                  max_iters: int):
    """
    Execute DRSCI on all benchmark instances in parallel.
    """
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DRSCI benchmark on {len(INSTANCES)} instances")
    print(f"Output directory: {output_dir}")
    print(f"DRSCI iterations per instance: {max_iters}")
    print(f"Parallel workers: {max_workers or 'auto'}")
    print("-" * 80)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {
            executor.submit(solve_instance_drsci, inst, output_dir, max_iters): inst
            for inst in INSTANCES
        }

        for future in as_completed(futures):
            instance = futures[future]
            try:
                r = future.result()
                results.append(r)

                if r["success"]:
                    gap_str = (
                        f" | Gap: {r['gap_percent']:+.2f}% (ref: {r['reference_cost']:.2f})"
                        if r["gap_percent"] is not None
                        else ""
                    )

                    print(
                        f"✓ {r['instance']}: "
                        f"Cost={r['cost']:.2f}, "
                        f"Time={r['runtime']:.2f}s, "
                        f"Iters={r['iterations']}{gap_str}"
                    )
                else:
                    print(f"✗ {instance}: ERROR - {r['error']}")

            except Exception as e:
                print(f"✗ {instance}: EXCEPTION - {e}")
                results.append({"instance": instance, "error": str(e), "success": False})

    # --------------- Summary ---------------
    print("-" * 80)
    print("\nSummary:")

    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")

    if successful:
        avg_cost = sum(r["cost"] for r in successful) / len(successful)
        avg_runtime = sum(r["runtime"] for r in successful) / len(successful)
        print(f"Average cost: {avg_cost:.2f}")
        print(f"Average runtime: {avg_runtime:.2f}s")

        gaps = [r["gap_percent"] for r in successful if r["gap_percent"] is not None]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"Average gap vs reference: {avg_gap:+.2f}%")

    if failed:
        print("\nFailed instances:")
        for r in failed:
            print(f"  - {r['instance']}: {r['error']}")


# ---------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run DRSCI benchmark on multiple VRP instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "output_path",
        type=str,
        help="Directory to store .sol files",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=10,
        help="Maximum DRSCI iterations per instance",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Number of parallel worker processes",
    )

    args = parser.parse_args()

    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
