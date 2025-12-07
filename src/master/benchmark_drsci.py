"""
benchmark_drsci.py

Benchmark script for running the FULL DRSCI solver on many VRP instances.

Pipeline of DRSCI:
    - Vertex-based clustering (iter 0)
    - Route-based decomposition (iter >= 1)
    - Routing of subproblems
    - Global LS
    - Set Covering (SCP)
    - Duplicate removal + LS repair
    - Final LS refine
    - Iterates until convergence or max_iters

This benchmark:
    ✓ Runs instances in parallel (TUM cluster friendly)
    ✓ Saves final DRSCI routes to .sol
    ✓ Computes gap vs reference solutions
    ✓ Prints clean summary table like the PyVRP benchmark
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
CURRENT = Path(__file__).parent         # master/
PROJECT_ROOT = CURRENT.parent.parent    # core/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from master.utils.solution_helpers import (
    find_existing_solution,
    calculate_gap,
    _write_solution,
)
from master.utils.loader import load_instance

# ---------------------------------------------------------
# Imports inside worker will load run_drsci correctly
# ---------------------------------------------------------


# ---------------------------------------------------------
# Instances to benchmark
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
# Worker initialization for multiprocessing
# ---------------------------------------------------------
def _init_worker():
    """Ensure worker processes know project root."""
    import sys
    from pathlib import Path

    CURRENT = Path(__file__).parent
    PROJECT_ROOT = CURRENT.parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Solve a single instance with DRSCI
# ---------------------------------------------------------
def solve_instance_drsci(
    instance_name: str,
    output_dir: Path,
    max_iters: int,
    seed: int = 0,
) -> dict:
    """
    Executes full DRSCI on one VRP instance.
    Returns dict with cost, runtime, iteration count, and feasibility.
    """
    try:
        # Import inside worker so parallel jobs do not fail
        from master.run_drsci import drsci_solve

        output_dir.mkdir(parents=True, exist_ok=True)

        # ----------------- RUN DRSCI -----------------
        result = drsci_solve(
            instance=instance_name,
            max_iters=max_iters,
            seed=seed,
        )

        best_cost = result["best_cost"]
        routes = result["routes"]
        runtime = result["runtime"]
        iters = result["iterations"]

        # ----------------- GAP vs REF -----------------
        ref = find_existing_solution(instance_name)
        reference_cost = ref[1] if ref else None
        gap = calculate_gap(best_cost, reference_cost) if reference_cost else None

        # ----------------- WRITE .SOL -----------------
        inst = load_instance(instance_name)
        sol_path = output_dir / f"{Path(instance_name).stem}_drsci.sol"

        _write_solution(
            where=output_dir,
            instance_name=instance_name,
            data=inst,
            result=routes,
            solver="DRSCI",
            runtime=runtime,
            stopping_criteria=f"DRSCI(iters={iters})",
            gap_percent=gap,
            filename_override=sol_path.name,
        )

        return {
            "instance": instance_name,
            "success": True,
            "cost": best_cost,
            "runtime": runtime,
            "iterations": iters,
            "gap_percent": gap,
            "reference_cost": reference_cost,
            "sol_path": str(sol_path),
        }

    except Exception as e:
        return {
            "instance": instance_name,
            "success": False,
            "error": str(e),
        }


# ---------------------------------------------------------
# Run full benchmark for all instances
# ---------------------------------------------------------
def run_benchmark(
    output_path: str,
    max_workers: Optional[int],
    max_iters: int,
):
    """
    Run DRSCI on all benchmark instances in parallel.
    """
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DRSCI benchmark on {len(INSTANCES)} instances.")
    print(f"Max DRSCI iterations  : {max_iters}")
    print(f"Parallel workers       : {max_workers or 'auto'}")
    print(f"Output directory       : {output_dir}")
    print("-" * 80)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:

        futures = {
            executor.submit(
                solve_instance_drsci,
                inst,
                output_dir,
                max_iters,
            ): inst
            for inst in INSTANCES
        }

        for future in as_completed(futures):
            instance = futures[future]
            try:
                r = future.result()
                results.append(r)

                if r["success"]:
                    gap_txt = (
                        f" | Gap: {r['gap_percent']:+.2f}% (ref={r['reference_cost']:.1f})"
                        if r["gap_percent"] is not None
                        else ""
                    )
                    print(
                        f"✓ {instance:<20} "
                        f"Cost={r['cost']:.2f} "
                        f"Time={r['runtime']:.2f}s "
                        f"Iters={r['iterations']}{gap_txt}"
                    )
                else:
                    print(f"✗ {instance} ERROR: {r['error']}")

            except Exception as e:
                print(f"✗ {instance} EXCEPTION: {e}")
                results.append({
                    "instance": instance,
                    "success": False,
                    "error": str(e),
                })

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    print("-" * 80)
    print("\nSummary:\n")

    ok = [r for r in results if r["success"]]
    bad = [r for r in results if not r["success"]]

    print(f"Successful: {len(ok)}/{len(results)}")
    print(f"Failed    : {len(bad)}/{len(results)}\n")

    if ok:
        avg_cost = sum(r["cost"] for r in ok) / len(ok)
        avg_runtime = sum(r["runtime"] for r in ok) / len(ok)
        print(f"Average final cost    : {avg_cost:.2f}")
        print(f"Average runtime       : {avg_runtime:.2f}s")

        gaps = [r["gap_percent"] for r in ok if r["gap_percent"] is not None]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"Average gap vs ref    : {avg_gap:+.2f}%")

    if bad:
        print("\nFailed instances:")
        for r in bad:
            print(f"  - {r['instance']}: {r['error']}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the FULL DRSCI algorithm on multiple VRP instances."
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
        help="Maximum DRSCI outer iterations per instance",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Parallel worker processes (default auto)",
    )

    args = parser.parse_args()

    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        max_iters=args.max_iters,
    )


if __name__ == "__main__":
    main()
