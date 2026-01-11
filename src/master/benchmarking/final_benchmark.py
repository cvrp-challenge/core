"""
final_benchmark.py

Benchmark script for running the probabilistic DRSCI solver on all 20 test instances.

Pipeline of Probabilistic DRSCI:
    - Probabilistic selection of clustering methods (VB/RB)
    - Probabilistic routing solver selection
    - Routing of subproblems
    - Local Search
    - Periodic Set Covering (SCP)
    - Iterates until convergence or max iterations
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
CURRENT = Path(__file__).parent         # master/benchmarking/
PROJECT_ROOT = CURRENT.parent.parent    # core/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from master.utils.solution_helpers import (
    calculate_gap,
    _write_solution,
)
from master.utils.loader import load_instance

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


def _load_bks_from_file(instance_name: str) -> Optional[int]:
    """
    Load BKS (Best Known Solution) cost for an instance from test-bks.json file.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n916-k207.vrp")
        
    Returns:
        BKS cost as int if found, None otherwise
    """
    bks_file = PROJECT_ROOT / "instances" / "test-instances" / "test-bks.json"
    
    if not bks_file.exists():
        return None
    
    try:
        with open(bks_file, "r") as f:
            bks_data = json.load(f)
        
        instance_stem = Path(instance_name).stem
        bks_cost = bks_data.get(instance_stem)
        
        if bks_cost is None:
            return None
        
        return int(bks_cost)
    except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
        return None


# ---------------------------------------------------------
# Worker initialization
# ---------------------------------------------------------
def _init_worker():
    import sys
    from pathlib import Path

    CURRENT = Path(__file__).parent
    PROJECT_ROOT = CURRENT.parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Solve a single instance
# ---------------------------------------------------------
def solve_instance_probabilistic(
    instance_name: str,
    output_dir: Path,
    seed: int,
    scp_solvers: list,
    scp_switch_prob: float,
    time_limit_scp: float,
    scp_every: int,
    time_limit_total: float,
    max_no_improvement_iters: int,
    min_avg_cluster_size: int,
    max_avg_cluster_size: int,
    routing_solvers: Optional[list],
    routing_solver_options: Optional[dict],
    routing_no_improvement: Optional[int],
    ls_neighbourhood: str,
    ls_after_routing_max_neighbours: int,
    ls_max_neighbours_restricted: int,
    randomize_polar_angle: bool,
    bks_output_dir: str,
) -> dict:
    try:
        from master.run_drsci_probabilistic import run_drsci_probabilistic

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{instance_name} is starting.", flush=True)

        result = run_drsci_probabilistic(
            instance_name=instance_name,
            seed=seed,
            scp_solvers=scp_solvers,
            scp_switch_prob=scp_switch_prob,
            time_limit_scp=time_limit_scp,
            scp_every=scp_every,
            time_limit_total=time_limit_total,
            max_no_improvement_iters=max_no_improvement_iters,
            min_avg_cluster_size=min_avg_cluster_size,
            max_avg_cluster_size=max_avg_cluster_size,
            routing_solvers=routing_solvers,
            routing_solver_options=routing_solver_options,
            routing_no_improvement=routing_no_improvement,
            ls_neighbourhood=ls_neighbourhood,
            ls_after_routing_max_neighbours=ls_after_routing_max_neighbours,
            ls_max_neighbours_restricted=ls_max_neighbours_restricted,
            randomize_polar_angle=randomize_polar_angle,
            bks_output_dir=bks_output_dir,
        )

        best_cost = result["best_cost"]
        routes = result["routes"]
        runtime = result["runtime"]
        iterations = result.get("iterations", 0)
        route_pool_size = result.get("route_pool_size", 0)

        # Get BKS from test-bks.json for gap calculation
        bks_cost = _load_bks_from_file(instance_name)
        gap = calculate_gap(best_cost, bks_cost) if bks_cost else None

        inst = load_instance(instance_name)
        sol_name = f"{Path(instance_name).stem}_probabilistic"

        _write_solution(
            where=output_dir,
            instance_name=sol_name,
            data=inst,
            result=routes,
            solver="Probabilistic-DRSCI",
            runtime=runtime,
            stopping_criteria=f"Probabilistic-DRSCI({iterations} iterations)",
            gap_percent=gap,
            cost=best_cost,
        )

        return {
            "instance": instance_name,
            "success": True,
            "cost": best_cost,
            "runtime": runtime,
            "iterations": iterations,
            "route_pool_size": route_pool_size,
            "gap_percent": gap,
            "bks_cost": bks_cost,
        }

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] {instance_name}: {e}", flush=True)
        print(tb, flush=True)
        return {
            "instance": instance_name,
            "success": False,
            "error": str(e),
            "traceback": tb,
        }

# ---------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------
def run_benchmark(
    output_path: str,
    max_workers: Optional[int],
    seed: int,
    scp_solvers: list,
    scp_switch_prob: float,
    time_limit_scp: float,
    scp_every: int,
    time_limit_total: float,
    max_no_improvement_iters: int,
    min_avg_cluster_size: int,
    max_avg_cluster_size: int,
    routing_solvers: Optional[list],
    routing_solver_options: Optional[dict],
    routing_no_improvement: Optional[int],
    ls_neighbourhood: str,
    ls_after_routing_max_neighbours: int,
    ls_max_neighbours_restricted: int,
    randomize_polar_angle: bool,
    bks_output_dir: str,
):
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Probabilistic DRSCI benchmark on {len(INSTANCES)} instances.")
    print(f"SCP solvers            : {scp_solvers}")
    print(f"SCP switch probability : {scp_switch_prob}")
    print(f"Time limit (total)     : {time_limit_total}s")
    print(f"Time limit (SCP)       : {time_limit_scp}s")
    print(f"SCP every              : {scp_every} iterations")
    print(f"Max no-improvement     : {max_no_improvement_iters} iterations")
    print(f"Seed                   : {seed}")
    print(f"Parallel workers       : {max_workers or 'auto'}")
    print("-" * 80)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {
            executor.submit(
                solve_instance_probabilistic,
                inst,
                output_dir,
                seed,
                scp_solvers,
                scp_switch_prob,
                time_limit_scp,
                scp_every,
                time_limit_total,
                max_no_improvement_iters,
                min_avg_cluster_size,
                max_avg_cluster_size,
                routing_solvers,
                routing_solver_options,
                routing_no_improvement,
                ls_neighbourhood,
                ls_after_routing_max_neighbours,
                ls_max_neighbours_restricted,
                randomize_polar_angle,
                bks_output_dir,
            ): inst
            for inst in INSTANCES
        }

        for future in as_completed(futures):
            r = future.result()
            results.append(r)

            if r["success"]:
                gap_txt = (
                    f" | Gap: {r['gap_percent']:+.2f}%"
                    if r["gap_percent"] is not None
                    else ""
                )
                bks_txt = (
                    f" | BKS: {r['bks_cost']}"
                    if r["bks_cost"] is not None
                    else ""
                )
                print(
                    f"✓ {r['instance']:<30} "
                    f"Cost={r['cost']:>10} "
                    f"Time={r['runtime']:>8.2f}s "
                    f"Iters={r['iterations']:>4}{gap_txt}{bks_txt}"
                )
            else:
                print(f"✗ {r['instance']} ERROR: {r['error']}")

    print("-" * 80)
    print("Benchmark complete.\n")

    # Print summary statistics
    successful = [r for r in results if r["success"]]
    if successful:
        avg_cost = sum(r["cost"] for r in successful) / len(successful)
        avg_runtime = sum(r["runtime"] for r in successful) / len(successful)
        avg_iterations = sum(r["iterations"] for r in successful) / len(successful)
        gaps = [r["gap_percent"] for r in successful if r["gap_percent"] is not None]
        avg_gap = sum(gaps) / len(gaps) if gaps else None

        print("Summary Statistics:")
        print(f"  Successful runs    : {len(successful)}/{len(results)}")
        if avg_gap is not None:
            print(f"  Average gap        : {avg_gap:+.2f}%")
        print(f"  Average cost       : {avg_cost:.2f}")
        print(f"  Average runtime    : {avg_runtime:.2f}s")
        print(f"  Average iterations : {avg_iterations:.1f}")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Probabilistic DRSCI algorithm on 20 test instances."
    )

    parser.add_argument("output_path", type=str)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scp_solvers",
        nargs="+",
        choices=["gurobi_mip", "gurobi_lp", "hexaly"],
        default=["gurobi_mip"],
    )
    parser.add_argument("--scp_switch_prob", type=float, default=0.0)
    parser.add_argument("--time_limit_scp", type=float, default=300.0)
    parser.add_argument("--scp_every", type=int, default=3)
    parser.add_argument("--time_limit_total", type=float, default=3600.0)
    parser.add_argument("--max_no_improvement_iters", type=int, default=20)
    parser.add_argument("--min_avg_cluster_size", type=int, default=100)
    parser.add_argument("--max_avg_cluster_size", type=int, default=2500)
    parser.add_argument(
        "--routing_solvers",
        nargs="+",
        choices=["pyvrp", "filo1", "filo2"],
        default=None,
    )
    parser.add_argument("--routing_no_improvement", type=int, default=None)
    parser.add_argument("--ls_neighbourhood", type=str, default="dri_spatial")
    parser.add_argument("--ls_after_routing_max_neighbours", type=int, default=250)
    parser.add_argument("--ls_max_neighbours_restricted", type=int, default=250)
    parser.add_argument("--randomize_polar_angle", action="store_true", default=True)
    parser.add_argument("--no_randomize_polar_angle", dest="randomize_polar_angle", action="store_false")
    parser.add_argument("--bks_output_dir", type=str, default="output")

    args = parser.parse_args()

    # Parse routing_solver_options if needed (for future extension)
    routing_solver_options = None

    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        seed=args.seed,
        scp_solvers=args.scp_solvers,
        scp_switch_prob=args.scp_switch_prob,
        time_limit_scp=args.time_limit_scp,
        scp_every=args.scp_every,
        time_limit_total=args.time_limit_total,
        max_no_improvement_iters=args.max_no_improvement_iters,
        min_avg_cluster_size=args.min_avg_cluster_size,
        max_avg_cluster_size=args.max_avg_cluster_size,
        routing_solvers=args.routing_solvers,
        routing_solver_options=routing_solver_options,
        routing_no_improvement=args.routing_no_improvement,
        ls_neighbourhood=args.ls_neighbourhood,
        ls_after_routing_max_neighbours=args.ls_after_routing_max_neighbours,
        ls_max_neighbours_restricted=args.ls_max_neighbours_restricted,
        randomize_polar_angle=args.randomize_polar_angle,
        bks_output_dir=args.output_path,
    )


if __name__ == "__main__":
    main()
