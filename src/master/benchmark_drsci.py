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
# Instances to benchmark
# ---------------------------------------------------------
INSTANCES = [
    "X-n502-k39.vrp",
    # "X-n524-k153.vrp",
    # "X-n561-k42.vrp",
    # "X-n641-k35.vrp",
    # "X-n685-k75.vrp",
    # "X-n716-k35.vrp",
    # "X-n749-k98.vrp",
    # "X-n801-k40.vrp",
    # "X-n856-k95.vrp",
    # "X-n916-k207.vrp",
    # "XLTEST-n1048-k138.vrp",
    # "XLTEST-n1794-k408.vrp",
    # "XLTEST-n2541-k62.vrp",
    # "XLTEST-n3147-k210.vrp",
    # "XLTEST-n4153-k259.vrp",
    # "XLTEST-n6034-k1685.vrp",
    # "XLTEST-n6734-k1347.vrp",
    # "XLTEST-n8028-k691.vrp",
    # "XLTEST-n8766-k55.vrp",
    # "XLTEST-n10001-k798.vrp",
]

CLUSTERING_METHODS = [
    "sk_ac_avg",
    # "sk_ac_complete",
    # "sk_ac_min",
    # "sk_kmeans",
    # "fcm",
    "k_medoids_pyclustering",
]

K_PER_METHOD = {
    "sk_ac_avg": [2, 4],
    "sk_ac_complete": [2, 4],
    "sk_ac_min": [2, 4],
    "sk_kmeans": [2, 4],
    "fcm": [2, 4],
    "k_medoids_pyclustering": [2, 4],
}


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
def solve_instance_drsci(
    instance_name: str,
    output_dir: Path,
    seed: int,
    time_limit_per_cluster: float,
    ls_neighbourhood: str,
    ls_after_routing_max_neighbours: int,
    ls_max_neighbours_restricted: int,
    scp_time_limit: float,
    use_combined_dissimilarity: bool,
    scp_solver: str,
    routing_solver: str,
) -> dict:
    try:
        from master.run_drsci import run_drsci_for_instance

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{instance_name} is starting.", flush=True)

        result = run_drsci_for_instance(
            instance_name=instance_name,
            seed=seed,
            time_limit_per_cluster=time_limit_per_cluster,
            ls_neighbourhood=ls_neighbourhood,
            ls_after_routing_max_neighbours=ls_after_routing_max_neighbours,
            ls_max_neighbours_restricted=ls_max_neighbours_restricted,
            scp_time_limit=scp_time_limit,
            use_combined_dissimilarity=use_combined_dissimilarity,
            methods=CLUSTERING_METHODS,
            k_per_method=K_PER_METHOD,
            scp_solver=scp_solver,
            routing_solver=routing_solver,
        )

        best_cost = result["best_cost"]
        routes = result["routes"]
        runtime = result["runtime"]
        stages = result.get("stages", 0)

        ref = find_existing_solution(instance_name)
        reference_cost = ref[1] if ref else None
        gap = calculate_gap(best_cost, reference_cost) if reference_cost else None

        inst = load_instance(instance_name)
        sol_name = f"{Path(instance_name).stem}_drsci"

        _write_solution(
            where=output_dir,
            instance_name=sol_name,
            data=inst,
            result=routes,
            solver=f"DRSCI({routing_solver}+{scp_solver})",
            runtime=runtime,
            stopping_criteria=f"DRSCI({stages} Stages)",
            gap_percent=gap,
            cost=best_cost,
        )

        return {
            "instance": instance_name,
            "success": True,
            "cost": best_cost,
            "runtime": runtime,
            "stages": stages,
            "gap_percent": gap,
            "reference_cost": reference_cost,
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
    time_limit_per_cluster: float,
    ls_neighbourhood: str,
    ls_after_routing_max_neighbours: int,
    ls_max_neighbours_restricted: int,
    scp_time_limit: float,
    use_combined_dissimilarity: bool,
    scp_solver: str,
    routing_solver: str,
):
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DRSCI benchmark on {len(INSTANCES)} instances.")
    print(f"Routing solver          : {routing_solver}")
    print(f"SCP solver              : {scp_solver}")
    print(f"Seed                    : {seed}")
    print(f"Time limit per cluster  : {time_limit_per_cluster}s")
    print(f"SCP time limit          : {scp_time_limit}s")
    print(f"Parallel workers        : {max_workers or 'auto'}")
    print("-" * 80)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {
            executor.submit(
                solve_instance_drsci,
                inst,
                output_dir,
                seed,
                time_limit_per_cluster,
                ls_neighbourhood,
                ls_after_routing_max_neighbours,
                ls_max_neighbours_restricted,
                scp_time_limit,
                use_combined_dissimilarity,
                scp_solver,
                routing_solver,
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
                print(
                    f"✓ {r['instance']:<25} "
                    f"Cost={r['cost']:.2f} "
                    f"Time={r['runtime']:.2f}s "
                    f"Stages={r['stages']}{gap_txt}"
                )
            else:
                print(f"✗ {r['instance']} ERROR: {r['error']}")

    print("-" * 80)
    print("Benchmark complete.\n")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the FULL DRSCI algorithm on multiple VRP instances."
    )

    parser.add_argument("output_path", type=str)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_limit_per_cluster", type=float, default=15.0)
    parser.add_argument("--ls_neighbourhood", type=str, default="dri_spatial")
    parser.add_argument("--ls_after_routing_max_neighbours", type=int, default=40)
    parser.add_argument("--ls_max_neighbours_restricted", type=int, default=40)
    parser.add_argument("--scp_time_limit", type=float, default=600.0)
    parser.add_argument("--use_combined_dissimilarity", action="store_true")

    parser.add_argument(
        "--scp_solver",
        choices=["gurobi", "hexaly"],
        default="gurobi",
    )

    parser.add_argument(
        "--routing_solver",
        choices=["pyvrp", "hexaly", "filo1", "filo2"],
        default="pyvrp",
    )


    args = parser.parse_args()

    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        seed=args.seed,
        time_limit_per_cluster=args.time_limit_per_cluster,
        ls_neighbourhood=args.ls_neighbourhood,
        ls_after_routing_max_neighbours=args.ls_after_routing_max_neighbours,
        ls_max_neighbours_restricted=args.ls_max_neighbours_restricted,
        scp_time_limit=args.scp_time_limit,
        use_combined_dissimilarity=args.use_combined_dissimilarity,
        scp_solver=args.scp_solver,
        routing_solver=args.routing_solver,
    )

if __name__ == "__main__":
    main()
