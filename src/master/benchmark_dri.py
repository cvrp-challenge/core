"""
benchmark_dri.py

Benchmark clustering methods using the pipeline:
    clustering → routing → global LS → objective

This corresponds to the "DRI" framework:
    Decompose → Route → Improve

Excluded:
    - Set Covering (SCP)
    - Route-based decomposition (RBD)
    - Duplicate removal
    - DRSCI outer-iterations

Used for evaluating the effect of clustering + LS quality.

Runs in parallel (TUM cluster compatible).

Outputs:
    - cost per method per instance
    - .sol file containing the globally improved routes
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
CURRENT = Path(__file__).parent
PROJECT_ROOT = CURRENT.parent.parent   # core/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------
from master.clustering.run_clustering import run_clustering
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp
from master.improve.ls_controller import improve_with_local_search
from master.utils.loader import load_instance
from master.utils.solution_helpers import _write_solution, find_existing_solution, calculate_gap


# ---------------------------------------------------------
# Clustering methods and dissimilarity types to benchmark
# ---------------------------------------------------------
CLUSTERING_METHODS = [
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
    "fcm",
    "k_medoids_pyclustering",
]

DISSIMILARITY_TYPES = [
    "spatial",
    "combined",
]

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
# Worker initialization
# ---------------------------------------------------------
def _init_worker():
    """Make project root visible to subprocesses."""
    import sys
    from pathlib import Path

    CURRENT = Path(__file__).parent
    PROJECT_ROOT = CURRENT.parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Evaluate ONE method on ONE instance
# ---------------------------------------------------------
def evaluate_method_on_instance(
    instance_name: str,
    method: str,
    output_dir: Path,
    k: Optional[int] = None,
    dissimilarity: str = "spatial",
    ls_max: int = 40,
) -> dict:
    """
    Runs: clustering → routing → LS.
    Writes a .sol file and returns result dict.
    """

    try:
        inst = load_instance(instance_name)

        # Determine K if not specified
        if k is None:
            k = int(instance_name.split("-k")[-1].split(".")[0])

        # Determine dissimilarity type
        if dissimilarity == "combined":
            use_combined = True
            ls_neigh = "dri_combined"
        else:
            use_combined = False
            ls_neigh = "dri_spatial"

        # ------------------ 1) Clustering ------------------
        clusters, _ = run_clustering(method, instance_name, k, use_combined=use_combined)

        # ------------------ 2) Routing ---------------------
        routing = solve_clusters_with_pyvrp(
            instance_name=instance_name,
            clusters=clusters,
            time_limit_per_cluster=60.0,
            seed=0,
        )

        routes_after_routing = routing["routes"]
        routing_cost = routing["total_cost"]

        # ------------------ 3) Global LS -------------------
        ls_start = time.time()
        ls_result = improve_with_local_search(
            instance_name=instance_name,
            routes_vrplib=routes_after_routing,
            neighbourhood=ls_neigh,
            max_neighbours=ls_max,
            seed=0,
        )
        ls_runtime = time.time() - ls_start

        improved_routes = ls_result["routes_improved"]
        improved_cost = ls_result["improved_cost"]

        # ------------------ 4) Gap vs reference ------------
        ref = find_existing_solution(instance_name)
        reference_cost = ref[1] if ref else None
        gap = calculate_gap(improved_cost, reference_cost) if reference_cost else None

        # ------------------ 5) Write .sol output -----------
        # Use instance name with method suffix for unique filenames
        sol_instance_name = f"{Path(instance_name).stem}_{method}_dri"
        stopping = "60s per cluster"

        _write_solution(
            where=output_dir,
            instance_name=sol_instance_name,
            data=inst,
            result=improved_routes,
            solver="PyVRP (HGS)",
            runtime=ls_runtime + routing["total_runtime"],
            stopping_criteria=stopping,
            gap_percent=gap,
            clustering_method=method,
            dissimilarity=dissimilarity,
        )

        return {
            "instance": instance_name,
            "method": method,
            "cost": improved_cost,
            "runtime": ls_runtime + routing["total_runtime"],
            "routes": improved_routes,
            "gap_percent": gap,
            "reference_cost": reference_cost,
            "success": True,
        }

    except Exception as e:
        return {
            "instance": instance_name,
            "method": method,
            "error": str(e),
            "success": False,
        }


# ---------------------------------------------------------
# Benchmark across all instances & methods
# ---------------------------------------------------------
def run_benchmark(
    output_path: str,
    max_workers: Optional[int],
    methods: List[str],
    k_override: Optional[int],
):
    """
    Parallel benchmark of all clustering methods using the full DRI pipeline.
    """

    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DRI benchmark on {len(INSTANCES)} instances.")
    print(f"Methods: {methods}")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {max_workers or 'auto'}")
    print("-" * 80)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:

        futures = {
            executor.submit(
                evaluate_method_on_instance,
                inst,
                method,
                output_dir,
                k_override,
                dissimilarity,
            ): (inst, method, dissimilarity)
            for inst in INSTANCES
            for method in methods
            for dissimilarity in DISSIMILARITY_TYPES
        }

        for future in as_completed(futures):
            inst, method, dissimilarity = futures[future]

            try:
                r = future.result()
                results.append(r)

                if r["success"]:
                    gap_txt = (
                        f" | Gap: {r['gap_percent']:+.2f}%"
                        if r["gap_percent"] is not None else ""
                    )
                    print(
                        f"✓ {inst} [{method}]  "
                        f"Cost={r['cost']:.2f}, Runtime={r['runtime']:.2f}s{gap_txt}"
                    )
                else:
                    print(f"✗ {inst} [{method}] ERROR - {r['error']}")

            except Exception as e:
                print(f"✗ {inst} [{method}] EXCEPTION - {e}")
                results.append({
                    "instance": inst,
                    "method": method,
                    "error": str(e),
                    "success": False,
                })

    # ---------------- Summary Table ----------------
    print("\n" + "-" * 80)
    print("Summary Table (lower is better):\n")

    table: Dict[str, Dict[str, float]] = {}

    for r in results:
        inst = r["instance"]
        method = r["method"]
        if inst not in table:
            table[inst] = {}
        table[inst][method] = r["cost"] if r["success"] else float("inf")

    header = "Instance".ljust(22) + " ".join(m.ljust(20) for m in methods)
    print(header)
    print("-" * len(header))

    for inst in INSTANCES:
        row = inst.ljust(22)
        for m in methods:
            val = table.get(inst, {}).get(m)
            row += (
                f"{val:.1f}".ljust(20)
                if val is not None
                else "n/a".ljust(20)
            )
        print(row)

    print("\nBenchmark complete.\n")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DRI (Decompose → Route → Improve via LS)."
    )

    parser.add_argument("output_path", type=str,
                        help="Directory to store .sol files")

    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel workers")

    parser.add_argument("--methods", type=str, nargs="+",
                        default=CLUSTERING_METHODS,
                        help="Clustering methods to compare")

    parser.add_argument("--k", type=int, default=None,
                        help="Override number of clusters (default: from instance name)")

    args = parser.parse_args()

    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        methods=args.methods,
        k_override=args.k,
    )


if __name__ == "__main__":
    main()
