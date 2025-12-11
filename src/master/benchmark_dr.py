"""
benchmark_dr.py

Benchmark clustering methods using the DR pipeline:
    Decompose (clustering) → Route (PyVRP) → Evaluate cost

Purposely excludes:
    - Local Search (LS)
    - Set Covering (SCP)
    - Route-Based Decomposition (RBD)
    - Duplicate removal
    - DRSCI iterations

This benchmark compares clustering methods based purely on
their decomposition quality after routing.

Runs in parallel using ProcessPoolExecutor (TUM server compatible).

Outputs:
    - cost per method per instance
    - .sol file containing resulting routes
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Dict

# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
CURRENT = Path(__file__).parent
PROJECT_ROOT = CURRENT.parent.parent  # core/

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
from master.clustering.run_clustering import run_clustering
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp
from master.utils.loader import load_instance
from master.utils.solution_helpers import (
    _write_solution,
    find_existing_solution,
    calculate_gap,
)

# ---------------------------------------------------------
# Clustering methods, dissimilarity types and k-clusters to benchmark
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

K_CLUSTERS = [
    2,
    4,
    6,
    9,
    12,
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
# Worker Initialization
# ---------------------------------------------------------
def _init_worker():
    """Ensure subprocesses know the project root."""
    import sys
    from pathlib import Path

    CURRENT = Path(__file__).parent
    PROJECT_ROOT = CURRENT.parent.parent

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Single-instance & single-method evaluation
# ---------------------------------------------------------
def evaluate_method_on_instance(
    instance_name: str,
    method: str,
    output_dir: Path,
    k: Optional[int] = None,
    dissimilarity: str = "spatial",
) -> dict:
    """
    Run clustering → routing for a single method on a single instance.
    Returns cost, runtime, routes, feasibility, and gap vs reference.
    """
    try:
        inst = load_instance(instance_name)

        # Determine number of clusters (fix value for fallback)
        if k is None:
            k = 3

        # Determine dissimilarity type
        if dissimilarity == "combined":
            use_combined = True
        else:
            use_combined = False
        
        # 1) Clustering
        clusters, _ = run_clustering(method, instance_name, k, use_combined=use_combined)

        # 2) Routing subproblems
        import time
        routing_start = time.time()
        routing = solve_clusters_with_pyvrp(
            instance_name=instance_name,
            clusters=clusters,
            time_limit_per_cluster=10.0,
            seed=0,
        )
        routing_runtime = time.time() - routing_start

        cost = routing["total_cost"]
        routes = routing["routes"]

        # 3) Gap vs reference
        ref = find_existing_solution(instance_name)
        reference_cost = ref[1] if ref else None
        gap = calculate_gap(cost, reference_cost) if reference_cost else None

        _write_solution(
            where=output_dir,
            instance_name=f"{Path(instance_name).stem}_{method}_{dissimilarity}_k={k}",
            data=inst,
            result=routes,
            solver=f"PyVRP (HGS)",
            runtime=routing_runtime,
            stopping_criteria="10s per cluster",
            gap_percent=gap,
            clustering_method=method,
            dissimilarity=dissimilarity,
            k_clusters=k,
        )

        return {
            "instance": instance_name,
            "method": method,
            "cost": cost,
            "dissimilarity": dissimilarity,
            "k_clusters": k,
            "routes": routes,
            "runtime": routing_runtime,
            "feasible": True,
            "gap_percent": gap,
            "reference_cost": reference_cost,
            "success": True,
        }

    except Exception as e:
        return {
            "instance": instance_name,
            "method": method,
            "dissimilarity": dissimilarity,
            "k_clusters": k,
            "error": str(e),
            "success": False,
        }


# ---------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------
def run_benchmark(
    output_path: str,
    max_workers: Optional[int],
    methods: List[str],
    dissimilarity_types: List[str],
    k_values: Optional[List[int]],
):
    """
    Run all clustering methods on all instances in parallel.
    """
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default k values if not provided
    if k_values is None:
        k_values = K_CLUSTERS

    print(f"Running DR benchmark on {len(INSTANCES)} instances")
    print(f"Methods: {methods}")
    print(f"K values: {k_values}")
    print(f"Dissimilarity types: {dissimilarity_types}")
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
                k,
                dissimilarity,
            ): (inst, method, dissimilarity, k)
            for inst in INSTANCES
            for method in methods
            for dissimilarity in dissimilarity_types
            for k in k_values
        }

        for future in as_completed(futures):
            inst, method, dissimilarity, k = futures[future]

            try:
                r = future.result()
                results.append(r)

                if r["success"]:
                    gap_str = (
                        f" | Gap: {r['gap_percent']:+.2f}%"
                        if r["gap_percent"] is not None
                        else ""
                    )
                    print(
                        f"✓ {inst} [{method}]  "
                        f"Cost={r['cost']:.2f},  "
                        f"Time={r['runtime']:.2f}s"
                        f"{gap_str}"
                    )
                else:
                    print(f"✗ {inst} [{method}]: ERROR - {r['error']}")

            except Exception as e:
                print(f"✗ {inst} [{method}]: EXCEPTION - {e}")
                results.append({
                    "instance": inst,
                    "method": method,
                    "dissimilarity": dissimilarity,
                    "k_clusters": k,
                    "error": str(e),
                    "success": False,
                })

    # ---------------- Summary ----------------
    print("-" * 80)
    print("\nSummary:\n")

    table: Dict[str, Dict[str, float]] = {}

    for r in results:
        inst = r["instance"]
        method = r["method"]
        dissimilarity = r["dissimilarity"]
        k = r["k_clusters"]
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
        description="Benchmark DR (Decompose → Route) across different clustering methods."
    )

    parser.add_argument("output_path", type=str,
                        help="Directory to store .sol files")

    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel workers (default auto)")

    parser.add_argument("--methods", type=str, nargs="+",
                        default=CLUSTERING_METHODS,
                        help="Clustering methods to compare")

    parser.add_argument("--dissimilarity", type=str, nargs="+",
                        default=DISSIMILARITY_TYPES,
                        help="Dissimilarity types to compare")

    parser.add_argument("--k", type=int, default=K_CLUSTERS,
                        help="Override number of clusters (default: derive from instance name)")

    args = parser.parse_args()

    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        methods=args.methods,
        dissimilarity_types=args.dissimilarity,
        k_values=args.k,
    )


if __name__ == "__main__":
    main()
