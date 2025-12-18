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
import math
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
from master.routing.routing_controller import solve_clusters_with_pyvrp
from master.improve.ls_controller import improve_with_local_search
from master.utils.loader import load_instance
from master.utils.solution_helpers import (
    _write_solution,
    find_existing_solution,
    calculate_gap,
)


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

        # Determine number of clusters (fix value for fallback)
        if k is None:
            k = 3

        # Determine dissimilarity type
        if dissimilarity == "combined":
            use_combined = True
            ls_neigh = "dri_combined"
        else:
            use_combined = False
            ls_neigh = "dri_spatial"
        
        # 1) Clustering
        clusters, _ = run_clustering(method, instance_name, k, use_combined=use_combined)

        import sys
        sys.stdout.flush()
        print(f"{instance_name} [{method}] k={k} {dissimilarity}...", flush=True)

        # 2) Routing subproblems
        routing = solve_clusters_with_pyvrp(
            instance_name=instance_name,
            clusters=clusters,
            time_limit_per_cluster=20.0,
            seed=0,
        )

        # routing is now a PyVRP Result object
        routing_cost = routing.cost()
        routing_runtime = routing.runtime

        # Convert PyVRP Result to VRPLIB routes for LS
        # Use the same conversion as _solution_to_vrplib_routes in ls_controller.py
        # route.visits() returns location indices (0=depot, 1+=clients)
        # Filter out depot (0) before converting to VRPLIB format
        routes_after_routing = []
        for route in routing.best.routes():
            location_indices = list(route.visits())
            # Filter out depot (location 0) - only keep client locations
            clients = [idx for idx in location_indices if idx > 0]
            if not clients:
                continue
            # Convert location indices to VRPLIB node IDs: location_index + 1 = VRPLIB_node_ID
            # Location 1 -> VRPLIB node 2, location 2 -> VRPLIB node 3, etc.
            seq = [1] + [(idx + 1) for idx in clients] + [1]
            routes_after_routing.append(seq)

        # 3) Global LS
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
        
        # Convert improved routes from VRPLIB format (customers 2 to n) to format (customers 1 to n-1)
        # VRPLIB: depot=1, customers=2,3,4,...,n
        # Target: depot=0, customers=1,2,3,...,n-1
        # Conversion: subtract 1 from customer nodes (nodes > 1), keep depot (0) as is
        improved_routes_corrected = []
        for route in improved_routes:
            corrected_route = [0 if node == 1 else node - 1 for node in route]
            improved_routes_corrected.append(corrected_route)
        improved_routes = improved_routes_corrected
        
        # Recalculate cost for the converted routes
        # PyVRP uses integer distances (rounded), so we need to match that for consistency
        # The cost from LS was calculated using integer distances, so we recalculate with rounding
        coords = inst["node_coord"]  # shape (n, 2), index 0 -> node 1 in VRPLIB
        edge_mat = inst.get("edge_weight")  # May be None, shape (n, n), 0-based indices
        
        def dist(u: int, v: int) -> int:
            # Convert from new format (depot=0, customers=1 to n-1) to VRPLIB array indices
            # New format: depot=0 -> VRPLIB node 1 -> array index 0
            # New format: customer i -> VRPLIB node (i+1) -> array index i
            u_idx = 0 if u == 0 else u  # depot 0 -> index 0, customer i -> index i
            v_idx = 0 if v == 0 else v
            
            if edge_mat is not None:
                # Use precomputed distance matrix if available
                # PyVRP rounds distances to integers: int(round(float(dist)))
                return int(round(float(edge_mat[u_idx, v_idx])))
            else:
                # Fallback to Euclidean distance, rounded to integer (matching PyVRP)
                dx = float(coords[u_idx, 0] - coords[v_idx, 0])
                dy = float(coords[u_idx, 1] - coords[v_idx, 1])
                return int(round(math.hypot(dx, dy)))
        
        improved_cost = 0
        for route in improved_routes:
            if len(route) < 2:
                continue
            for u, v in zip(route, route[1:]):
                improved_cost += dist(u, v)

        # 4) Gap vs reference
        ref = find_existing_solution(instance_name)
        reference_cost = ref[1] if ref else None
        gap = calculate_gap(improved_cost, reference_cost) if reference_cost else None

        # 5) Write .sol output
        # Use instance name with method suffix for unique filenames
        sol_instance_name = f"{Path(instance_name).stem}_{method}_{dissimilarity}_k={k}_dri"
        stopping = "20s per cluster + LS"

        try:
            _write_solution(
                where=output_dir,
                instance_name=sol_instance_name,
                data=getattr(routing, 'data', inst),  # Use ProblemData if available, else instance dict
                result=improved_routes,
                solver="PyVRP (HGS)",
                runtime=ls_runtime + routing_runtime,
                stopping_criteria=stopping,
                gap_percent=gap,
                clustering_method=method,
                dissimilarity=dissimilarity,
                k_clusters=k,
                cost=improved_cost,  # Required when result is a list of routes
            )
            instance_stem = Path(instance_name).stem
            print(f"✓ {instance_stem} completed using {method}, diss={dissimilarity}, and k={k}", flush=True)
        except Exception as write_error:
            sys.stdout.flush()
            print(f"[ERROR] {instance_name} [{method}] k={k}: Failed to write solution: {write_error}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        return {
            "instance": instance_name,
            "method": method,
            "cost": improved_cost,
            "dissimilarity": dissimilarity,
            "k_clusters": k,
            "runtime": ls_runtime + routing_runtime,
            "routes": improved_routes,
            "gap_percent": gap,
            "reference_cost": reference_cost,
            "success": True,
        }

    except Exception as e:
        import sys
        import traceback
        error_traceback = traceback.format_exc()
        sys.stdout.flush()
        print(f"[ERROR] {instance_name} [{method}] k={k}: Exception in evaluate_method_on_instance: {e}", flush=True)
        print(error_traceback, flush=True)
        sys.stdout.flush()
        return {
            "instance": instance_name,
            "method": method,
            "dissimilarity": dissimilarity,
            "k_clusters": k,
            "error": str(e),
            "traceback": error_traceback,
            "success": False,
        }


# ---------------------------------------------------------
# Benchmark across all instances & methods
# ---------------------------------------------------------
def run_benchmark(
    output_path: str,
    max_workers: Optional[int],
    methods: List[str],
    dissimilarity_types: List[str],
    k_values: Optional[List[int]],
):
    """
    Parallel benchmark of all clustering methods using the full DRI pipeline.
    """
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default k values if not provided
    if k_values is None:
        k_values = K_CLUSTERS

    print(f"Running DRI benchmark on {len(INSTANCES)} instances")
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
                    print(f"✗ {inst} [{method}]: ERROR - {r.get('error', 'Unknown error')}")
                    import traceback
                    if 'traceback' in r:
                        print(r['traceback'])

            except Exception as e:
                import traceback
                print(f"✗ {inst} [{method}]: EXCEPTION in future.result() - {e}")
                traceback.print_exc()
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
        description="Benchmark DRI (Decompose → Route → Improve via LS) across different clustering methods."
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

    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="Cluster sizes to test (default: K_CLUSTERS = [2, 4, 6, 9, 12])")

    args = parser.parse_args()

    run_benchmark(
        output_path=args.output_path,
        max_workers=args.max_workers,
        methods=args.methods,
        dissimilarity_types=args.dissimilarity,
        k_values=args.k,  # Will be None if not provided, then defaults to K_CLUSTERS in run_benchmark
    )


if __name__ == "__main__":
    main()
