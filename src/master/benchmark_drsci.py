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

CLUSTERING_METHODS = [
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
    "fcm",
    "k_medoids_pyclustering",
]

K_PER_METHOD = {
    "sk_ac_avg": [2, 4, 6, 8],
    "sk_ac_complete": [2, 4, 6, 8],
    "sk_ac_min": [2, 4, 6, 8],
    "sk_kmeans": [2, 4, 6, 8],
    "fcm": [2, 4, 6, 8],
    "k_medoids_pyclustering": [2, 4, 6, 8],
}


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
    seed: int = 42,
    time_limit_per_cluster: float = 30.0,
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 40,
    ls_max_neighbours_restricted: int = 40,
    scp_time_limit: float = 600.0,
    use_combined_dissimilarity: bool = False,
) -> dict:
    """
    Executes full DRSCI on one VRP instance.
    Returns dict with cost, runtime, and feasibility.
    """
    try:
        # Import inside worker so parallel jobs do not fail
        from master.run_drsci import run_drsci_for_instance

        output_dir.mkdir(parents=True, exist_ok=True)

        # Print start message
        print(f"{instance_name} is starting.", flush=True)

        # ----------------- RUN DRSCI -----------------
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
        )

        best_cost = result["best_cost"]
        routes = result["routes"]  # VRPLIB format: depot=1, customers=2+
        runtime = result["runtime"]
        stages = result.get("stages", 0)

        # ----------------- GAP vs REF -----------------
        ref = find_existing_solution(instance_name)
        reference_cost = ref[1] if ref else None
        gap = calculate_gap(best_cost, reference_cost) if reference_cost else None

        # ----------------- WRITE .SOL -----------------
        inst = load_instance(instance_name)
        instance_stem = Path(instance_name).stem
        sol_instance_name = f"{instance_stem}_drsci"

        _write_solution(
            where=output_dir,
            instance_name=sol_instance_name,
            data=inst,
            result=routes,
            solver="DRSCI",
            runtime=runtime,
            stopping_criteria=f"DRSCI({stages} Stages)",
            gap_percent=gap,
            cost=best_cost,  # Required when result is a list of routes
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
        import sys
        import traceback
        error_traceback = traceback.format_exc()
        sys.stdout.flush()
        print(f"[ERROR] {instance_name}: Exception in solve_instance_drsci: {e}", flush=True)
        print(error_traceback, flush=True)
        sys.stdout.flush()
        return {
            "instance": instance_name,
            "success": False,
            "error": str(e),
            "traceback": error_traceback,
        }


# ---------------------------------------------------------
# Run full benchmark for all instances
# ---------------------------------------------------------
def run_benchmark(
    output_path: str,
    max_workers: Optional[int],
    seed: int = 42,
    time_limit_per_cluster: float = 30.0,
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 40,
    ls_max_neighbours_restricted: int = 40,
    scp_time_limit: float = 600.0,
    use_combined_dissimilarity: bool = False,
):
    """
    Run DRSCI on all benchmark instances in parallel.
    """
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running DRSCI benchmark on {len(INSTANCES)} instances.")
    print(f"Seed                    : {seed}")
    print(f"Time limit per cluster  : {time_limit_per_cluster}s")
    print(f"LS neighbourhood        : {ls_neighbourhood}")
    print(f"LS max neighbours       : {ls_after_routing_max_neighbours}")
    print(f"LS max neighbours (SCP) : {ls_max_neighbours_restricted}")
    print(f"SCP time limit          : {scp_time_limit}s")
    print(f"Combined dissimilarity  : {use_combined_dissimilarity}")
    print(f"Parallel workers        : {max_workers or 'auto'}")
    print(f"Output directory        : {output_dir}")
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
                        f"✓ {instance:<30} "
                        f"Cost={r['cost']:.2f} "
                        f"Time={r['runtime']:.2f}s "
                        f"Stages={r.get('stages', 0)}{gap_txt}"
                    )
                else:
                    print(f"✗ {instance} ERROR: {r.get('error', 'Unknown error')}")
                    if 'traceback' in r:
                        print(r['traceback'])

            except Exception as e:
                import traceback
                print(f"✗ {instance} EXCEPTION: {e}")
                traceback.print_exc()
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
        avg_stages = sum(r.get("stages", 0) for r in ok) / len(ok)
        print(f"Average final cost    : {avg_cost:.2f}")
        print(f"Average runtime       : {avg_runtime:.2f}s")
        print(f"Average stages        : {avg_stages:.1f}")

        gaps = [r["gap_percent"] for r in ok if r["gap_percent"] is not None]
        if gaps:
            avg_gap = sum(gaps) / len(gaps)
            print(f"Average gap vs ref    : {avg_gap:+.2f}%")

    if bad:
        print("\nFailed instances:")
        for r in bad:
            print(f"  - {r['instance']}: {r.get('error', 'Unknown error')}")

    print("\nBenchmark complete.\n")


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
        "--max_workers",
        type=int,
        default=None,
        help="Parallel worker processes (default auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for DRSCI",
    )
    parser.add_argument(
        "--time_limit_per_cluster",
        type=float,
        default=30.0,
        help="Time limit per cluster in seconds (default: 30.0)",
    )
    parser.add_argument(
        "--ls_neighbourhood",
        type=str,
        default="dri_spatial",
        help="Local search neighbourhood (default: dri_spatial)",
    )
    parser.add_argument(
        "--ls_after_routing_max_neighbours",
        type=int,
        default=40,
        help="Max neighbours for LS after routing (default: 40)",
    )
    parser.add_argument(
        "--ls_max_neighbours_restricted",
        type=int,
        default=40,
        help="Max neighbours for LS in SCP phase (default: 40)",
    )
    parser.add_argument(
        "--scp_time_limit",
        type=float,
        default=600.0,
        help="Time limit for SCP solver in seconds (default: 600.0)",
    )
    parser.add_argument(
        "--use_combined_dissimilarity",
        action="store_true",
        help="Use combined dissimilarity instead of spatial only",
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
    )


if __name__ == "__main__":
    main()
