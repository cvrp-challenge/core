import argparse
import json
import random
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


def _load_bks_from_file(instance_name: str) -> Optional[int]:
    """
    Load BKS (Best Known Solution) cost for an instance from test-bks.json file.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n916-k207.vrp")
        
    Returns:
        BKS cost as int if found, None otherwise
    """
    bks_file = PROJECT_ROOT / "instances" / "challenge-instances" / "challenge-bks.json"
    
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
    run_id: int,
    output_dir: Path,
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
    enable_logging: bool,
    log_mode: str,
    log_to_console: bool,
    run_log_name: Optional[str],
    periodic_sol_dump: bool,
    sol_dump_interval: float,
) -> dict:
    try:
        from master.run_drsci_probabilistic import run_drsci_probabilistic

        # Generate a unique seed for this worker
        seed = random.randint(0, 2**31 - 1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{instance_name} (run {run_id}) is starting with seed {seed}.", flush=True)

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
            enable_logging=enable_logging,
            log_mode=log_mode,
            log_to_console=log_to_console,
            run_log_name=run_log_name,
            periodic_sol_dump=periodic_sol_dump,
            sol_dump_interval=sol_dump_interval,
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
        sol_name = f"{Path(instance_name).stem}_probabilistic_run{run_id}"

        _write_solution(
            where=output_dir,
            instance_name=sol_name,
            data=inst,
            result=routes,
            solver="Probabilistic-DRSCI",
            runtime=runtime,
            stopping_criteria=f"Probabilistic-DRSCI({iterations} iterations, seed={seed})",
            gap_percent=gap,
            cost=best_cost,
        )

        return {
            "instance": instance_name,
            "run_id": run_id,
            "seed": seed,
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
        print(f"[ERROR] {instance_name} (run {run_id}): {e}", flush=True)
        print(tb, flush=True)
        return {
            "instance": instance_name,
            "run_id": run_id,
            "success": False,
            "error": str(e),
            "traceback": tb,
        }

# ---------------------------------------------------------
# Run intensify benchmark
# ---------------------------------------------------------
def run_intensify(
    instance_name: str,
    num_runs: int,
    output_path: str,
    max_workers: Optional[int],
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
    enable_logging: bool,
    log_mode: str,
    log_to_console: bool,
    run_log_name: Optional[str],
    periodic_sol_dump: bool,
    sol_dump_interval: float,
):
    output_dir = Path(output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Probabilistic DRSCI intensify on {instance_name} ({num_runs} runs).")
    print(f"SCP solvers            : {scp_solvers}")
    print(f"SCP switch probability : {scp_switch_prob}")
    print(f"Time limit (total)     : {time_limit_total}s")
    print(f"Time limit (SCP)       : {time_limit_scp}s")
    print(f"SCP every              : {scp_every} iterations")
    print(f"Max no-improvement     : {max_no_improvement_iters} iterations")
    print(f"Seed                   : random (each worker generates its own)")
    print(f"Parallel workers       : {max_workers or 'auto'}")
    print("-" * 80)

    results = []

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker) as executor:
        futures = {
            executor.submit(
                solve_instance_probabilistic,
                instance_name,
                run_id,
                output_dir,
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
                enable_logging,
                log_mode,
                log_to_console,
                run_log_name,
                periodic_sol_dump,
                sol_dump_interval,
            ): run_id
            for run_id in range(1, num_runs + 1)
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
                    f"✓ Run {r['run_id']:>3} "
                    f"Cost={r['cost']:>10} "
                    f"Time={r['runtime']:>8.2f}s "
                    f"Iters={r['iterations']:>4} "
                    f"Seed={r['seed']:>10}{gap_txt}{bks_txt}"
                )
            else:
                print(f"✗ Run {r['run_id']} ERROR: {r['error']}")

    print("-" * 80)
    print("Intensify complete.\n")

    # Print summary statistics
    successful = [r for r in results if r["success"]]
    if successful:
        costs = [r["cost"] for r in successful]
        runtimes = [r["runtime"] for r in successful]
        iterations = [r["iterations"] for r in successful]
        gaps = [r["gap_percent"] for r in successful if r["gap_percent"] is not None]
        
        best_run = min(successful, key=lambda x: x["cost"])
        worst_run = max(successful, key=lambda x: x["cost"])
        
        avg_cost = sum(costs) / len(costs)
        avg_runtime = sum(runtimes) / len(runtimes)
        avg_iterations = sum(iterations) / len(iterations)
        avg_gap = sum(gaps) / len(gaps) if gaps else None

        print("Summary Statistics:")
        print(f"  Successful runs    : {len(successful)}/{len(results)}")
        print(f"  Best cost          : {best_run['cost']} (run {best_run['run_id']}, seed {best_run['seed']})")
        print(f"  Worst cost         : {worst_run['cost']} (run {worst_run['run_id']}, seed {worst_run['seed']})")
        print(f"  Average cost       : {avg_cost:.2f}")
        print(f"  Cost std dev       : {((sum((c - avg_cost) ** 2 for c in costs) / len(costs)) ** 0.5):.2f}")
        if avg_gap is not None:
            print(f"  Average gap        : {avg_gap:+.2f}%")
        print(f"  Average runtime    : {avg_runtime:.2f}s")
        print(f"  Average iterations : {avg_iterations:.1f}")

# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Intensify the Probabilistic DRSCI algorithm by running a single instance multiple times with different seeds."
    )

    parser.add_argument("instance_name", type=str, help="Name of the instance to run (e.g., 'XL-n1281-k29.vrp')")
    parser.add_argument("num_runs", type=int, help="Number of parallel runs to execute")
    parser.add_argument("output_path", type=str)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument(
        "--scp_solvers",
        nargs="+",
        choices=["gurobi_mip", "gurobi_lp", "hexaly"],
        default=["gurobi_mip"],
    )
    parser.add_argument("--scp_switch_prob", type=float, default=0.0)
    parser.add_argument("--time_limit_scp", type=float, default=600.0)
    parser.add_argument("--scp_every", type=int, default=3)
    parser.add_argument("--time_limit_total", type=float, default=50000.0)
    parser.add_argument("--max_no_improvement_iters", type=int, default=75)
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
    parser.add_argument("--ls_max_neighbours_restricted", type=int, default=100)
    parser.add_argument("--randomize_polar_angle", action="store_true", default=True)
    parser.add_argument("--no_randomize_polar_angle", dest="randomize_polar_angle", action="store_false")
    parser.add_argument("--bks_output_dir", type=str, default="output")
    # ---------------- Logging options ----------------
    parser.add_argument("--enable_logging", action="store_true", default=True)
    parser.add_argument("--log_mode", choices=["run", "instance"], default="instance")
    parser.add_argument("--log_to_console", action="store_true", default=True)
    parser.add_argument("--run_log_name", type=str, default=None)

    # -------------- Periodic .sol dump options --------------
    parser.add_argument("--periodic_sol_dump", action="store_true", default=True)
    parser.add_argument("--sol_dump_interval", type=float, default=3600.0)

    args = parser.parse_args()

    # Parse routing_solver_options if needed (for future extension)
    routing_solver_options = None

    run_intensify(
        instance_name=args.instance_name,
        num_runs=args.num_runs,
        output_path=args.output_path,
        max_workers=args.max_workers,
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
        enable_logging=args.enable_logging,
        log_mode=args.log_mode,
        log_to_console=args.log_to_console,
        run_log_name=args.run_log_name,
        periodic_sol_dump=args.periodic_sol_dump,
        sol_dump_interval=args.sol_dump_interval,
    )


if __name__ == "__main__":
    main()
