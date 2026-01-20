import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from datetime import datetime

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
from master.utils.logging_utils import (
    get_run_logger,
    get_instance_logger,
)

# ---------------------------------------------------------
# Instances to benchmark
# ---------------------------------------------------------
INSTANCES = [
    # "XL-n1048-k237.vrp", ###
    "XL-n1094-k157.vrp", ### good -- 0.0178%
    # "XL-n1141-k112.vrp", ###
    # "XL-n1188-k96.vrp", ###
    # "XL-n1234-k55.vrp", ###
    # "XL-n1281-k29.vrp", ### bad
    # "XL-n1328-k19.vrp", ###
    # "XL-n1374-k278.vrp", ###
    # "XL-n1421-k232.vrp", ### bad
    # "XL-n1468-k151.vrp", ###
    "XL-n1514-k106.vrp", ### good
    "XL-n1561-k75.vrp", ### good
    # "XL-n1608-k39.vrp", ###
    # "XL-n1654-k11.vrp", ###
    "XL-n1701-k562.vrp", ###
    # "XL-n1748-k271.vrp", ### bad
    "XL-n1794-k163.vrp", ### good -- 0.051%
    # "XL-n1841-k126.vrp", ###
    # "XL-n1888-k82.vrp", ###
    "XL-n1934-k46.vrp", ### good
    # "XL-n1981-k13.vrp", ### bad
    # "XL-n2028-k617.vrp", ### bad
    # "XL-n2074-k264.vrp", ### bad
    # "XL-n2121-k186.vrp", ###
    # "XL-n2168-k138.vrp", ###
    "XL-n2214-k131.vrp", ### good -- 0.054%
    # "XL-n2261-k54.vrp", ###
    # "XL-n2307-k34.vrp", ###
    # "XL-n2354-k631.vrp", ### bad
    # "XL-n2401-k408.vrp", ### bad
    # "XL-n2447-k290.vrp", ### bad
    # "XL-n2494-k194.vrp", ### bad
    "XL-n2541-k121.vrp", ### good
    "XL-n2587-k66.vrp", ### good
    # "XL-n2634-k17.vrp", ###
    # "XL-n2681-k540.vrp", ### bad
    "XL-n2727-k546.vrp", ### good -- 0.02%
    # "XL-n2774-k286.vrp", ### bad
    # "XL-n2821-k208.vrp", ###
    # "XL-n2867-k120.vrp", ###
    # "XL-n2914-k95.vrp", ### bad
    # "XL-n2961-k55.vrp", ### bad
    # "XL-n3007-k658.vrp", ###
    # "XL-n3054-k461.vrp", ### bad
    # "XL-n3101-k311.vrp", ###
    # "XL-n3147-k232.vrp", ###
    # "XL-n3194-k161.vrp", ### bad
    # "XL-n3241-k115.vrp", ###
    # "XL-n3287-k30.vrp", ###
    # "XL-n3334-k934.vrp", ###
    # "XL-n3408-k524.vrp", ### bad
    "XL-n3484-k436.vrp", ### good -- 0.01%
    # "XL-n3561-k229.vrp", ###
    # "XL-n3640-k211.vrp", ###
    # "XL-n3721-k77.vrp", ### bad
    # "XL-n3804-k29.vrp", ###
    # "XL-n3888-k1010.vrp" ### bad
    "XL-n3975-k687.vrp", ###
    # "XL-n4063-k347.vrp", ###
    # "XL-n4153-k291.vrp", ###
    "XL-n4245-k203.vrp", ### good
    # "XL-n4340-k148.vrp", ###
    "XL-n4436-k48.vrp", ###
    "XL-n4535-k1134.vrp", ### good -- 0.0029%
    # "XL-n4635-k790.vrp", ###
    # "XL-n4738-k487.vrp", ###
    # "XL-n4844-k321.vrp", ###
    # "XL-n4951-k203.vrp", ###
    # "XL-n5061-k184.vrp", ###
    # "XL-n5174-k55.vrp", ### bad
    # "XL-n5288-k1246.vrp", ### bad
    "XL-n5406-k783.vrp", ### good
    "XL-n5526-k553.vrp", ### good -- 0.08%
    # "XL-n5649-k401.vrp", ###
    # "XL-n5774-k290.vrp", ###
    "XL-n5902-k122.vrp", ###
    # "XL-n6034-k61.vrp", ### bad
    # "XL-n6168-k1922.vrp", ### bad
    # "XL-n6305-k1042.vrp", ### bad
    # "XL-n6445-k628.vrp", ###
    # "XL-n6588-k473.vrp", ### bad
    # "XL-n6734-k330.vrp", ###
    # "XL-n6884-k148.vrp", ### bad
    # "XL-n7037-k38.vrp", ### bad
    # "XL-n7193-k1683.vrp", ### bad
    "XL-n7353-k1471.vrp", ### good -- 0.034%
    # "XL-n7516-k859.vrp", ### bad
    # "XL-n7683-k602.vrp", ###
    # "XL-n7854-k365.vrp", ###
    # "XL-n8028-k294.vrp", ### bad
    # "XL-n8207-k108.vrp", ### bad
    # "XL-n8389-k2028.vrp", ### bad
    "XL-n8575-k1297.vrp", ###
    "XL-n8766-k1032.vrp", ### bad
    "XL-n8960-k634.vrp", ###
    "XL-n9160-k379.vrp", ### bad
    # "XL-n9363-k209.vrp", ###
    # "XL-n9571-k55.vrp", ### bad
    "XL-n9784-k2774.vrp", ### bad (exclude)
    "XL-n10001-k1570.vrp" ### bad (exclude)
]

def _log_run_configuration(logger, *, instance_name, run_id, seed, **params):
    logger.init("=" * 80)
    logger.init(f"[RUN CONFIGURATION] instance={instance_name} run={run_id}")
    logger.init(f"seed = {seed}")
    for k, v in params.items():
        logger.init(f"{k} = {v}")
    logger.init("=" * 80)

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
    warm_start_solutions: Optional[list[str]],
) -> dict:
    try:
        from master.run_drsci_probabilistic import run_drsci_probabilistic

        # Generate a unique seed for this worker
        seed = random.randint(0, 2**31 - 1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"{instance_name} is starting with seed {seed}.", flush=True)

        if log_mode == "instance":
            logger = get_instance_logger(
                instance_name=instance_name,
                output_dir=str(output_dir),
                to_console=log_to_console,
                instance_suffix=None,
            )
        else:
            logger = get_run_logger(
                output_dir=str(output_dir),
                run_log_name=run_log_name,
                to_console=log_to_console,
            )

        _log_run_configuration(
            logger,
            instance_name=instance_name,
            run_id=0,  # benchmark has no per-run index
            seed=seed,

            scp_solvers=scp_solvers,
            scp_switch_prob=scp_switch_prob,
            time_limit_scp=time_limit_scp,
            scp_every=scp_every,
            time_limit_total=time_limit_total,
            max_no_improvement_iters=max_no_improvement_iters,

            min_avg_cluster_size=min_avg_cluster_size,
            max_avg_cluster_size=max_avg_cluster_size,

            routing_solvers=routing_solvers or ["pyvrp", "filo1", "filo2"],
            routing_no_improvement=routing_no_improvement,

            ls_neighbourhood=ls_neighbourhood,
            ls_after_routing_max_neighbours=ls_after_routing_max_neighbours,
            ls_max_neighbours_restricted=ls_max_neighbours_restricted,

            randomize_polar_angle=randomize_polar_angle,

            warm_start_solutions=warm_start_solutions,

            log_mode=log_mode,
            log_to_console=log_to_console,
            periodic_sol_dump=periodic_sol_dump,
            sol_dump_interval=sol_dump_interval,
        )

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
            warm_start_solutions=warm_start_solutions,
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
    warm_start_solutions: Optional[list[str]],
):
    experiment_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = Path(output_path).resolve() / f"challenge_{experiment_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    bks_output_dir = str(output_dir)  

    # Use provided instances or default to INSTANCES list
    instances_to_run = INSTANCES

    print(f"Running Probabilistic DRSCI benchmark on {len(instances_to_run)} instances.")
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
                inst,
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
                warm_start_solutions,
            ): inst
            for inst in instances_to_run
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
    parser.add_argument(
        "--scp_solvers",
        nargs="+",
        choices=["gurobi_mip", "gurobi_lp", "hexaly"],
        default=["gurobi_mip"],
    )
    parser.add_argument("--scp_switch_prob", type=float, default=0.0)
    parser.add_argument("--time_limit_scp", type=float, default=10000.0)
    parser.add_argument("--scp_every", type=int, default=3)
    parser.add_argument("--time_limit_total", type=float, default=35000.0)
    parser.add_argument("--max_no_improvement_iters", type=int, default=50)
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
    parser.add_argument("--ls_after_routing_max_neighbours", type=int, default=300)
    parser.add_argument("--ls_max_neighbours_restricted", type=int, default=200)
    parser.add_argument("--randomize_polar_angle", action="store_true", default=True)
    parser.add_argument("--no_randomize_polar_angle", dest="randomize_polar_angle", action="store_false")
    parser.add_argument("--bks_output_dir", type=str, default="output")

    parser.add_argument("--enable_logging", action="store_true", default=True)
    parser.add_argument("--log_mode", choices=["run", "instance"], default="instance")
    parser.add_argument("--log_to_console", action="store_true", default=False)
    parser.add_argument("--run_log_name", type=str, default=None)
    parser.add_argument("--periodic_sol_dump", action="store_true", default=True)
    parser.add_argument("--sol_dump_interval", type=float, default=3600.0)
    parser.add_argument(
        "--warm_start_solutions",
        nargs="*",
        type=str,
        default=None,
        help="Paths to .sol files whose routes are injected into the initial route pool",
    )


    args = parser.parse_args()

    # Parse routing_solver_options if needed (for future extension)
    routing_solver_options = None

    run_benchmark(
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
        warm_start_solutions=args.warm_start_solutions,
    )


if __name__ == "__main__":
    main()
