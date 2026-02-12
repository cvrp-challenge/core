from __future__ import annotations

import time
import random
import math
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Mapping, Tuple
from collections import Counter

# ---------------------------------------------------------
# Project path setup (same pattern as run_drsci.py)
# ---------------------------------------------------------
CURRENT = Path(__file__).parent
SRC_ROOT = CURRENT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from master.clustering.run_clustering import run_clustering
from master.clustering.route_based import route_based_decomposition
from master.routing.routing_controller import solve_clusters
from master.routing.solver import solve as routing_solve
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.setcover.route_dominance_filter import filter_route_pool
from master.utils.loader import load_instance
from master.utils.solution_helpers import load_routes_from_sol_for_pool
from master.setcover.route_pool_filtering import filter_route_pool_for_scp

# from master.utils.termination import Checkpoint, install_termination_handlers
from master.utils.logging_utils import get_run_logger, get_instance_logger

from master.utils.helpers_run_probabilistic import (
    Route,
    Routes,
    RouteKey,
    Tag,
    _route_key,
    _tag_new_routes,
    _result_to_vrplib_routes,
    _compute_integer_cost,
    _select_scp_solver_name,
    _load_bks_from_file,
    _format_gap_to_bks,
    _convert_customer_ids_for_output,
    _write_sol_if_bks_beaten,
    _write_sol_unconditional,
    print_final_route_summary,
)


# ============================================================
# SCP SOLVER DISPATCHER
# ============================================================

def lazy_import_scp(scp_solver: str):
    if scp_solver == "gurobi_mip":
        from master.setcover.scp_solver_gurobi_MIP import solve_scp
    elif scp_solver == "gurobi_lp":
        from master.setcover.scp_solver_gurobi_LP import solve_scp
    elif scp_solver == "hexaly":
        from master.setcover.scp_solver_hexaly import solve_scp
    else:
        raise ValueError(f"Unknown SCP solver: {scp_solver}")
    return solve_scp


def _format_cluster_sizes(
    clusters: Dict[Any, List[int]], *, depot_id: int = 1
) -> str:
    sizes = [len([nid for nid in v if nid != depot_id]) for v in clusters.values()]
    if not sizes:
        return "sizes=[] | min=0 max=0 avg=0.0"
    avg = sum(sizes) / len(sizes)
    return f"sizes={sizes} | min={min(sizes)} max={max(sizes)} avg={avg:.1f}"


# ============================================================
# METHOD SETS
# ============================================================

VB_METHODS = [
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
    "fcm",
    "k_medoids_pyclustering",
]

RB_METHODS = [
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
]

SOLVERS = ["pyvrp", "filo1", "filo2", "ails2"]

Route = List[int]
Routes = List[Route]

RouteKey = Tuple[int, ...]
Tag = Dict[str, Any]

def run_drsci_probabilistic(
    instance_name: str,
    *,
    seed: int = 0,
    scp_solvers: List[str] = ["gurobi_mip"],
    scp_switch_prob: float = 0.0,
    time_limit_scp: float = 300.0,
    scp_every: int = 5,
    time_limit_total: float = 600.0,
    max_no_improvement_iters: int = 20,
    k_min: int = 2,  # backward compatibility
    k_max: int = 8,  # backward compatibility
    min_avg_cluster_size: int = 100,
    max_avg_cluster_size: int = 2500,
    routing_solvers: Optional[List[str]] = None,
    routing_solver_options: Optional[Mapping[str, Any]] = None,
    routing_no_improvement: Optional[int] = None,
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 100,
    ls_max_neighbours_restricted: int = 100,
    randomize_polar_angle: bool = True,
    bks_output_dir: str = "output",
    periodic_sol_dump: bool = True,
    sol_dump_interval: float = 3600.0,
    enable_logging: bool = True,
    log_mode: str = "instance",  # "run" or "instance"
    log_to_console: bool = True,
    run_log_name: Optional[str] = None,
    warm_start_solutions: Optional[List[Path]] = None,
    scp_pruning_mode: str = "diversity",
) -> Dict[str, Any]:

    if scp_every <= 0:
        raise ValueError("scp_every must be >= 1")

    rng = random.Random(seed)
    start_time = time.time()

    inst = load_instance(instance_name)
    instance_base = Path(instance_name).stem

    global_route_pool: Routes = []
    route_tags: Dict[RouteKey, Tag] = {}
    
    # TERMINATION LOGIC COMMENTED OUT
    # ckpt = Checkpoint(
    #     instance_name=instance_name,
    #     output_dir=bks_output_dir,
    #     write_sol_fn=_write_sol_unconditional,
    # )
    # install_termination_handlers(ckpt)
    ckpt = None  # Dummy to avoid NameError

    logger = None
    if enable_logging:
        if log_mode == "run":
            logger = get_run_logger(
                output_dir=bks_output_dir,
                run_log_name=run_log_name,
                to_console=log_to_console,
            )
        elif log_mode == "instance":
            logger = get_instance_logger(
                instance_name=instance_name,
                output_dir=bks_output_dir,
                to_console=log_to_console,
                instance_suffix=run_log_name,
            )
        else:
            raise ValueError(f"Unknown log_mode: {log_mode}")

    # ----------------------------------------------------
    # Unified logging helper (print + optional logger)
    # ----------------------------------------------------
    def _log(msg: str, level: str = "info") -> None:
        if logger:
            getattr(logger, level)(msg)

    # ----------------------------------------------------
    # WARM START: load routes from existing .sol files
    # ----------------------------------------------------
    if warm_start_solutions:
        n_customers = len(inst["demand"]) - 1

        for sol_path in warm_start_solutions:
            sol_path = Path(sol_path)

            routes_from_sol = load_routes_from_sol_for_pool(
                sol_path,
                n_customers=n_customers,
                depot_id=1,
            )

            global_route_pool.extend(routes_from_sol)

            _tag_new_routes(
                route_tags,
                routes_from_sol,
                tag={
                    "mode": "warm_start",
                    "method": "sol",
                    "solver": "external",
                    "iteration": 0,
                    "stage": "initial_pool",
                },
            )

        # Safety: filter immediately
        global_route_pool = filter_route_pool(
            global_route_pool,
            depot_id=1,
            verbose=False,
        )

        msg = (
            f"[{instance_base} WARM-START] "
            f"loaded {len(global_route_pool)} routes from "
            f"{len(warm_start_solutions)} .sol files"
        )
        print(msg, flush=True)
        _log(msg)


    # Load BKS
    bks_cost = _load_bks_from_file(instance_name)

    # --------------------------------------------------------
    # Adaptive k-range
    # --------------------------------------------------------
    num_customers = len(inst["demand"]) - 1
    k_min_auto = max(2, math.ceil(num_customers / max_avg_cluster_size))
    k_max_auto = max(k_min_auto, 11)
    k_peak = 10

    values = list(range(k_min_auto, k_max_auto + 1))
    k_weights = [
        1.0 if k <= k_peak else (k_max_auto - k) / (k_max_auto - k_peak)
        for k in values
    ]

    print(
        f"[K-RANGE] customers={num_customers} | "
        f"k_min={k_min_auto} | k_max={k_max_auto}",
        flush=True,
    )

    best_routes: Optional[Routes] = None
    best_cost = float("inf")

    iteration = 0
    no_improvement_iters = 0
    last_snapshot_time = start_time
    snapshot_counter = 0


    # TERMINATION LOGIC COMMENTED OUT
    def maybe_checkpoint():
        # nonlocal last_dump
        # if not periodic_sol_dump:
        #     return
        # if ckpt.dirty and time.time() - last_dump >= sol_dump_interval:
        #     ckpt.dump(suffix="PERIODIC")
        #     last_dump = time.time()
        pass

    def log_best_route_summary():
        if not best_routes or not logger:
            return

        logger.info("")
        logger.info("[BEST SOLUTION ROUTE SUMMARY]")
        logger.info("-" * 80)

        tags = []
        for r in best_routes:
            key = _route_key(r, depot_id=1)
            tag = route_tags.get(key)
            if tag is None:
                tags.append(("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "N/A"))
            else:
                tags.append(
                    (
                        str(tag.get("mode")).upper(),
                        str(tag.get("method")),
                        str(tag.get("solver", "UNKNOWN")),
                        str(tag.get("stage")),
                        str(tag.get("ls", "N/A")),
                    )
                )

        counter = Counter(tags)

        for (mode, method, solver, stage, ls), count in sorted(
            counter.items(), key=lambda x: (-x[1], x[0])
        ):
            logger.info(
                f"{count:4d} routes | {mode:<5} | {method:<15} | "
                f"solver={solver:<10} | stage={stage} | ls={ls}"
            )


    def maybe_periodic_snapshot(force: bool = False):
        nonlocal last_snapshot_time, snapshot_counter

        if not periodic_sol_dump:
            return

        now = time.time()
        if not force and (now - last_snapshot_time) < sol_dump_interval:
            return

        if best_routes is None:
            return

        snapshot_counter += 1
        last_snapshot_time = now

        suffix = f"SNAPSHOT_{snapshot_counter:03d}"

        # 1) Write solution snapshot
        _write_sol_unconditional(
            instance_name=instance_name,
            routes=best_routes,
            cost=best_cost,
            output_dir=bks_output_dir,
            suffix=suffix,
        )

        # 2) Log best routes with tags
        if logger:
            logger.info("")
            logger.info("=" * 80)
            logger.info(f"[{instance_base}] PERIODIC SNAPSHOT {suffix}")
            logger.info("=" * 80)

            for i, r in enumerate(best_routes, 1):
                body = _convert_customer_ids_for_output([n for n in r if n != 1])
                tag = route_tags.get(_route_key(r, depot_id=1), {})
                logger.info(
                    f"Route #{i}: {' '.join(map(str, body))} || "
                    f"mode={tag.get('mode')} method={tag.get('method')} "
                    f"solver={tag.get('solver')} stage={tag.get('stage')} "
                    f"ls={tag.get('ls', 'N/A')} iter={tag.get('iteration')}"
                )

            log_best_route_summary()
            
            # 3) Route pool summary
            counter = Counter(
                (
                    str(t.get("mode")).upper(),
                    str(t.get("method")),
                    str(t.get("solver", "UNKNOWN")),
                    str(t.get("stage")),
                    str(t.get("ls", "N/A")),
                )
                for t in (
                    route_tags.get(_route_key(r, depot_id=1), {})
                    for r in global_route_pool
                )
            )

            logger.info("")
            logger.info("[ROUTE POOL SUMMARY]")
            for (mode, method, solver, stage, ls), v in sorted(counter.items(), key=lambda x: -x[1]):
                logger.info(f"{v:5d} routes | {mode} | {method} | solver={solver} | stage={stage} | ls={ls}")


    available_solvers = routing_solvers or SOLVERS
    if not available_solvers:
        raise ValueError("routing_solvers must contain at least one solver")

    solver_opts = dict(routing_solver_options or {})
    solver_opts.setdefault("use_stall", False)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    try:
        while True:
            # Check gap at beginning of iteration - overrule termination if gap < 0.001%
            gap_override = False
            if bks_cost is not None and best_cost != float("inf"):
                gap_percent = ((best_cost - bks_cost) / bks_cost) * 100
                if gap_percent < 0.001:
                    gap_override = True
                    msg = (
                        f"[{instance_base} GAP-OVERRIDE] "
                        f"gap={gap_percent:.4f}% < 0.001% - continuing despite termination criteria"
                    )
                    print(f"\033[92m{msg}\033[0m", flush=True)
                    _log(msg, level="info")
            
            # Termination criteria (skipped if gap override is active)
            if not gap_override:
                if time.time() - start_time >= time_limit_total:
                    msg = f"[{instance_base} STOP] time limit reached"
                    print(f"\033[91m{msg}\033[0m", flush=True)
                    _log(msg, level="warning")
                    break

                if no_improvement_iters >= max_no_improvement_iters:
                    msg = f"[{instance_base} STOP] no improvement limit reached"
                    print(f"\033[91m{msg}\033[0m", flush=True)
                    _log(msg, level="warning")
                    break

            iteration += 1
            improved_this_iter = False

            angle_offset = (
                rng.uniform(0, 2 * math.pi) if randomize_polar_angle else 0.0
            )

            mode = "vb" if rng.random() < 0.5 else "rb"
            k = rng.choices(values, weights=k_weights, k=1)[0]

            use_ails2_ls = rng.random() < 0.5
            ls_method = "ails2" if use_ails2_ls else "pyvrp"

            solver_weights = {"pyvrp": 0.2, "filo1": 0.2, "filo2": 0.2, "ails2": 0.4}
            weights = [
                solver_weights.get(s.lower(), 1.0 / len(available_solvers))
                for s in available_solvers
            ]
            routing_solver_key = rng.choices(
                available_solvers, weights=weights, k=1
            )[0].lower()

            method = rng.choice(VB_METHODS if mode == "vb" else RB_METHODS)

            if iteration == 1:
                mode = "vb"
                method = "sk_kmeans"
                routing_solver_key = "filo2"
                use_ails2_ls = False
                ls_method = "pyvrp"
                k = 1
            
            msg = (
                f"[{instance_base} ITERATION {iteration}] "
                f"mode={mode.upper()} method={method} k={k} solver={routing_solver_key} ls={ls_method}"
            )
            print(f"\033[94m{msg}\033[0m", flush=True)
            _log(msg)

            run_scp_now = (iteration % scp_every == 0)

            # ---------------- DECOMPOSITION ----------------
            if mode == "vb":
                clusters, _ = run_clustering(
                    method=method,
                    instance_name=instance_name,
                    k=k,
                    use_combined=False,
                    angle_offset=angle_offset,
                )
            else:
                if best_routes is None:
                    msg = f"[{instance_base} RB-SKIP] no incumbent solution yet"
                    print(msg, flush=True)
                    _log(msg)
                    no_improvement_iters += 1
                    continue

                clusters = route_based_decomposition(
                    instance_name=instance_name,
                    global_routes=best_routes,
                    k=k,
                    method=method,
                    use_angle=False,
                    use_load=False,
                )

            msg = (
                f"[{instance_base} CLUSTER] method={method} k={k} | "
                f"{_format_cluster_sizes(clusters, depot_id=1)}"
            )
            print(msg, flush=True)
            _log(msg)

            msg = f"[{instance_base} ROUTING] Solving clusters with {routing_solver_key}"
            print(msg, flush=True)
            _log(msg)

            override_no_improvement = (
                solver_opts.get("no_improvement", routing_no_improvement)
            )

            routing = solve_clusters(
                instance_name=instance_name,
                clusters=clusters,
                solver=routing_solver_key,
                solver_options=(solver_opts if routing_solver_key != "pyvrp" else None),
                seed=seed,
                no_improvement=override_no_improvement,
            )

            routes = _result_to_vrplib_routes(routing)
            if not routes:
                no_improvement_iters += 1
                continue

            if use_ails2_ls:
                ls_res = improve_with_local_search(
                    instance_name=instance_name,
                    routes_vrplib=routes,
                    ls_solver="ails2",
                    ails2_time_limit=100.0,  # 10 seconds for AILS2 improvement
                    seed=seed
                )
                improvement_stage = "post_ails2_ls"
            else:
                ls_res = improve_with_local_search(
                    instance_name=instance_name,
                    routes_vrplib=routes,
                    ls_solver="pyvrp",
                    neighbourhood=ls_neighbourhood,                    
                    max_neighbours=ls_after_routing_max_neighbours,
                    seed=seed
                )
                improvement_stage = "post_ls"
            
            candidate_routes = ls_res["routes_improved"]

            candidate_cost = _compute_integer_cost(inst, candidate_routes)

            _tag_new_routes(
                route_tags,
                candidate_routes,
                tag={
                    "mode": mode,
                    "method": method,
                    "solver": routing_solver_key,
                    "iteration": iteration,
                    "stage": improvement_stage,
                    "ls": ls_method,
                },
            )

            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_routes = candidate_routes
                # ckpt.update(best_routes, best_cost)  # TERMINATION LOGIC COMMENTED OUT
                improved_this_iter = True

                gap_str = _format_gap_to_bks(best_cost, bks_cost)
                msg = f"[{instance_base} IMPROVED-VB/RB] best_cost={best_cost}{gap_str}"
                print(msg, flush=True)
                _log(msg)

                _write_sol_if_bks_beaten(
                    instance_name=instance_name,
                    routes=best_routes,
                    cost=best_cost,
                    output_dir=bks_output_dir,
                )

            global_route_pool.extend(candidate_routes)
            global_route_pool = filter_route_pool(global_route_pool, depot_id=1, verbose=False)

            # ---------------- SCP ----------------
            if run_scp_now:
                scp_solver_name = _select_scp_solver_name(
                    rng, scp_solvers, scp_switch_prob
                )
                solve_scp = lazy_import_scp(scp_solver_name)

                _log(
                    f"[{instance_base} SCP] pruning_mode={scp_pruning_mode}"
                )

                scp_route_pool = filter_route_pool_for_scp(
                    routes=global_route_pool,
                    route_tags=route_tags,
                    inst=inst,
                    depot_id=1,
                    max_routes=4000,
                    min_utilization=0.30,
                    scp_every=scp_every,
                    elite_after_scp_rounds=2,
                    min_pool_size_for_elite=1500,
                    pruning_mode=scp_pruning_mode,
                )

                before = len(global_route_pool)
                after = len(scp_route_pool)
                removed = before - after

                if removed > 0:
                    msg = (
                        f"[{instance_base} SCP] solver={scp_solver_name} | "
                        f"route_pool={after} (removed {removed} / {before})"
                    )
                else:
                    msg = (
                        f"[{instance_base} SCP] solver={scp_solver_name} | "
                        f"route_pool={after}"
                    )

                print(msg, flush=True)
                _log(msg)

                scp_res = solve_scp(
                    instance_name=instance_name,
                    route_pool=scp_route_pool,
                    time_limit=time_limit_scp,
                    verbose=False,
                )

                # Optimality logging
                if logger:
                    if scp_res.get("optimal", False):
                        logger.info(f"[{instance_base} SCP] optimal solution found")
                    else:
                        logger.warning(
                            f"[{instance_base} SCP] not optimal (status={scp_res.get('status')})"
                        )

                # Duplicate removal + repair LS
                repaired = remove_duplicates(
                    instance_name=instance_name,
                    routes=scp_res["selected_routes"],
                    max_iters=50,
                    ls_neighbourhood=ls_neighbourhood,
                    ls_max_neighbours_restricted=ls_max_neighbours_restricted,
                    seed=seed,
                )["routes"]

                final_ls = improve_with_local_search(
                    instance_name=instance_name,
                    routes_vrplib=repaired,
                    neighbourhood=ls_neighbourhood,
                    max_neighbours=ls_max_neighbours_restricted,
                    seed=seed,
                )

                scp_routes = final_ls["routes_improved"]
                scp_cost = _compute_integer_cost(inst, scp_routes)

                # Enrich route pool
                global_route_pool.extend(scp_routes)
                global_route_pool = filter_route_pool(global_route_pool, depot_id=1, verbose=False)

                # Tag SCP-produced routes
                _tag_new_routes(
                    route_tags,
                    scp_routes,
                    tag={
                        "mode": "scp",
                        "method": scp_solver_name,
                        "iteration": iteration,
                        "stage": "scp_post_ls",
                    },
                )

                if scp_cost < best_cost:
                    maybe_periodic_snapshot()
                    best_cost = scp_cost
                    best_routes = scp_routes
                    # ckpt.update(best_routes, best_cost)  # TERMINATION LOGIC COMMENTED OUT
                    improved_this_iter = True

                    gap_str = _format_gap_to_bks(best_cost, bks_cost)
                    msg = f"[{instance_base} IMPROVED-SCP] best_cost={best_cost}{gap_str}"
                    print(msg, flush=True)
                    _log(msg)

                    _write_sol_if_bks_beaten(
                        instance_name=instance_name,
                        routes=best_routes,
                        cost=best_cost,
                        output_dir=bks_output_dir,
                    )
                else:
                    gap_str = _format_gap_to_bks(best_cost, bks_cost)
                    msg = (
                        f"[{instance_base} SCP-NO-IMPROVEMENT] "
                        f"cost={scp_cost} (best={best_cost}){gap_str}"
                    )
                    print(msg, flush=True)
                    _log(msg)


            maybe_periodic_snapshot()
            if improved_this_iter:
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1
                gap_str = _format_gap_to_bks(best_cost, bks_cost)
                msg = (
                    f"[{instance_base} NO-IMPROVEMENT] "
                    f"best_cost={best_cost} | streak={no_improvement_iters}{gap_str}"
                )
                print(msg, flush=True)
                _log(msg)


    except KeyboardInterrupt:
        msg = f"[{instance_base} INTERRUPT] Keyboard interrupt received"
        print(f"\033[91m{msg}\033[0m", flush=True)
        _log(msg, level="warning")
        raise
    
    # ========================================================
    # FINAL SCP (always run once before returning)
    # ========================================================
    if global_route_pool:
        print(
            f"[{instance_base} FINAL SCP] running final consolidation",
            flush=True,
        )

        scp_solver_name = _select_scp_solver_name(
            rng, scp_solvers, scp_switch_prob
        )
        solve_scp = lazy_import_scp(scp_solver_name)

        # ---- FILTER ROUTE POOL FOR SCP ----
        before = len(global_route_pool)

        scp_route_pool = filter_route_pool_for_scp(
            routes=global_route_pool,
            route_tags=route_tags,
            inst=inst,
            depot_id=1,
            max_routes=4000,
            min_utilization=0.30,
            scp_every=scp_every,
            elite_after_scp_rounds=2,
            min_pool_size_for_elite=1500,
            pruning_mode=scp_pruning_mode,
        )

        after = len(scp_route_pool)
        removed = before - after

        if removed > 0:
            msg = (
                f"[{instance_base} FINAL SCP] solver={scp_solver_name} | "
                f"route_pool={after} (removed {removed} / {before})"
            )
        else:
            msg = (
                f"[{instance_base} FINAL SCP] solver={scp_solver_name} | "
                f"route_pool={after}"
            )

        print(msg, flush=True)
        _log(msg)

        # ---- SCP SOLVE (FILTERED POOL ONLY) ----
        scp_res = solve_scp(
            instance_name=instance_name,
            route_pool=scp_route_pool,
            time_limit=time_limit_scp,
            verbose=False,
        )


        if logger:
            if scp_res.get("optimal", False):
                logger.info(f"[{instance_base} SCP] optimal solution found")
            else:
                logger.warning(
                    f"[{instance_base} SCP] not optimal (status={scp_res.get('status')})"
                )

        repaired = remove_duplicates(
            instance_name=instance_name,
            routes=scp_res["selected_routes"],
            max_iters=50,
            ls_neighbourhood=ls_neighbourhood,
            ls_max_neighbours_restricted=ls_max_neighbours_restricted,
            seed=seed,
        )["routes"]

        final_ls = improve_with_local_search(
            instance_name=instance_name,
            routes_vrplib=repaired,
            neighbourhood=ls_neighbourhood,
            max_neighbours=ls_max_neighbours_restricted,
            seed=seed,
        )

        final_routes = final_ls["routes_improved"]
        final_cost = _compute_integer_cost(inst, final_routes)

        global_route_pool.extend(final_routes)
        global_route_pool = filter_route_pool(global_route_pool, depot_id=1, verbose=False)

        _tag_new_routes(
            route_tags,
            final_routes,
            tag={
                "mode": "scp",
                "method": scp_solver_name,
                "iteration": iteration,
                "stage": "final_scp_post_ls",
            },
        )

        if final_cost < best_cost:
            best_cost = final_cost
            best_routes = final_routes
            # ckpt.update(best_routes, best_cost)  # TERMINATION LOGIC COMMENTED OUT
            maybe_checkpoint()

            gap_str = _format_gap_to_bks(best_cost, bks_cost)
            print(
                f"[{instance_base} FINAL IMPROVED] best_cost={best_cost}{gap_str}",
                flush=True,
            )

            _write_sol_if_bks_beaten(
                instance_name=instance_name,
                routes=best_routes,
                cost=best_cost,
                output_dir=bks_output_dir,
            )
        else:
            gap_str = _format_gap_to_bks(best_cost, bks_cost)
            print(
                f"[{instance_base} FINAL NO-IMPROVEMENT] "
                f"cost={final_cost} (best={best_cost}){gap_str}",
                flush=True,
            )


    # ========================================================
    # FINAL LOGGING OUTPUT (mirrors console output)
    # ========================================================
    if logger:

        def log_section(title: str):
            logger.info("")
            logger.info("=" * 80)
            logger.info(title)
            logger.info("=" * 80)

        # ----------------------------------------------------
        # 1) FINAL BEST SOLUTION ROUTES WITH TAGS
        # ----------------------------------------------------
        if best_routes:
            log_section(f"[{instance_base}] FINAL BEST ROUTES WITH TAGS")

            for i, r in enumerate(best_routes, 1):
                body_vrplib = [n for n in r if n != 1]
                body = _convert_customer_ids_for_output(body_vrplib)
                tag = route_tags.get(_route_key(r, depot_id=1))

                if tag is None:
                    tag_str = "mode=UNKNOWN method=UNKNOWN solver=UNKNOWN stage=UNKNOWN ls=N/A"
                else:
                    tag_str = (
                        f"mode={str(tag.get('mode')).upper()} "
                        f"method={tag.get('method')} "
                        f"solver={tag.get('solver', 'UNKNOWN')} "
                        f"stage={tag.get('stage')} "
                        f"ls={tag.get('ls', 'N/A')} "
                        f"iter={tag.get('iteration')}"
                    )

                logger.info(f"Route #{i}: {' '.join(map(str, body))} || {tag_str}")

        # ----------------------------------------------------
        # 2) SUMMARY: BEST SOLUTION ORIGINS
        # ----------------------------------------------------
        if best_routes:
            log_section(f"[{instance_base}] SUMMARY — BEST SOLUTION ROUTE ORIGINS")

            final_tags = []
            for r in best_routes:
                key = _route_key(r, depot_id=1)
                tag = route_tags.get(key)
                if tag is None:
                    final_tags.append(("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "N/A"))
                else:
                    final_tags.append(
                        (
                            str(tag.get("mode")).upper(),
                            str(tag.get("method")),
                            str(tag.get("solver", "UNKNOWN")),
                            str(tag.get("stage")),
                            str(tag.get("ls", "N/A")),
                        )
                    )

            counter = Counter(final_tags)
            for (mode, method, solver, stage, ls), count in sorted(
                counter.items(), key=lambda x: (-x[1], x[0])
            ):
                logger.info(
                    f"{count:4d} routes | {mode} | {method} | solver={solver} | stage={stage} | ls={ls}"
                )

        # ----------------------------------------------------
        # 3) SUMMARY: ROUTE POOL COMPOSITION
        # ----------------------------------------------------
        log_section(f"[{instance_base}] SUMMARY — ROUTE POOL COMPOSITION")

        pool_keys = [_route_key(r, depot_id=1) for r in global_route_pool]
        pool_tags = []

        for k in pool_keys:
            t = route_tags.get(k)
            if t is None:
                pool_tags.append(("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", "N/A"))
            else:
                pool_tags.append(
                    (
                        str(t.get("mode")).upper(),
                        str(t.get("method")),
                        str(t.get("solver", "UNKNOWN")),
                        str(t.get("stage")),
                        str(t.get("ls", "N/A")),
                    )
                )

        pool_counter = Counter(pool_tags)
        for (mode, method, solver, stage, ls), count in sorted(
            pool_counter.items(), key=lambda x: (-x[1], x[0])
        ):
            logger.info(
                f"{count:4d} routes | {mode} | {method} | solver={solver} | stage={stage} | ls={ls}"
            )

        # ----------------------------------------------------
        # 4) FULL ROUTE POOL DUMP
        # ----------------------------------------------------
        log_section(f"[{instance_base}] FULL ROUTE POOL DUMP")

        for i, r in enumerate(global_route_pool, 1):
            body_vrplib = [n for n in r if n != 1]
            body = _convert_customer_ids_for_output(body_vrplib)
            tag = route_tags.get(_route_key(r, depot_id=1))

            if tag is None:
                tag_str = "mode=UNKNOWN method=UNKNOWN solver=UNKNOWN stage=UNKNOWN ls=N/A"
            else:
                tag_str = (
                    f"mode={str(tag.get('mode')).upper()} "
                    f"method={tag.get('method')} "
                    f"solver={tag.get('solver', 'UNKNOWN')} "
                    f"stage={tag.get('stage')} "
                    f"ls={tag.get('ls', 'N/A')} "
                    f"iter={tag.get('iteration')}"
                )

            logger.info(f"[POOL #{i:05d}] {' '.join(map(str, body))} || {tag_str}")
    # ----------------------------------------------------
    # CONSOLE SUMMARY — FINAL ROUTE ORIGINS
    # ----------------------------------------------------
    if best_routes:
        print_final_route_summary(
            best_routes=best_routes,
            route_tags=route_tags,
            depot_id=1,
        )

    maybe_periodic_snapshot(force=True)
    return {
        "instance": instance_name,
        "best_cost": best_cost,
        "routes": best_routes or [],
        "iterations": iteration,
        "runtime": time.time() - start_time,
        "route_pool_size": len(global_route_pool),
    }
