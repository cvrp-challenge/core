from __future__ import annotations

import time
import random
from typing import Dict, List, Optional, Any, Mapping, Tuple
import math
import json

import sys
from pathlib import Path
import os
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
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.setcover.route_dominance_filter import filter_route_pool
from master.utils.loader import load_instance


# ============================================================
# SCP SOLVER DISPATCHER (same pattern as run_drsci.py)
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


def _format_cluster_sizes(clusters: Dict[Any, List[int]], *, depot_id: int = 1) -> str:
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

SOLVERS = [
    "pyvrp",
    "filo1",
    "filo2",
]


Route = List[int]
Routes = List[Route]

# ---------------- TAGGING TYPES ----------------
RouteKey = Tuple[int, ...]   # depot stripped, order preserved
Tag = Dict[str, Any]         # keep flexible for debugging / extra metadata


# ============================================================
# HELPERS
# ============================================================

def _route_key(route: Route, *, depot_id: int = 1) -> RouteKey:
    """
    Canonical key for tagging:
    - remove depot visits
    - keep order (do NOT sort)
    """
    return tuple(n for n in route if n != depot_id)


def _tag_new_routes(
    route_tags: Dict[RouteKey, Tag],
    routes: Routes,
    *,
    tag: Tag,
    depot_id: int = 1,
) -> None:
    """
    Attach tag to each route if it doesn't already have one.
    We use setdefault so we preserve the first-known origin for that exact route.
    """
    for r in routes:
        key = _route_key(r, depot_id=depot_id)
        if not key:
            continue
        route_tags.setdefault(key, dict(tag))


def _result_to_vrplib_routes(result) -> Routes:
    best = result.best
    if best is None:
        return []

    routes: Routes = []
    for r in best.routes():
        # PyVRP returns location indices; keep non-negative visits.
        visits = [v for v in r.visits() if v >= 0]
        if visits:
            routes.append([1] + [v + 1 for v in visits] + [1])
    return routes


def _compute_integer_cost(instance: dict, routes: Routes) -> int:
    coords = instance["node_coord"]
    edge_mat = instance.get("edge_weight")

    def dist(u: int, v: int) -> int:
        u_idx, v_idx = u - 1, v - 1
        if edge_mat is not None:
            return int(round(float(edge_mat[u_idx, v_idx])))
        dx = coords[u_idx][0] - coords[v_idx][0]
        dy = coords[u_idx][1] - coords[v_idx][1]
        return int(round((dx * dx + dy * dy) ** 0.5))

    total = 0
    for r in routes:
        for a, b in zip(r, r[1:]):
            total += dist(a, b)
    return total


def _select_scp_solver_name(rng: random.Random, scp_solvers: List[str], scp_switch_prob: float) -> str:
    if not scp_solvers:
        raise ValueError("scp_solvers must contain at least one solver name.")
    if rng.random() < scp_switch_prob and len(scp_solvers) > 1:
        return rng.choice(scp_solvers)
    return scp_solvers[0]


def _load_bks_from_file(instance_name: str) -> Optional[int]:
    """
    Load BKS (Best Known Solution) cost for an instance from bks.json file.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n916-k207.vrp")
        
    Returns:
        BKS cost as int if found, None otherwise
    """
    # Get the path to bks.json file
    # CURRENT is src/master, so we go up to core, then instances/test-instances
    bks_file = CURRENT.parent.parent / "instances" / "challenge-instances" / "challenge-bks.json"
    
    if not bks_file.exists():
        return None
    
    try:
        with open(bks_file, "r") as f:
            bks_data = json.load(f)
        
        instance_stem = Path(instance_name).stem
        bks_cost = bks_data.get(instance_stem)
        
        # Return None if BKS is null or not found
        if bks_cost is None:
            return None
        
        return int(bks_cost)
    except (json.JSONDecodeError, KeyError, ValueError, FileNotFoundError):
        return None


def _format_gap_to_bks(current_cost: float, bks_cost: Optional[int]) -> str:
    """
    Format the gap to BKS as a percentage string.
    
    Args:
        current_cost: Current solution cost
        bks_cost: Best Known Solution cost, or None if not available
        
    Returns:
        Formatted gap string like " | Gap: 10.0000%" or " | Gap: -10.0000%"
        Returns empty string if BKS is not available
        Negative gaps (better than BKS) are displayed in green
    """
    if bks_cost is None:
        return ""
    
    gap = ((current_cost - bks_cost) / bks_cost) * 100
    if gap < 0:
        # Green color for negative gaps (better than BKS)
        return f" | Gap: \033[38;2;41;209;47m{gap:.4f}%\033[0m"
    return f" | Gap: {gap:.4f}%"


def _convert_customer_ids_for_output(customers: List[int]) -> List[int]:
    """
    Convert customer IDs from VRPLIB format (2 to n) to official checker format (1 to n-1).
    This is a display-only conversion that doesn't affect calculations.
    
    Args:
        customers: List of customer IDs in VRPLIB format (2, 3, ..., n)
        
    Returns:
        List of customer IDs in official checker format (1, 2, ..., n-1)
    """
    return [c - 1 for c in customers]


def _write_sol_if_bks_beaten(
    *,
    instance_name: str,
    routes: Routes,
    cost: int,
    output_dir: str,
):
    """
    Write solution file if the current cost beats the BKS from bks.json.
    
    Args:
        instance_name: Name of the instance (e.g., "X-n916-k207.vrp")
        routes: List of routes (VRPLIB format)
        cost: Current solution cost
        output_dir: Directory to write the solution file
    """
    bks_cost = _load_bks_from_file(instance_name)
    
    # If BKS is not found or current cost doesn't beat BKS, return early
    if bks_cost is None or cost >= bks_cost:
        return

    os.makedirs(output_dir, exist_ok=True)
    base = Path(instance_name).stem
    sol_path = Path(output_dir) / f"BKS_{base}_{cost}.sol"

    # Official checker format: depot not mentioned, customers from 1 to n-1
    with open(sol_path, "w") as f:
        for idx, r in enumerate(routes, start=1):
            # Remove depot (assumes VRPLIB format [1, ..., 1])
            customers_vrplib = [v for v in r if v != 1]
            # Convert to official checker format (1 to n-1)
            customers = _convert_customer_ids_for_output(customers_vrplib)

            f.write(
                f"Route #{idx}: " + " ".join(map(str, customers)) + "\n"
            )

        f.write(f"Cost: {cost}\n")

    instance_base = Path(instance_name).stem

    print(
        # making the print appear in green for better visibility
        f"\033[38;2;41;209;47m[{instance_base} BKS] 🎉 BKS beaten! cost={cost} < BKS={bks_cost} "
        f"→ wrote {sol_path}\033[0m",
        flush=True,
    )
    
def _write_sol_unconditional(
    *,
    instance_name: str,
    routes: Routes,
    cost: int,
    output_dir: str,
    suffix: str = "INTERRUPTED",
):
    """
    Always write a .sol file, regardless of BKS.
    Used for interrupts / emergency checkpoints.
    """
    os.makedirs(output_dir, exist_ok=True)
    base = Path(instance_name).stem
    sol_path = Path(output_dir) / f"{base}_{suffix}_{cost}.sol"

    with open(sol_path, "w") as f:
        for idx, r in enumerate(routes, start=1):
            customers_vrplib = [v for v in r if v != 1]
            customers = _convert_customer_ids_for_output(customers_vrplib)
            f.write(f"Route #{idx}: " + " ".join(map(str, customers)) + "\n")
        f.write(f"Cost: {cost}\n")

    print(
        f"\033[93m[{base} INTERRUPT] wrote best-so-far solution → {sol_path}\033[0m",
        flush=True,
    )


def print_final_route_summary(
    *,
    best_routes: Routes,
    route_tags: Dict[RouteKey, Tag],
    depot_id: int = 1,
) -> None:
    """
    Summarize ONLY the routes that appear in the final solution.
    """
    final_tags = []

    for r in best_routes:
        key = _route_key(r, depot_id=depot_id)
        tag = route_tags.get(key)

        if tag is None:
            final_tags.append(("UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"))
        else:
            final_tags.append(
                (
                    str(tag.get("mode")).upper(),
                    str(tag.get("method")),
                    str(tag.get("solver", "UNKNOWN")),
                    str(tag.get("stage")),
                )
            )

    counter = Counter(final_tags)

    print("\n[FINAL ROUTE SUMMARY]")
    for (mode, method, solver, stage), count in sorted(
        counter.items(), key=lambda x: (-x[1], x[0])
    ):
        print(f"  {count:4d} routes | {mode} | {method} | solver={solver} | stage={stage}")

# ============================================================
# MAIN PROBABILISTIC DRSCI DRIVER
# ============================================================

# LUCCA: What about dissimilarity metrics?
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
    k_min: int = 2,                     # kept for backward compatibility
    k_max: int = 8,                     # kept for backward compatibility
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
) -> Dict[str, Any]:

    if scp_every <= 0:
        raise ValueError("scp_every must be >= 1")

    rng = random.Random(seed)
    start_time = time.time()

    inst = load_instance(instance_name)
    instance_base = Path(instance_name).stem
    
    # Load BKS for gap calculation
    bks_cost = _load_bks_from_file(instance_name)

    # --------------------------------------------------------
    # Adaptive k-range based on instance size
    # --------------------------------------------------------
    num_customers = len(inst["demand"]) - 1  # exclude depot

    k_min_auto = math.ceil(num_customers / max_avg_cluster_size)
    k_max_auto = 16
    k_peak = 10

    k_min_auto = max(2, k_min_auto)
    k_max_auto = max(k_min_auto, k_max_auto)

    values = list(range(k_min_auto, k_max_auto + 1))

    k_weights = []
    for k in values:
        if k <= k_peak:
            w = 1.0
        else:
            # linear decay from 1 at k=10 to 0 at k=25
            w = (k_max_auto - k) / (k_max_auto - k_peak)
        k_weights.append(w)

    print(
        f"[K-RANGE] customers={num_customers} | "
        f"k_min={k_min_auto} | k_max={k_max_auto}",
        flush=True,
    )

    global_route_pool: Routes = []

    # ---------------- TAGGING STORAGE (parallel to pool) ----------------
    route_tags: Dict[RouteKey, Tag] = {}

    best_routes: Optional[Routes] = None
    best_cost = float("inf")

    no_improvement_iters = 0
    iteration = 0

    # Default to SOLVERS list if not provided
    available_solvers = routing_solvers if routing_solvers is not None else SOLVERS
    if not available_solvers:
        raise ValueError("routing_solvers must contain at least one solver name.")

    # !!!!!Hexaly stalling enabling!!!!!
    # Keep user-provided options, but default use_stall=False if not given.
    _routing_solver_options = dict(routing_solver_options or {})
    _routing_solver_options.setdefault("use_stall", False)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    try:
        while True:
            # stopping criteria fulfilled?
            if time.time() - start_time >= time_limit_total:
                print(f"\033[91m[{instance_base} STOP] time limit reached\033[0m", flush=True)
                break

            if no_improvement_iters >= max_no_improvement_iters:
                print(f"\033[91m[{instance_base} STOP] no improvement limit reached\033[0m", flush=True)
                break

            iteration += 1

            if randomize_polar_angle:
                angle_offset = rng.uniform(0.0, 2 * math.pi)
            else:
                angle_offset = 0.0

            improved_this_iter = False

            # select mode (vb or rb)
            mode = "vb" if rng.random() < 0.5 else "rb"

            # select number of clusters 
            k = rng.choices(values, weights=k_weights, k=1)[0]

            # select routing solver (weighted: 20% pyvrp, 40% filo1, 40% filo2) <- finetune here
            solver_weights = {"pyvrp": 0.2, "filo1": 0.4, "filo2": 0.4}
            weights = [solver_weights.get(s.lower(), 1.0 / len(available_solvers)) for s in available_solvers]
            routing_solver_key = rng.choices(available_solvers, weights=weights, k=1)[0].lower()

            # select clustering method
            if mode == "vb":
                method = rng.choice(VB_METHODS)   # 1/6 each
            else:
                method = rng.choice(RB_METHODS)   # 1/4 each

            # special selection for first iteration
            if iteration == 1:
                mode = "vb"
                method = "sk_kmeans"
                routing_solver_key = "filo2"
                k = 1

            print(f"\033[94m[{instance_base} ITERATION {iteration}] mode={mode.upper()} method={method} k={k} solver={routing_solver_key}\033[0m", flush=True)

            # is scp running this iteration:
            run_scp_now = (iteration % scp_every == 0)

            # DECOMPOSITION
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
                    print(f"[{instance_base} RB-SKIP] no incumbent solution yet", flush=True)
                    no_improvement_iters += 1
                    continue

                clusters = route_based_decomposition(
                    instance_name=instance_name,
                    global_routes=best_routes,
                    k=k,
                    method=method,
                    use_angle=False,   # x,y only
                    use_load=False,    # x,y only
                )

            # Print cluster sizes for THIS iteration (customers-only, matches routing)
            print(
                f"[{instance_base} CLUSTER] method={method} k={k} | {_format_cluster_sizes(clusters, depot_id=1)}",
                flush=True,
            )

            # ROUTING

            # Note: We do NOT attempt any PyVRP stall stopping here. Any solver-specific early-stop behavior 
            # (e.g., Hexaly stall_time) belongs to routing_controller + solver adapters 
            # Routing solvers now use adaptive no-improvement iterations based on cluster size.
            # The value scales from 10000 (n<=100) to 100000 (n>=1000), linear in between.
            # If routing_no_improvement is provided, it overrides the adaptive behavior for all clusters.
            # If routing_solver_options contains "no_improvement", it also overrides.
            # Otherwise, no_improvement=None means use fully adaptive behavior per cluster.
            override_no_improvement = None
            if _routing_solver_options and "no_improvement" in _routing_solver_options:
                override_no_improvement = _routing_solver_options["no_improvement"]
            elif routing_no_improvement is not None:
                override_no_improvement = routing_no_improvement
            
            
            print(f"[{instance_base} ROUTING] Solving Clusters with {routing_solver_key}", flush=True)
            routing = solve_clusters(
                instance_name=instance_name,
                clusters=clusters,
                solver=routing_solver_key,
                solver_options=(_routing_solver_options if routing_solver_key != "pyvrp" else None),
                seed=seed,
                no_improvement=override_no_improvement,  # None = fully adaptive, value = override
            )

            routes = _result_to_vrplib_routes(routing)
            if not routes:
                print(f"[{instance_base} SKIP] routing produced no routes", flush=True)
                no_improvement_iters += 1
                continue

            # IMPROVEMENT:LOCAL SEARCH (POST-ROUTING)
            ls_res = improve_with_local_search(
                instance_name=instance_name,
                routes_vrplib=routes,
                neighbourhood=ls_neighbourhood,
                max_neighbours=ls_after_routing_max_neighbours,
                seed=seed,
            )

            candidate_routes = ls_res["routes_improved"]
            candidate_cost = _compute_integer_cost(inst, candidate_routes)

            # ---------------- TAGGING: VB/RB produced routes ----------------
            _tag_new_routes(
                route_tags,
                candidate_routes,
                tag={
                    "mode": mode,
                    "method": method,
                    "solver": routing_solver_key,
                    "iteration": iteration,
                    "stage": "post_ls",
                },
                depot_id=1,
            )

            # Update incumbent immediately if VB/RB candidate improves
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_routes = candidate_routes
                improved_this_iter = True
                gap_str = _format_gap_to_bks(best_cost, bks_cost)
                print(f"[{instance_base} IMPROVED-VB/RB] best_cost={best_cost}{gap_str}", flush=True)

                _write_sol_if_bks_beaten(
                    instance_name=instance_name,
                    routes=best_routes,
                    cost=best_cost,
                    output_dir=bks_output_dir,
                )

            global_route_pool.extend(candidate_routes)
            global_route_pool = filter_route_pool(
                global_route_pool,
                depot_id=1,
                verbose=False,
            )

            # ====================================================
            # PERIODIC SCP (every scp_every iterations)
            # ====================================================
            if run_scp_now:
                scp_solver_name = _select_scp_solver_name(rng, scp_solvers, scp_switch_prob)
                solve_scp = lazy_import_scp(scp_solver_name)
                print(f"[{instance_base} SCP] solver={scp_solver_name} | Route Pool Size={len(global_route_pool)}", flush=True)

                scp_res = solve_scp(
                    instance_name=instance_name,
                    route_pool=global_route_pool,
                    time_limit=time_limit_scp,
                    verbose=False,
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

                scp_routes = final_ls["routes_improved"]
                scp_cost = _compute_integer_cost(inst, scp_routes)

                global_route_pool.extend(scp_routes)
                global_route_pool = filter_route_pool(
                    global_route_pool,
                    depot_id=1,
                    verbose=False,
                )

                # ---------------- TAGGING: SCP-produced routes ----------------
                # These routes are not tied to a single clustering method anymore.
                _tag_new_routes(
                    route_tags,
                    scp_routes,
                    tag={
                        "mode": "scp",
                        "method": scp_solver_name,
                        "iteration": iteration,
                        "stage": "scp_post_ls",
                    },
                    depot_id=1,
                )

                if scp_cost < best_cost:
                    best_cost = scp_cost
                    best_routes = scp_routes
                    improved_this_iter = True
                    gap_str = _format_gap_to_bks(best_cost, bks_cost)
                    print(f"[{instance_base} IMPROVED-SCP] best_cost={best_cost}{gap_str}", flush=True)

                    _write_sol_if_bks_beaten(
                        instance_name=instance_name,
                        routes=best_routes,
                        cost=best_cost,
                        output_dir=bks_output_dir,
                    )
                else:
                    gap_str = _format_gap_to_bks(best_cost, bks_cost)
                    print(f"[{instance_base} SCP-NO-IMPROVEMENT] cost={scp_cost} (best={best_cost}){gap_str}", flush=True)
            else:
                print(f"[{instance_base} SCP-SKIP] accumulating routes only", flush=True)

            # ----------------------------------------------------
            # Stagnation counter (based on entire iteration result)
            # ----------------------------------------------------
            if improved_this_iter:
                no_improvement_iters = 0
            else:
                no_improvement_iters += 1
                gap_str = _format_gap_to_bks(best_cost, bks_cost)
                print(f"[{instance_base} NO-IMPROVEMENT] best_cost={best_cost} | streak={no_improvement_iters}{gap_str}", flush=True)

    except KeyboardInterrupt:
        print(
            f"\n\033[91m[{instance_base} INTERRUPT] Keyboard interrupt received.\033[0m",
            flush=True,
        )

        if best_routes is not None and math.isfinite(best_cost):
            _write_sol_unconditional(
                instance_name=instance_name,
                routes=best_routes,
                cost=int(best_cost),
                output_dir=bks_output_dir,
            )
        else:
            print(
                f"[{instance_base} INTERRUPT] No incumbent solution to write.",
                flush=True,
            )

        # Important: re-raise so multiprocessing / caller handles shutdown
        raise


    # ========================================================
    # FINAL SCP (always run once before returning)
    # ========================================================
    if global_route_pool:
        print(f"[{instance_base} FINAL SCP] running final consolidation", flush=True)

        scp_solver_name = _select_scp_solver_name(rng, scp_solvers, scp_switch_prob)
        solve_scp = lazy_import_scp(scp_solver_name)
        print(f"[{instance_base} FINAL SCP] solver={scp_solver_name}", flush=True)

        scp_res = solve_scp(
            instance_name=instance_name,
            route_pool=global_route_pool,
            time_limit=time_limit_scp,
            verbose=False,
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
        global_route_pool = filter_route_pool(
            global_route_pool,
            depot_id=1,
            verbose=False,
        )

        # ---------------- TAGGING: FINAL SCP routes ----------------
        _tag_new_routes(
            route_tags,
            final_routes,
            tag={
                "mode": "scp",
                "method": scp_solver_name,
                "iteration": iteration,
                "stage": "final_scp_post_ls",
            },
            depot_id=1,
        )

        if final_cost < best_cost:
            best_cost = final_cost
            best_routes = final_routes
            gap_str = _format_gap_to_bks(best_cost, bks_cost)
            print(f"[{instance_base} FINAL IMPROVED] best_cost={best_cost}{gap_str}", flush=True)
            _write_sol_if_bks_beaten(
                    instance_name=instance_name,
                    routes=best_routes,
                    cost=best_cost,
                    output_dir=bks_output_dir,
            )
        else:
            gap_str = _format_gap_to_bks(best_cost, bks_cost)
            print(f"[{instance_base} FINAL NO-IMPROVEMENT] cost={final_cost} (best={best_cost}){gap_str}", flush=True)

    # ========================================================
    # PRINT ROUTE TAGGING SUMMARY + FINAL ROUTES WITH TAGS
    # ========================================================

    # Summary should reflect the routes currently present in the pool (not all ever seen).
    pool_keys = [_route_key(r, depot_id=1) for r in global_route_pool]
    pool_tags = []
    for k in pool_keys:
        t = route_tags.get(k)
        if t is None:
            # Route in pool but not tagged (should be rare); keep visible.
            pool_tags.append(("unknown", "unknown", "unknown", "unknown"))
        else:
            pool_tags.append((str(t.get("mode")), str(t.get("method")), str(t.get("solver", "unknown")), str(t.get("stage"))))

    counter = Counter(pool_tags)

    print(f"\n[{instance_base} ROUTE SUMMARY]")
    for (mode, method, solver, stage), count in counter.items():
        print(f"  {count:4d} routes | {mode.upper()} | {method} | solver={solver} | stage={stage}")

    # Print final returned routes in sol-like format, with tags.
    print("\n" + "-" * 80)
    print(f"[{instance_base} FINAL ROUTES WITH TAGS]")

    if best_routes:
        for i, r in enumerate(best_routes, 1):
            # Remove depot and convert to official checker format (1 to n-1)
            body_vrplib = [n for n in r if n != 1]
            body = _convert_customer_ids_for_output(body_vrplib)
            tag = route_tags.get(_route_key(r, depot_id=1))
            if tag is None:
                tag_str = "mode=UNKNOWN method=UNKNOWN solver=UNKNOWN stage=UNKNOWN"
            else:
                tag_str = (
                    f"mode={str(tag.get('mode')).upper()} "
                    f"method={tag.get('method')} "
                    f"solver={tag.get('solver', 'UNKNOWN')} "
                    f"iter={tag.get('iteration')} "
                    f"stage={tag.get('stage')}"
                )
            print(f"Route #{i}: {' '.join(map(str, body))} || {tag_str}")
    
    if best_routes:
        print_final_route_summary(
            best_routes=best_routes,
            route_tags=route_tags,
            depot_id=1,
        )


    return {
        "instance": instance_name,
        "best_cost": best_cost,
        "routes": best_routes or [],
        "iterations": iteration,
        "runtime": time.time() - start_time,
        "route_pool_size": len(global_route_pool),
    }


# ============================================================
# DEBUG ENTRY POINT
# ============================================================

if __name__ == "__main__":
    res = run_drsci_probabilistic(
        instance_name="X-n916-k207.vrp",
        seed=321,
        time_limit_total=1200.0,
        time_limit_scp=300.0,
        scp_every=3,
        min_avg_cluster_size=125,
        max_avg_cluster_size=2500,
        max_no_improvement_iters=10,
        ls_max_neighbours_restricted=200,
        ls_after_routing_max_neighbours=200,
    )

    print("\n[DEBUG] Best cost:", res["best_cost"])
    print("[DEBUG] #routes:", len(res["routes"]))
    print("[DEBUG] Iterations:", res["iterations"])
