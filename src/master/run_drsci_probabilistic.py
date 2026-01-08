from __future__ import annotations

import time
import random
from typing import Dict, List, Optional, Any, Mapping, Tuple
import math

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


def _write_sol_if_bks_beaten(
    *,
    instance_name: str,
    routes: Routes,
    cost: int,
    bks_cost: int,
    output_dir: str,
):
    if cost >= bks_cost:
        return

    os.makedirs(output_dir, exist_ok=True)
    base = Path(instance_name).stem
    sol_path = Path(output_dir) / f"{base}_BKS_BEAT_{cost}.sol"

    with open(sol_path, "w") as f:
        for idx, r in enumerate(routes, start=1):
            # Remove depot (assumes VRPLIB format [1, ..., 1])
            customers = [v for v in r if v != 1]

            f.write(
                f"Route #{idx}: " + " ".join(map(str, customers)) + "\n"
            )

        f.write(f"Cost: {cost}\n")

    print(
        f"[BKS] 🎉 BKS beaten! cost={cost} < BKS={bks_cost} "
        f"→ wrote {sol_path}",
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
            final_tags.append(("UNKNOWN", "UNKNOWN", "UNKNOWN"))
        else:
            final_tags.append(
                (
                    str(tag.get("mode")).upper(),
                    str(tag.get("method")),
                    str(tag.get("stage")),
                )
            )

    counter = Counter(final_tags)

    print("\n[FINAL ROUTE SUMMARY]")
    for (mode, method, stage), count in sorted(
        counter.items(), key=lambda x: (-x[1], x[0])
    ):
        print(f"  {count:4d} routes | {mode} | {method} | stage={stage}")

# ============================================================
# MAIN PROBABILISTIC DRSCI DRIVER
# ============================================================

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
    min_cluster_size: int = 100,
    max_cluster_size: int = 2500,
    routing_solver: str = "pyvrp",
    routing_solver_options: Optional[Mapping[str, Any]] = None,
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 100,
    ls_max_neighbours_restricted: int = 100,
    randomize_polar_angle: bool = True,
    bks_cost: Optional[int] = None,
    bks_output_dir: str = "output",
) -> Dict[str, Any]:

    if scp_every <= 0:
        raise ValueError("scp_every must be >= 1")

    rng = random.Random(seed)
    start_time = time.time()

    inst = load_instance(instance_name)

    # --------------------------------------------------------
    # Adaptive k-range based on instance size
    # --------------------------------------------------------
    num_customers = len(inst["demand"]) - 1  # exclude depot

    k_min_auto = math.ceil(num_customers / max_cluster_size)
    k_max_auto = math.floor(num_customers / min_cluster_size)

    k_min_auto = max(2, k_min_auto)
    k_max_auto = max(k_min_auto, k_max_auto)

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

    routing_solver_key = routing_solver.lower()

    # !!!!!Hexaly stalling enabling!!!!!
    # Keep user-provided options, but default use_stall=False if not given.
    _routing_solver_options = dict(routing_solver_options or {})
    _routing_solver_options.setdefault("use_stall", False)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    while True:
        # -------------------------------
        # STOPPING CRITERIA
        # -------------------------------
        if time.time() - start_time >= time_limit_total:
            print("[STOP] time limit reached", flush=True)
            break

        if no_improvement_iters >= max_no_improvement_iters:
            print("[STOP] no improvement limit reached", flush=True)
            break

        iteration += 1

        if randomize_polar_angle:
            angle_offset = rng.uniform(0.0, 2 * math.pi)
        else:
            angle_offset = 0.0

        print(
            f"[ANGLE] polar_offset={angle_offset:.3f} rad "
            f"({angle_offset * 180 / math.pi:.1f}°)",
            flush=True,
        )

        improved_this_iter = False

        # -------------------------------
        # SELECT MODE (VB / RB)
        # -------------------------------
        if iteration == 1:
            mode = "vb"
        else:
            mode = "vb" if rng.random() < 0.5 else "rb"

        # -------------------------------
        # SELECT METHOD
        # -------------------------------
        if mode == "vb":
            method = rng.choice(VB_METHODS)   # 1/6 each
        else:
            method = rng.choice(RB_METHODS)   # 1/4 each

        # -------------------------------
        # SELECT k
        # -------------------------------
        # NOTE: we use auto range; keep k_min/k_max parameters for backward compat only.
        k = rng.randint(k_min_auto, k_max_auto)

        print(f"[ITER {iteration}] mode={mode.upper()} method={method} k={k}", flush=True)

        # ----------------------------------------------------
        # SCP schedule: ONLY every scp_every iterations
        # ----------------------------------------------------
        run_scp_now = (iteration % scp_every == 0)

        # ====================================================
        # DECOMPOSITION
        # ====================================================
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
                print("[RB-SKIP] no incumbent solution yet", flush=True)
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
            f"[CLUSTER] method={method} k={k} | {_format_cluster_sizes(clusters, depot_id=1)}",
            flush=True,
        )

        # ====================================================
        # ROUTING
        # ====================================================
        # Note:
        # - We do NOT attempt any PyVRP stall stopping here.
        # - Any solver-specific early-stop behavior (e.g., Hexaly stall_time) belongs
        #   to routing_controller + solver adapters.
        routing = solve_clusters(
            instance_name=instance_name,
            clusters=clusters,
            solver=routing_solver_key,
            solver_options=(_routing_solver_options if routing_solver_key != "pyvrp" else None),
            seed=seed,
        )

        routes = _result_to_vrplib_routes(routing)
        if not routes:
            print("[SKIP] routing produced no routes", flush=True)
            no_improvement_iters += 1
            continue

        # ====================================================
        # LOCAL SEARCH (POST-ROUTING)
        # ====================================================
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
            print(f"[IMPROVED-VB/RB] best_cost={best_cost}", flush=True)

            if bks_cost is not None:
                _write_sol_if_bks_beaten(
                    instance_name=instance_name,
                    routes=best_routes,
                    cost=best_cost,
                    bks_cost=bks_cost,
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
            print(f"[SCP] solver={scp_solver_name}", flush=True)

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
                print(f"[IMPROVED-SCP] best_cost={best_cost}", flush=True)

                if bks_cost is not None:
                    _write_sol_if_bks_beaten(
                        instance_name=instance_name,
                        routes=best_routes,
                        cost=best_cost,
                        bks_cost=bks_cost,
                        output_dir=bks_output_dir,
                    )
            else:
                print(f"[SCP-NO-IMPROVEMENT] cost={scp_cost} (best={best_cost})", flush=True)
        else:
            print("[SCP-SKIP] accumulating routes only", flush=True)

        # ----------------------------------------------------
        # Stagnation counter (based on entire iteration result)
        # ----------------------------------------------------
        if improved_this_iter:
            no_improvement_iters = 0
        else:
            no_improvement_iters += 1
            print(f"[NO-IMPROVEMENT] best_cost={best_cost} | streak={no_improvement_iters}", flush=True)

    # ========================================================
    # FINAL SCP (always run once before returning)
    # ========================================================
    if global_route_pool:
        print("[FINAL SCP] running final consolidation", flush=True)

        scp_solver_name = _select_scp_solver_name(rng, scp_solvers, scp_switch_prob)
        solve_scp = lazy_import_scp(scp_solver_name)
        print(f"[FINAL SCP] solver={scp_solver_name}", flush=True)

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
            print(f"[FINAL IMPROVED] best_cost={best_cost}", flush=True)
        else:
            print(f"[FINAL NO-IMPROVEMENT] cost={final_cost} (best={best_cost})", flush=True)

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
            pool_tags.append(("unknown", "unknown", "unknown"))
        else:
            pool_tags.append((str(t.get("mode")), str(t.get("method")), str(t.get("stage"))))

    counter = Counter(pool_tags)

    print("\n[ROUTE SUMMARY]")
    for (mode, method, stage), count in counter.items():
        print(f"  {count:4d} routes | {mode.upper()} | {method} | stage={stage}")

    # Print final returned routes in sol-like format, with tags.
    print("\n" + "-" * 80)
    print("[FINAL ROUTES WITH TAGS]")

    if best_routes:
        for i, r in enumerate(best_routes, 1):
            body = [n for n in r if n != 1]
            tag = route_tags.get(_route_key(r, depot_id=1))
            if tag is None:
                tag_str = "mode=UNKNOWN method=UNKNOWN stage=UNKNOWN"
            else:
                tag_str = (
                    f"mode={str(tag.get('mode')).upper()} "
                    f"method={tag.get('method')} "
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
        instance_name="XLTEST-n2541-k62.vrp",
        seed=1,
        scp_solvers=["gurobi_mip"],
        scp_switch_prob=0.0,
        time_limit_total=300.0,
        time_limit_scp=600.0,
        scp_every=2,
        min_cluster_size=300,
        max_cluster_size=2500,
        routing_solver="pyvrp",
        routing_solver_options=None,
        bks_cost=101100,
    )

    print("\n[DEBUG] Best cost:", res["best_cost"])
    print("[DEBUG] #routes:", len(res["routes"]))
    print("[DEBUG] Iterations:", res["iterations"])
