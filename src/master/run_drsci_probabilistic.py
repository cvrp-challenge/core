from __future__ import annotations

import time
import random
from typing import Dict, List, Optional, Any, Mapping
import math

import sys
from pathlib import Path

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
    """
    Formats cluster sizes in a way that matches what routing_controller will see:
    - customers only (exclude depot_id)
    """
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
]  # |VB| = 6

RB_METHODS = [
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
]  # |RB| = 4


Route = List[int]
Routes = List[Route]


# ============================================================
# HELPERS
# ============================================================

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
    k_min: int = 2,
    k_max: int = 8,
    routing_solver: str = "pyvrp",
    routing_solver_options: Optional[Mapping[str, Any]] = None,
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 100,
    ls_max_neighbours_restricted: int = 100,
    randomize_polar_angle: bool = True,
) -> Dict[str, Any]:
    """
    Probabilistic DRSCI driver.

    Important design choice:
      - This file does NOT implement any solver-specific stall / stagnation logic.
      - PyVRP stagnation stopping is NOT attempted here (and should not be).
      - If a routing backend supports stall_time (e.g. Hexaly), that should be handled
        inside routing_controller/solver adapters — this driver stays clean.
    """

    if scp_every <= 0:
        raise ValueError("scp_every must be >= 1")
    if k_min < 2 or k_max < k_min:
        raise ValueError("Require k_min >= 2 and k_max >= k_min")

    rng = random.Random(seed)
    start_time = time.time()

    inst = load_instance(instance_name)

    global_route_pool: Routes = []
    best_routes: Optional[Routes] = None
    best_cost = float("inf")

    no_improvement_iters = 0
    iteration = 0

    routing_solver_key = routing_solver.lower()
    routing_solver_options = {
        "use_stall": False   # disables Hexaly stalling
    }
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
        k = rng.randint(k_min, k_max)

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
            solver_options=(routing_solver_options if routing_solver_key != "pyvrp" else None),
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

        # Update incumbent immediately if VB/RB candidate improves
        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_routes = candidate_routes
            improved_this_iter = True
            print(f"[IMPROVED-VB/RB] best_cost={best_cost}", flush=True)

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

            if scp_cost < best_cost:
                best_cost = scp_cost
                best_routes = scp_routes
                improved_this_iter = True
                print(f"[IMPROVED-SCP] best_cost={best_cost}", flush=True)
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

        if final_cost < best_cost:
            best_cost = final_cost
            best_routes = final_routes
            print(f"[FINAL IMPROVED] best_cost={best_cost}", flush=True)
        else:
            print(f"[FINAL NO-IMPROVEMENT] cost={final_cost} (best={best_cost})", flush=True)

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
        scp_solvers=["gurobi_mip", "hexaly"],
        scp_switch_prob=0.5,
        time_limit_total=300.0,
        time_limit_scp=600.0,
        scp_every=2,
        k_min=2,
        k_max=4,
        routing_solver="hexaly",
        routing_solver_options=None,
    )

    print("\n[DEBUG] Best cost:", res["best_cost"])
    print("[DEBUG] #routes:", len(res["routes"]))
    print("[DEBUG] Iterations:", res["iterations"])
