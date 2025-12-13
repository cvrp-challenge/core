"""
Multi-K DRSCI for CVRP (Robin Version)
--------------------------------------

Pipeline for each vertex-based method m and cluster count k:

    VB STEP (m, k)
        1) Vertex-based clustering
        2) Route VB clusters with PyVRP
        3) Local Search on routed VB solution        <- DRI-style LS (immediate)
        4) Add VB routes to global pool
        5) SCP + duplicate removal + LS (global refine)
           -> update best_routes / best_cost
           -> add repaired routes back to pool

    RBD STEP (m, k)
        6) Route-based decomposition on current best solution
        7) Route RBD clusters with PyVRP
        8) Local Search on routed RBD solution       <- DRI-style LS (immediate)
        9) Add RBD routes to global pool
       10) SCP + duplicate removal + LS (global refine)
           -> update best_routes / best_cost
           -> add repaired routes back to pool

After all methods and all k's:

    FINAL STEP
       11) Global SCP + duplicate removal + LS
       12) Final best solution

This is a multi-granularity DRSCI system with:
    - multiple k values per vertex-based method,
    - immediate LS after each routing phase (DRI-style),
    - global SCP-based refinement,
    - route pool enrichment with repaired routes.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------
CURRENT = os.path.dirname(__file__)          # core/src/master
ROOT = os.path.abspath(os.path.join(CURRENT, "."))        # core/src/master
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))  # core/src

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
from master.clustering.run_clustering import run_clustering
from master.clustering.route_based import route_based_decomposition
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.utils.loader import load_instance


def lazy_import_scp():
    from master.setcover.scp_solver import solve_scp
    return solve_scp


Route = List[int]
Routes = List[Route]


# ---------------------------------------------------------
# Vertex-based methods in deterministic sequence
# ---------------------------------------------------------
VB_CLUSTER_METHODS = [
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
    "fcm",
    "pyclust_k_medoids",
]


# ---------------------------------------------------------
# Default multi-k configuration (customize via k_per_method)
# All k values should come from {2, 4, 6, 9, 15}.
# ---------------------------------------------------------
K_PER_METHOD_DEFAULT: Dict[str, List[int]] = {
    "sk_ac_avg": [4, 6],
    "sk_ac_complete": [4],
    "sk_ac_min": [6],
    "sk_kmeans": [4, 6],
    "fcm": [6],
    "pyclust_k_medoids": [4, 6],
}


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def deduplicate_routes(route_pool: Routes) -> Routes:
    """
    Deduplicate a list of routes (each route is a list[int]).
    """
    seen = set()
    unique: Routes = []
    for r in route_pool:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def add_repaired_routes_to_pool(pool: Routes, repaired: Routes) -> Routes:
    """
    Add repaired routes (from SCP + duplicate removal + LS) back into
    the global route pool, only if they are new.
    """
    seen = set(tuple(r) for r in pool)
    added = 0

    for r in repaired:
        key = tuple(r)
        if key not in seen:
            pool.append(r)
            seen.add(key)
            added += 1

    if added > 0:
        print(f"[Pool] Added {added} repaired routes to global pool.")

    return pool


def _scp_plus_ls(
    *,
    instance_name: str,
    route_pool: Routes,
    solve_scp,
    ls_neighbourhood: str,
    ls_max_neighbours_restricted: int,
    seed: int,
    scp_time_limit: float,
) -> tuple[Routes, float]:
    """
    Run:
        - SCP on the given route_pool,
        - duplicate removal + LS repair,
        - global LS refine.

    Returns (routes, cost) of the refined solution.
    """
    scp_res = solve_scp(
        instance_name=instance_name,
        route_pool=route_pool,
        time_limit=scp_time_limit,
        verbose=False,
    )
    selected = scp_res["selected_routes"]

    # Duplicate removal + repair LS
    dup_res = remove_duplicates(
        instance_name=instance_name,
        routes=selected,
        verbose=False,
        max_iters=50,
        ls_neighbourhood=ls_neighbourhood,
        ls_max_neighbours_restricted=ls_max_neighbours_restricted,
        seed=seed,
    )
    repaired = dup_res["routes"]

    # Global LS refine
    ls_res = improve_with_local_search(
        instance_name=instance_name,
        routes_vrplib=repaired,
        neighbourhood=ls_neighbourhood,
        max_neighbours=ls_max_neighbours_restricted,
        seed=seed,
    )

    routes_final = ls_res["routes_improved"]
    cost_final = ls_res["improved_cost"]

    return routes_final, cost_final


# =====================================================================
# DRSCI solver with multi-k + DRI-style LS after routing
# =====================================================================
def drsci_solve(
    *,
    instance: str,
    seed: int = 0,
    max_iters: int = 1,  # kept for compatibility; not used
    time_limit_per_cluster: float = 20.0,
    ls_neighbourhood: str = "dri_spatial",
    ls_after_routing_max_neighbours: int = 40,
    ls_max_neighbours_restricted: int = 40,
    scp_time_limit: float = 600.0,
    k_per_method: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Multi-K DRSCI main entry point.

    Parameters
    ----------
    instance : str
        VRPLIB instance name, e.g. "XLTEST-n2541-k62.vrp".
    seed : int
        Random seed for PyVRP and LS.
    time_limit_per_cluster : float
        Time limit per cluster-subproblem for PyVRP.
    ls_neighbourhood : str
        Neighbourhood name passed to improve_with_local_search.
    ls_after_routing_max_neighbours : int
        max_neighbours for LS immediately after routing (DRI step).
    ls_max_neighbours_restricted : int
        max_neighbours for LS in SCP+repair refinement.
    scp_time_limit : float
        Time limit in seconds per SCP solve.
    k_per_method : dict[str, list[int]], optional
        Mapping method -> list of k values (e.g. [4, 6]).
        If None, K_PER_METHOD_DEFAULT is used.
    """
    start_time = time.time()
    solve_scp = lazy_import_scp()

    # -----------------------------------------------------
    # Resolve k-per-method configuration
    # -----------------------------------------------------
    if k_per_method is None:
        k_per_method = {m: list(K_PER_METHOD_DEFAULT[m]) for m in K_PER_METHOD_DEFAULT}

    for m in VB_CLUSTER_METHODS:
        if m not in k_per_method:
            raise ValueError(f"Missing k list for method '{m}'.")
        k_list = k_per_method[m]
        if not isinstance(k_list, list) or len(k_list) == 0:
            raise ValueError(f"k_per_method['{m}'] must be a non-empty list.")
        if any(k < 2 for k in k_list):
            raise ValueError(f"All k values for method '{m}' must be >= 2.")

    print(f"\n=== DRSCI Multi-K RUN on {instance} ===")
    print(f"VB methods                  : {VB_CLUSTER_METHODS}")
    print(f"k per method                : {k_per_method}")
    print(f"Routing TL per cluster      : {time_limit_per_cluster}s")
    print(f"LS after routing (max nnbs) : {ls_after_routing_max_neighbours}")
    print(f"SCP time limit              : {scp_time_limit}s")
    print(f"LS refine max neighbours    : {ls_max_neighbours_restricted}")
    print(f"Seed                        : {seed}")
    print("------------------------------------------------------------")

    inst = load_instance(instance)
    dim = int(inst["dimension"])
    print(f"Instance dimension          : {dim}")

    # -----------------------------------------------------
    # Global DRSCI state
    # -----------------------------------------------------
    global_route_pool: Routes = []
    best_routes: Optional[Routes] = None
    best_cost: float = float("inf")
    stages_executed = 0

    # -----------------------------------------------------
    # MAIN LOOP: for each VB method and each k in its list
    # -----------------------------------------------------
    for method in VB_CLUSTER_METHODS:
        print("\n" + "=" * 70)
        print(f"=== METHOD: {method} ===")
        print("=" * 70)

        for k in k_per_method[method]:
            print("\n" + "-" * 70)
            print(f"[VB] method={method}, k={k}")
            print("-" * 70)
            stages_executed += 1

            # =================================================
            # (1) Vertex-based clustering
            # =================================================
            clusters_vb, _ = run_clustering(
                method=method,
                instance_name=instance,
                k=k,
            )
            total_assigned = sum(len(v) for v in clusters_vb.values())
            print(f"[VB] #clusters={len(clusters_vb)}, assigned_customers={total_assigned}")

            # =================================================
            # (2) Route VB clusters with PyVRP
            # =================================================
            routing_vb = solve_clusters_with_pyvrp(
                instance_name=instance,
                clusters=clusters_vb,
                time_limit_per_cluster=time_limit_per_cluster,
                seed=seed,
            )
            routes_vb = routing_vb["routes"]
            print(f"[VB] Routed {len(routes_vb)} routes, cost={routing_vb['total_cost']}")

            # =================================================
            # (3) LS immediately after VB routing (DRI-style)
            # =================================================
            ls_vb = improve_with_local_search(
                instance_name=instance,
                routes_vrplib=routes_vb,
                neighbourhood=ls_neighbourhood,
                max_neighbours=ls_after_routing_max_neighbours,
                seed=seed,
            )
            improved_vb_routes = ls_vb["routes_improved"]
            print(f"[VB] LS-after-routing cost={ls_vb['improved_cost']}")

            # =================================================
            # (4) Add improved VB routes to global route pool
            # =================================================
            global_route_pool.extend(improved_vb_routes)
            global_route_pool = deduplicate_routes(global_route_pool)
            print(f"[VB] Global pool size after VB add = {len(global_route_pool)}")

            # =================================================
            # (5) SCP + duplicate removal + LS refine (global)
            # =================================================
            vb_routes_final, vb_cost = _scp_plus_ls(
                instance_name=instance,
                route_pool=global_route_pool,
                solve_scp=solve_scp,
                ls_neighbourhood=ls_neighbourhood,
                ls_max_neighbours_restricted=ls_max_neighbours_restricted,
                seed=seed,
                scp_time_limit=scp_time_limit,
            )
            print(f"[VB] Candidate post-SCP cost = {vb_cost}")

            # --- enrich pool with repaired VB routes
            global_route_pool = add_repaired_routes_to_pool(global_route_pool, vb_routes_final)
            global_route_pool = deduplicate_routes(global_route_pool)
            print(f"[VB] Pool size after adding repaired VB routes = {len(global_route_pool)}")

            # --- update best solution
            if vb_cost < best_cost:
                delta = best_cost - vb_cost if best_cost < float("inf") else 0.0
                best_cost = vb_cost
                best_routes = vb_routes_final
                print(f"[VB] NEW BEST solution: cost={best_cost:.2f} (Δ={delta:.2f})")
            else:
                print(f"[VB] No improvement over current best={best_cost:.2f}")

            # =================================================
            # RBD step only if we already have a best solution
            # =================================================
            if best_routes is None:
                print("[RB] Skipping RBD – no best solution available yet.")
                continue

            stages_executed += 1
            print("\n[RB] Route-based decomposition step")
            print(f"[RB] k={k}, method='sk_kmeans'")

            # (6) Route-based decomposition on best_routes
            clusters_rb = route_based_decomposition(
                instance_name=instance,
                global_routes=best_routes,
                k=k,
                method="sk_kmeans",
                use_angle=True,
                use_load=True,
            )
            total_assigned_rb = sum(len(v) for v in clusters_rb.values())
            print(f"[RB] #RBD clusters={len(clusters_rb)}, assigned_customers={total_assigned_rb}")

            # (7) Route RBD clusters
            routing_rb = solve_clusters_with_pyvrp(
                instance_name=instance,
                clusters=clusters_rb,
                time_limit_per_cluster=time_limit_per_cluster,
                seed=seed,
            )
            routes_rb = routing_rb["routes"]
            print(f"[RB] Routed {len(routes_rb)} routes, cost={routing_rb['total_cost']}")

            # (8) LS immediately after RBD routing (DRI-style)
            ls_rb = improve_with_local_search(
                instance_name=instance,
                routes_vrplib=routes_rb,
                neighbourhood=ls_neighbourhood,
                max_neighbours=ls_after_routing_max_neighbours,
                seed=seed,
            )
            improved_rb_routes = ls_rb["routes_improved"]
            print(f"[RB] LS-after-routing cost={ls_rb['improved_cost']}")

            # (9) Add improved RBD routes to global pool
            global_route_pool.extend(improved_rb_routes)
            global_route_pool = deduplicate_routes(global_route_pool)
            print(f"[RB] Global pool size after RBD add = {len(global_route_pool)}")

            # (10) SCP + duplicate removal + LS refine
            rb_routes_final, rb_cost = _scp_plus_ls(
                instance_name=instance,
                route_pool=global_route_pool,
                solve_scp=solve_scp,
                ls_neighbourhood=ls_neighbourhood,
                ls_max_neighbours_restricted=ls_max_neighbours_restricted,
                seed=seed,
                scp_time_limit=scp_time_limit,
            )
            print(f"[RB] Candidate post-SCP cost = {rb_cost}")

            # --- enrich pool with repaired RBD routes
            global_route_pool = add_repaired_routes_to_pool(global_route_pool, rb_routes_final)
            global_route_pool = deduplicate_routes(global_route_pool)
            print(f"[RB] Pool size after adding repaired RBD routes = {len(global_route_pool)}")

            # --- update best solution
            if rb_cost < best_cost:
                delta = best_cost - rb_cost
                best_cost = rb_cost
                best_routes = rb_routes_final
                print(f"[RB] NEW BEST solution: cost={best_cost:.2f} (Δ={delta:.2f})")
            else:
                print(f"[RB] No improvement over current best={best_cost:.2f}")

    # ---------------------------------------------------------
    # FINAL GLOBAL SCP + LS over full pool
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== FINAL GLOBAL SCP + LS ===")
    print("=" * 70)

    if not global_route_pool:
        print("[Final] WARNING: global route pool is empty – no solution.")
        runtime = time.time() - start_time
        return {
            "instance": instance,
            "best_cost": float("inf"),
            "routes": [],
            "runtime": runtime,
            "iterations": stages_executed,
            "route_pool_size": 0,
        }

    final_routes, final_cost = _scp_plus_ls(
        instance_name=instance,
        route_pool=global_route_pool,
        solve_scp=solve_scp,
        ls_neighbourhood=ls_neighbourhood,
        ls_max_neighbours_restricted=ls_max_neighbours_restricted,
        seed=seed,
        scp_time_limit=scp_time_limit,
    )
    print(f"[Final] Final SCP+LS cost = {final_cost:.2f}")

    if final_cost < best_cost:
        print(f"[Final] Improved best from {best_cost:.2f} → {final_cost:.2f}")
        best_cost = final_cost
        best_routes = final_routes

    runtime = time.time() - start_time

    print("\n=== DRSCI COMPLETED ===")
    print(f"Best cost        : {best_cost:.2f}")
    print(f"#best routes     : {len(best_routes) if best_routes else 0}")
    print(f"Runtime          : {runtime:.2f}s")
    print(f"Stages executed  : {stages_executed}")
    print(f"Final pool size  : {len(global_route_pool)}")

    return {
        "instance": instance,
        "best_cost": best_cost,
        "routes": best_routes if best_routes is not None else [],
        "runtime": runtime,
        "iterations": stages_executed,
        "route_pool_size": len(global_route_pool),
    }


# ---------------------------------------------------------------------
# Debug entry point (optional)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    result = drsci_solve(
        instance="X-n502-k39.vrp",
        seed=0,
        time_limit_per_cluster=20.0,
        k_per_method={
            "sk_ac_avg": [4],
            "sk_ac_complete": [4],
            "sk_ac_min": [4],
            "sk_kmeans": [4],
            "fcm": [4],
            "pyclust_k_medoids": [4],
        },
    )
    print("\n[DEBUG] Best cost:", result["best_cost"], "#routes:", len(result["routes"]))
