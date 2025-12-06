"""
run_drsci.py

Provides a single callable function:

    drsci_solve(instance, max_iters, seed)

This runs the full DRSCI iterative pipeline including:
- Customer-based decomposition (iteration 0)
- Route-based decomposition (iteration >= 1)
- PyVRP routing of clusters
- Global LS
- Set Covering (SCP)
- Duplicate removal + LS repair
- Final LS refine
"""

import sys
import os
import time

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
CURRENT = os.path.dirname(__file__)        # core/src/master
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
from master.setcover.scp_solver import solve_scp
from master.setcover.duplicate_removal import remove_duplicates


# ==========================================================
# DRSCI CONFIGURATION
# ==========================================================

# Customer-based decomposition (iteration 0)
K_CD = 9
CD_METHOD = "sk_ac_avg"

# Route-based decomposition (iteration >= 1)
RBD_K = K_CD
RBD_METHOD = "ac_avg"
RBD_USE_ANGLE = True
RBD_USE_LOAD = True

# Local Search settings
LS_NEIGHBOURHOOD = "dri_spatial"
LS_MAX_NEIGHBOURS = 40

# SCP settings
SCP_TIME_LIMIT = 60.0  # seconds

# Convergence condition
IMPROVEMENT_TOL_REL = 1e-3


# ==========================================================
# PUBLIC API: drsci_solve()
# ==========================================================

def drsci_solve(instance: str, max_iters: int = 10, seed: int = 0):
    """
    Run the full DRSCI algorithm on a single VRP instance.

    Parameters
    ----------
    instance : str
        VRPLIB instance filename (e.g., "X-n101-k25.vrp")

    max_iters : int
        Maximum number of DRSCI outer-iterations

    seed : int
        Random seed for LS and PyVRP

    Returns
    -------
    dict with keys:
        "best_cost"   : best solution cost
        "routes"      : list of VRPLIB routes
        "iterations"  : number of iterations executed
        "runtime"     : total wall-clock time
    """

    t0 = time.time()

    global_routes = None
    best_routes = None
    best_cost = float("inf")
    prev_cost = float("inf")

    for it in range(max_iters):

        # -------------------------------------------------------------
        # 1) Decomposition step
        # -------------------------------------------------------------
        if it == 0:
            clusters, _ = run_clustering(CD_METHOD, instance, K_CD)
        else:
            clusters = route_based_decomposition(
                instance_name=instance,
                global_routes=global_routes,
                k=RBD_K,
                method=RBD_METHOD,
                use_angle=RBD_USE_ANGLE,
                use_load=RBD_USE_LOAD,
            )

        # -------------------------------------------------------------
        # 2) Routing of all decomposed clusters
        # -------------------------------------------------------------
        routing_result = solve_clusters_with_pyvrp(
            instance_name=instance,
            clusters=clusters,
            time_limit_per_cluster=5.0,
            seed=seed,
        )
        routes_routing = routing_result["routes"]

        # -------------------------------------------------------------
        # 3) Global LS
        # -------------------------------------------------------------
        ls1 = improve_with_local_search(
            instance_name=instance,
            routes_vrplib=routes_routing,
            neighbourhood=LS_NEIGHBOURHOOD,
            max_neighbours=LS_MAX_NEIGHBOURS,
            seed=seed,
        )
        routes_ls1 = ls1["routes_improved"]
        cost_ls1 = ls1["improved_cost"]

        # -------------------------------------------------------------
        # 4) Set Covering Problem (SCP)
        # -------------------------------------------------------------
        scp = solve_scp(
            instance_name=instance,
            route_pool=routes_ls1,
            time_limit=SCP_TIME_LIMIT,
            verbose=False,
        )
        selected_routes = scp.get("selected_routes", None)

        if not selected_routes:
            selected_routes = routes_ls1

        # -------------------------------------------------------------
        # 5) Duplicate removal + LS repair
        # -------------------------------------------------------------
        dup = remove_duplicates(
            instance_name=instance,
            routes=selected_routes,
            verbose=False,
            max_iters=50,
            ls_neighbourhood=LS_NEIGHBOURHOOD,
            ls_max_neighbours_restricted=LS_MAX_NEIGHBOURS,
            seed=seed,
        )
        routes_after_dup = dup["routes"]

        # -------------------------------------------------------------
        # 6) Final LS refine for this iteration
        # -------------------------------------------------------------
        ls2 = improve_with_local_search(
            instance_name=instance,
            routes_vrplib=routes_after_dup,
            neighbourhood=LS_NEIGHBOURHOOD,
            max_neighbours=LS_MAX_NEIGHBOURS,
            seed=seed,
        )
        global_routes = ls2["routes_improved"]
        current_cost = ls2["improved_cost"]

        # -------------------------------------------------------------
        # 7) Update best solution and check convergence
        # -------------------------------------------------------------
        if current_cost < best_cost:
            best_cost = current_cost
            best_routes = [list(r) for r in global_routes]

        if prev_cost < float("inf"):
            rel_impr = (prev_cost - current_cost) / max(prev_cost, 1e-9)
            if rel_impr < IMPROVEMENT_TOL_REL:
                break

        prev_cost = current_cost

    total_time = time.time() - t0

    return {
        "best_cost": best_cost,
        "routes": best_routes,
        "iterations": it + 1,
        "runtime": total_time,
    }
