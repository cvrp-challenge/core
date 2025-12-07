"""
run_drsci.py

Unified DRSCI pipeline with consistent clustering method naming.

Supports:
    - Any clustering method for customer-based CD (custom_* , sk_* , pyclust_* , fcm)
    - Fast package-based clustering methods for RBD (sk_* , pyclust_* , fcm)
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
# Imports
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

# --- Customer-based decomposition ---
K_CD = 9
CD_METHOD = "sk_ac_avg"   # free choice: custom_*, sk_*, pyclust_*, fcm

# --- Route-based decomposition (fast methods only) ---
RBD_K = K_CD
RBD_METHOD = "sk_ac_avg"   # free choice: sk_*, pyclust_*, fcm
RBD_USE_ANGLE = True
RBD_USE_LOAD = True

# --- Local Search ---
LS_NEIGHBOURHOOD = "dri_spatial"
LS_MAX_NEIGHBOURS = 40

# --- SCP ---
SCP_TIME_LIMIT = 60.0

# --- Convergence ---
IMPROVEMENT_TOL_REL = 1e-3


# ==========================================================
# Validation of method names for RBD
# ==========================================================

FAST_RBD_METHODS = {
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
    "fcm",
    "pyclust_k_medoids",
}

def _validate_rbd_method(method: str):
    if method not in FAST_RBD_METHODS:
        raise ValueError(
            f"RBD_METHOD '{method}' is not allowed.\n"
            f"RBD must use FAST methods only:\n"
            f"{FAST_RBD_METHODS}\n\n"
            f"Your slow methods (custom_* , custom_k_medoids) can be used only for CD."
        )


# ==========================================================
# PUBLIC API: drsci_solve()
# ==========================================================

def drsci_solve(instance: str, max_iters: int = 10, seed: int = 0):
    """
    Run the DRSCI algorithm with full control over clustering methods.
    """

    _validate_rbd_method(RBD_METHOD)

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
            # Customer-based decomposition
            clusters, _ = run_clustering(CD_METHOD, instance, K_CD)
        else:
            # Route-based decomposition
            clusters = route_based_decomposition(
                instance_name=instance,
                global_routes=global_routes,
                k=RBD_K,
                method=RBD_METHOD,
                use_angle=RBD_USE_ANGLE,
                use_load=RBD_USE_LOAD,
            )

        # -------------------------------------------------------------
        # 2) Routing of clusters
        # -------------------------------------------------------------
        routing = solve_clusters_with_pyvrp(
            instance_name=instance,
            clusters=clusters,
            time_limit_per_cluster=5.0,
            seed=seed,
        )
        routes_routing = routing["routes"]

        # -------------------------------------------------------------
        # 3) Global Local Search
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
        # 4) SCP
        # -------------------------------------------------------------
        scp = solve_scp(
            instance_name=instance,
            route_pool=routes_ls1,
            time_limit=SCP_TIME_LIMIT,
            verbose=False,
        )
        selected_routes = scp.get("selected_routes") or routes_ls1

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
        routes_clean = dup["routes"]

        # -------------------------------------------------------------
        # 6) Final LS
        # -------------------------------------------------------------
        ls2 = improve_with_local_search(
            instance_name=instance,
            routes_vrplib=routes_clean,
            neighbourhood=LS_NEIGHBOURHOOD,
            max_neighbours=LS_MAX_NEIGHBOURS,
            seed=seed,
        )
        global_routes = ls2["routes_improved"]
        current_cost = ls2["improved_cost"]

        # -------------------------------------------------------------
        # 7) Convergence & best solution tracking
        # -------------------------------------------------------------
        if current_cost < best_cost:
            best_cost = current_cost
            best_routes = [list(r) for r in global_routes]

        if prev_cost < float("inf"):
            rel_improvement = (prev_cost - current_cost) / max(prev_cost, 1e-9)
            if rel_improvement < IMPROVEMENT_TOL_REL:
                break

        prev_cost = current_cost

    total_time = time.time() - t0

    return {
        "best_cost": best_cost,
        "routes": best_routes,
        "iterations": it + 1,
        "runtime": total_time,
    }
