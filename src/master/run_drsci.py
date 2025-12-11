"""
DRSCI-style iterative decomposition for CVRP instances.

Your customized version:

  • Systematic cluster sizes:
        C_s = [2, 4, 8, 16, 32, ..., K_max]
    (C=1 explicitly excluded.)

  • Decomposition logic:
        For each C in C_s:
            For each vertex-based method (fixed order):
                VB step → RB step → VB step → RB step → ...

        RB = classic route-based clustering based on current best solution.

  • All key parameters (cluster routing TL, LS neighbourhood, LS granularity,
    stopping criteria) are keyword arguments of the run function.

  • No randomness in decomposition (deterministic method ordering).

  • Full customer coverage guaranteed in clustering (handled in run_clustering.py).
"""

import sys
import os
import time
import math
import re

# ---------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------
CURRENT = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(CURRENT, "."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

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
from utils.loader import load_instance


def lazy_import_scp():
    from master.setcover.scp_solver import solve_scp
    return solve_scp


# ---------------------------------------------------------
# Fixed list of VB clustering methods (deterministic order)
# ---------------------------------------------------------
VB_CLUSTER_METHODS = [
    "sk_ac_avg",
    "sk_kmeans",
    "fcm",
    "pyclust_k_medoids",
    "sk_ac_complete",
    "sk_ac_min",
]


# =====================================================================
# Helper functions
# =====================================================================

def extract_kmax_from_filename(filename: str) -> int:
    """
    Extract K_max (number of vehicles in best-known solution) from VRPLIB filename.
    Example: X-n502-k39.vrp → 39
    """
    m = re.search(r"-k(\d+)", filename)
    if not m:
        raise ValueError(f"Could not extract K_max from filename '{filename}'.")
    return int(m.group(1))


def build_cluster_sizes(k_max: int):
    """
    Build systematic cluster sizes:
        C_s = [2, 4, 8, ..., k_max]
    excluding C = 1 explicitly.
    """
    sizes = set()
    c = 2
    while c <= k_max:
        sizes.add(c)
        c *= 2
    sizes.add(k_max)
    result = sorted(sizes)
    return [c for c in result if c >= 2]


def deduplicate_routes(route_pool):
    """
    Deduplicate list of routes (each route is a list[int]).
    """
    seen = set()
    unique = []
    for r in route_pool:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique.append(r)
    return unique


# =====================================================================
# Main DRSCI method (parameterized)
# =====================================================================

def run_drsci_for_instance(
    *,
    instance_name: str,
    time_limit_per_cluster: float = 40.0,
    ls_neighbourhood: str = "dri_spatial",
    max_neighbours_ls: int = 40,
    max_no_improve: int = 50,
    seed: int = 0,
):
    """
    Run DRSCI for a single instance.

    Parameters
    ----------
    instance_name : str
        VRPLIB instance name (e.g., "X-n502-k39.vrp")
    time_limit_per_cluster : float
        Time limit for routing each cluster subproblem with PyVRP.
    ls_neighbourhood : str
        Local-search neighbourhood definition (e.g., "dri_spatial").
    max_neighbours_ls : int
        Granular neighbourhood size in LS (e.g., 40 neighbours per node).
    max_no_improve : int
        Stop if no VB/RB step improves best solution for this many steps.
    seed : int
        Random seed (only affects routing stochasticity, clustering is deterministic).
    """

    solve_scp = lazy_import_scp()

    print(f"\n=== DRSCI RUN on instance {instance_name} ===")

    # ---------------------------------------------------------
    # Load instance and basic metadata
    # ---------------------------------------------------------
    inst = load_instance(instance_name)
    demands = inst["demand"]    # list: index = node id
    q = inst["capacity"]

    # customers = all nodes with demand > 0
    customers = [i for i, d in enumerate(demands) if d > 0]
    N = len(customers)
    total_demand = sum(demands[i] for i in customers)

    # K_max
    try:
        K_MAX = extract_kmax_from_filename(instance_name)
        print(f"K_MAX extracted from filename: {K_MAX}")
    except ValueError:
        K_MAX = math.ceil(total_demand / q)
        print(f"K_MAX computed from demand/capacity: {K_MAX}")

    TOTAL_TIME_LIMIT = 10.0 * N

    print(f"Number of customers N       : {N}")
    print(f"Vehicle capacity q          : {q}")
    print(f"Total demand                : {total_demand}")
    print(f"K_MAX                       : {K_MAX}")
    print(f"Total runtime limit (10*N)  : {TOTAL_TIME_LIMIT:.1f}s")
    print(f"Vertex-based methods        : {VB_CLUSTER_METHODS}")
    print(f"LS neighbourhood            : {ls_neighbourhood}")
    print(f"LS max neighbours           : {max_neighbours_ls}")
    print(f"Routing TL per cluster      : {time_limit_per_cluster}s")
    print(f"Stopping rule (no improve)  : {max_no_improve}\n")

    # Build cluster size schedule
    cluster_sizes = build_cluster_sizes(K_MAX)
    print(f"Systematic C_s = {cluster_sizes}\n")

    # ---------------------------------------------------------
    # Global DRSCI state
    # ---------------------------------------------------------
    global_route_pool = []
    best_routes = None
    best_cost = float("inf")
    no_improve = 0
    step_counter = 0

    start_time = time.time()

    # Outer loop: over cluster sizes C in C_s
    for C in cluster_sizes:
        print("\n" + "=" * 70)
        print(f"=== Cluster size C = {C} ===")
        print("=" * 70)

        # For each VB clustering method once
        for vb_method in VB_CLUSTER_METHODS:

            # Check stopping
            elapsed = time.time() - start_time
            if elapsed >= TOTAL_TIME_LIMIT:
                print("\n[STOP] Time limit reached.")
                return best_routes
            if no_improve >= max_no_improve:
                print("\n[STOP] max_no_improve steps reached.")
                return best_routes

            # -------------------------------------------------
            # === VB STEP ===
            # -------------------------------------------------
            step_counter += 1
            print("\n" + "-" * 70)
            print(f"[Step {step_counter}] VB: method={vb_method}, C={C}")
            print("-" * 70)

            # --- Clustering (vertex-based)
            clusters_vb, _ = run_clustering(
                vb_method,
                instance_name,
                C,
            )
            total_assigned = sum(len(members) for members in clusters_vb.values())
            print(f"[VB] #clusters={len(clusters_vb)}, customers assigned={total_assigned}")

            # --- Routing
            print("[VB] Routing subproblems...")
            routing_vb = solve_clusters_with_pyvrp(
                instance_name=instance_name,
                clusters=clusters_vb,
                time_limit_per_cluster=time_limit_per_cluster,
                seed=seed,
            )
            routes_vb = routing_vb["routes"]
            print(f"[VB]   routes: {len(routes_vb)}, cost={routing_vb['total_cost']}")

            # Add to pool
            global_route_pool.extend(routes_vb)
            global_route_pool = deduplicate_routes(global_route_pool)
            print(f"[VB] Global route pool now = {len(global_route_pool)} routes")

            # --- SCP
            print("[VB] Solving SCP...")
            scp_vb = solve_scp(
                instance_name=instance_name,
                route_pool=global_route_pool,
                time_limit=600,
                verbose=False,
            )
            selected_vb = scp_vb["selected_routes"]
            print(f"[VB] SCP selected {len(selected_vb)} routes")

            # --- Duplicate removal + LS repair
            print("[VB] Duplicate removal + LS repair...")
            repaired_vb = remove_duplicates(
                instance_name=instance_name,
                routes=selected_vb,
                verbose=False,
                max_iters=50,
                ls_neighbourhood=ls_neighbourhood,
                ls_max_neighbours_restricted=max_neighbours_ls,
                seed=seed,
            )["routes"]

            print(f"[VB] Repaired route count={len(repaired_vb)}")

            # --- Global LS ---
            print("[VB] Global LS...")
            ls_res_vb = improve_with_local_search(
                instance_name=instance_name,
                routes_vrplib=repaired_vb,
                neighbourhood=ls_neighbourhood,
                max_neighbours=max_neighbours_ls,
                seed=seed,
            )

            vb_cost = ls_res_vb["improved_cost"]
            vb_routes = ls_res_vb["routes_improved"]
            print(f"[VB] LS improved cost = {vb_cost}")

            # --- Update best solution ---
            if vb_cost < best_cost:
                delta = best_cost - vb_cost if best_cost < float("inf") else None
                best_cost = vb_cost
                best_routes = vb_routes
                no_improve = 0
                print(f"[VB] New BEST solution (improved by {delta if delta else 0:.2f}) -> {best_cost}")
            else:
                no_improve += 1
                print(f"[VB] No improvement ({no_improve}/{max_no_improve})")

            elapsed = time.time() - start_time
            print(f"[VB] Time elapsed={elapsed:.1f}s / {TOTAL_TIME_LIMIT:.1f}s")

            if elapsed >= TOTAL_TIME_LIMIT or no_improve >= max_no_improve:
                return best_routes

            # -------------------------------------------------
            # === RB STEP ===
            # -------------------------------------------------

            # Skip RB if no meaningful best solution yet
            if best_routes is None:
                continue

            step_counter += 1
            print("\n" + "-" * 70)
            print(f"[Step {step_counter}] RB: classic route-based, C={C}")
            print("-" * 70)

            # --- Route-based clustering on best solution ---
            clusters_rb = route_based_decomposition(
                instance_name=instance_name,
                global_routes=best_routes,
                k=C,
                method="sk_ac_avg",
                use_angle=True,
                use_load=True,
            )

            total_assigned_rb = sum(len(members) for members in clusters_rb.values())
            print(f"[RB] #clusters={len(clusters_rb)}, customers assigned={total_assigned_rb}")

            # --- Routing RB clusters
            print("[RB] Routing subproblems...")
            routing_rb = solve_clusters_with_pyvrp(
                instance_name=instance_name,
                clusters=clusters_rb,
                time_limit_per_cluster=time_limit_per_cluster,
                seed=seed,
            )
            routes_rb = routing_rb["routes"]
            print(f"[RB]   routes: {len(routes_rb)}, cost={routing_rb['total_cost']}")

            # Add to pool
            global_route_pool.extend(routes_rb)
            global_route_pool = deduplicate_routes(global_route_pool)
            print(f"[RB] Global route pool now = {len(global_route_pool)} routes")

            # --- SCP ---
            print("[RB] Solving SCP...")
            scp_rb = solve_scp(
                instance_name=instance_name,
                route_pool=global_route_pool,
                time_limit=600,
                verbose=False,
            )
            selected_rb = scp_rb["selected_routes"]
            print(f"[RB] SCP selected {len(selected_rb)} routes")

            # --- Duplicate removal + LS repair
            print("[RB] Duplicate removal + LS repair...")
            repaired_rb = remove_duplicates(
                instance_name=instance_name,
                routes=selected_rb,
                verbose=False,
                max_iters=50,
                ls_neighbourhood=ls_neighbourhood,
                ls_max_neighbours_restricted=max_neighbours_ls,
                seed=seed,
            )["routes"]

            print(f"[RB] Repaired route count={len(repaired_rb)}")

            # --- Global LS ---
            print("[RB] Global LS...")
            ls_res_rb = improve_with_local_search(
                instance_name=instance_name,
                routes_vrplib=repaired_rb,
                neighbourhood=ls_neighbourhood,
                max_neighbours=max_neighbours_ls,
                seed=seed,
            )

            rb_cost = ls_res_rb["improved_cost"]
            rb_routes = ls_res_rb["routes_improved"]
            print(f"[RB] LS improved cost = {rb_cost}")

            # --- Update best solution ---
            if rb_cost < best_cost:
                delta = best_cost - rb_cost
                best_cost = rb_cost
                best_routes = rb_routes
                no_improve = 0
                print(f"[RB] New BEST solution (improved by {delta:.2f}) -> {best_cost}")
            else:
                no_improve += 1
                print(f"[RB] No improvement ({no_improve}/{max_no_improve})")

            elapsed = time.time() - start_time
            print(f"[RB] Time elapsed={elapsed:.1f}s / {TOTAL_TIME_LIMIT:.1f}s")

            if elapsed >= TOTAL_TIME_LIMIT or no_improve >= max_no_improve:
                return best_routes

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 70)
    print("=== FINAL DRSCI SUMMARY ===")
    print("=" * 70)

    if best_routes is not None:
        print(f"Best solution cost   : {best_cost}")
        print(f"Best number of routes: {len(best_routes)}")
    else:
        print("No feasible solution found.")

    print("\n=== END ===")

    return best_routes



# =====================================================================
# Example direct call 
# (Use for debugging; replace with argparse in your experiment runner.)
# =====================================================================
if __name__ == "__main__":
    run_drsci_for_instance(
        instance_name="X-n502-k39.vrp",
        time_limit_per_cluster=40.0,
        ls_neighbourhood="dri_spatial",
        max_neighbours_ls=40,
        max_no_improve=50,
        seed=0,
    )
