"""
Iterative DRSCI-style pipeline for a single large instance.

This version uses a FIXED number of clusters |C| = 2 for all iterations.
This improves stability and avoids extremely large clusters that can lead
to infeasible SCP coverage (e.g., missing customers in route pool).

DRSCI loop:
  - choose decomposition method (vertex-based or route-based)
  - cluster into |C| = 2 clusters
  - solve cluster routing via PyVRP
  - add routes to global pool
  - run SCP
  - duplicate removal + global LS
  - update best solution
  - stop after 10*N seconds or 30 no-improvement iterations
"""

import sys
import os
import time
import math
import random
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
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.clustering.route_based import route_based_decomposition
from utils.loader import load_instance


def lazy_import_scp():
    from master.setcover.scp_solver import solve_scp
    return solve_scp


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INSTANCE = "X-n502-k39.vrp"

# Vertex-based clustering methods used for diversification
VB_CLUSTER_METHODS = [
    "sk_ac_avg",
    "sk_kmeans",
    "fcm",
    "pyclust_k_medoids",
    "sk_ac_complete",
    "sk_ac_min",
]

SEED = 0
random.seed(SEED)

# Routing and LS parameters
TIME_LIMIT_PER_CLUSTER = 40.0          # increased due to cluster size ~250
LS_NEIGHBOURHOOD = "dri_spatial"
MAX_NEIGHBOURS_LS = 40

# DRSCI iteration stopping rules
MAX_NO_IMPROVE = 30
PROB_VERTEX_BASED = 0.5                # 50% chance for VB / RB

# FIXED number of clusters per iteration
FIXED_C = 2


# =====================================================================
# ========================   HELPER FUNCTIONS   ========================
# =====================================================================

def extract_kmax_from_filename(filename: str):
    """
    Extracts Kmax (#vehicles) from VRPLIB filename pattern '-kXX'.
    """
    match = re.search(r'-k(\d+)', filename)
    if not match:
        raise ValueError(f"No '-kXX' part found in filename: {filename}")
    return int(match.group(1))


def deduplicate_routes(route_pool):
    seen = set()
    unique = []
    for r in route_pool:
        tup = tuple(r)
        if tup not in seen:
            seen.add(tup)
            unique.append(r)
    return unique


# =====================================================================
# ============================   MAIN   ================================
# =====================================================================

def main():
    solve_scp = lazy_import_scp()

    print(f"\n=== DRSCI-STYLE ITERATIVE RUN on {INSTANCE} ===")

    # ---------------------------------------------------------
    # Step 1: Try reading K_MAX from filename
    # ---------------------------------------------------------
    try:
        K_MAX_filename = extract_kmax_from_filename(INSTANCE)
        print(f"K_MAX extracted from filename: {K_MAX_filename}")
    except ValueError:
        K_MAX_filename = None
        print("Filename does not contain -kXX.")

    # ---------------------------------------------------------
    # Step 2: Load instance and compute N, q, total demand
    # ---------------------------------------------------------
    inst = load_instance(INSTANCE)

    # VRPLIB structure in your project:
    # inst["demand"]     = dict {node_id: demand}
    # inst["capacity"]   = vehicle capacity
    demands = inst["demand"]
    q = inst["capacity"]

    # depot is node 0 → customers = all except 0
    N = len(demands) - 1
    total_demand = sum(demands[i] for i in demands if i != 0)

    print(f"Number of customers N: {N}")
    print(f"Vehicle capacity q   : {q}")
    print(f"Total demand         : {total_demand}")

    # If filename gave us K_MAX, use it; otherwise fallback
    if K_MAX_filename is not None:
        K_MAX = K_MAX_filename
    else:
        K_MAX = math.ceil(total_demand / q)

    print(f"Using K_MAX = {K_MAX}")

    # ---------------------------------------------------------
    # Step 3: DRSCI global runtime limit = 10 * N
    # ---------------------------------------------------------
    TOTAL_TIME_LIMIT = 10.0 * N
    print(f"Total runtime limit (10*N): {TOTAL_TIME_LIMIT:.1f}s\n")

    # ---------------------------------------------------------
    # FIXED cluster size per iteration
    # ---------------------------------------------------------
    print(f"Using FIXED number of clusters per iteration: |C| = {FIXED_C}")
    print("This avoids huge clusters and SCP failures.\n")

    # ---------------------------------------------------------
    # DRSCI state
    # ---------------------------------------------------------
    global_route_pool = []
    best_routes = None
    best_cost = float("inf")
    no_improve = 0
    iteration = 0

    start_time = time.time()

    # ---------------------------------------------------------
    # Main DRSCI Loop
    # ---------------------------------------------------------
    while True:
        elapsed = time.time() - start_time
        if elapsed >= TOTAL_TIME_LIMIT:
            print("\n[STOP] Global time limit reached.")
            break
        if no_improve >= MAX_NO_IMPROVE:
            print("\n[STOP] No improvement for too long.")
            break

        iteration += 1
        print("\n" + "=" * 70)
        print(f"=== Iteration {iteration} ===")
        print("=" * 70)

        current_K = FIXED_C
        print(f"[Iter {iteration}] Using |C| = {current_K}")

        # -----------------------------------------------------
        # Step 1: Choose decomposition type
        # -----------------------------------------------------
        if best_routes is not None and random.random() >= PROB_VERTEX_BASED:
            # ROUTE-BASED
            print(f"[Iter {iteration}] Decomposition: ROUTE-BASED")

            clusters = route_based_decomposition(
                instance_name=INSTANCE,
                global_routes=best_routes,
                k=current_K,
                method="sk_ac_avg",
                use_angle=True,
                use_load=True,
            )
        else:
            # VERTEX-BASED
            method = random.choice(VB_CLUSTER_METHODS)
            print(f"[Iter {iteration}] Decomposition: VERTEX-BASED ({method})")

            clusters, medoids = run_clustering(
                method,
                INSTANCE,
                current_K,
            )

        # stats
        total_assigned = sum(len(members) for members in clusters.values())
        print(f"[Iter {iteration}] #clusters={len(clusters)}, customers assigned={total_assigned}")

        # -----------------------------------------------------
        # Step 2: Solve each cluster via PyVRP
        # -----------------------------------------------------
        print(f"\n[Iter {iteration}] Stage 2: Routing subproblems")

        routing = solve_clusters_with_pyvrp(
            instance_name=INSTANCE,
            clusters=clusters,
            time_limit_per_cluster=TIME_LIMIT_PER_CLUSTER,
            seed=SEED,
        )

        routes = routing["routes"]
        print(f"[Iter {iteration}] Cluster routes total: {len(routes)}")
        print(f"[Iter {iteration}] Cluster routing cost: {routing['total_cost']}")

        # -----------------------------------------------------
        # Step 3: Add to global route pool
        # -----------------------------------------------------
        global_route_pool.extend(routes)
        global_route_pool = deduplicate_routes(global_route_pool)
        print(f"[Iter {iteration}] Global route pool size = {len(global_route_pool)}")

        # -----------------------------------------------------
        # Step 4: SCP on pooled routes
        # -----------------------------------------------------
        print(f"\n[Iter {iteration}] Stage 3: SCP")

        scp = solve_scp(
            instance_name=INSTANCE,
            route_pool=global_route_pool,
            time_limit=600,
            verbose=False,
        )
        selected = scp["selected_routes"]
        print(f"[Iter {iteration}] SCP selected {len(selected)} routes")

        # -----------------------------------------------------
        # Step 5: Duplicate removal + LS repair
        # -----------------------------------------------------
        print(f"\n[Iter {iteration}] Stage 4: Duplicate removal + LS repair")

        repaired = remove_duplicates(
            instance_name=INSTANCE,
            routes=selected,
            verbose=False,
            max_iters=50,
            ls_neighbourhood=LS_NEIGHBOURHOOD,
            ls_max_neighbours_restricted=MAX_NEIGHBOURS_LS,
            seed=SEED,
        )["routes"]

        print(f"[Iter {iteration}] Repaired routes: {len(repaired)}")

        # -----------------------------------------------------
        # Step 6: Global LS
        # -----------------------------------------------------
        print(f"\n[Iter {iteration}] Stage 5: Global LS")

        ls_res = improve_with_local_search(
            instance_name=INSTANCE,
            routes_vrplib=repaired,
            neighbourhood=LS_NEIGHBOURHOOD,
            max_neighbours=MAX_NEIGHBOURS_LS,
            seed=SEED,
        )

        improved_cost = ls_res["improved_cost"]
        improved_routes = ls_res["routes_improved"]

        print(f"[Iter {iteration}] LS improved cost: {improved_cost}")

        # -----------------------------------------------------
        # Step 7: Update best solution
        # -----------------------------------------------------
        if improved_cost < best_cost:
            delta = best_cost - improved_cost if best_cost < float("inf") else None
            best_cost = improved_cost
            best_routes = improved_routes
            no_improve = 0

            if delta is None:
                print(f"[Iter {iteration}] New best solution = {best_cost}")
            else:
                print(f"[Iter {iteration}] Improved by {delta:.2f}, new best = {best_cost}")
        else:
            no_improve += 1
            print(f"[Iter {iteration}] No improvement ({no_improve}/{MAX_NO_IMPROVE})")

        print(f"[Iter {iteration}] Current best: {best_cost}, routes={len(best_routes) if best_routes else 0}")
        print(f"[Iter {iteration}] Time elapsed: {time.time()-start_time:.1f}s\n")

    # ---------------------------------------------------------
    # FINAL OUTPUT
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)

    if best_routes is None:
        print("No feasible solution found.")
    else:
        print(f"Best cost   : {best_cost}")
        print(f"Route count : {len(best_routes)}")
        for r in best_routes:
            print(" ", r)

    print("\n=== END ===")


if __name__ == "__main__":
    main()
