"""
run_scp_and_repair.py

Testing pipeline for:
    1) Clustering (customer decomposition)
    2) Routing each cluster via PyVRP
    3) Global LS improve
    4) SCP (optional)
    5) Duplicate removal + LS repair
    6) Route-Based Decomposition (RBD)
    7) RBD routing + LS

This is the first complete single-iteration DRSCI-style pipeline.
"""

import sys
import os

# ---------------------------------------------------------
# Correct sys.path handling
# ---------------------------------------------------------
CURRENT = os.path.dirname(__file__)        # core/src/master
ROOT = os.path.abspath(os.path.join(CURRENT, "."))        # core/src/master
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))  # core/src

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Imports from your project
# ---------------------------------------------------------
from master.clustering.run_clustering import run_clustering
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.clustering.route_based import route_based_decomposition


# ---------------------------------------------------------
# Lazy import wrapper for SCP (because gurobi might not be installed)
# ---------------------------------------------------------
def lazy_import_scp():
    from master.setcover.scp_solver import solve_scp
    return solve_scp


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
INSTANCE = "X-n101-k25.vrp"

# Customer-based decomposition parameters
K = 3
CLUSTER_METHOD = "sk_ac_avg"

# Optional SCP (requires gurobi)
RUN_SCP = False  # set True after Gurobi is installed

# LS (Local Search) configuration
SEED = 0
LS_NEIGHBOURHOOD = "dri_spatial"  # same used in your LS controller

# Route-Based Decomposition (RBD) configuration
RUN_RBD = True
RBD_METHOD = "ac_avg"  # safe choice for now, until kmeans override is updated
RBD_K = K               # number of RBD clusters (usually same as K)


# =====================================================================
# ============================   MAIN   ================================
# =====================================================================

def main():
    # ---------------------------------------------------------
    # Stage 1: Customer-based decomposition (clustering)
    # ---------------------------------------------------------
    print("\n=== Stage 1: Clustering ===")
    clusters, medoids = run_clustering(CLUSTER_METHOD, INSTANCE, K)
    for cid, members in clusters.items():
        print(f"  Cluster {cid}: {len(members)} customers")

    # ---------------------------------------------------------
    # Stage 2: Routing subproblems
    # ---------------------------------------------------------
    print("\n=== Stage 2: Subproblem Routing ===")
    routing_result = solve_clusters_with_pyvrp(
        instance_name=INSTANCE,
        clusters=clusters,
        time_limit_per_cluster=5.0,
        seed=SEED,
    )
    routes = routing_result["routes"]
    print(f"Total merged routes: {len(routes)}")
    print(f"Total cost (sum of cluster solves): {routing_result['total_cost']}")

    # ---------------------------------------------------------
    # Stage 3: Global LS Improve
    # ---------------------------------------------------------
    print("\n=== Stage 3: Global LS Improve ===")
    ls_result = improve_with_local_search(
        instance_name=INSTANCE,
        routes_vrplib=routes,
        neighbourhood=LS_NEIGHBOURHOOD,
        max_neighbours=40,
        seed=SEED,
    )
    improved_routes = ls_result["routes_improved"]
    print("Cost before LS:", ls_result["initial_cost"])
    print("Cost after  LS:", ls_result["improved_cost"])

    # ---------------------------------------------------------
    # Stage 4: Set Covering Problem (optional)
    # ---------------------------------------------------------
    if RUN_SCP:
        print("\n=== Stage 4: Set Covering Problem ===")
        solve_scp = lazy_import_scp()   # Import SCP solver only if needed

        scp_result = solve_scp(
            instance_name=INSTANCE,
            route_pool=improved_routes,
            time_limit=30,
            verbose=True,
        )
        selected_routes = scp_result["selected_routes"]
        print(f"SCP chose {len(selected_routes)} routes "
              f"(out of {len(improved_routes)}).")
    else:
        print("\n=== Stage 4: SCP SKIPPED (Gurobi not active) ===")
        selected_routes = improved_routes

    # ---------------------------------------------------------
    # Stage 5: Duplicate removal + LS Repair
    # ---------------------------------------------------------
    print("\n=== Stage 5: Duplicate Removal + LS Repair ===")
    dup_result = remove_duplicates(
        instance_name=INSTANCE,
        routes=selected_routes,
        verbose=True,  # Turn off later for speed
        max_iters=50,
        ls_neighbourhood=LS_NEIGHBOURHOOD,
        ls_max_neighbours_restricted=40,
        seed=SEED,
    )

    final_routes = dup_result["routes"]
    print("\n=== Summary after SCP + Duplicate Removal ===")
    print("Iterations :", dup_result["iterations"])
    print("LS calls   :", dup_result["ls_calls"])
    print("Initial duplicates:", dup_result["initial_duplicates"])
    print("Final duplicates  :", dup_result["final_duplicates"])
    print("Missing customers :", dup_result["missing_customers"])

    print(f"\nSolution after Stage 5 has {len(final_routes)} routes:")
    for r in final_routes:
        print(" ", r)

    # ---------------------------------------------------------
    # Stage 6: Route-Based Decomposition (RBD)
    # ---------------------------------------------------------
    if RUN_RBD:
        print("\n=== Stage 6: Route-Based Decomposition (RBD) ===")
        rbd_clusters = route_based_decomposition(
            instance_name=INSTANCE,
            global_routes=final_routes,
            k=RBD_K,
            method=RBD_METHOD,
            use_angle=True,
            use_load=True,
        )

        print("RBD produced customer clusters:")
        for cid, members in rbd_clusters.items():
            print(f"  RBD cluster {cid}: {len(members)} customers")

        # -----------------------------------------------------
        # Stage 7: Routing subproblems for RBD + LS Improve
        # -----------------------------------------------------
        print("\n=== Stage 7: Routing RBD clusters ===")
        rbd_routing_result = solve_clusters_with_pyvrp(
            instance_name=INSTANCE,
            clusters=rbd_clusters,
            time_limit_per_cluster=5.0,
            seed=SEED,
        )
        rbd_routes = rbd_routing_result["routes"]
        print(f"RBD routing: {len(rbd_routes)} routes, "
              f"total cost = {rbd_routing_result['total_cost']}")

        print("\n=== Stage 7b: LS Improve on RBD solution ===")
        rbd_ls_result = improve_with_local_search(
            instance_name=INSTANCE,
            routes_vrplib=rbd_routes,
            neighbourhood=LS_NEIGHBOURHOOD,
            max_neighbours=40,
            seed=SEED,
        )
        rbd_final_routes = rbd_ls_result["routes_improved"]

        print("RBD LS cost before:", rbd_ls_result["initial_cost"])
        print("RBD LS cost after :", rbd_ls_result["improved_cost"])

        print(f"\n=== FINAL RBD SOLUTION ===")
        print(f"#routes: {len(rbd_final_routes)}")
        for r in rbd_final_routes:
            print(" ", r)
    else:
        print("\n=== Stage 6: RBD SKIPPED ===")


if __name__ == "__main__":
    main()
