# run_decomp_route_improve.py
"""
End-to-end pipeline:
    1) Run selected clustering (decomposition step)
    2) Solve each cluster with PyVRP (route step)
    3) Improve the merged global solution using PyVRP LocalSearch (improve step)

Later this will be extended with:
    - route-based decomposition
    - set covering phase
    - iteration loop (DRSCI)
"""

import sys
import os

# Ensure project root packages
CURRENT = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(CURRENT, "."))    # core/src/master
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))  # core/src

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from master.clustering.run_clustering import run_clustering
from master.routing.routing_controller import solve_clusters_with_pyvrp
from master.improve.ls_controller import improve_with_local_search


# Simple configuration
INSTANCE = "XLTEST-n1141-k94.vrp"
METHOD = "sk_ac_avg"        # any of: "fcm", "ac_min", "k_medoids",
                            #          "sk_kmeans", "ac_max", "ac_avg",
                            #          "sk_ac_avg", "sk_ac_complete", "sk_ac_min"
K = 9

def main():
    # --------------------------------------------------------------
    # Stage 1: Clustering (Decomposition)
    # --------------------------------------------------------------
    print("\n=== DRSCI Stage 1: Clustering ===")
    clusters, medoids = run_clustering(METHOD, INSTANCE, K)
    print("Clusters built:")
    for cid, members in clusters.items():
        print(f"  Cluster {cid}: {len(members)} customers")

    # --------------------------------------------------------------
    # Stage 2: Routing Subproblems
    # --------------------------------------------------------------
    print("\n=== DRSCI Stage 2: Routing Subproblems ===")
    routing_result = solve_clusters_with_pyvrp(
        instance_name=INSTANCE,
        clusters=clusters,
        time_limit_per_cluster=5.0
    )

    print("\nMerged routing solution:")
    print("Total cluster cost:", routing_result["total_cost"])
    print("Number of routes:", len(routing_result["routes"]))

    # --------------------------------------------------------------
    # Stage 3: Improvement (Local Search)
    # --------------------------------------------------------------
    print("\n=== DRSCI Stage 3: Improve Global Solution ===")

    ls_result = improve_with_local_search(
        instance_name=INSTANCE,
        routes_vrplib=routing_result["routes"],
        neighbourhood="dri_spatial",    # or "dri_combined"
        max_neighbours=40,              # adjust depending on instance size
        seed=0
    )

    print("\nLocal Search Results:")
    print("Initial cost :", ls_result["initial_cost"])
    print("Improved cost:", ls_result["improved_cost"])
    print("LS moves     :", ls_result["ls_moves"])
    print("LS improving :", ls_result["ls_improving_moves"])

    print("\n=== Final Improved Routes ===")
    for r in ls_result["routes_improved"]:
        print(" ", r)


if __name__ == "__main__":
    main()
