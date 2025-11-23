# run_decomposition_and_routing.py
"""
End-to-end pipeline:
    1) Run selected clustering
    2) Solve each cluster with PyVRP
    3) Output merged solution

Later this will be extended with:
    - route-based decomposition
    - set covering phase
    - iteration loop (DRSCI)
"""

import sys
import os

# Ensure project root packages
CURRENT = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(CURRENT, "."))  # core/src/master
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))  # core/src

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from master.clustering.run_clustering import run_clustering
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp

# Simple configuration
INSTANCE = "XLTEST-n1141-k94.vrp"
METHOD = "sk_ac_avg"      # any: "fcm", "ac_min", "k_medoids", "sk_kmeans", "sk_kmeans", 
                        #     "ac_max", "ac_avg", "sk_ac_avg", "sk_ac_complete", "sk_ac_min"
K = 9

def main():
    print("\n=== DRSCI Stage 1: Clustering ===")
    clusters, medoids = run_clustering(METHOD, INSTANCE, K)
    print("Clusters built:")
    for cid, members in clusters.items():
        print(f"  Cluster {cid}: {len(members)} customers")

    print("\n=== DRSCI Stage 2: Routing Subproblems ===")
    result = solve_clusters_with_pyvrp(
        instance_name=INSTANCE,
        clusters=clusters,
        time_limit_per_cluster=5.0
    )

    print("\n=== Final Global Solution ===")
    print("Total cluster cost:", result["total_cost"])
    print("Routes:")
    for r in result["routes"]:
        print(" ", r)

if __name__ == "__main__":
    main()
