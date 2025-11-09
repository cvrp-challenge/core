# D:\PS_CVRP\core\src\master\test_clustering.py

import os, sys

# --- Make sure Python finds all subpackages like utils, clustering, etc. ---
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from clustering.k_medoids import k_medoids
from clustering.fcm import fuzzy_c_medoids
from clustering.fcm import fuzzy_c_medoids_debug



if __name__ == "__main__":
    # Choose a small benchmark instance for testing
    instance_name = "XLTEST-n1094-k6.vrp"
    k = 8  # number of clusters

    print(f"→ Running K-Medoids clustering on instance '{instance_name}' with k={k}")
    clusters = k_medoids(instance_name, k)

    print("\n✅ Clustering complete!")
    print(f"Number of clusters: {len(clusters)}\n")

    # Show a quick summary
    for idx, (medoid, members) in enumerate(clusters.items(), start=1):
        print(f"Cluster {idx}:")
        print(f"  Medoid: {medoid}")
        print(f"  Members: {len(members)} customers")
        print(f"  All members: {members}")

    print(f"→ Running Fuzzy C-Medoids clustering on instance '{instance_name}' with k={k}")
    U, medoids, τ_P = fuzzy_c_medoids_debug(instance_name, k)

    print("\n✅ Fuzzy C-Medoids complete!")
    print(f"Number of clusters: {len(medoids)}\n")

    # --- Print summary like K-Medoids ---
    for p, m in enumerate(medoids):
        # list all customers assigned (by max membership)
        assigned = [i for i in U.keys() if max(U[i], key=U[i].get) == p]
        avg_membership = sum(U[i][p] for i in U.keys()) / len(U)
        top5 = sorted(U.items(), key=lambda x: x[1][p], reverse=True)[:5]

        τ = τ_P[p]
        print(f"Cluster {p+1}:")
        print(f"  Medoid: {m}")
        print(f"  Size (argmax-membership customers): {len(assigned)}")
        print(f"  Avg membership μ̄: {avg_membership:.3f}")
        print(f"  τ_P = (x̄={τ[0]:.2f}, ȳ={τ[1]:.2f}, θ̄={τ[2]:.3f}, q̄={τ[3]:.2f})")
        print(f"  Top 5 strongest members: {[(i, round(U[i][p],3)) for i,_ in top5]}\n")
