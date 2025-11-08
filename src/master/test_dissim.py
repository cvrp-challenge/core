# Orchestrator for the entire pipeline
# master/main.py

from utils.loader import load_instance
from clustering.dissimilarity.polar_coordinates import compute_polar_angle
from clustering.dissimilarity.spatial import spatial_dissimilarity
from clustering.dissimilarity.demand import demand_dissimilarity
from clustering.dissimilarity.combined import combined_dissimilarity


def main():
    instance_name = "X-n101-k25.vrp"

    # Load once (cached loader ensures no reload)
    instance = load_instance(instance_name)
    print(f"→ Loaded instance '{instance['name']}' with {len(instance['node_coord']) - 1} customers.")
    print(f"  Vehicle capacity: {instance['capacity']}\n")

    # === 1️⃣ Polar angles ===
    angles = compute_polar_angle(instance_name, instance)
    print("🧭 First 3 polar angles:")
    for k, v in list(angles.items())[:3]:
        print(f"  Node {k:>3}: θ = {v:.6f}")
    print()

    # === 2️⃣ Spatial dissimilarity ===
    S_s = spatial_dissimilarity(instance_name, instance)
    print("📐 First 3 spatial dissimilarities:")
    for (i, j), val in list(S_s.items())[:3]:
        print(f"  Pair ({i:>3}, {j:>3}): S^s_ij = {val:.6f}")
    print()

    # === 3️⃣ Demand dissimilarity ===
    S_d = demand_dissimilarity(instance_name, instance)
    print("📦 First 3 demand dissimilarities:")
    for (i, j), val in list(S_d.items())[:3]:
        print(f"  Pair ({i:>3}, {j:>3}): S^d_ij = {val:.6f}")
    print()

    # === 4️⃣ Combined dissimilarity ===
    S_sd = combined_dissimilarity(instance_name)
    print("⚡ First 3 combined dissimilarities:")
    for (i, j), val in list(S_sd.items())[:3]:
        print(f"  Pair ({i:>3}, {j:>3}): S^sd_ij = {val:.6f}")
    print()

    print("✅ All dissimilarities computed successfully.\n")


if __name__ == "__main__":
    main()
