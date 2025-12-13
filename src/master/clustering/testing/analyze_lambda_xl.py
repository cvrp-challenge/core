# master/analyze_lambda_xl.py

import os
import math
import numpy as np
from master.utils.loader import load_instance
from master.clustering.dissimilarity.polar_coordinates import compute_polar_angle
from master.clustering.dissimilarity.spatial import compute_lambda, angular_spread_circular


def compute_lambda_factor(instance_name: str) -> dict:
    """Compute λ base, angular range, and adaptive factor for one instance."""
    instance = load_instance(instance_name)

    coords_arr = instance["node_coord"]
    coords_full = {i + 1: tuple(coords_arr[i]) for i in range(len(coords_arr))}
    DEPOT_ID = 1
    coords = {i: coords_full[i] for i in coords_full if i != DEPOT_ID}

    angles = compute_polar_angle(instance_name, instance)
    lam = compute_lambda(coords)

    theta_range = angular_spread_circular(angles)
    w = min(math.pi, theta_range)
    p = 2.0  # variance matching
    factor = (math.pi / w) ** p
    factor = max(1.0, min(factor, 8.0))
    lam_eff = lam * factor

    return {
        "instance": instance_name,
        "lam": lam,
        "theta_range": theta_range,
        "factor": factor,
        "lam_eff": lam_eff,
    }


def analyze_xl_folder():
    base_dir = os.path.dirname(__file__)
    xl_path = os.path.abspath(os.path.join(base_dir, "../../instances/test-instances/xl"))
    vrp_files = [f for f in os.listdir(xl_path) if f.lower().endswith(".vrp")]

    print(f"Found {len(vrp_files)} XL instances.\n")

    results = []
    for f in sorted(vrp_files):
        try:
            data = compute_lambda_factor(f)
            results.append(data)
            print(
                f"{f:25s} | λ={data['lam']:.1f} | θ_range={data['theta_range']:.4f} | "
                f"factor={data['factor']:.2f} | λ_eff={data['lam_eff']:.1f}"
            )
        except Exception as e:
            print(f"⚠️  Failed for {f}: {e}")

    if not results:
        print("\nNo instances processed successfully.")
        return

    # Aggregate statistics
    factors = [r["factor"] for r in results]
    lam_vals = [r["lam"] for r in results]
    lam_eff_vals = [r["lam_eff"] for r in results]

    print("\n=== Summary ===")
    print(f"Instances processed : {len(results)}")
    print(f"λ factor min        : {min(factors):.3f}")
    print(f"λ factor max        : {max(factors):.3f}")
    print(f"λ factor mean       : {np.mean(factors):.3f}")
    print(f"Base λ range        : {min(lam_vals):.1f} – {max(lam_vals):.1f}")
    print(f"λ_eff range         : {min(lam_eff_vals):.1f} – {max(lam_eff_vals):.1f}")


if __name__ == "__main__":
    analyze_xl_folder()


# master/main.py

import math
from master.utils.loader import load_instance
from master.clustering.dissimilarity.polar_coordinates import compute_polar_angle
from master.clustering.dissimilarity.spatial import (
    compute_lambda,
    spatial_dissimilarity,
    angular_spread_circular,
)
import statistics


def analyze_instance(instance_name: str):
    print(f"\n=== Analyzing instance: {instance_name} ===")

    # ------------------------------------------------------------------
    # Load instance
    instance = load_instance(instance_name)
    coords_arr = instance["node_coord"]
    coords = {i + 1: tuple(coords_arr[i]) for i in range(len(coords_arr))}
    DEPOT_ID = 1
    depot = coords[DEPOT_ID]
    coords_no_depot = {i: coords[i] for i in coords if i != DEPOT_ID}
    print(f"→ Loaded {len(coords_no_depot)} customers (depot at {depot})")

    # ------------------------------------------------------------------
    # Compute polar angles and λ
    angles = compute_polar_angle(instance_name, instance)
    lam = compute_lambda(coords_no_depot)
    theta_range = angular_spread_circular(angles)
    w = min(math.pi, theta_range)
    p = 2.0
    factor = (math.pi / w) ** p
    factor = max(1.0, min(factor, 8.0))
    lam_eff = lam * factor

    print(f"λ (base)      = {lam:.4f}")
    print(f"θ range (rad) = {theta_range:.4f}")
    print(f"λ factor      = {factor:.4f}")
    print(f"λ_eff (used)  = {lam_eff:.4f}")

    # ------------------------------------------------------------------
    # Show some polar coordinates
    print("\n🧭 First 10 polar angles (radians):")
    for k, v in list(angles.items())[:10]:
        print(f"  Node {k:>3}: θ = {v:>8.4f}")

    # ------------------------------------------------------------------
    # Compute some spatial dissimilarities
    S_s = spatial_dissimilarity(instance_name, instance)
    values = list(S_s.values())
    mean_s = statistics.mean(values)
    min_s = min(values)
    max_s = max(values)
    print(f"\n📐 Spatial dissimilarity stats:")
    print(f"  Count = {len(values)} pairs")
    print(f"  Mean  = {mean_s:.4f}")
    print(f"  Min   = {min_s:.4f}")
    print(f"  Max   = {max_s:.4f}")

    print("\n📐 First 5 spatial dissimilarities:")
    for (i, j), val in list(S_s.items())[:5]:
        print(f"  Pair ({i:>3},{j:>3}): S^s_ij = {val:.4f}")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    # Centered depot instance
    #analyze_instance("X-n110-k13.vrp")

    # Cornered depot instance
    #analyze_instance("X-n106-k14.vrp")

    #analyze_instance("XLTEST-n1141-k94.vrp")

    #analyze_instance("XLTEST-n1421-k9.vrp")

    #analyze_instance("XLTEST-n10001-k798.vrp")

    analyze_instance("XLTEST-n9571-k979.vrp")
