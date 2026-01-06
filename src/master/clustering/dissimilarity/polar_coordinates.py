# clustering/dissimilarity/polar_coordinates.py

import math
from typing import Dict, Optional
from master.utils.loader import load_instance


def compute_polar_angle(
    instance_name: str,
    instance: Optional[dict] = None,
    *,
    angle_offset: float = 0.0,
) -> Dict[int, float]:

    """
    Computes polar angles θ_i of all customer nodes relative to the depot (node 1).

    θ_i = arctan((y_i - y_0) / (x_i - x_0))
    Excludes the depot itself.
    """
    if instance is None:
        instance = load_instance(instance_name)

    # Convert NumPy array (0-based) → dict with 1-based node IDs
    coords_arr = instance["node_coord"]
    coords = {i + 1: tuple(coords_arr[i]) for i in range(len(coords_arr))}

    DEPOT_ID = 1
    x0, y0 = coords[DEPOT_ID]

    angles: Dict[int, float] = {}
    for i, (x_i, y_i) in coords.items():
        if i == DEPOT_ID:
            continue
        raw = math.atan2(y_i - y0, x_i - x0)
        theta = raw + angle_offset

        # wrap to (-pi, pi]
        theta = math.atan2(math.sin(theta), math.cos(theta))

        angles[i] = theta


    return angles


if __name__ == "__main__":
    instance = load_instance("X-n101-k25.vrp")
    angles = compute_polar_angle("X-n101-k25.vrp", instance)
    print("First 3 polar angles:", list(angles.items())[:3])
