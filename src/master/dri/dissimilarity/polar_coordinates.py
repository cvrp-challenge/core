# cvrp_dri/dissimilarity/polar_coordinates.py

import os
import math
import vrplib
from typing import Dict


def compute_polar_angle(instance_name: str) -> Dict[int, float]:
    """
    Loads a VRP instance via vrplib and computes the polar angles θ_i
    of all customer nodes relative to the depot.

    Automatically searches for the instance in:
        core/instances/test-instances/x
        core/instances/test-instances/xl

    θ_i = arctan((y_i - y_0) / (x_i - x_0))
    """

    # Base directory of this file
    base_dir = os.path.dirname(__file__)
    core_root = os.path.abspath(os.path.join(base_dir, "../../../../"))
    instances_root = os.path.join(core_root, "instances", "test-instances")

    # Try both subfolders
    possible_paths = [
        os.path.join(instances_root, "x", instance_name),
        os.path.join(instances_root, "xl", instance_name),
    ]
    instance_path = next((p for p in possible_paths if os.path.exists(p)), None)

    if instance_path is None:
        raise FileNotFoundError(
            f"Instance '{instance_name}' not found in: {possible_paths}"
        )

    print(f"→ Loading instance from: {instance_path}")

    # Load the VRP instance
    instance = vrplib.read_instance(instance_path)
    coords = instance["node_coord"]
    depot = coords[0]  # depot is index 0 in vrplib format

    x0, y0 = depot
    angles = {}

    # Compute polar angles for all customers
    for i in range(1, len(coords)):  # skip depot
        x_i, y_i = coords[i]
        theta_i = math.atan2(y_i - y0, x_i - x0)
        angles[i] = theta_i

    return angles


if __name__ == "__main__":
    # Example usage
    angles = compute_polar_angle("X-n101-k25.vrp")
    print("First 3 polar angles:", list(angles.items())[:3])
