# cvrp_dri/data/polar_coordinates.py

import os
import math
import vrplib
from typing import Dict

def compute_polar_angle(instance_name: str) -> Dict[int, float]:
    """
    Loads a VRP instance via vrplib and computes the polar angles θ_i
    of all customer nodes relative to the depot.

    Automatically searches both 'x' and 'xl' in:
        core/instances/test-instances/
    """

    # Find base directory dynamically (this file → up to /core/)
    base_dir = os.path.dirname(__file__)
    core_root = os.path.abspath(os.path.join(base_dir, "../../../../"))  # go up 4 levels
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

    # Load instance and compute polar angles
    instance = vrplib.read_instance(instance_path)
    coords = instance["node_coord"]
    depot = coords[0]  # depot is index 0 in vrplib format

    x0, y0 = depot
    angles = {}

    for i in range(1, len(coords)):  # customers only
        x_i, y_i = coords[i]
        theta_i = math.atan2(y_i - y0, x_i - x0)
        angles[i] = theta_i

    return angles


if __name__ == "__main__":
    angles = compute_polar_angle("X-n101-k25.vrp")
    print("First 3 polar angles:", list(angles.items())[:3])



# # cvrp_dri/data/polar_coordinates.py

# import math
# from typing import Dict, Tuple

# def compute_polar_angle(
#     coords: Dict[int, Tuple[float, float]],
#     depot: Tuple[float, float]
# ) -> Dict[int, float]:
#     """
#     Computes the polar angle θ_i for each customer i with respect to the depot.

#     As defined in the DRI framework (Kerscher & Minner, 2025, Eq. 8):
#         θ_i = arctan((y_i - y_0) / (x_i - x_0))

#     Args:
#         coords: {node_id: (x_i, y_i)} for all customers
#         depot: (x_0, y_0) coordinates of the depot

#     Returns:
#         angles: {node_id: θ_i} in radians
#     """
#     x0, y0 = depot
#     angles = {}

#     for node_id, (x_i, y_i) in coords.items():
#         theta_i = math.atan2(y_i - y0, x_i - x0)  # correct sign & quadrant automatically
#         angles[node_id] = theta_i

#     return angles


# if __name__ == "__main__":
#     # Example usage
#     coords = {
#         2: (146.0, 180.0),
#         3: (792.0, 5.0),
#         4: (658.0, 510.0),
#     }
#     depot = (145.0, 215.0)

#     polar_angles = compute_polar_angle(coords, depot)
#     for k, v in list(polar_angles.items())[:3]:
#         print(f"Node {k}: θ={v:.4f} rad")

