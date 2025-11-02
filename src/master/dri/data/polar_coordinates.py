# cvrp_dri/dissimilarity/spatial_dissimilarity.py

import math
import vrplib
from typing import Dict, Tuple
from cvrp_dri.data.polar_coordinates import compute_polar_angle


def compute_lambda(coords: Dict[int, Tuple[float, float]]) -> float:
    n = len(coords)
    return sum(x + y for x, y in coords.values()) / (2 * n)


def spatial_dissimilarity(instance_name: str) -> Dict[Tuple[int, int], float]:
    """
    Computes spatial dissimilarity S^s_ij for all customer pairs in a VRP instance:
        S^s_ij = sqrt((x_j - x_i)^2 + (y_j - y_i)^2 + λ * (θ_j - θ_i)^2)
    """
    instance = vrplib.read_instance("instances/" + instance_name)
    coords_full = instance["node_coord"]
    coords = {i: coords_full[i] for i in range(1, len(coords_full))}  # exclude depot

    angles = compute_polar_angle(instance_name)
    lam = compute_lambda(coords)
    S = {}

    nodes = list(coords.keys())
    for idx_i, i in enumerate(nodes):
        x_i, y_i = coords[i]
        theta_i = angles[i]
        for j in nodes[idx_i + 1:]:
            x_j, y_j = coords[j]
            theta_j = angles[j]
            dist = math.sqrt((x_j - x_i)**2 + (y_j - y_i)**2 + lam * (theta_j - theta_i)**2)
            S[(i, j)] = S[(j, i)] = dist

    return S


if __name__ == "__main__":
    S = spatial_dissimilarity("X-n101-k25.vrp")
    print("First 3 spatial dissimilarities:", list(S.items())[:3])



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

