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



# # cvrp_dri/dissimilarity/spatial_dissimilarity.py

# import math
# from typing import Dict, Tuple

# def compute_lambda(coords: Dict[int, Tuple[float, float]]) -> float:
#     """
#     Computes λ as defined in Eq. (35) of Kerscher & Minner (2025):
#         λ = (1 / (2n)) * Σ_i (x_i + y_i)
#     """
#     n = len(coords)
#     if n == 0:
#         raise ValueError("Coordinate dictionary is empty.")
#     total = sum(x + y for x, y in coords.values())
#     return total / (2 * n)


# def spatial_dissimilarity(
#     coords: Dict[int, Tuple[float, float]],
#     angles: Dict[int, float]
# ) -> Dict[Tuple[int, int], float]:
#     """
#     Computes pairwise spatial dissimilarity S^s_ij as defined in Eq. (34):
#         S^s_ij = sqrt((x_j - x_i)^2 + (y_j - y_i)^2 + λ * (θ_j - θ_i)^2)

#     Args:
#         coords: {node_id: (x, y)} for all customers
#         angles: {node_id: θ_i} polar angles in radians (from depot)

#     Returns:
#         dissimilarity: {(i, j): S^s_ij} for all unordered pairs i < j
#     """
#     lam = compute_lambda(coords)
#     nodes = list(coords.keys())
#     S = {}

#     for idx_i, i in enumerate(nodes):
#         x_i, y_i = coords[i]
#         theta_i = angles[i]

#         for j in nodes[idx_i + 1:]:
#             x_j, y_j = coords[j]
#             theta_j = angles[j]

#             dist = math.sqrt(
#                 (x_j - x_i)**2 + (y_j - y_i)**2 + lam * (theta_j - theta_i)**2
#             )
#             S[(i, j)] = dist
#             S[(j, i)] = dist  # symmetric

#     return S


# if __name__ == "__main__":
#     # Example usage
#     coords = {
#         2: (146.0, 180.0),
#         3: (792.0, 5.0),
#         4: (658.0, 510.0),
#     }
#     angles = {
#         2: -1.5408,
#         3: -0.1797,
#         4: 1.0297,
#     }

#     S = spatial_dissimilarity(coords, angles)
#     print("λ =", compute_lambda(coords))
#     print("First 3 dissimilarities:", list(S.items())[:3])
