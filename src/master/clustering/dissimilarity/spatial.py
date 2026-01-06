# clustering/dissimilarity/spatial.py

import math
from typing import Dict, Tuple, Optional
from master.utils.loader import load_instance
from master.clustering.dissimilarity.polar_coordinates import compute_polar_angle


# ----------------------------------------------------------------------
# --- Helper functions -------------------------------------------------
# ----------------------------------------------------------------------

def shortest_angle_diff(a: float, b: float) -> float:
    """Shortest wrapped angular difference in (-pi, pi]."""
    d = a - b
    return math.atan2(math.sin(d), math.cos(d))


def angular_spread_circular(angles: Dict[int, float]) -> float:
    """Minimal arc covering all angles, robust to wrap-around."""
    th = sorted(angles.values())
    if not th:
        return 2 * math.pi
    th2 = th + [t + 2 * math.pi for t in th]
    m = len(th)
    best = 2 * math.pi
    j = 0
    for i in range(m):
        while j + 1 < i + m and th2[j + 1] - th2[i] <= 2 * math.pi:
            j += 1
        best = min(best, th2[j] - th2[i])
        if j == i:
            j += 1
    return max(best, 1e-6)


def compute_lambda(coords: Dict[int, Tuple[float, float]]) -> float:
    """λ = (1 / (2n)) * Σ_i (x_i + y_i)."""
    n = len(coords)
    if n == 0:
        raise ValueError("Coordinate dictionary is empty.")
    return sum(x + y for x, y in coords.values()) / (2 * n)


# ----------------------------------------------------------------------
# --- Main dissimilarity computation -----------------------------------
# ----------------------------------------------------------------------

def spatial_dissimilarity(
    instance_name: str,
    instance: Optional[dict] = None,
    *,
    angle_offset: float = 0.0,
) -> Dict[Tuple[int, int], float]:
    """
    Computes spatial dissimilarity S^s_ij:
        S^s_ij = sqrt((x_j - x_i)^2 + (y_j - y_i)^2 + λ_eff * (Δθ_ij)^2)

    λ_eff is adapted to depot position via observed angular spread.
    Only stores (i, j) for i < j for efficiency.
    """
    if instance is None:
        instance = load_instance(instance_name)

    coords_arr = instance["node_coord"]
    coords_full = {i + 1: tuple(coords_arr[i]) for i in range(len(coords_arr))}
    DEPOT_ID = 1

    # Exclude depot
    coords = {i: coords_full[i] for i in coords_full if i != DEPOT_ID}

    # Polar angles & base λ
    angles = compute_polar_angle(
        instance_name,
        instance,
        angle_offset=angle_offset,
    )

    lam = compute_lambda(coords)

    # ---- Adaptive λ scaling (variance correction) ---------------------
    theta_range = angular_spread_circular(angles)
    w = min(math.pi, theta_range)          # cap at π since wrapped diffs ≤ π
    p = 2.0                                # variance matching (quadratic term)
    factor = (math.pi / w) ** p            # boost when angular spread small
    factor = max(1.0, min(factor, 8.0))    # never reduce λ; optional cap
    lam_eff = lam * factor

    # ---- Compute dissimilarities -------------------------------------
    nodes = list(coords.keys())
    S: Dict[Tuple[int, int], float] = {}

    for idx_i, i in enumerate(nodes):
        x_i, y_i = coords[i]
        theta_i = angles[i]
        for j in nodes[idx_i + 1:]:
            x_j, y_j = coords[j]
            theta_j = angles[j]
            dtheta = shortest_angle_diff(theta_j, theta_i)

            S[(i, j)] = math.sqrt(
                (x_j - x_i) ** 2 + (y_j - y_i) ** 2 + lam_eff * (dtheta) ** 2
            )

    return S


# ----------------------------------------------------------------------
# --- Stand-alone test -------------------------------------------------
# ----------------------------------------------------------------------

if __name__ == "__main__":
    instance = load_instance("X-n101-k25.vrp")
    S = spatial_dissimilarity("X-n101-k25.vrp", instance)
    print("First 3 spatial dissimilarities:", list(S.items())[:3])
