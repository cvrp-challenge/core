# cvrp_dri/data/loader.py

import re
from typing import Dict, Tuple


def load_instance(filepath: str) -> Tuple[Dict[int, Tuple[float, float]], Tuple[float, float], Dict[int, int], int]:
    """
    Loads coordinates, demands, and capacity from a .vrp instance file.
    Supports both classic CVRPLIB and large-scale formats.

    Returns:
        coords_customers (dict): {node_id: (x, y)} for customers only
        coords_depot (tuple): (x, y) coordinate of the depot
        demands (dict): {node_id: demand}
        capacity (int): vehicle capacity Q
    """
    with open(filepath, "r") as f:
        lines = [line.strip() for line in f if line.strip() != ""]

    coords = {}
    demands = {}
    capacity = None
    depot_id = None
    depot_coords = None

    # --- Find section indices ---
    def find_section(keyword):
        for i, l in enumerate(lines):
            if keyword in l:
                return i
        return None

    node_start = find_section("NODE_COORD_SECTION")
    demand_start = find_section("DEMAND_SECTION")
    depot_start = find_section("DEPOT_SECTION")

    # --- Extract capacity ---
    for line in lines:
        if "CAPACITY" in line:
            match = re.findall(r"\d+", line)
            if match:
                capacity = int(match[0])
            break

    if node_start is None or demand_start is None or depot_start is None or capacity is None:
        raise ValueError(f"Missing one or more required sections in {filepath}")

    # --- Read NODE_COORD_SECTION ---
    for line in lines[node_start + 1: demand_start]:
        if "DEMAND_SECTION" in line or "EOF" in line:
            break
        parts = re.split(r"\s+", line)
        if len(parts) >= 3:
            node_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            coords[node_id] = (x, y)

    # --- Read DEMAND_SECTION ---
    for line in lines[demand_start + 1: depot_start]:
        if "DEPOT_SECTION" in line or "EOF" in line:
            break
        parts = re.split(r"\s+", line)
        if len(parts) >= 2:
            node_id, demand = int(parts[0]), int(parts[1])
            demands[node_id] = demand

    # --- Read DEPOT_SECTION ---
    for line in lines[depot_start + 1:]:
        if line.startswith("-1") or line.startswith("EOF"):
            break
        depot_id = int(line)
        break  # usually only one depot

    # --- Extract depot coordinate and remove from customer sets ---
    if depot_id is not None:
        depot_coords = coords.get(depot_id, None)
        coords_customers = {k: v for k, v in coords.items() if k != depot_id}
        demands = {k: v for k, v in demands.items() if k != depot_id}
    else:
        raise ValueError("Depot not found in DEPOT_SECTION")

    if not coords_customers or not demands or depot_coords is None:
        raise ValueError(f"Failed to parse all required data from {filepath}")

    return coords_customers, depot_coords, demands, capacity


if __name__ == "__main__":
    # Example usage
    coords, depot, demands, Q = load_instance(
        r"D:\PS_CVRP\core\instances\test-instances\xl\XLTEST-n5174-k170.vrp"
    )
    print(f"Loaded {len(coords)} customers, capacity={Q}")
    print(f"Depot: {depot}")
    print("First 3 coords:", list(coords.items())[:3])
    print("First 3 demands:", list(demands.items())[:3])
