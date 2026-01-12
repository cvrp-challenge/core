import os
import json
import math
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN

# -------- configuration --------
INSTANCE_DIR = r"C:\Users\robin\Documents\PS_CVRP\core\instances\challenge-instances"
OUTPUT_JSON = os.path.join(INSTANCE_DIR, "instance_metadata.json")

# --------------------------------

def parse_vrp(filepath):
    coords = []
    demands = []
    capacity = None

    with open(filepath, "r") as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[1])
        elif line == "NODE_COORD_SECTION":
            section = "coords"
        elif line == "DEMAND_SECTION":
            section = "demands"
        elif line == "DEPOT_SECTION":
            section = None
        elif section == "coords":
            _, x, y = line.split()
            coords.append((float(x), float(y)))
        elif section == "demands":
            _, d = line.split()
            demands.append(int(d))

    return coords, demands, capacity


# ---------------- inference logic ----------------

def infer_depot_position(depot):
    x, y = depot
    if x == 0 and y == 0:
        return "cornered"
    if abs(x - 500) < 5 and abs(y - 500) < 5:
        return "centered"
    return "random"


def infer_customer_position(customers):
    pts = np.array(customers)

    # DBSCAN clustering
    db = DBSCAN(eps=80, min_samples=15).fit(pts)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels)

    if n_clusters <= 1:
        return "random"
    if noise_ratio < 0.15:
        return "clustered"
    return "random-clustered"


def infer_demand_type(demands):
    demands = demands[1:]  # exclude depot
    mn, mx = min(demands), max(demands)
    mean = np.mean(demands)

    large = sum(d >= 50 for d in demands) / len(demands)

    if mn == mx == 1:
        return "unitary"

    if large > 0.4:
        if mn >= 50:
            return "large-only"
        return "few-large-many-small"

    if mx <= 10:
        return "small-demands"

    return "mixed"


def infer_avg_route_size(capacity, demands):
    demands = demands[1:]
    avg_demand = np.mean(demands)
    approx_route_len = capacity / avg_demand

    if approx_route_len < 5:
        return "ultra-short"
    if approx_route_len < 10:
        return "very-short"
    if approx_route_len < 15:
        return "short"
    if approx_route_len < 25:
        return "medium"
    if approx_route_len < 40:
        return "long"
    return "very-long"


# ---------------- main pipeline ----------------

metadata = {}

for file in sorted(os.listdir(INSTANCE_DIR)):
    if not file.endswith(".vrp"):
        continue

    path = os.path.join(INSTANCE_DIR, file)
    coords, demands, capacity = parse_vrp(path)

    depot = coords[0]
    customers = coords[1:]

    metadata[file] = {
        "n_nodes": len(coords),
        "capacity": capacity,
        "depot_positioning": infer_depot_position(depot),
        "customer_positioning": infer_customer_position(customers),
        "demand_distribution": infer_demand_type(demands),
        "avg_route_size_estimate": infer_avg_route_size(capacity, demands)
    }

# write JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata written to {OUTPUT_JSON}")
