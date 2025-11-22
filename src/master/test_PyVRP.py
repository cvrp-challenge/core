# test_PyVRP.py
"""
Test script for routing_controller + PyVRP integration.

This verifies:
 1) Instance can be loaded (vrplib)
 2) Clusters are passed correctly
 3) routing_controller builds PyVRP models for each cluster
 4) PyVRP successfully solves each subproblem
 5) Returned routes use correct VRPLIB IDs (depot = 1)

Run using:
    python test_PyVRP.py
"""
import sys
import os

# Current file is: core/src/master/test_PyVRP.py
CURRENT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT, ".."))   # core/src

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("[test] sys.path now contains:", PROJECT_ROOT)

from master.utils.loader import load_instance
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp



# -------------------------------------------------------------------
# Test configuration
# -------------------------------------------------------------------

INSTANCE_NAME = "X-n101-k25.vrp"

# A very small artificial cluster split for testing
# Customers in VRPLIB numbering: depot = 1, customers = 2..101
TEST_CLUSTERS = {
    1: [2, 3, 4, 5, 6],         # small cluster
    2: [7, 8, 9, 10, 11, 12],   # small cluster
}

TIME_LIMIT = 2.0    # seconds per cluster
SEED = 0


# -------------------------------------------------------------------
# Test Logic
# -------------------------------------------------------------------

print("\n===================== PyVRP Routing Test =====================\n")

# 1) Load instance
print(f"[test] Loading instance: {INSTANCE_NAME}")
inst = load_instance(INSTANCE_NAME)

print("[test] Instance keys:", inst.keys())
print("[test] DIMENSION =", inst.get("dimension"))
print("[test] Vehicle capacity =", inst.get("capacity"))

coords = inst["node_coord"]
print(f"[test] Total nodes in instance (incl. depot): {len(coords)}")
print(f"[test] Depot coordinates = {coords[0]} (VRPLIB node 1)")
print(f"[test] Customer #2 coords = {coords[1]}")
print()


# 2) Print test clusters
print("[test] Clusters to solve:")
for cid, nodes in TEST_CLUSTERS.items():
    print(f"  Cluster {cid}: customers = {nodes}")
print()


# 3) Call routing_controller
print("[test] Calling solve_clusters_with_pyvrp() ...\n")

result = solve_clusters_with_pyvrp(
    instance_name=INSTANCE_NAME,
    clusters=TEST_CLUSTERS,
    time_limit_per_cluster=TIME_LIMIT,
    seed=SEED,
)

# 4) Inspect output
print("\n==================== RESULT ====================\n")
print("[test] Cluster costs:")
for cid, cst in result["cluster_costs"].items():
    print(f"  Cluster {cid}: cost = {cst}")

print(f"\n[test] Total merged cost = {result['total_cost']}")

print("\n[test] Routes returned:")
for r_idx, route in enumerate(result["routes"], start=1):
    print(f"  Route {r_idx}: {route}")

print("\n===================== END TEST =====================\n")
