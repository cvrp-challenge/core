"""
Iterative DRSCI-style pipeline for a single large instance.

This script aims to match the DRSCI framework described by Kerscher (2025):

  - Dynamic per-instance parameters:
        * TOTAL_TIME_LIMIT = 10 * N   (N = #customers)
        * K_MAX = number from '-kXX' in filename (fast), 
          else fallback: ceil(total_demand / q)

  - Systematic cluster sizes:
        C_s = {1, 2, 4, ..., K_MAX}

  - In each iteration:
        * choose |C| from C_s in systematic order
        * with prob 0.5: vertex-based clustering
        * with prob 0.5: route-based clustering (if best solution available)
        * solve all clusters with PyVRP
        * add routes to global pool
        * solve SCP on full pool
        * duplicate removal + global LS repair
        * update best solution

  - Termination:
        * total time >= TOTAL_TIME_LIMIT
        * no improvement for MAX_NO_IMPROVE iterations

Outputs only console text — no solution files written.
"""

import sys
import os
import time
import math
import random
import re

# ---------------------------------------------------------
# sys.path setup
# ---------------------------------------------------------
CURRENT = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(CURRENT, "."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
from master.clustering.run_clustering import run_clustering
from master.dri.routing.routing_controller import solve_clusters_with_pyvrp
from master.improve.ls_controller import improve_with_local_search
from master.setcover.duplicate_removal import remove_duplicates
from master.clustering.route_based import route_based_decomposition
from utils.loader import load_instance


def lazy_import_scp():
    from master.setcover.scp_solver import solve_scp
    return solve_scp


# ---------------------------------------------------------
# Instance configuration
# ---------------------------------------------------------
INSTANCE = "X-n502-k39.vrp"

    # "X-n502-k39.vrp",
    # "X-n524-k153.vrp",
    # "X-n561-k42.vrp",
    # "X-n641-k35.vrp",
    # "X-n685-k75.vrp",
    # "X-n716-k35.vrp",
    # "X-n749-k98.vrp",
    # "X-n801-k40.vrp",
    # "X-n856-k95.vrp",
    # "X-n916-k207.vrp",
    # "XLTEST-n1048-k138.vrp",
    # "XLTEST-n1794-k408.vrp",
    # "XLTEST-n2541-k62.vrp",
    # "XLTEST-n3147-k210.vrp",
    # "XLTEST-n4153-k259.vrp",
    # "XLTEST-n6034-k1685.vrp",
    # "XLTEST-n6734-k1347.vrp",
    # "XLTEST-n8028-k691.vrp",
    # "XLTEST-n8766-k55.vrp",
    # "XLTEST-n10001-k798.vrp"

# Vertex-based clustering methods
VB_CLUSTER_METHODS = [
    "sk_ac_avg",
    "sk_kmeans",
    "fcm",
    "pyclust_k_medoids",
    "sk_ac_complete",
    "sk_ac_min",
]

SEED = 0
random.seed(SEED)

TIME_LIMIT_PER_CLUSTER = 20.0
LS_NEIGHBOURHOOD = "dri_spatial"
MAX_NEIGHBOURS_LS = 40

MAX_NO_IMPROVE = 30
PROB_VERTEX_BASED = 0.5


# =====================================================================
# ========================   HELPER FUNCTIONS   ========================
# =====================================================================

def extract_kmax_from_filename(filename: str):
    """
    Extract the Kmax (#vehicles) from VRPLIB-style filenames: '-kXX'.
    Returns int if found, else raises ValueError.
    """
    match = re.search(r'-k(\d+)', filename)
    if not match:
        raise ValueError(f"No '-kXX' found in filename {filename}")
    return int(match.group(1))


def build_initial_cluster_set(K_max: int):
    """
    Systematic cluster candidate set:
        C_s = {1, 2, 4, 8, ..., K_max}
    """
    cand = set()
    value = 1
    while value <= K_max:
        cand.add(value)
        value *= 2
    cand.add(K_max)
    return sorted(cand)


def deduplicate_routes(route_pool):
    """
    Remove duplicate routes from route pool.
    A route is a list[int], so we convert to tuple for hashing.
    """
    unique_routes = []
    seen = set()
    for r in route_pool:
        tup = tuple(r)
        if tup not in seen:
            seen.add(tup)
            unique_routes.append(r)
    return unique_routes


# =====================================================================
# ============================   MAIN   ================================
# =====================================================================

def main():
    solve_scp = lazy_import_scp()

    # ---------------------------------------------------------
    # Step 1: Try to extract K_MAX from filename
    # ---------------------------------------------------------
    print(f"\n=== DRSCI-STYLE ITERATIVE RUN on {INSTANCE} ===")

    filename_kmax = None
    try:
        filename_kmax = extract_kmax_from_filename(INSTANCE)
        print(f"Found K_MAX={filename_kmax} from filename.")
    except ValueError:
        print("Filename does not contain '-kXX'. Falling back to demand/capacity.")

    # ---------------------------------------------------------
    # Step 2: Load instance to compute N and fallback K_MAX
    # ---------------------------------------------------------
    inst = load_instance(INSTANCE)

    # --- Correct for VRPLIB structures used in your project ---
    # inst["demand"]     = {node_id: demand}
    # inst["capacity"]   = vehicle capacity
    # depot is node 0

    demands = inst["demand"]
    q = inst["capacity"]

    # number of customers (exclude depot 0)
    N = len(demands) - 1

    # total customer demand
    total_demand = sum(demands[i] for i in demands if i != 0)

    print(f"Number of customers N: {N}")
    print(f"Vehicle capacity q   : {q}")
    print(f"Total demand         : {total_demand}")



    # If filename provided Kmax, use it; else compute
    if filename_kmax is not None:
        K_MAX = filename_kmax
    else:
        K_MAX = math.ceil(total_demand / q)

    print(f"Using K_MAX (upper bound on routes): {K_MAX}")

    # ---------------------------------------------------------
    # Step 3: Compute TOTAL_TIME_LIMIT = 10 * N
    # ---------------------------------------------------------
    TOTAL_TIME_LIMIT = 10.0 * N
    print(f"Total runtime limit (10*N): {TOTAL_TIME_LIMIT:.1f}s\n")

    # ---------------------------------------------------------
    # Step 4: Build initial systematic cluster sizes
    # ---------------------------------------------------------
    cluster_candidates = build_initial_cluster_set(K_MAX)
    num_candidates = len(cluster_candidates)
    cluster_idx = 0

    print(f"Initial systematic cluster sizes C_s = {cluster_candidates}\n")

    # ---------------------------------------------------------
    # Global DRSCI state
    # ---------------------------------------------------------
    global_route_pool = []
    best_routes = None
    best_cost = float("inf")
    no_improve = 0
    iteration = 0

    start_time = time.time()

    # ---------------------------------------------------------
    # Main DRSCI Loop
    # ---------------------------------------------------------
    while True:
        now = time.time()
        elapsed = now - start_time

        if elapsed >= TOTAL_TIME_LIMIT:
            print("\n[STOP] Global time limit reached.")
            break
        if no_improve >= MAX_NO_IMPROVE:
            print("\n[STOP] Max iterations without improvement reached.")
            break

        iteration += 1
        print("\n" + "=" * 70)
        print(f"=== Iteration {iteration} ===")
        print("=" * 70)

        # ---------------- Step 1: select |C| ----------------
        current_K = cluster_candidates[cluster_idx]
        print(f"[Iter {iteration}] Using |C| = {current_K}")

        cluster_idx = (cluster_idx + 1) % num_candidates

        # ---------------- Step 2: VB vs RB ------------------
        use_vertex_based = True
        if best_routes is not None:
            if random.random() >= PROB_VERTEX_BASED:
                use_vertex_based = False

        if use_vertex_based or best_routes is None:
            # ----- VERTEX-BASED -----
            method = random.choice(VB_CLUSTER_METHODS)
            print(f"[Iter {iteration}] Decomposition: VERTEX-BASED ({method})")

            clusters, medoids = run_clustering(
                method,
                INSTANCE,
                current_K,
            )
        else:
            # ----- ROUTE-BASED -----
            print(f"[Iter {iteration}] Decomposition: ROUTE-BASED (sk_ac_avg on routes)")

            clusters = route_based_decomposition(
                instance_name=INSTANCE,
                global_routes=best_routes,
                k=current_K,
                method="sk_ac_avg",
                use_angle=True,
                use_load=True,
            )

        # report clusters
        total_customers_assigned = sum(len(members) for members in clusters.values())
        print(f"[Iter {iteration}] #clusters: {len(clusters)}, customers assigned: {total_customers_assigned}")

        # ---------------- Step 3: solve all clusters --------
        print(f"\n[Iter {iteration}] Stage 2: Routing subproblems")

        routing = solve_clusters_with_pyvrp(
            instance_name=INSTANCE,
            clusters=clusters,
            time_limit_per_cluster=TIME_LIMIT_PER_CLUSTER,
            seed=SEED,
        )

        routes = routing["routes"]
        print(f"[Iter {iteration}] Cluster routes total : {len(routes)}")
        print(f"[Iter {iteration}] Cluster routing cost : {routing['total_cost']}")

        # ---------------- Step 4: add to route pool ---------
        global_route_pool.extend(routes)
        global_route_pool = deduplicate_routes(global_route_pool)
        print(f"[Iter {iteration}] Global route pool size (unique): {len(global_route_pool)}")

        # ---------------- Step 5: SCP on pooled routes ------
        print(f"\n[Iter {iteration}] Stage 3: SCP on pooled routes")

        scp = solve_scp(
            instance_name=INSTANCE,
            route_pool=global_route_pool,
            time_limit=600,
            verbose=False,
        )

        selected_routes = scp["selected_routes"]
        print(f"[Iter {iteration}] SCP selected {len(selected_routes)} routes.")

        # ---------------- Step 6: duplicate removal + LS ----
        print(f"\n[Iter {iteration}] Stage 4: Duplicate removal + LS repair")

        dup = remove_duplicates(
            instance_name=INSTANCE,
            routes=selected_routes,
            verbose=False,
            max_iters=50,
            ls_neighbourhood=LS_NEIGHBOURHOOD,
            ls_max_neighbours_restricted=MAX_NEIGHBOURS_LS,
            seed=SEED,
        )

        repaired_routes = dup["routes"]
        print(f"[Iter {iteration}] Repaired route count : {len(repaired_routes)}")

        # ---------------- Step 7: Global LS -----------------
        print(f"\n[Iter {iteration}] Stage 5: Global LS")

        ls_res = improve_with_local_search(
            instance_name=INSTANCE,
            routes_vrplib=repaired_routes,
            neighbourhood=LS_NEIGHBOURHOOD,
            max_neighbours=MAX_NEIGHBOURS_LS,
            seed=SEED,
        )

        iter_initial_cost = ls_res["initial_cost"]
        iter_improved_cost = ls_res["improved_cost"]
        iter_routes = ls_res["routes_improved"]

        print(f"[Iter {iteration}] LS cost before: {iter_initial_cost}")
        print(f"[Iter {iteration}] LS cost after : {iter_improved_cost}")

        # ---------------- Step 8: update best ---------------
        if iter_improved_cost < best_cost:
            improvement = (best_cost - iter_improved_cost
                           if best_cost < float("inf") else None)

            best_cost = iter_improved_cost
            best_routes = iter_routes
            no_improve = 0

            if improvement is None:
                print(f"[Iter {iteration}] New best solution: {best_cost:.2f}")
            else:
                print(f"[Iter {iteration}] Improved by {improvement:.2f}, new best = {best_cost:.2f}")
        else:
            no_improve += 1
            print(f"[Iter {iteration}] No improvement ({no_improve}/{MAX_NO_IMPROVE})")

        elapsed = time.time() - start_time
        print(f"[Iter {iteration}] Elapsed time: {elapsed:.1f}s / {TOTAL_TIME_LIMIT:.1f}s")
        print(f"[Iter {iteration}] Best cost so far: {best_cost:.2f}")
        print(f"[Iter {iteration}] Best route count: "
              f"{len(best_routes) if best_routes is not None else 'N/A'}")

    # =================================================================
    # Final summary
    # =================================================================
    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)

    if best_routes is None:
        print("No feasible solution found.")
    else:
        print(f"Best cost   : {best_cost:.2f}")
        print(f"Route count : {len(best_routes)}")
        print("Routes:")
        for r in best_routes:
            print(" ", r)

    print("\n=== END ===")


if __name__ == "__main__":
    main()
