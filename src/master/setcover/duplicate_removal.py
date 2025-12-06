# master/setcover/duplicate_removal.py
"""
Duplicate removal + repair local search for DRSCI.

This module takes a set of routes (typically the solution of the SCP),
detects customers that are visited more than once, and iteratively:

    1) Removes one "redundant" visit (based on a savings criterion).
    2) Repairs the solution using PyVRP LocalSearch:
        - First with restricted (dissimilarity-based) neighbourhoods.
        - If that causes missing customers, fallback to a "full-ish"
          LS with a much larger neighbourhood.

The process stops when every customer is visited at most once, or when a
maximum number of iterations is reached.

It uses:
    - instance loader from master.utils.loader
    - PyVRP LS wrapper from master.improve.ls_controller
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

from master.utils.loader import load_instance
from master.improve.ls_controller import improve_with_local_search


Route = List[int]
Routes = List[Route]


# ======================================================================
# Helpers: counts, duplicates, distances
# ======================================================================

def _compute_customer_counts(routes: Routes) -> Dict[int, int]:
    """
    Count how many times each customer node appears in all routes.

    Depot (=1) is ignored.
    """
    counts: Dict[int, int] = {}
    for r in routes:
        for nid in r:
            if nid == 1:
                continue
            counts[nid] = counts.get(nid, 0) + 1
    return counts


def _find_duplicates(routes: Routes) -> Dict[int, List[Tuple[int, int]]]:
    """
    Find customers that appear in more than one position.

    Returns:
        duplicates: dict
            duplicates[i] = list of (route_idx, pos_idx) where node i appears.
    """
    occ: Dict[int, List[Tuple[int, int]]] = {}
    for r_idx, route in enumerate(routes):
        for pos, nid in enumerate(route):
            if nid == 1:
                continue
            occ.setdefault(nid, []).append((r_idx, pos))

    duplicates = {i: positions for i, positions in occ.items() if len(positions) > 1}
    return duplicates


def _build_distance_function(instance: dict):
    """
    Build a simple Euclidean distance function dist(u, v) from instance coords.
    """
    coords = instance["node_coord"]  # index 0 -> node 1

    def dist(u: int, v: int) -> float:
        x1, y1 = coords[u - 1]
        x2, y2 = coords[v - 1]
        return math.hypot(x2 - x1, y2 - y1)

    return dist


def _compute_savings_for_occurrence(
    routes: Routes,
    r_idx: int,
    pos_idx: int,
    dist,
) -> float:
    """
    Approximate savings of removing the customer at (r_idx, pos_idx).

    We use:
        savings = d(prev, i) + d(i, next) - d(prev, next)

    Where 'i' is the customer at position pos_idx in route r_idx.
    """
    route = routes[r_idx]
    nid = route[pos_idx]

    # We assume standard VRPLIB routes: [1, ..., 1],
    # so prev and next always exist for customer entries.
    if nid == 1:
        return -1e9  # should never happen; depot removal not allowed

    prev_nid = route[pos_idx - 1]
    next_nid = route[pos_idx + 1]

    return dist(prev_nid, nid) + dist(nid, next_nid) - dist(prev_nid, next_nid)


def _remove_occurrence(routes: Routes, r_idx: int, pos_idx: int) -> Routes:
    """
    Remove the node at (r_idx, pos_idx) from the routes.

    If a route becomes [1, 1] or shorter (no customers), remove the route.
    Returns a *new* routes list (does not mutate input in place).
    """
    new_routes: Routes = []
    for idx, route in enumerate(routes):
        if idx != r_idx:
            new_routes.append(list(route))
            continue

        # Remove the node at pos_idx
        new_route = route[:pos_idx] + route[pos_idx + 1:]

        # If route is effectively empty (just depots), drop it
        if len(new_route) <= 2:
            # skip adding this route
            continue

        new_routes.append(new_route)

    return new_routes


# ======================================================================
# Main API: duplicate removal + LS repair
# ======================================================================

def remove_duplicates(
    instance_name: str,
    routes: Routes,
    *,
    max_iters: int = 100,
    ls_neighbourhood: str = "dri_spatial",  # "dri_spatial" or "dri_combined"
    ls_max_neighbours_restricted: int = 40,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Iteratively remove duplicate customer visits and repair with LS.

    Args
    ----
    instance_name : str
        VRPLIB instance name (e.g., "X-n101-k25.vrp").
    routes : list[list[int]]
        Current solution routes in VRPLIB format [1, ..., 1].
    max_iters : int
        Maximum number of duplicate-removal iterations.
    ls_neighbourhood : str
        Which DRI-based neighbourhood type to use for LS:
            "dri_spatial"  -> spatial dissimilarity
            "dri_combined" -> spatial + demand dissimilarity
    ls_max_neighbours_restricted : int
        Max neighbours per client in the restricted LS calls.
    seed : int
        Random seed for PyVRP LS.
    verbose : bool
        If True, prints progress and basic statistics.

    Returns
    -------
    dict with keys:
        "routes"             : final routes (VRPLIB format)
        "iterations"         : number of removal iterations performed
        "ls_calls"           : number of LS calls made
        "initial_duplicates" : number of customers with duplicates at start
        "final_duplicates"   : number of customers with duplicates at end
        "missing_customers"  : list of customers with zero visits at end
    """
    if verbose:
        print("\n[dup-removal] Starting duplicate removal + LS repair.")

    inst = load_instance(instance_name)
    dim = int(inst["dimension"])
    all_customers = list(range(2, dim + 1))  # vrplib customers
    dist = _build_distance_function(inst)

    # Initial counts / duplicates
    counts = _compute_customer_counts(routes)
    duplicates = {i for i, c in counts.items() if c > 1}
    initial_duplicates = len(duplicates)

    if verbose:
        print(f"[dup-removal] Initial #customers with duplicates: {initial_duplicates}")

    if initial_duplicates == 0:
        if verbose:
            print("[dup-removal] No duplicates; nothing to do.")
        missing_customers = [i for i in all_customers if counts.get(i, 0) == 0]
        return {
            "routes": routes,
            "iterations": 0,
            "ls_calls": 0,
            "initial_duplicates": 0,
            "final_duplicates": 0,
            "missing_customers": missing_customers,
        }

    current_routes = [list(r) for r in routes]
    ls_calls = 0
    iterations = 0

    for it in range(max_iters):
        iterations = it + 1

        # Recompute duplicates at the start of this iteration
        dup_map = _find_duplicates(current_routes)
        if not dup_map:
            if verbose:
                print(f"[dup-removal] Iteration {iterations}: no duplicates left.")
            break

        if verbose:
            print(f"[dup-removal] Iteration {iterations}: "
                  f"{len(dup_map)} customers with duplicates.")

        # ------------------------------------------------------------------
        # 1) Choose which occurrence to remove based on max savings
        # ------------------------------------------------------------------
        best_savings = -1e18
        best_choice: Optional[Tuple[int, int, int]] = None  # (cust, r_idx, pos_idx)

        for cust, occs in dup_map.items():
            # For each occurrence of this customer, compute savings
            for (r_idx, pos_idx) in occs:
                s = _compute_savings_for_occurrence(
                    current_routes, r_idx, pos_idx, dist
                )
                if s > best_savings:
                    best_savings = s
                    best_choice = (cust, r_idx, pos_idx)

        if best_choice is None:
            if verbose:
                print("[dup-removal] No removable occurrence found; stopping early.")
            break

        cust_star, r_idx_star, pos_idx_star = best_choice

        if verbose:
            print(f"[dup-removal] Removing customer {cust_star} at "
                  f"(route {r_idx_star}, pos {pos_idx_star}), "
                  f"savings ≈ {best_savings:.2f}")

        # ------------------------------------------------------------------
        # 2) Remove that occurrence
        # ------------------------------------------------------------------
        current_routes = _remove_occurrence(current_routes, r_idx_star, pos_idx_star)

        # ------------------------------------------------------------------
        # 3) Repair using LS (restricted neighbourhood first)
        # ------------------------------------------------------------------
        # restricted LS
        ls_result_restricted = improve_with_local_search(
            instance_name=instance_name,
            routes_vrplib=current_routes,
            neighbourhood=ls_neighbourhood,
            max_neighbours=ls_max_neighbours_restricted,
            seed=seed,
        )
        ls_calls += 1
        current_routes = ls_result_restricted["routes_improved"]

        # Check for missing customers after restricted LS
        counts = _compute_customer_counts(current_routes)
        missing = [i for i in all_customers if counts.get(i, 0) == 0]

        if missing:
            if verbose:
                print(f"[dup-removal] Restricted LS caused missing customers "
                      f"{missing}. Falling back to full LS.")

            # Fallback: full-ish LS (large neighbourhood)
            ls_result_full = improve_with_local_search(
                instance_name=instance_name,
                routes_vrplib=current_routes,
                neighbourhood=ls_neighbourhood,
                # approximate "full" by allowing as many neighbours
                # as there are locations; ls_controller will clip
                # internally if needed
                max_neighbours=dim,
                seed=seed,
            )
            ls_calls += 1
            current_routes = ls_result_full["routes_improved"]

            # Re-check missing customers after full LS
            counts = _compute_customer_counts(current_routes)
            missing = [i for i in all_customers if counts.get(i, 0) == 0]

            if missing:
                if verbose:
                    print("[dup-removal] Even full LS could not restore all "
                          f"customers. Missing: {missing}. Stopping.")
                break

        # After LS, we continue to next iteration and re-evaluate duplicates.

    # Final stats
    counts = _compute_customer_counts(current_routes)
    final_duplicates = sum(1 for c in counts.values() if c > 1)
    missing_customers = [i for i in all_customers if counts.get(i, 0) == 0]

    if verbose:
        print(f"[dup-removal] Finished after {iterations} iterations, "
              f"LS calls: {ls_calls}")
        print(f"[dup-removal] Final #customers with duplicates: {final_duplicates}")
        if missing_customers:
            print(f"[dup-removal] WARNING: customers missing in final solution: "
                  f"{missing_customers}")

    return {
        "routes": current_routes,
        "iterations": iterations,
        "ls_calls": ls_calls,
        "initial_duplicates": initial_duplicates,
        "final_duplicates": final_duplicates,
        "missing_customers": missing_customers,
    }

