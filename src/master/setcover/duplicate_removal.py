# master/setcover/duplicate_removal.py
"""
Duplicate removal + repair local search for DRSCI.

This module takes a set of routes (typically the solution of the SCP),
detects customers that are visited more than once, and:

    1) Removes redundant visits (based on a savings criterion) so that
       each customer appears at most once.
    2) Repairs / improves the solution using PyVRP LocalSearch on a
       *feasible* solution (no duplicate customers).

Important change vs. previous version:
    - Local search is only called AFTER all duplicates are removed.
      PyVRP expects a valid solution; it cannot fix duplicate clients
      and will crash if we feed duplicated routes.
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
    Find customers that appear in more than one position (over all routes).

    Returns
    -------
    duplicates : dict
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

    if nid == 1:
        # depot removal is not allowed / not meaningful
        return -1e9

    prev_nid = route[pos_idx - 1]
    next_nid = route[pos_idx + 1]

    return dist(prev_nid, nid) + dist(nid, next_nid) - dist(prev_nid, next_nid)


def _apply_removals(routes: Routes, removals: List[Tuple[int, int]]) -> Routes:
    """
    Apply a list of (route_idx, pos_idx) removals to the routes.

    Removals are done without mutating the input, and routes that
    become [1, 1] (empty) are dropped.
    """
    # Group removals per route for efficiency
    to_remove_per_route: Dict[int, List[int]] = {}
    for r_idx, pos_idx in removals:
        to_remove_per_route.setdefault(r_idx, []).append(pos_idx)

    new_routes: Routes = []
    for r_idx, route in enumerate(routes):
        if r_idx not in to_remove_per_route:
            # no removals in this route
            new_routes.append(list(route))
            continue

        positions = sorted(to_remove_per_route[r_idx], reverse=True)
        new_route = list(route)
        for pos in positions:
            # Guard against out-of-range due to earlier deletions
            if 0 <= pos < len(new_route):
                del new_route[pos]

        # Drop routes that are just depots or empty
        if len(new_route) <= 2:
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
    ls_neighbourhood: str = "dri_spatial",
    ls_max_neighbours_restricted: int = 40,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Remove duplicate customer visits and repair with LS, safely.

    The new behavior is:

        1) Repeatedly:
            - Detect all customers that appear more than once.
            - For each such customer, keep exactly one occurrence
              (the one with the smallest removal savings) and mark
              all other occurrences for removal.
            - Apply all removals in one batch.

           This loop stops when no duplicates are left, or when
           `max_iters` is reached.

        2) Run PyVRP LocalSearch ONCE on the deduplicated solution.

    At no point is LocalSearch called on a solution that still has
    duplicate customer visits.

    Returns
    -------
    dict with keys:
        "routes"             : final routes (VRPLIB format)
        "iterations"         : number of duplicate-removal iterations
        "ls_calls"           : number of LS calls made
        "initial_duplicates" : number of customers with duplicates at start
        "final_duplicates"   : number of customers with duplicates at end
        "missing_customers"  : list of customers with zero visits at end
    """
    if verbose:
        print("\n[dup-removal] Starting duplicate removal + LS repair.")

    inst = load_instance(instance_name)
    dim = int(inst["dimension"])
    # VRPLIB: depot is node 1, customers = 2..dim
    all_customers = list(range(2, dim + 1))
    dist = _build_distance_function(inst)

    # Initial counts / duplicates
    counts = _compute_customer_counts(routes)
    dup_customers = {i for i, c in counts.items() if c > 1}
    initial_duplicates = len(dup_customers)

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
    iterations = 0

    # ------------------------------------------------------------------
    # Phase 1: PURE duplicate removal (no LS yet)
    # ------------------------------------------------------------------
    for it in range(max_iters):
        dup_map = _find_duplicates(current_routes)
        if not dup_map:
            if verbose:
                print(f"[dup-removal] Iteration {it}: no duplicates left.")
            break

        iterations = it + 1
        if verbose:
            print(f"[dup-removal] Iteration {iterations}: "
                  f"{len(dup_map)} customers with duplicates.")

        removals: List[Tuple[int, int]] = []

        # For each duplicated customer, decide which occurrences to remove
        for cust, occs in dup_map.items():
            if len(occs) <= 1:
                continue

            scored: List[Tuple[float, int, int]] = []
            for (r_idx, pos_idx) in occs:
                s = _compute_savings_for_occurrence(current_routes, r_idx, pos_idx, dist)
                scored.append((s, r_idx, pos_idx))

            # Sort by descending savings: first entries are most profitable to remove
            scored.sort(reverse=True)

            # Keep exactly ONE occurrence: the one with the smallest savings
            # (worst to remove), which is the last in this sorted list.
            # All others are marked for removal.
            for (s, r_idx, pos_idx) in scored[:-1]:
                removals.append((r_idx, pos_idx))

        if not removals:
            # Nothing to remove => stop
            if verbose:
                print("[dup-removal] No removable occurrences found; stopping.")
            break

        # Apply removals in one batch
        current_routes = _apply_removals(current_routes, removals)

    # After Phase 1, check duplicates and missing customers
    counts = _compute_customer_counts(current_routes)
    final_dup_customers = {i for i, c in counts.items() if c > 1}
    final_duplicates = len(final_dup_customers)
    missing_customers = [i for i in all_customers if counts.get(i, 0) == 0]

    if verbose:
        print(f"[dup-removal] After removal: {final_duplicates} customers still duplicated.")
        if missing_customers:
            print(f"[dup-removal] WARNING: customers missing before LS: {missing_customers}")

    # At this point, LS will only be called if there's at least a feasible-ish structure.
    ls_calls = 0

    # ------------------------------------------------------------------
    # Phase 2: LocalSearch on deduplicated solution
    # ------------------------------------------------------------------
    if current_routes:
        ls_result = improve_with_local_search(
            instance_name=instance_name,
            routes_vrplib=current_routes,
            neighbourhood=ls_neighbourhood,
            max_neighbours=ls_max_neighbours_restricted,
            seed=seed,
        )
        ls_calls += 1
        current_routes = ls_result["routes_improved"]

        # Recompute stats after LS
        counts = _compute_customer_counts(current_routes)
        final_dup_customers = {i for i, c in counts.items() if c > 1}
        final_duplicates = len(final_dup_customers)
        missing_customers = [i for i in all_customers if counts.get(i, 0) == 0]

        if verbose:
            print(f"[dup-removal] After LS: {final_duplicates} customers with duplicates.")
            if missing_customers:
                print(f"[dup-removal] WARNING: customers missing in final solution: "
                      f"{missing_customers}")

    if verbose:
        print(f"[dup-removal] Finished after {iterations} iterations, LS calls: {ls_calls}")

    return {
        "routes": current_routes,
        "iterations": iterations,
        "ls_calls": ls_calls,
        "initial_duplicates": initial_duplicates,
        "final_duplicates": final_duplicates,
        "missing_customers": missing_customers,
    }
