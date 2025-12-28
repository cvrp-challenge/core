"""
Route dominance filters for SCP route pools.

All filters here are SAFE:
they do not change the optimal SCP solution.

Currently implemented:
  1) Same customer set dominance (order ignored):
     keep only the cheapest route per customer set.

This filter SUBSUMES exact duplicate removal.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Iterable


Route = List[int]
Routes = List[Route]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _customer_set(route: Iterable[int], depot_id: int) -> frozenset[int]:
    """Return customer set, ignoring depot and order."""
    return frozenset(n for n in route if n != depot_id)


# ------------------------------------------------------------------
# Dominance rule 1: same customer set → keep cheapest
# ------------------------------------------------------------------

def filter_same_customer_set_keep_cheapest(
    routes: Routes,
    costs: List[float],
    *,
    depot_id: int = 1,
) -> Tuple[Routes, List[float], Dict[int, int]]:
    assert len(routes) == len(costs)

    best_idx: Dict[frozenset[int], int] = {}

    for i, r in enumerate(routes):
        key = _customer_set(r, depot_id)
        prev = best_idx.get(key)

        if prev is None or costs[i] < costs[prev]:
            best_idx[key] = i

    kept_old_indices = sorted(best_idx.values())

    filtered_routes = [routes[i] for i in kept_old_indices]
    filtered_costs = [costs[i] for i in kept_old_indices]

    old_to_new = {
        old_i: new_i
        for new_i, old_i in enumerate(kept_old_indices)
    }

    return filtered_routes, filtered_costs, old_to_new


# ------------------------------------------------------------------
# Public pool-level API (THIS is what DRSCI should call)
# ------------------------------------------------------------------

def filter_route_pool(
    routes: Routes,
    costs: List[float] | None = None,
    *,
    depot_id: int = 1,
    verbose: bool = False,
) -> Routes:
    """
    Apply all enabled SAFE dominance rules to a route pool.

    This replaces `_deduplicate_routes`.

    Args:
        routes: route pool (VRPLIB format)
        costs : parallel cost list (optional).
                If None, all costs are treated as equal.
        depot_id: depot node id
        verbose: print pool reduction

    Returns:
        filtered route pool
    """
    n_before = len(routes)

    if costs is None:
        # Equal costs are fine for same-set dominance
        costs = [0.0] * len(routes)

    routes, costs, _ = filter_same_customer_set_keep_cheapest(
        routes,
        costs,
        depot_id=depot_id,
    )

    if verbose and len(routes) < n_before:
        print(
            f"[ROUTE DOMINANCE] same-customer-set: "
            f"{n_before} -> {len(routes)}",
            flush=True,
        )

    return routes

# ------------------------------------------------------------------
# Local sanity check
# ------------------------------------------------------------------

if __name__ == "__main__":
    routes = [
        [1, 2, 3, 1],
        [1, 2, 3, 1],
        [1, 3, 2, 1],
        [1, 4, 5, 1],
        [1, 5, 4, 1],
        [1, 6, 1],
    ]
    costs = [6, 6, 9, 7, 8, 2]

    filtered = filter_route_pool(routes, costs, verbose=True)

    assert len(filtered) == 3
    print("✅ dominance filter sanity check passed")