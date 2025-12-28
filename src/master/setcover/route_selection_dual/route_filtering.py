"""
Route Filtering (Dual-based)
----------------------------

Implements paper-style filtering rules to shrink a route pool before solving a
restricted set partitioning / covering master.

Inputs:
- route_pool: VRPLIB routes
- costs[r]
- route_cust[r] = set of customers covered by route r
- reduced_costs[r] = c_r - sum_{i in route} pi_i  (computed from LP duals)
- incumbent routes (must be preserved)

Filtering rules (typical):
A) keep all routes with negative reduced cost
B) for each customer i: keep top-k routes covering i by reduced cost
C) keep best N routes by raw cost (cheap baseline routes)
+ Always keep incumbents
+ Cap to N_final

Robustness:
- Verifies coverage after filtering
- If some customer becomes uncovered, repairs by adding cheapest routes that cover them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


Route = List[int]   # VRPLIB format
Routes = List[Route]


@dataclass(frozen=True)
class FilterStats:
    n_input: int
    n_output: int

    kept_negative_rc: int
    kept_cost_seed: int
    kept_topk_per_customer: int
    kept_incumbent: int

    uncovered_after_filter: int
    repaired_added: int

    rc_threshold: float
    k_per_customer: int
    n_cost_seed: int
    n_final: int


def _custset(route: Route, depot_id: int = 1) -> frozenset[int]:
    return frozenset(n for n in route if n != depot_id)


def _build_incumbent_index_set(
    route_pool: Routes,
    incumbent_routes: Optional[Routes],
    depot_id: int = 1,
) -> Set[int]:
    """
    Map incumbent routes to indices in the current route_pool by customer-set signature.
    If an incumbent route does not exist in pool, it cannot be preserved here (should be added earlier).
    """
    if not incumbent_routes:
        return set()

    sig_to_idx: Dict[frozenset[int], int] = {}
    for idx, r in enumerate(route_pool):
        sig_to_idx[_custset(r, depot_id)] = idx

    keep = set()
    for r in incumbent_routes:
        sig = _custset(r, depot_id)
        if sig in sig_to_idx:
            keep.add(sig_to_idx[sig])
    return keep


def _ensure_coverage(
    *,
    customers: List[int],
    route_cust: List[Set[int]],
    costs: List[float],
    selected: Set[int],
) -> Tuple[Set[int], int]:
    """
    Ensure every customer is covered by at least one route in `selected`.
    If not, greedily add cheapest routes that cover uncovered customers.

    Returns:
      (new_selected, num_added)
    """
    all_cust = set(customers)

    covered: Set[int] = set()
    for r in selected:
        covered |= route_cust[r]

    uncovered = all_cust - covered
    if not uncovered:
        return selected, 0

    num_added = 0
    # Greedy: add cheapest route that covers at least one uncovered, repeat.
    # (This is robust and fast enough for filtering; can be improved later.)
    num_routes = len(route_cust)

    while uncovered:
        best_r = None
        best_cost = float("inf")
        best_new = 0

        for r in range(num_routes):
            if r in selected:
                continue
            newly = uncovered & route_cust[r]
            if not newly:
                continue
            c = costs[r]
            # tie-breaker: more newly covered
            if c < best_cost or (c == best_cost and len(newly) > best_new):
                best_cost = c
                best_r = r
                best_new = len(newly)

        if best_r is None:
            # Should not happen if original pool covers all customers
            break

        selected.add(best_r)
        covered |= route_cust[best_r]
        uncovered = all_cust - covered
        num_added += 1

    return selected, num_added


def filter_routes_dual(
    *,
    customers: List[int],
    route_pool: Routes,
    route_cust: List[Set[int]],
    costs: List[float],
    reduced_costs: List[float],
    incumbent_routes: Optional[Routes] = None,
    depot_id: int = 1,
    rc_threshold: float = -1e-6,
    k_per_customer: int = 10,
    n_cost_seed: int = 8000,
    n_final: int = 16000,
    verbose: bool = True,
) -> Tuple[Routes, List[int], FilterStats]:
    """
    Apply dual-based filtering to shrink the pool.

    Returns:
      filtered_routes: Routes
      filtered_indices: indices of kept routes in the original route_pool
      stats: FilterStats
    """
    n = len(route_pool)
    if not (len(route_cust) == len(costs) == len(reduced_costs) == n):
        raise ValueError("[filter] route_pool, route_cust, costs, reduced_costs must align in length.")

    selected: Set[int] = set()

    # -----------------------
    # Incumbent preservation
    # -----------------------
    incumbent_idx = _build_incumbent_index_set(route_pool, incumbent_routes, depot_id=depot_id)
    selected |= incumbent_idx

    # -----------------------
    # Rule A: negative reduced costs
    # -----------------------
    neg_rc = {r for r, rc in enumerate(reduced_costs) if rc < rc_threshold}
    selected |= neg_rc

    # -----------------------
    # Rule C: cost seed (best raw-cost routes)
    # -----------------------
    if n_cost_seed > 0:
        by_cost = sorted(range(n), key=lambda r: costs[r])
        cost_seed = set(by_cost[: min(n_cost_seed, n)])
        selected |= cost_seed
    else:
        cost_seed = set()

    # -----------------------
    # Rule B: per-customer top-k by reduced cost
    # -----------------------
    kept_topk = 0
    if k_per_customer > 0:
        # Build for each customer: candidate list
        # We do it via scanning routes once (cheaper than building a full cust->routes map here)
        cust_to_routes: Dict[int, List[int]] = {i: [] for i in customers}
        for r, custs in enumerate(route_cust):
            for i in custs:
                if i in cust_to_routes:
                    cust_to_routes[i].append(r)

        for i in customers:
            rlist = cust_to_routes[i]
            if not rlist:
                continue
            # Keep best k by reduced cost (tie-break by cost)
            rlist_sorted = sorted(rlist, key=lambda r: (reduced_costs[r], costs[r]))
            topk = rlist_sorted[: min(k_per_customer, len(rlist_sorted))]
            before = len(selected)
            selected |= set(topk)
            kept_topk += max(0, len(selected) - before)

    # -----------------------
    # Cap to n_final (but keep incumbent!)
    # -----------------------
    if len(selected) > n_final:
        # Keep incumbent always, then fill remaining slots by:
        # 1) negative reduced cost first (best rc)
        # 2) then remaining by raw cost
        keep = set(incumbent_idx)

        remaining_budget = max(0, n_final - len(keep))
        if remaining_budget == 0:
            selected = keep
        else:
            # candidates sorted by (reduced_cost, cost)
            candidates = [r for r in selected if r not in keep]
            candidates.sort(key=lambda r: (reduced_costs[r], costs[r]))

            chosen = candidates[:remaining_budget]
            selected = keep | set(chosen)

    # -----------------------
    # Coverage repair
    # -----------------------
    covered: Set[int] = set()
    for r in selected:
        covered |= route_cust[r]
    uncovered_cnt = len(set(customers) - covered)

    selected_before_repair = set(selected)
    selected, added = _ensure_coverage(
        customers=customers,
        route_cust=route_cust,
        costs=costs,
        selected=selected,
    )

    # Cap again if repair pushed above n_final
    if len(selected) > n_final:
        keep = set(incumbent_idx)
        remaining_budget = max(0, n_final - len(keep))
        candidates = [r for r in selected if r not in keep]
        # Prefer repaired routes if they were needed for coverage: they are in selected but might have bad rc;
        # we still sort by (reduced_cost, cost) to stay consistent.
        candidates.sort(key=lambda r: (reduced_costs[r], costs[r]))
        selected = keep | set(candidates[:remaining_budget])

        # Final safety: ensure coverage again (should normally still hold)
        selected, _ = _ensure_coverage(
            customers=customers,
            route_cust=route_cust,
            costs=costs,
            selected=selected,
        )

    filtered_indices = sorted(selected)
    filtered_routes = [route_pool[i] for i in filtered_indices]

    stats = FilterStats(
        n_input=n,
        n_output=len(filtered_indices),
        kept_negative_rc=len(neg_rc),
        kept_cost_seed=len(cost_seed),
        kept_topk_per_customer=kept_topk,
        kept_incumbent=len(incumbent_idx),
        uncovered_after_filter=uncovered_cnt,
        repaired_added=added,
        rc_threshold=rc_threshold,
        k_per_customer=k_per_customer,
        n_cost_seed=n_cost_seed,
        n_final=n_final,
    )

    if verbose:
        print(
            "[FILTER] "
            f"in={stats.n_input} out={stats.n_output} | "
            f"inc={stats.kept_incumbent} negRC={stats.kept_negative_rc} "
            f"costSeed={stats.kept_cost_seed} topKadds≈{stats.kept_topk_per_customer} | "
            f"uncovered_after={stats.uncovered_after_filter} repaired_add={stats.repaired_added} | "
            f"cap={stats.n_final}",
            flush=True,
        )

    return filtered_routes, filtered_indices, stats


def filter_by_reduced_cost(
    routes: Routes,
    *args,
    **kwargs,
) -> Routes:
    """
    Temporary stub.

    This is only here so the dual DRSCI pipeline runs.
    For now: do NOT filter anything.
    """
    return routes

