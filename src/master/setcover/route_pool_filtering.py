from __future__ import annotations

from typing import List, Dict, Any, Tuple
from collections import Counter

Route = List[int]
RouteKey = Tuple[int, ...]
Tag = Dict[str, Any]


def _route_badness(
    *,
    utilization: float,
    length: int,
    from_scp: bool,
    in_best: bool,
    age: int,
) -> float:
    """
    Higher = worse (removed earlier).

    Step B ranking is ONLY used among routes that are already removable.
    Coverage safety is enforced elsewhere.
    """
    # Defensive guards
    length = max(length, 1)
    utilization = max(0.0, min(1.0, utilization))

    badness = 0.0

    # Low utilization is bad
    badness += 2.0 * (1.0 - utilization)

    # Very short routes are weak once redundant
    badness += 1.0 / length

    # Routes never used in best solution are weak
    if not in_best:
        badness += 2.5

    # SCP-derived routes are strong
    if not from_scp:
        badness += 2.0

    # Old routes that never helped slowly decay
    badness += 0.1 * age

    return badness


def filter_route_pool_for_scp(
    *,
    routes: List[Route],
    route_tags: Dict[RouteKey, Tag],
    inst: dict,
    depot_id: int = 1,
    max_routes: int = 5000,
    min_utilization: float = 0.30,

    # ---- elite maturity controls (Step A) ----
    scp_every: int,
    elite_after_scp_rounds: int = 2,
    min_pool_size_for_elite: int = 1500,

    # ---- Step B (ranking-based pruning) ----
    enable_step_b: bool = False,
) -> List[Route]:
    """
    SCP route pool filtering with strict safety guarantees.

    POLICY (strict):
    - If len(routes) <= max_routes:
        -> return routes UNCHANGED
    - Only if len(routes) > max_routes:
        -> apply filtering + truncation

    Invariants:
    - global route pool is NEVER mutated
    - routes are NEVER removed below max_routes
    - routes are NEVER removed if that would break customer coverage
    - elite routes are preserved once mature
    - Step B ONLY ranks removable routes, never decides removability
    """

    pool_size = len(routes)

    # --------------------------------------------------
    # HARD POLICY: do nothing below the cap
    # --------------------------------------------------
    if pool_size <= max_routes:
        return list(routes)

    capacity = inst["capacity"]
    demand = inst["demand"]

    # --------------------------------------------------
    # Precompute customer coverage counts
    # --------------------------------------------------
    coverage_count: Counter[int] = Counter()
    route_customers: Dict[int, List[int]] = {}

    for idx, r in enumerate(routes):
        customers = [c for c in r if c != depot_id]
        route_customers[idx] = customers
        for c in customers:
            coverage_count[c] += 1

    elite_routes: List[Route] = []

    # candidate_routes contains:
    # (idx, route, badness)
    candidate_routes: List[Tuple[int, Route, float]] = []

    # --------------------------------------------------
    # Compute elite maturity threshold
    # --------------------------------------------------
    elite_iteration_threshold = elite_after_scp_rounds * scp_every

    # --------------------------------------------------
    # Classify routes (ONLY because pool_size > max_routes)
    # --------------------------------------------------
    for idx, r in enumerate(routes):
        customers = route_customers[idx]
        if not customers:
            continue

        load = sum(demand[c - 1] for c in customers)
        utilization = load / capacity
        length = len(customers)

        key = tuple(customers)
        tag = route_tags.get(key, {})

        stage = tag.get("stage")
        iteration = tag.get("iteration", 0)

        is_elite = (
            stage in {"scp_post_ls", "final_scp_post_ls"}
            and iteration >= elite_iteration_threshold
            and pool_size >= min_pool_size_for_elite
        )

        # --------------------------------------------------
        # Elite routes are always kept once mature
        # --------------------------------------------------
        if is_elite:
            elite_routes.append(r)
            continue

        # --------------------------------------------------
        # Utilization filter (ONLY above cap)
        # --------------------------------------------------
        if utilization < min_utilization:
            continue

        # --------------------------------------------------
        # Coverage-safe single-customer pruning (ONLY above cap)
        # --------------------------------------------------
        if length < 2:
            if all(coverage_count[c] > 1 for c in customers):
                continue

        # --------------------------------------------------
        # Step B scoring (ranking ONLY)
        # --------------------------------------------------
        if enable_step_b:
            from_scp = stage in {"scp_post_ls", "final_scp_post_ls"}
            in_best = tag.get("in_best", False)
            age = max(0, iteration)

            badness = _route_badness(
                utilization=utilization,
                length=length,
                from_scp=from_scp,
                in_best=in_best,
                age=age,
            )
        else:
            badness = 0.0

        candidate_routes.append((idx, r, badness))

    # --------------------------------------------------
    # Enforce hard SCP size limit (coverage-safe)
    # --------------------------------------------------
    if len(elite_routes) >= max_routes:
        return elite_routes[:max_routes]

    remaining_slots = max_routes - len(elite_routes)

    if len(candidate_routes) <= remaining_slots:
        return elite_routes + [r for _, r, _ in candidate_routes]

    # --------------------------------------------------
    # Step B: rank removable routes (worst first)
    # --------------------------------------------------
    if enable_step_b:
        candidate_routes.sort(key=lambda x: x[2], reverse=True)

    # --------------------------------------------------
    # Coverage-safe truncation
    # --------------------------------------------------
    kept_routes: List[Route] = []
    local_coverage = coverage_count.copy()

    for idx, r, _ in candidate_routes:
        customers = route_customers[idx]

        # Can we remove it safely?
        if all(local_coverage[c] > 1 for c in customers):
            continue

        # Must keep
        kept_routes.append(r)
        for c in customers:
            local_coverage[c] -= 1

        if len(kept_routes) >= remaining_slots:
            break

    # Fill remaining slots deterministically if needed
    if len(kept_routes) < remaining_slots:
        for _, r, _ in candidate_routes:
            if r in kept_routes:
                continue
            kept_routes.append(r)
            if len(kept_routes) >= remaining_slots:
                break

    return elite_routes + kept_routes
