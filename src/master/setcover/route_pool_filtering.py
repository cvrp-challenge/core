from __future__ import annotations

from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

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
    Ranking ONLY among removable routes.
    """
    length = max(length, 1)
    utilization = max(0.0, min(1.0, utilization))

    badness = 0.0
    badness += 2.0 * (1.0 - utilization)     # low utilization
    badness += 1.0 / length                  # short routes
    if not in_best:
        badness += 2.5
    if not from_scp:
        badness += 2.0
    badness += 0.1 * age

    return badness


def _ensure_coverage(
    *,
    kept_routes: List[Route],
    all_routes: List[Route],
    depot_id: int,
    customers: List[int],
) -> List[Route]:
    """
    Hard safety net: ensure every customer appears in at least one route.
    """
    covered = set()
    for r in kept_routes:
        covered.update(c for c in r if c != depot_id)

    missing = [c for c in customers if c not in covered]
    if not missing:
        return kept_routes

    by_customer = defaultdict(list)
    for r in all_routes:
        custs = [c for c in r if c != depot_id]
        for c in custs:
            by_customer[c].append(r)

    repaired = list(kept_routes)
    for c in missing:
        if by_customer[c]:
            repaired.append(by_customer[c][0])
        else:
            # last-resort singleton route
            repaired.append([depot_id, c, depot_id])

    return repaired

def _rank_by_quality(
    *,
    candidate_routes: List[Tuple[int, Route]],
    route_customers: Dict[int, List[int]],
    route_tags: Dict[RouteKey, Tag],
    inst: dict,
    depot_id: int,
) -> List[Tuple[int, Route, float]]:
    ranked = []
    capacity = inst["capacity"]
    demand = inst["demand"]

    for idx, r in candidate_routes:
        customers = route_customers[idx]
        load = sum(demand[c - 1] for c in customers)
        utilization = load / capacity
        length = len(customers)

        tag = route_tags.get(tuple(customers), {})
        badness = _route_badness(
            utilization=utilization,
            length=length,
            from_scp=tag.get("stage") in {"scp_post_ls", "final_scp_post_ls"},
            in_best=tag.get("in_best", False),
            age=tag.get("iteration", 0),
        )
        ranked.append((idx, r, badness))

    return sorted(ranked, key=lambda x: x[2], reverse=True)

def _rank_by_diversity(
    *,
    candidate_routes: List[Tuple[int, Route]],
    route_customers: Dict[int, List[int]],
    best_routes: List[Route],
    depot_id: int,
) -> List[Tuple[int, Route, float]]:
    def edges(route):
        return {
            (route[i], route[i + 1])
            for i in range(len(route) - 1)
        }

    best_edges = set()
    for r in best_routes:
        best_edges |= edges(r)

    ranked = []
    for idx, r in candidate_routes:
        r_edges = edges(r)
        overlap = (
            len(r_edges & best_edges) / max(len(r_edges), 1)
        )
        ranked.append((idx, r, overlap))

    # LOWER overlap = better → remove worst first
    return sorted(ranked, key=lambda x: x[2], reverse=True)

def filter_route_pool_for_scp(
    *,
    routes: List[Route],
    route_tags: Dict[RouteKey, Tag],
    inst: dict,
    depot_id: int = 1,
    max_routes: int = 5000,
    min_utilization: float = 0.30,

    # ---- elite maturity controls ----
    scp_every: int,
    elite_after_scp_rounds: int = 2,
    min_pool_size_for_elite: int = 1500,

    # ---- pruning strategy ----
    pruning_mode: str = "quality",  # "none" | "quality" | "diversity"
) -> List[Route]:
    """
    SCP route pool filtering with strict feasibility guarantees.

    POLICY:
    - If len(routes) <= max_routes: return routes unchanged
    - Otherwise:
        Step A: safety, elites, utilization
        Step B: rank removable routes according to pruning_mode
        Coverage-safe truncation
    """

    pool_size = len(routes)

    # --------------------------------------------------
    # HARD POLICY: no pruning below cap
    # --------------------------------------------------
    if pool_size <= max_routes:
        return list(routes)

    capacity = inst["capacity"]
    demand = inst["demand"]
    all_customers = list(range(1, len(demand) + 1))

    # --------------------------------------------------
    # Precompute customers per route
    # --------------------------------------------------
    route_customers: Dict[int, List[int]] = {
        idx: [c for c in r if c != depot_id]
        for idx, r in enumerate(routes)
    }

    # --------------------------------------------------
    # Elite maturity threshold
    # --------------------------------------------------
    elite_iteration_threshold = elite_after_scp_rounds * scp_every

    elite_routes: List[Route] = []
    candidates: List[Tuple[int, Route]] = []

    # --------------------------------------------------
    # Step A: classify routes (NO ranking yet)
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

        if is_elite:
            elite_routes.append(r)
            continue

        if utilization < min_utilization:
            continue

        if length < 2:
            continue

        candidates.append((idx, r))

    # --------------------------------------------------
    # Elites alone exceed cap
    # --------------------------------------------------
    if len(elite_routes) >= max_routes:
        return elite_routes[:max_routes]

    remaining_slots = max_routes - len(elite_routes)

    # --------------------------------------------------
    # Coverage over eligible routes
    # --------------------------------------------------
    eligible = elite_routes + [r for _, r in candidates]
    coverage = Counter(c for r in eligible for c in r if c != depot_id)

    # --------------------------------------------------
    # Step B: ranking strategy
    # --------------------------------------------------
    ranked: List[Tuple[int, Route, float]] = []

    if pruning_mode == "none":
        ranked = [(idx, r, 0.0) for idx, r in candidates]

    elif pruning_mode == "quality":
        for idx, r in candidates:
            customers = route_customers[idx]
            load = sum(demand[c - 1] for c in customers)
            utilization = load / capacity
            length = len(customers)

            tag = route_tags.get(tuple(customers), {})
            badness = _route_badness(
                utilization=utilization,
                length=length,
                from_scp=tag.get("stage") in {"scp_post_ls", "final_scp_post_ls"},
                in_best=tag.get("in_best", False),
                age=max(0, tag.get("iteration", 0)),
            )
            ranked.append((idx, r, badness))

        # worst first
        ranked.sort(key=lambda x: x[2], reverse=True)

    elif pruning_mode == "diversity":
        # Build edge set of current incumbent
        best_edges = set()
        for key, tag in route_tags.items():
            if tag.get("in_best", False):
                route = [depot_id, *key, depot_id]
                best_edges |= {
                    (route[i], route[i + 1]) for i in range(len(route) - 1)
                }

        for idx, r in candidates:
            edges = {(r[i], r[i + 1]) for i in range(len(r) - 1)}
            overlap = (
                len(edges & best_edges) / max(len(edges), 1)
                if best_edges else 0.0
            )
            ranked.append((idx, r, overlap))

        # highest overlap = worst
        ranked.sort(key=lambda x: x[2], reverse=True)

    else:
        raise ValueError(f"Unknown pruning_mode: {pruning_mode}")

    # --------------------------------------------------
    # Coverage-safe truncation
    # --------------------------------------------------
    kept: List[Route] = []
    local_coverage = coverage.copy()

    for idx, r, _ in ranked:
        customers = route_customers[idx]
        can_remove = all(local_coverage[c] > 1 for c in customers)

        if can_remove:
            for c in customers:
                local_coverage[c] -= 1
            continue

        kept.append(r)
        if len(kept) >= remaining_slots:
            break

    # --------------------------------------------------
    # Deterministic fill if needed
    # --------------------------------------------------
    if len(kept) < remaining_slots:
        for _, r, _ in ranked:
            if r not in kept:
                kept.append(r)
            if len(kept) >= remaining_slots:
                break

    final_pool = elite_routes + kept

    # --------------------------------------------------
    # HARD SAFETY: enforce full coverage
    # --------------------------------------------------
    final_pool = _ensure_coverage(
        kept_routes=final_pool,
        all_routes=routes,
        depot_id=depot_id,
        customers=all_customers,
    )

    return final_pool
