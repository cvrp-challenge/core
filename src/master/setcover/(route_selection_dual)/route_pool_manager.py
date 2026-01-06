"""
Route Pool Manager
------------------

Centralized management of the global route pool for SCP / SP pipelines.

Responsibilities:
- Store and track all generated routes
- Deduplicate routes (customer-set based)
- Track incumbent (best-known) routes
- Enforce pool size limits safely
- Preserve provenance information (where routes came from)

Extended (for dual-based pipeline):
- Expose structural views (customers, route_cust, costs)
- Attach instance data explicitly (no implicit loading)
- Remains solver-agnostic (NO optimization logic)

Key invariant (IMPORTANT):
- Route costs are cached and MUST be invalidated whenever the route set changes.
"""

from __future__ import annotations

from typing import List, Dict, Set, Optional


Route = List[int]   # VRPLIB format: [1, ..., 1]
Routes = List[Route]


def _customer_set(route: Route, depot_id: int = 1) -> frozenset[int]:
    return frozenset(n for n in route if n != depot_id)


class RoutePoolManager:
    """
    Pool manager with strict API discipline:
    - routes is a PROPERTY (list) => never call routes()
    - costs is cached, but invalidated whenever pool changes
    """

    def __init__(self) -> None:
        self._routes: Routes = []
        self._custset_to_idx: Dict[frozenset[int], int] = {}

        self._sources: Dict[int, Set[str]] = {}
        self._is_incumbent: Set[int] = set()

        self._instance: Optional[dict] = None
        self._depot_id: int = 1
        self._customers: Optional[List[int]] = None

        self._costs_cache: Optional[List[float]] = None  # aligned with self._routes

    # ------------------------------------------------------------------
    # Cache discipline
    # ------------------------------------------------------------------
    def _invalidate_costs(self) -> None:
        self._costs_cache = None

    # ------------------------------------------------------------------
    # Basic access
    # ------------------------------------------------------------------
    @property
    def routes(self) -> Routes:
        # return copy to avoid accidental external mutation
        return list(self._routes)

    def size(self) -> int:
        return len(self._routes)

    @property
    def customers(self) -> List[int]:
        if self._customers is None:
            raise RuntimeError("Instance not attached to RoutePoolManager.")
        return self._customers

    @property
    def route_cust(self) -> List[Set[int]]:
        depot = self._depot_id
        return [set(_customer_set(r, depot)) for r in self._routes]

    @property
    def costs(self) -> List[float]:
        if self._instance is None or self._customers is None:
            raise RuntimeError("Instance not attached to RoutePoolManager.")

        if self._costs_cache is None:
            inst = self._instance
            coords = inst["node_coord"]
            edge_mat = inst.get("edge_weight")

            def dist(u: int, v: int) -> float:
                if edge_mat is not None:
                    return float(edge_mat[u - 1, v - 1])
                x1, y1 = coords[u - 1]
                x2, y2 = coords[v - 1]
                return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            costs: List[float] = []
            for r in self._routes:
                c = 0.0
                for a, b in zip(r, r[1:]):
                    c += dist(a, b)
                costs.append(c)

            self._costs_cache = costs

        # Hard safety check: alignment must always hold
        if len(self._costs_cache) != len(self._routes):
            raise RuntimeError(
                "Costs cache is out of sync with route list. "
                "This indicates a missing cache invalidation."
            )

        return self._costs_cache

    # ------------------------------------------------------------------
    # Instance attachment
    # ------------------------------------------------------------------
    def attach_instance(self, instance: dict, depot_id: int = 1) -> None:
        self._instance = instance
        self._depot_id = depot_id

        dim = int(instance["dimension"])
        self._customers = list(range(depot_id + 1, dim + 1))

        self._invalidate_costs()

    # ------------------------------------------------------------------
    # Adding routes (dedup by customer set)
    # ------------------------------------------------------------------
    def add_routes(
        self,
        routes: Routes,
        *,
        source: str,
        mark_incumbent: bool = False,
    ) -> None:
        if self._instance is None:
            # Not strictly required, but avoids depot mismatch surprises later
            pass

        changed = False

        for r in routes:
            cs = _customer_set(r, self._depot_id)

            if cs in self._custset_to_idx:
                idx = self._custset_to_idx[cs]
                self._sources[idx].add(source)
                if mark_incumbent:
                    self._is_incumbent.add(idx)
                continue

            idx = len(self._routes)
            self._routes.append(r)
            self._custset_to_idx[cs] = idx
            self._sources[idx] = {source}
            if mark_incumbent:
                self._is_incumbent.add(idx)

            changed = True

        if changed:
            self._invalidate_costs()

    # ------------------------------------------------------------------
    # Incumbent handling (needed by run_drsci_dual.py)
    # ------------------------------------------------------------------
    def mark_incumbent(self, routes: Routes, *, source: str = "INCUMBENT") -> None:
        """
        Mark routes as incumbent. If they are not in pool yet, add them.
        """
        # First ensure they are in pool
        self.add_routes(routes, source=source, mark_incumbent=True)

        # Then mark indices (already done by add_routes for new ones; for existing ones we need to find them)
        for r in routes:
            cs = _customer_set(r, self._depot_id)
            idx = self._custset_to_idx.get(cs)
            if idx is not None:
                self._is_incumbent.add(idx)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def incumbent_routes(self) -> Routes:
        return [self._routes[i] for i in sorted(self._is_incumbent)]

    def source_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = {}
        for srcs in self._sources.values():
            for s in srcs:
                stats[s] = stats.get(s, 0) + 1
        return stats

    def summary(self) -> str:
        return f"RoutePool(size={self.size()}, incumbent={len(self._is_incumbent)})"
