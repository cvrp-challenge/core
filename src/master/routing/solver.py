"""
Central routing solver interface.

This module exposes a single :func:`solve` entry point that abstracts away
solver-specific plumbing.

Supported backends:
    - pyvrp   (built-in)
    - hexaly  (via solver_hexaly.py)
    - filo1 / filo2 (via solver_filo.py)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping
import sys
import warnings

# -----------------------------------------------------------------------------
# Repo paths
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_INSTANCE_SUBDIRS = (
    REPO_ROOT,
    REPO_ROOT / "instances",
    REPO_ROOT / "instances" / "challenge-instances",
    REPO_ROOT / "instances" / "test-instances" / "x",
    REPO_ROOT / "instances" / "test-instances" / "xl",
)

# -----------------------------------------------------------------------------
# Public result type
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class SolveOutput:
    solver: str
    instance: Path
    cost: float
    runtime: float
    num_iterations: int
    feasible: bool
    data: Any | None
    raw_result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return str(self.raw_result)

# -----------------------------------------------------------------------------
# PyVRP options (kept here, but PyVRP itself is lazy-imported)
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class PyVRPSolveOptions:
    seed: int = 1
    max_runtime: float | None = 30.0
    max_iterations: int | None = None
    no_improvement: int | None = None
    per_client: bool = False
    round_func: str = "none"
    collect_stats: bool = True
    display: bool = False
    params_file: Path | None = None

    @classmethod
    def from_kwargs(cls, options: Mapping[str, Any] | None) -> "PyVRPSolveOptions":
        options = options or {}
        allowed = {f.name for f in fields(cls)}
        unknown = set(options) - allowed
        if unknown:
            raise TypeError(f"Unknown PyVRP option(s): {', '.join(sorted(unknown))}")
        return cls(**options)

# -----------------------------------------------------------------------------
# Solver registry
# -----------------------------------------------------------------------------
SolverCallable = Callable[[Path, Mapping[str, Any]], SolveOutput]
_SOLVER_REGISTRY: Dict[str, SolverCallable] = {}

def register_solver(name: str) -> Callable[[SolverCallable], SolverCallable]:
    def decorator(func: SolverCallable) -> SolverCallable:
        _SOLVER_REGISTRY[name.lower()] = func
        return func
    return decorator

# -----------------------------------------------------------------------------
# Lazy backend loader
# -----------------------------------------------------------------------------
def _lazy_import_backend(solver_key: str) -> None:
    """
    Ensure solver backend module is imported so its @register_solver decorator runs.
    """
    if solver_key == "hexaly":
        importlib.import_module("master.routing.solver_hexaly")

    elif solver_key in {"filo1", "filo2"}:
        importlib.import_module("master.routing.solver_filo")

    elif solver_key == "pyvrp":
        # PyVRP backend is defined below in this file
        pass

# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
def solve(
    instance: str | Path,
    *,
    solver: str = "pyvrp",
    solver_options: Mapping[str, Any] | None = None,
) -> SolveOutput:

    solver_key = solver.lower()

    if solver_key not in _SOLVER_REGISTRY:
        _lazy_import_backend(solver_key)

    if solver_key not in _SOLVER_REGISTRY:
        available = ", ".join(sorted(_SOLVER_REGISTRY))
        raise ValueError(
            f"Unknown solver '{solver}'. Available solvers: {available}."
        )

    instance_path = _resolve_instance_path(instance)
    adapter = _SOLVER_REGISTRY[solver_key]
    return adapter(instance_path, solver_options or {})

# -----------------------------------------------------------------------------
# PyVRP backend (LAZY IMPORT INSIDE)
# -----------------------------------------------------------------------------
@register_solver("pyvrp")
def _solve_with_pyvrp(
    instance_path: Path, options: Mapping[str, Any]
) -> SolveOutput:

    # Lazy imports — PyVRP is only required if we actually use it
    _pyvrp = importlib.import_module("pyvrp")
    _pyvrp_stop = importlib.import_module("pyvrp.stop")

    ProblemData = _pyvrp.ProblemData
    SolveParams = _pyvrp.SolveParams
    read = _pyvrp.read
    pyvrp_solve = _pyvrp.solve

    MaxIterations = _pyvrp_stop.MaxIterations
    MaxRuntime = _pyvrp_stop.MaxRuntime
    MultipleCriteria = _pyvrp_stop.MultipleCriteria
    NoImprovement = _pyvrp_stop.NoImprovement

    cfg = PyVRPSolveOptions.from_kwargs(options)

    data = read(instance_path, cfg.round_func)

    terms = []
    if cfg.max_runtime is not None:
        terms.append(MaxRuntime(cfg.max_runtime))
    if cfg.max_iterations is not None:
        terms.append(MaxIterations(cfg.max_iterations))
    if cfg.no_improvement is not None:
        terms.append(NoImprovement(cfg.no_improvement))

    if not terms:
        warnings.warn("No stopping criterion set, defaulting to 30s runtime.")
        stop = MaxRuntime(30.0)
    elif len(terms) == 1:
        stop = terms[0]
    else:
        stop = MultipleCriteria(terms)

    params = SolveParams.from_file(cfg.params_file) if cfg.params_file else SolveParams()

    result = pyvrp_solve(
        data,
        stop=stop,
        seed=cfg.seed,
        collect_stats=cfg.collect_stats,
        display=cfg.display,
        params=params,
    )

    return SolveOutput(
        solver="pyvrp",
        instance=instance_path,
        cost=result.cost(),
        runtime=result.runtime,
        num_iterations=result.num_iterations,
        feasible=result.is_feasible(),
        data=data,
        raw_result=result,
        metadata={},
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_instance_path(instance: str | Path) -> Path:
    p = Path(instance)
    if p.exists():
        return p.resolve()

    for root in _DEFAULT_INSTANCE_SUBDIRS:
        cand = (root / p).resolve()
        if cand.exists():
            return cand

    raise FileNotFoundError(f"Instance '{instance}' not found.")

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = ["solve", "SolveOutput", "PyVRPSolveOptions"]
