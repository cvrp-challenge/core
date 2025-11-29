"""
Central routing solver interface.

This module exposes a single :func:`solve` entry point that abstracts away
solver-specific plumbing.  It currently supports the PyVRP backend, but the
dispatcher is intentionally structured so new solvers can be registered with a
small adapter in the future.

Example
-------

    from master.routing.solver import solve

    result = solve(
        instance="instances/test-instances/x/X-n101-k25.vrp",
        solver="pyvrp",
        seed=42,
        max_runtime=60,
    )

    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Mapping
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_vendored_pyvrp() -> None:
    vendored = REPO_ROOT / "solver" / "pyvrp"
    path_str = str(vendored)
    if vendored.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_vendored_pyvrp()

_pyvrp = importlib.import_module("pyvrp")
_pyvrp_stop = importlib.import_module("pyvrp.stop")

ProblemData = _pyvrp.ProblemData
Result = _pyvrp.Result
SolveParams = _pyvrp.SolveParams
read = _pyvrp.read
pyvrp_solve = _pyvrp.solve

MaxIterations = _pyvrp_stop.MaxIterations
MaxRuntime = _pyvrp_stop.MaxRuntime
MultipleCriteria = _pyvrp_stop.MultipleCriteria
NoImprovement = _pyvrp_stop.NoImprovement
StoppingCriterion = _pyvrp_stop.StoppingCriterion
_DEFAULT_INSTANCE_SUBDIRS = (
    REPO_ROOT,
    REPO_ROOT / "instances",
    REPO_ROOT / "instances" / "test-instances",
    REPO_ROOT / "instances" / "test-instances" / "x",
    REPO_ROOT / "instances" / "test-instances" / "xl",
)


@dataclass(slots=True)
class SolveOutput:
    """
    Canonical return type for :func:`solve`.
    """

    solver: str
    instance: Path
    cost: float
    runtime: float
    num_iterations: int
    feasible: bool
    data: ProblemData
    raw_result: Result
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """
        Convenience helper: reuse PyVRP's textual summary when available.
        """

        return str(self.raw_result)


@dataclass(slots=True)
class PyVRPSolveOptions:
    """
    User-tunable options for the PyVRP backend.
    """

    seed: int = 1
    max_runtime: float | None = 30.0
    max_iterations: int | None = None
    no_improvement: int | None = None
    per_client: bool = False
    round_func: str = "none"
    collect_stats: bool = True
    display: bool = False
    params_file: Path | None = None
    stats_dir: Path | None = None
    solution_dir: Path | None = None

    @classmethod
    def from_kwargs(
        cls, options: Mapping[str, Any] | None
    ) -> PyVRPSolveOptions:
        options = options or {}
        allowed = {field_.name for field_ in fields(cls)}
        unknown = set(options).difference(allowed)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"Unknown PyVRP option(s): {names}")

        return cls(**options)

    def __post_init__(self) -> None:
        for attr in ("max_runtime",):
            value = getattr(self, attr)
            if value is not None and value <= 0:
                raise ValueError(f"{attr} must be > 0, got {value!r}.")

        for attr in ("max_iterations", "no_improvement", "seed"):
            value = getattr(self, attr)
            if value is not None and value < 0:
                raise ValueError(f"{attr} must be >= 0, got {value!r}.")

        self.params_file = _coerce_optional_path(self.params_file)
        self.stats_dir = _coerce_optional_path(self.stats_dir)
        self.solution_dir = _coerce_optional_path(self.solution_dir)

        if self.stats_dir and not self.collect_stats:
            raise ValueError(
                "stats_dir is set but collect_stats=False. Either enable "
                "statistics collection or omit stats_dir."
            )


SolverCallable = Callable[[Path, Mapping[str, Any]], SolveOutput]
_SOLVER_REGISTRY: Dict[str, SolverCallable] = {}


def register_solver(name: str) -> Callable[[SolverCallable], SolverCallable]:
    """
    Decorator to register backend adapters.
    """

    def decorator(func: SolverCallable) -> SolverCallable:
        _SOLVER_REGISTRY[name.lower()] = func
        return func

    return decorator


def solve(
    instance: str | Path,
    *,
    solver: str = "pyvrp",
    solver_options: Mapping[str, Any] | None = None,
) -> SolveOutput:
    """
    Runs the requested solver on the provided instance.

    Args:
        instance: Absolute path, relative path, or bare filename of the VRPLIB
            instance to solve.
        solver: Identifier of the solver backend. Defaults to ``"pyvrp"``.
        solver_options: Solver-specific keyword arguments. For PyVRP, see
            :class:`PyVRPSolveOptions`.
    """

    solver_key = solver.lower()
    if solver_key not in _SOLVER_REGISTRY:
        available = ", ".join(sorted(_SOLVER_REGISTRY))
        raise ValueError(
            f"Unknown solver '{solver}'. Available solvers: {available}."
        )

    instance_path = _resolve_instance_path(instance)
    adapter = _SOLVER_REGISTRY[solver_key]
    return adapter(instance_path, solver_options or {})


@register_solver("pyvrp")
def _solve_with_pyvrp(
    instance_path: Path, options: Mapping[str, Any]
) -> SolveOutput:
    cfg = PyVRPSolveOptions.from_kwargs(options)

    data = read(instance_path, cfg.round_func)
    stop = _build_stop_criterion(data, cfg)
    params = (
        SolveParams.from_file(cfg.params_file)
        if cfg.params_file
        else SolveParams()
    )

    result = pyvrp_solve(
        data,
        stop=stop,
        seed=cfg.seed,
        collect_stats=cfg.collect_stats,
        display=cfg.display,
        params=params,
    )

    _persist_outputs(instance_path, data, result, cfg)

    metadata = {
        "stop": stop.__class__.__name__,
        "round_func": cfg.round_func,
    }

    return SolveOutput(
        solver="pyvrp",
        instance=instance_path,
        cost=result.cost(),
        runtime=result.runtime,
        num_iterations=result.num_iterations,
        feasible=result.is_feasible(),
        data=data,
        raw_result=result,
        metadata=metadata,
    )


def _build_stop_criterion(
    data: ProblemData, cfg: PyVRPSolveOptions
) -> StoppingCriterion:
    terms: list[StoppingCriterion] = []

    if cfg.max_runtime is not None:
        runtime = _scale_by_clients(cfg.max_runtime, data, cfg.per_client)
        terms.append(MaxRuntime(runtime))

    if cfg.max_iterations is not None:
        iterations = _scale_by_clients(
            cfg.max_iterations, data, cfg.per_client
        )
        terms.append(MaxIterations(iterations))

    if cfg.no_improvement is not None:
        stagnation = _scale_by_clients(
            cfg.no_improvement, data, cfg.per_client
        )
        terms.append(NoImprovement(stagnation))

    if not terms:
        warnings.warn(
            "No stopping criterion configured; defaulting to MaxRuntime=30s.",
            stacklevel=2,
        )
        terms.append(MaxRuntime(30.0))

    if len(terms) == 1:
        return terms[0]

    return MultipleCriteria(terms)


def _scale_by_clients(
    value: float | int, data: ProblemData, per_client: bool
) -> float | int:
    if not per_client:
        return value

    return value * max(data.num_clients, 1)


def _persist_outputs(
    instance: Path,
    data: ProblemData,
    result: Result,
    cfg: PyVRPSolveOptions,
) -> None:
    if cfg.stats_dir:
        cfg.stats_dir.mkdir(parents=True, exist_ok=True)
        stats_path = cfg.stats_dir / f"{instance.stem}.csv"
        result.stats.to_csv(stats_path)

    if cfg.solution_dir:
        cfg.solution_dir.mkdir(parents=True, exist_ok=True)
        _write_solution(
            cfg.solution_dir / f"{instance.stem}.sol", data, result
        )


def _write_solution(where: Path, data: ProblemData, result: Result) -> None:
    """
    Persists the best observed solution in VRPLIB's textual format.
    """

    with open(where, "w", encoding="utf-8") as handle:
        if data.num_vehicle_types == 1:
            for idx, route in enumerate(result.best.routes(), 1):
                visits = [str(visit.location) for visit in route.schedule()]
                visits = visits[1:-1]  # drop depot markers
                handle.write(f"Route #{idx}: {' '.join(visits)}\n")

            handle.write(f"Cost: {round(result.cost(), 2)}\n")
            return

        type2vehicle = [
            (int(vehicle) for vehicle in vehicle_type.name.split(","))
            for vehicle_type in data.vehicle_types()
        ]

        routes = [f"Route #{idx + 1}:" for idx in range(data.num_vehicles)]
        for route in result.best.routes():
            visits = [str(visit.location) for visit in route.schedule()]
            visits = visits[1:-1]

            vehicle = next(type2vehicle[route.vehicle_type()])
            routes[vehicle] += " " + " ".join(visits)

        handle.writelines(route + "\n" for route in routes)
        handle.write(f"Cost: {round(result.cost(), 2)}\n")


def _resolve_instance_path(instance: str | Path) -> Path:
    candidate = Path(instance).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate

    if candidate.exists():
        return candidate.resolve()

    for root in _DEFAULT_INSTANCE_SUBDIRS:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Instance '{instance}' not found. Checked: "
        f"{', '.join(str(p) for p in _DEFAULT_INSTANCE_SUBDIRS)}"
    )


def _coerce_optional_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()

    return path


__all__ = ["solve", "SolveOutput", "PyVRPSolveOptions"]

