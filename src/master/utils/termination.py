from __future__ import annotations

import signal
import atexit
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# IMPORTANT:
# we do NOT import _write_sol_unconditional at top-level
# to avoid circular imports. It is injected.


@dataclass
class Checkpoint:
    instance_name: str
    output_dir: str
    write_sol_fn: callable

    best_routes: Optional[List[List[int]]] = None
    best_cost: float = float("inf")
    dirty: bool = False

    def update(self, routes: List[List[int]], cost: float) -> None:
        self.best_routes = routes
        self.best_cost = cost
        self.dirty = True

    def dump(self, suffix: str) -> None:
        if self.best_routes is None or not math.isfinite(self.best_cost):
            return

        self.write_sol_fn(
            instance_name=self.instance_name,
            routes=self.best_routes,
            cost=int(self.best_cost),
            output_dir=self.output_dir,
            suffix=suffix,
        )
        self.dirty = False


def install_termination_handlers(ckpt: Checkpoint) -> None:
    """
    Install handlers so best-so-far solution is written on:
    - SIGINT  (Ctrl+C)
    - SIGHUP  (hangup)
    - SIGUSR1 (SLURM pre-timeout, if configured)
    - SIGXCPU (CPU time limit)

    NOTE: SIGTERM is intentionally not handled (commented out).
    NOTE: SIGKILL cannot be caught.
    """

    def _handler(signum, frame):
        try:
            name = signal.Signals(signum).name
        except Exception:
            name = str(signum)

        inst = Path(ckpt.instance_name).stem
        print(
            f"\n\033[91m[{inst}] Received {name} → dumping checkpoint...\033[0m",
            flush=True,
        )

        ckpt.dump(suffix=name)
        raise SystemExit(128 + signum)

    for sig in (
        signal.SIGINT,
        # signal.SIGTERM,
        signal.SIGHUP,
        signal.SIGUSR1,
        signal.SIGXCPU,
    ):
        try:
            signal.signal(sig, _handler)
        except (ValueError, AttributeError, OSError):
            # ValueError: not main thread
            # AttributeError: signal not available on platform
            pass

    def _on_exit():
        if ckpt.dirty:
            ckpt.dump(suffix="ATEXIT")

    atexit.register(_on_exit)
