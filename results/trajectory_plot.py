#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

# Ensure headless rendering (no GUI backend needed).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = ROOT / "instances" / "challenge-instances"
LOGS_DIR = ROOT / "results" / "out" / "logs"
OUT_DIR = ROOT / "results" / "out" / "trajectories"
CHALLENGE_BKS_PATH = INSTANCES_DIR / "challenge-bks.json"


BEST_COST_RE = re.compile(
    r"best_cost=(?P<cost>[0-9]+(?:\.[0-9]+)?)\s+\|\s+Gap:\s+(?P<gap>[0-9.]+)%"
)


@dataclass(frozen=True)
class NewBestPoint:
    iteration: int
    best_cost: float
    gap_percent: float
    source: str  # "routing" or "scp"


def _load_challenge_bks() -> Dict[str, float]:
    with CHALLENGE_BKS_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: float(v) for k, v in raw.items()}


def _gap_to_bks_percent(cost: float, bks_cost: float) -> float:
    return 100.0 * (cost - bks_cost) / bks_cost


def _parse_log_trajectory(log_path: Path, instance_name: str, bks_cost: float) -> Tuple[List[NewBestPoint], int]:
    """
    Extract only "new best" improvements, where best_cost strictly decreases.

    Returns:
      - points (chronological by discovery)
      - max_iteration_seen
    """
    points: List[NewBestPoint] = []
    best_cost_so_far: Optional[float] = None
    current_iteration: Optional[int] = None
    max_iteration_seen = 0

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            # Update current iteration context.
            # Example: [XL-n2634-k17 ITERATION 9] mode=... (not the ITERATION-TIMES lines)
            if "ITERATION-TIMES" not in line:
                m_iter = re.search(r"\bITERATION\s+(?P<iter>\d+)\b", line)
                if m_iter:
                    current_iteration = int(m_iter.group("iter"))
                    max_iteration_seen = max(max_iteration_seen, current_iteration)

            if "best_cost=" not in line:
                continue

            m_best = BEST_COST_RE.search(line)
            if not m_best:
                continue

            cost = float(m_best.group("cost"))

            # Only keep improvements.
            if best_cost_so_far is not None and not (cost < best_cost_so_far - 1e-12):
                continue

            if "IMPROVED-SCP" in line or "[%s IMPROVED-SCP]" % instance_name in line:
                source = "scp"
            else:
                # Covers IMPROVED-VB/RB, etc. as "routing/improvement".
                source = "routing"

            iteration = current_iteration if current_iteration is not None else max_iteration_seen
            points.append(
                NewBestPoint(
                    iteration=iteration,
                    best_cost=cost,
                    gap_percent=_gap_to_bks_percent(cost, bks_cost),
                    source=source,
                )
            )
            best_cost_so_far = cost

    return points, max_iteration_seen


def _plot_trajectory(
    ax: Axes,
    instance_name: str,
    points: List[NewBestPoint],
    max_iteration_seen: int,
) -> None:
    # Dark theme styling (similar "designy" look to other plots).
    ax.set_facecolor("black")
    fig = ax.figure
    fig.patch.set_facecolor("black")

    # Best-so-far curve (with vertical drops at new-best iterations).
    if not points:
        ax.set_title(f"{instance_name} trajectory (no data)")
        return

    points_sorted = sorted(points, key=lambda p: p.iteration)

    xs: List[int] = [points_sorted[0].iteration]
    ys: List[float] = [points_sorted[0].gap_percent]
    prev_gap = points_sorted[0].gap_percent

    for p in points_sorted[1:]:
        # Horizontal hold at previous best up to this iteration.
        xs.append(p.iteration)
        ys.append(prev_gap)
        # Vertical drop to new best at this iteration.
        xs.append(p.iteration)
        ys.append(p.gap_percent)
        prev_gap = p.gap_percent

    # Extend to the end of the run for visual clarity.
    last_iter = points_sorted[-1].iteration
    if max_iteration_seen > last_iter:
        xs.append(max_iteration_seen)
        ys.append(prev_gap)

    ax.plot(xs, ys, color="#d0d0d0", linewidth=2.0, alpha=0.9)

    # New-best markers.
    routing_color = "#00c2ff"  # cyan-blue
    scp_color = "#ffcc00"  # yellow

    routing_points = [p for p in points_sorted if p.source == "routing"]
    scp_points = [p for p in points_sorted if p.source == "scp"]

    if routing_points:
        ax.scatter(
            [p.iteration for p in routing_points],
            [p.gap_percent for p in routing_points],
            s=55,
            color=routing_color,
            edgecolors="none",
            label="routing / improvement",
            zorder=5,
        )
    if scp_points:
        ax.scatter(
            [p.iteration for p in scp_points],
            [p.gap_percent for p in scp_points],
            s=55,
            color=scp_color,
            edgecolors="none",
            label="scp",
            zorder=5,
        )

    # Reference line for BKS.
    ax.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.6)

    # Axes styling.
    ax.set_xlabel("Iteration", color="white")
    ax.set_ylabel("Gap to challenge BKS [%]", color="white")
    ax.tick_params(axis="both", colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444444")

    ax.grid(True, which="major", axis="y", linestyle="--", alpha=0.25, color="#666666")

    # Reasonable limits.
    y_min = min(0.0, min(p.gap_percent for p in points_sorted))
    y_max = max(p.gap_percent for p in points_sorted)
    pad = max(0.01, 0.08 * (y_max - y_min + 1e-9))
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.set_xlim(min(p.iteration for p in points_sorted), max_iteration_seen + 0.2)

    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor="white")

    ax.set_title(instance_name, color="white", pad=10)


def cmd_plot(args: argparse.Namespace) -> None:
    instance_name = args.instance
    log_path = LOGS_DIR / f"{instance_name}.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    bks_table = _load_challenge_bks()
    if instance_name not in bks_table:
        raise KeyError(f"Instance not found in {CHALLENGE_BKS_PATH}: {instance_name}")

    bks_cost = float(bks_table[instance_name])

    points, max_iter = _parse_log_trajectory(log_path, instance_name, bks_cost)
    if not points:
        raise RuntimeError(f"No new-best events found for {instance_name} in {log_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUT_DIR / f"{instance_name}_trajectory.png"
    csv_path = OUT_DIR / f"{instance_name}_trajectory_points.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iteration", "best_cost", "gap_percent", "source"])
        writer.writeheader()
        for p in sorted(points, key=lambda p: p.iteration):
            writer.writerow(
                {
                    "iteration": p.iteration,
                    "best_cost": f"{p.best_cost:.6f}",
                    "gap_percent": f"{p.gap_percent:.6f}",
                    "source": p.source,
                }
            )

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=200)
    _plot_trajectory(ax, instance_name, points, max_iter)
    fig.tight_layout(pad=0.4)
    fig.savefig(png_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {csv_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build trajectory chart for a single instance from benchmark logs.")
    p.add_argument(
        "instance",
        help="Instance name key used by challenge-bks.json (e.g. XL-n2634-k17)",
    )
    p.set_defaults(func=cmd_plot)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

