#!/usr/bin/env python
"""
Plot per-instance improvement-source split vs k (minimum number of routes).

Y-axis: percentage (0-100)
X-axis: k (from instances_characteristics.json field min_routes)

Lines:
- % of improvements from routing
- % of improvements from scp

Plus linear trend lines for each series and Pearson r, p on trend legend entries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
CHAR_PATH = ROOT / "instances" / "challenge-instances" / "instances_characteristics.json"
SUMMARY_PATH = ROOT / "results" / "summary.csv"
TRAJECTORIES_DIR = ROOT / "results" / "out" / "trajectories"
DEFAULT_OUT = ROOT / "results" / "out" / "improvement_split_vs_k.png"


def _trajectory_points_csv_path(instance: str) -> Path:
    """
    Support both trajectory naming conventions:
      - results/out/trajectories/<instance>_trajectory_points.csv
      - results/out/trajectories/<instance>/trajectory_points.csv
    """
    p_new = TRAJECTORIES_DIR / f"{instance}_trajectory_points.csv"
    if p_new.is_file():
        return p_new

    p_legacy = TRAJECTORIES_DIR / instance / "trajectory_points.csv"
    if p_legacy.is_file():
        return p_legacy

    raise FileNotFoundError(
        f"Missing trajectory points CSV for {instance}: "
        f"checked {p_new} and {p_legacy}"
    )


def _improvement_split_percent(path: Path) -> tuple[float, float]:
    """Return (%routing, %scp) based on number of improvement events by source."""
    routing = 0
    scp = 0
    total = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            source = (row.get("source") or "").strip().lower()
            if source == "routing":
                routing += 1
                total += 1
            elif source == "scp":
                scp += 1
                total += 1

    if total == 0:
        return 0.0, 0.0

    return 100.0 * routing / total, 100.0 * scp / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output image path (default: results/out/improvement_split_vs_k.png)",
    )
    args = parser.parse_args()

    with CHAR_PATH.open("r", encoding="utf-8") as f:
        chars: dict = json.load(f)

    ks: list[float] = []
    routing_pct: list[float] = []
    scp_pct: list[float] = []

    with SUMMARY_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = (row.get("instance") or "").strip()
            if not inst:
                continue

            key = inst if inst.endswith(".vrp") else f"{inst}.vrp"
            if key not in chars:
                raise KeyError(f"No entry in instances_characteristics.json for {key!r}")

            k_val = float(chars[key]["min_routes"])
            traj_path = _trajectory_points_csv_path(inst)
            r_pct, s_pct = _improvement_split_percent(traj_path)

            ks.append(k_val)
            routing_pct.append(r_pct)
            scp_pct.append(s_pct)

    n = len(ks)
    if n != 100:
        raise RuntimeError(f"Expected 100 instances from summary.csv, found {n}")

    x = np.asarray(ks, dtype=np.float64)
    y_r = np.asarray(routing_pct, dtype=np.float64)
    y_s = np.asarray(scp_pct, dtype=np.float64)

    order = np.argsort(x)
    xs = x[order]
    yr = y_r[order]
    ys = y_s[order]

    slope_r, intercept_r = np.polyfit(xs, yr, 1)
    slope_s, intercept_s = np.polyfit(xs, ys, 1)
    r_routing, p_routing = pearsonr(xs, yr)
    r_scp, p_scp = pearsonr(xs, ys)
    x_line = np.array([xs.min(), xs.max()], dtype=np.float64)
    y_line_r = slope_r * x_line + intercept_r
    y_line_s = slope_s * x_line + intercept_s

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        xs,
        yr,
        color="tab:blue",
        marker="o",
        markersize=3.5,
        linewidth=1.0,
        alpha=0.65,
        label="% improvements from routing",
    )
    ax.plot(
        xs,
        ys,
        color="gold",
        marker="o",
        markersize=3.5,
        linewidth=1.0,
        alpha=0.75,
        label="% improvements from scp",
    )

    ax.plot(
        x_line,
        y_line_r,
        color="navy",
        linewidth=2.5,
        label=f"Routing trend line (r={r_routing:.6f}, p={p_routing:.6f})",
    )
    ax.plot(
        x_line,
        y_line_s,
        color="darkorange",
        linewidth=2.5,
        label=f"SCP trend line (r={r_scp:.6f}, p={p_scp:.6f})",
    )

    ax.set_xlabel("k")
    ax.set_ylabel("Improvement Share (%)")
    ax.set_ylim(0.0, 100.0)
    ax.set_title("Improvement-source split vs k")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
