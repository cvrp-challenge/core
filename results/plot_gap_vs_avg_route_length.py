#!/usr/bin/env python
"""
Scatter plot: average route length n/k (from instances_characteristics.json) vs
gap_to_bks_percent (from results/summary.csv), with an OLS regression line and
Pearson correlation (r, p) annotated.
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
DEFAULT_OUT = ROOT / "results" / "out" / "gap_vs_avg_route_length.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output image path (default: results/out/gap_vs_avg_route_length.png)",
    )
    args = parser.parse_args()

    with CHAR_PATH.open(encoding="utf-8") as f:
        chars: dict = json.load(f)

    avg_route_lengths: list[float] = []
    gaps: list[float] = []
    with SUMMARY_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row["instance"].strip()
            key = inst if inst.endswith(".vrp") else f"{inst}.vrp"
            if key not in chars:
                raise KeyError(f"No entry in instances_characteristics.json for {key!r}")
            avg_route_lengths.append(float(chars[key]["avg_route_length"]))
            gaps.append(float(row["gap_to_bks_percent"]))

    n = len(avg_route_lengths)
    if n != 100:
        raise RuntimeError(f"Expected 100 instances in summary.csv, found {n}")

    x = np.asarray(avg_route_lengths, dtype=np.float64)
    y = np.asarray(gaps, dtype=np.float64)
    r, p_value = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.array([x.min(), x.max()])
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        x,
        y,
        s=28,
        c="tab:green",
        alpha=0.75,
        edgecolors="black",
        linewidths=0.35,
        zorder=3,
    )
    ax.plot(x_line, y_line, color="black", linewidth=2.0, zorder=2, label="OLS Line")
    ax.set_xlabel("Avg. Route Length (n/k)")
    ax.set_ylabel("Gap to BKS (%)")
    ax.set_title("Gap to best known solution vs average route length")
    ax.grid(True, alpha=0.3)
    stats_text = f"$r = {r:.6f}$\n$p = {p_value:.6f}$"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.legend(loc="lower right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
