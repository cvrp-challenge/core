#!/usr/bin/env python
"""
Box plots of gap_to_bks_percent (y) by demand distribution bucket
(from instances_characteristics.json field demand_distribution).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CHAR_PATH = ROOT / "instances" / "challenge-instances" / "instances_characteristics.json"
SUMMARY_PATH = ROOT / "results" / "summary.csv"
DEFAULT_OUT = ROOT / "results" / "out" / "gap_vs_demand_distribution.png"

BUCKET_ORDER = ("U", "1-10", "5-10", "1-100", "50-100", "Q", "SL")


def demand_distribution_bucket(raw: str) -> str:
    s = raw.strip()
    if s in BUCKET_ORDER:
        return s
    raise ValueError(f"Unmapped demand_distribution: {raw!r} (expected one of {BUCKET_ORDER})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output image path (default: results/out/gap_vs_demand_distribution.png)",
    )
    args = parser.parse_args()

    with CHAR_PATH.open(encoding="utf-8") as f:
        chars: dict = json.load(f)

    by_bucket: dict[str, list[float]] = {b: [] for b in BUCKET_ORDER}
    with SUMMARY_PATH.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst = row["instance"].strip()
            key = inst if inst.endswith(".vrp") else f"{inst}.vrp"
            if key not in chars:
                raise KeyError(f"No entry in instances_characteristics.json for {key!r}")
            raw = str(chars[key]["demand_distribution"])
            b = demand_distribution_bucket(raw)
            by_bucket[b].append(float(row["gap_to_bks_percent"]))

    n = sum(len(v) for v in by_bucket.values())
    if n != 100:
        raise RuntimeError(f"Expected 100 instances in summary.csv, found {n}")

    data = [by_bucket[b] for b in BUCKET_ORDER]
    labels = list(BUCKET_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    _ver = tuple(int(x) for x in matplotlib.__version__.split(".")[:2])
    _box_kw = {"patch_artist": True, "widths": 0.55}
    if _ver >= (3, 9):
        bp = ax.boxplot(data, tick_labels=labels, **_box_kw)
    else:
        bp = ax.boxplot(data, labels=labels, **_box_kw)
    for patch in bp["boxes"]:
        patch.set_facecolor("tab:green")
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
    for name in ("whiskers", "caps"):
        for line in bp[name]:
            line.set_color("black")
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(2.0)
    for line in bp["fliers"]:
        line.set_markerfacecolor("tab:green")
        line.set_markeredgecolor("black")
        line.set_markeredgewidth(0.35)
        line.set_alpha(0.75)

    ax.set_xlabel("Demand Distribution")
    ax.set_ylabel("Gap to BKS (%)")
    ax.set_title("Gap to best known solution vs demand distribution")
    ax.grid(True, axis="y", alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
