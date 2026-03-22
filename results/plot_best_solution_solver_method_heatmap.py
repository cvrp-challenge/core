#!/usr/bin/env python
"""
Heatmap: % of routes in final [BEST SOLUTION ROUTE SUMMARY] by composite method (VB/RB + algorithm)
and routing solver (last BEST block per log).

Aggregation (default): pool all routes across logs, then take % of the grand total.

With ``--per-instance-mean``: for each log compute % of that instance's final routes in each
cell, then average those percentages across instances (equal weight per instance; less bias
from large instances).

X-axis: pyvrp, filo1, filo2 (labels PyVRP, FILO1, FILO2)
Y-axis: 10 rows — VB methods (alphabetical) then RB methods (alphabetical)

Style: green sequential colormap, grid lines, annotated %.
Red frame around cell (VB_sk_kmeans, filo2).
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
SUMMARIZE_PATH = ROOT / "results" / "summarize_best_solution_route_logs.py"
LOGS_DIR = ROOT / "results" / "out" / "logs"
DEFAULT_OUT_POOLED = ROOT / "results" / "out" / "heatmap_best_solution_solver_method.png"
DEFAULT_OUT_PER_INSTANCE = (
    ROOT / "results" / "out" / "heatmap_best_solution_solver_method_per_instance_mean.png"
)

SOLVER_COLS = ("pyvrp", "filo1", "filo2")
HIGHLIGHT_ROW_KEY = "VB_sk_kmeans"
HIGHLIGHT_SOLVER = "filo2"


def _load_summarize():
    spec = importlib.util.spec_from_file_location("best_route_summary", SUMMARIZE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SUMMARIZE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _display_row_label(composite: str) -> str:
    """VB_sk_kmeans -> vb_sk_kmeans"""
    if "_" not in composite:
        return composite.lower()
    mode, _, rest = composite.partition("_")
    return f"{mode.lower()}_{rest}"


def _display_solver(s: str) -> str:
    if s.lower() == "pyvrp":
        return "PyVRP"
    return s.upper()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="Directory containing *.log files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default depends on --per-instance-mean)",
    )
    parser.add_argument(
        "--per-instance-mean",
        action="store_true",
        help=(
            "Average each log's cell %% (denominator = that instance's routes in final BEST), "
            "then mean across logs; writes heatmap_best_solution_solver_method_per_instance_mean.png by default"
        ),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow != 100 log files",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUT_PER_INSTANCE if args.per_instance_mean else DEFAULT_OUT_POOLED

    sm = _load_summarize()
    VB_METHODS = sm.VB_METHODS
    RB_METHODS = sm.RB_METHODS
    _iter_best_sections = sm._iter_best_sections
    _composite_clustering_key = sm._composite_clustering_key

    row_keys = [f"VB_{m}" for m in sorted(VB_METHODS)] + [f"RB_{m}" for m in sorted(RB_METHODS)]
    row_index = {k: i for i, k in enumerate(row_keys)}
    col_index = {s: j for j, s in enumerate(SOLVER_COLS)}

    log_files = sorted(args.logs_dir.glob("*.log"))
    if not log_files:
        raise SystemExit(f"No .log files under {args.logs_dir}")
    if not args.allow_missing and len(log_files) != 100:
        raise SystemExit(
            f"Expected 100 .log files, found {len(log_files)} (use --allow-missing)"
        )

    if args.per_instance_mean:
        pct_rows: list[np.ndarray] = []
        logs_skipped_no_section = 0
        for log_path in log_files:
            sections = _iter_best_sections(log_path)
            if not sections:
                logs_skipped_no_section += 1
                continue
            final = sections[-1]
            counts_k = np.zeros((len(row_keys), len(SOLVER_COLS)), dtype=np.float64)
            total_k = 0.0
            for n, mode, method, solver in final:
                total_k += n
                comp = _composite_clustering_key(mode, method)
                si = solver.lower()
                if comp not in row_index or si not in col_index:
                    continue
                counts_k[row_index[comp], col_index[si]] += n
            if total_k <= 0:
                logs_skipped_no_section += 1
                continue
            pct_rows.append(100.0 * counts_k / total_k)

        if not pct_rows:
            raise SystemExit("No routes parsed from final BEST sections; check logs.")

        pct = np.mean(np.stack(pct_rows, axis=0), axis=0)
        n_used = len(pct_rows)
        title_line2 = (
            f"Mean of per-instance % ({n_used} logs with final BEST; "
            f"denominator = each instance's route count)"
        )
        if logs_skipped_no_section:
            title_line2 += f"; skipped {logs_skipped_no_section} log(s) without data"
        cbar_label = "Mean share of routes (%)"
    else:
        counts = np.zeros((len(row_keys), len(SOLVER_COLS)), dtype=np.float64)
        total_routes = 0.0

        for log_path in log_files:
            sections = _iter_best_sections(log_path)
            if not sections:
                continue
            final = sections[-1]
            for n, mode, method, solver in final:
                total_routes += n
                comp = _composite_clustering_key(mode, method)
                si = solver.lower()
                if comp not in row_index or si not in col_index:
                    continue
                counts[row_index[comp], col_index[si]] += n

        if total_routes <= 0:
            raise SystemExit("No routes parsed from final BEST sections; check logs.")

        pct = 100.0 * counts / total_routes
        title_line2 = (
            f"Pooled: % of all routes ({len(log_files)} logs, last BEST each; "
            f"{int(total_routes)} routes total)"
        )
        cbar_label = "Share of routes (%)"

    vmax = float(np.max(pct)) if np.max(pct) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 10))
    # Greens range between full colormap and the previous light cap: not too dark, not too pale.
    greens_samples = plt.cm.Greens(np.linspace(0.12, 0.78, 256))
    cmap = mcolors.ListedColormap(greens_samples)
    cmap.set_bad(color="white")

    im = ax.imshow(pct, cmap=cmap, aspect="auto", vmin=0.0, vmax=vmax, origin="upper")

    ax.set_xticks(np.arange(len(SOLVER_COLS)))
    ax.set_yticks(np.arange(len(row_keys)))
    ax.set_xticklabels([_display_solver(s) for s in SOLVER_COLS])
    ax.set_yticklabels([_display_row_label(k) for k in row_keys])
    ax.set_title(
        "Route share in final best solution: solver × decomposition method\n" + title_line2,
        fontsize=12,
    )

    # Grid (cell borders)
    ax.set_xticks(np.arange(-0.5, len(SOLVER_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_keys), 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    hi_r = row_index.get(HIGHLIGHT_ROW_KEY)
    hi_c = col_index.get(HIGHLIGHT_SOLVER)
    if hi_r is not None and hi_c is not None:
        rect = Rectangle(
            (hi_c - 0.5, hi_r - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="red",
            linewidth=2.8,
        )
        ax.add_patch(rect)

    for i in range(len(row_keys)):
        for j in range(len(SOLVER_COLS)):
            val = pct[i, j]
            is_highlight = hi_r is not None and hi_c is not None and i == hi_r and j == hi_c
            ax.text(
                j,
                i,
                f"{val:.2f}%",
                ha="center",
                va="center",
                color="white" if is_highlight else "black",
                fontsize=9,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, rotation=270, labelpad=18)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
