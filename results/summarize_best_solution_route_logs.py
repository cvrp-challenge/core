#!/usr/bin/env python
"""
Aggregate statistics from the *final* [BEST SOLUTION ROUTE SUMMARY] in each run log.

Scans results/out/logs/*.log. For each file, only the **last** BEST block is used (final
best solution). Each block lists rows like:
  N routes | VB | <clustering_method> | solver=<name> | ...

Stops each block at a blank line or [ROUTE POOL SUMMARY] (does not include pool rows).

Outputs:
- RB vs VB: route-weighted split over those final rows (pooled across logs), and mean VB%
  per log (each instance weighted equally).
- Decomposition-aware clustering split: routes bucketed as ``VB_<vb_method>`` or
  ``RB_<rb_method>`` (10 categories: 6 value-based + 4 route-based), matching
  ``run_drsci_probabilistic.py`` — the same string ``method`` in a log means different
  procedures under VB vs RB.
- Solver split: % of routes by routing solver (weighted by route counts).

Requires exactly 100 *.log files by default (--allow-missing to relax).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "results" / "out" / "logs"
DEFAULT_OUT = ROOT / "results" / "out" / "best_solution_route_summary_stats.txt"

HEADER = "[BEST SOLUTION ROUTE SUMMARY]"
POOL = "[ROUTE POOL SUMMARY]"
# Message part after standard log prefix "... | INFO     | "
INFO_SPLIT = re.compile(r"\|\s*INFO\s+\|\s*")

# Must match src/master/run_drsci_probabilistic.py (VB = run_clustering, RB = route_based_decomposition).
VB_METHODS = (
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
    "fcm",
    "k_medoids_pyclustering",
)
RB_METHODS = (
    "sk_ac_avg",
    "sk_ac_complete",
    "sk_ac_min",
    "sk_kmeans",
)


def _composite_clustering_key(mode: str, method: str) -> str:
    """VB_<method> or RB_<method> — 10 semantic categories when method is in the known sets."""
    return f"{mode}_{method}"


def _expected_composite_keys_in_order() -> list[str]:
    return [f"VB_{m}" for m in VB_METHODS] + [f"RB_{m}" for m in RB_METHODS]


def _composite_sort_key(name: str) -> tuple[int, int, str]:
    if name.startswith("VB_"):
        m = name[3:]
        if m in VB_METHODS:
            return (0, VB_METHODS.index(m), name)
        return (2, 0, name)
    if name.startswith("RB_"):
        m = name[3:]
        if m in RB_METHODS:
            return (1, RB_METHODS.index(m), name)
        return (2, 0, name)
    return (2, 0, name)


def _message(line: str) -> str:
    m = INFO_SPLIT.search(line)
    if not m:
        return ""
    return line[m.end() :].strip()


def _parse_summary_row(msg: str) -> tuple[int, str, str, str] | None:
    """Return (n_routes, mode VB|RB, clustering_method, solver_name) or None."""
    if " routes |" not in msg:
        return None
    parts = [p.strip() for p in msg.split("|")]
    if len(parts) < 4:
        return None
    head = parts[0]
    if not head.endswith("routes"):
        return None
    try:
        n = int(head.replace("routes", "").strip())
    except ValueError:
        return None
    mode = parts[1]
    if mode not in ("VB", "RB"):
        return None
    method = parts[2]
    sol_field = parts[3]
    if not sol_field.startswith("solver="):
        return None
    # solver=filo2      | stage=... -> take token before space or |
    solver_token = sol_field.split()[0]
    solver = solver_token.split("=", 1)[1].strip()
    return (n, mode, method, solver)


def _iter_best_sections(path: Path) -> list[list[tuple[int, str, str, str]]]:
    """Return list of sections; each section is a list of parsed rows."""
    sections: list[list[tuple[int, str, str, str]]] = []
    in_block = False
    current: list[tuple[int, str, str, str]] = []

    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            msg = _message(line)
            if not msg:
                if in_block and current:
                    sections.append(current)
                    current = []
                    in_block = False
                continue

            if msg == HEADER:
                if in_block and current:
                    sections.append(current)
                in_block = True
                current = []
                continue

            if not in_block:
                continue

            if msg.startswith("---"):
                continue
            if msg == POOL or msg.startswith(POOL):
                if current:
                    sections.append(current)
                current = []
                in_block = False
                continue

            parsed = _parse_summary_row(msg)
            if parsed is not None:
                current.append(parsed)
            else:
                # Unexpected line inside block; close without treating as data.
                if current:
                    sections.append(current)
                current = []
                in_block = False

    if in_block and current:
        sections.append(current)

    return sections


def _section_rb_vb_counts(rows: list[tuple[int, str, str, str]]) -> tuple[int, int]:
    vb = sum(n for n, mode, _, _ in rows if mode == "VB")
    rb = sum(n for n, mode, _, _ in rows if mode == "RB")
    return rb, vb


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
        default=DEFAULT_OUT,
        help="Text report path",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional JSON output path with the same aggregates",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not require exactly 100 log files",
    )
    args = parser.parse_args()

    log_files = sorted(args.logs_dir.glob("*.log"))
    if not log_files:
        raise SystemExit(f"No .log files under {args.logs_dir}")

    if not args.allow_missing and len(log_files) != 100:
        raise SystemExit(
            f"Expected 100 .log files, found {len(log_files)} under {args.logs_dir} "
            "(use --allow-missing to override)"
        )

    per_log_vb_pcts: list[float] = []
    logs_without_best: list[str] = []

    total_routes = 0
    total_vb = 0
    total_rb = 0
    method_routes: Counter[str] = Counter()
    solver_routes: Counter[str] = Counter()
    unexpected_clustering: Counter[str] = Counter()

    per_file_section_count: list[tuple[str, int]] = []

    for log_path in log_files:
        sections = _iter_best_sections(log_path)
        per_file_section_count.append((log_path.name, len(sections)))
        final = sections[-1] if sections else None
        if final is None:
            logs_without_best.append(log_path.name)
            per_log_vb_pcts.append(0.0)
            continue

        rb, vb = _section_rb_vb_counts(final)
        t = rb + vb
        per_log_vb_pcts.append(100.0 * vb / t if t > 0 else 0.0)

        for n, _mode, method, solver in final:
            total_routes += n
            composite = _composite_clustering_key(_mode, method)
            method_routes[composite] += n
            if _mode == "VB" and method not in VB_METHODS:
                unexpected_clustering[composite] += n
            elif _mode == "RB" and method not in RB_METHODS:
                unexpected_clustering[composite] += n
            solver_routes[solver] += n
            if _mode == "VB":
                total_vb += n
            else:
                total_rb += n

    def pct(part: int, whole: int) -> float:
        return 100.0 * part / whole if whole else 0.0

    weighted_vb_pct = pct(total_vb, total_vb + total_rb)
    weighted_rb_pct = pct(total_rb, total_vb + total_rb)

    mean_vb_pct_per_log = mean(per_log_vb_pcts) if per_log_vb_pcts else 0.0
    n_logs_with_final_best = len(log_files) - len(logs_without_best)

    method_split = {m: pct(c, total_routes) for m, c in method_routes.items()}
    solver_split = {s: pct(c, total_routes) for s, c in solver_routes.most_common()}
    sorted_composites = sorted(method_routes.keys(), key=_composite_sort_key)

    lines: list[str] = []
    lines.append("BEST SOLUTION ROUTE SUMMARY — aggregate report (final best only)")
    lines.append(f"Logs directory: {args.logs_dir}")
    lines.append(f"Log files: {len(log_files)}")
    lines.append(
        f"Logs with a final BEST section: {n_logs_with_final_best} "
        f"(missing: {len(logs_without_best)})"
    )
    lines.append(f"Total routes (sum of N in final BEST rows): {total_routes}")
    if logs_without_best:
        lines.append("Logs with no [BEST SOLUTION ROUTE SUMMARY]:")
        for name in logs_without_best:
            lines.append(f"  - {name}")
        lines.append("")
    lines.append("")
    lines.append("=== RB vs VB ===")
    lines.append(
        f"Route-weighted (pooled final BEST rows): VB {weighted_vb_pct:.4f}%  |  RB {weighted_rb_pct:.4f}%"
    )
    lines.append(
        f"Mean VB% per log (each instance weighted equally, n={len(log_files)}): "
        f"{mean_vb_pct_per_log:.4f}%"
    )
    lines.append("")
    lines.append(
        "=== Clustering split: VB_method vs RB_method (10 categories; % of routes) ==="
    )
    lines.append(
        "    Keys are decomposition-style: VB_* = value-based clustering, RB_* = route-based."
    )
    for m in sorted_composites:
        p = method_split[m]
        lines.append(f"  {m}: {p:.4f}%  ({method_routes[m]} routes)")
    if unexpected_clustering:
        lines.append("")
        lines.append("  (routes with method name not in expected VB_METHODS/RB_METHODS for that mode:)")
        for u, cnt in unexpected_clustering.most_common():
            lines.append(f"    {u}: {cnt} routes")
    lines.append("")
    lines.append("=== Solver split (% of routes in final BEST rows) ===")
    for s, p in sorted(solver_split.items(), key=lambda x: (-solver_split[x[0]], x[0])):
        lines.append(f"  {s}: {p:.4f}%  ({solver_routes[s]} routes)")
    lines.append("")
    lines.append("=== Per-file: total BEST sections in log (final = last of these) ===")
    for name, k in per_file_section_count:
        lines.append(f"  {name}: {k} section(s)")

    report = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")

    payload = {
        "logs_dir": str(args.logs_dir),
        "n_log_files": len(log_files),
        "n_logs_with_final_best_section": n_logs_with_final_best,
        "logs_missing_best_section": logs_without_best,
        "total_routes_in_final_best_rows": total_routes,
        "rb_vb": {
            "weighted_vb_percent": weighted_vb_pct,
            "weighted_rb_percent": weighted_rb_pct,
            "mean_vb_percent_per_log": mean_vb_pct_per_log,
        },
        "vb_methods": list(VB_METHODS),
        "rb_methods": list(RB_METHODS),
        "expected_composite_keys": _expected_composite_keys_in_order(),
        "clustering_split_percent_by_mode_and_method": method_split,
        "clustering_route_counts_by_mode_and_method": dict(method_routes),
        "clustering_unexpected_route_counts": dict(unexpected_clustering)
        if unexpected_clustering
        else {},
        "solver_split_percent": solver_split,
        "solver_route_counts": dict(solver_routes),
        "sections_per_log": {n: k for n, k in per_file_section_count},
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
