#!/usr/bin/env python
import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = ROOT / "instances" / "challenge-instances"
LOGS_DIR = ROOT / "results" / "out" / "logs"
SOLUTIONS_DIR = ROOT / "results" / "out" / "solutions"


@dataclass
class InstanceLogSummary:
    instance: str
    best_cost: Optional[float]
    best_gap: Optional[float]
    iterations: int


INSTANCE_RE = re.compile(r"instance=(?P<name>[^ ]+)")
BEST_RE = re.compile(r"best_cost=(?P<cost>[0-9]+(?:\.[0-9]+)?)\s+\|\s+Gap:\s+(?P<gap>[0-9.]+)%")


def list_instance_names() -> List[str]:
    """Return base names (without extension) of all challenge instances."""
    return sorted(
        p.stem
        for p in INSTANCES_DIR.glob("XL-*.vrp")
        if p.is_file()
    )


def list_solution_instance_names() -> List[str]:
    """Return base names (without `_probabilistic.sol`) for all final solutions."""
    result = []
    for p in SOLUTIONS_DIR.glob("XL-*_*probabilistic.sol"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith("_probabilistic.sol"):
            result.append(name[: -len("_probabilistic.sol")])
    return sorted(result)


def compute_coverage() -> Tuple[List[str], List[str]]:
    """Return (instances_with_solution, instances_without_solution)."""
    all_instances = list_instance_names()
    solved = set(list_solution_instance_names())
    with_solution = sorted(name for name in all_instances if name in solved)
    without_solution = sorted(name for name in all_instances if name not in solved)
    return with_solution, without_solution


def parse_log(path: Path) -> InstanceLogSummary:
    """Extract simple KPIs from a single log file."""
    best_cost: Optional[float] = None
    best_gap: Optional[float] = None
    iterations = 0
    instance_name: Optional[str] = None

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if instance_name is None:
                    m_inst = INSTANCE_RE.search(line)
                    if m_inst:
                        # e.g. instance=XL-n6168-k1922.vrp -> XL-n6168-k1922
                        instance_name = m_inst.group("name").split(".")[0]

                if "ITERATION " in line:
                    iterations += 1

                if "best_cost=" in line:
                    m = BEST_RE.search(line)
                    if m:
                        best_cost = float(m.group("cost"))
                        best_gap = float(m.group("gap"))
    except OSError:
        pass

    if instance_name is None:
        # Fall back to log file stem, assuming XL-*.log
        instance_name = path.stem

    return InstanceLogSummary(
        instance=instance_name,
        best_cost=best_cost,
        best_gap=best_gap,
        iterations=iterations,
    )


def iter_log_summaries() -> Iterable[InstanceLogSummary]:
    for log_path in LOGS_DIR.glob("XL-*.log"):
        if log_path.is_file():
            yield parse_log(log_path)


def cmd_coverage(_: argparse.Namespace) -> None:
    """Print which instances have / lack final `_probabilistic.sol` solutions."""
    with_solution, without_solution = compute_coverage()

    print("=== Coverage summary ===")
    print(f"Total instances: {len(with_solution) + len(without_solution)}")
    print(f"With solutions:  {len(with_solution)}")
    print(f"Without solutions: {len(without_solution)}")
    print()

    if without_solution:
        print("Instances without `_probabilistic.sol` solution:")
        for name in without_solution:
            print(f"  - {name}.vrp")


def cmd_summary(args: argparse.Namespace) -> None:
    """Summarize logs into a simple table (stdout or CSV)."""
    rows: List[Dict[str, object]] = []
    for s in iter_log_summaries():
        rows.append(
            {
                "instance": s.instance,
                "best_cost": s.best_cost if s.best_cost is not None else "",
                "best_gap_percent": s.best_gap if s.best_gap is not None else "",
                "iterations": s.iterations,
            }
        )

    rows.sort(key=lambda r: r["instance"])

    if args.output:
        out_path = Path(args.output)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["instance", "best_cost", "best_gap_percent", "iterations"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary for {len(rows)} instances to {out_path}")
    else:
        # simple text table
        print(f"{'instance':30} {'best_cost':>12} {'gap[%]':>8} {'iters':>6}")
        print("-" * 60)
        for r in rows:
            inst = str(r["instance"])
            cost = "" if r["best_cost"] == "" else f"{r['best_cost']:.0f}"
            gap = "" if r["best_gap_percent"] == "" else f"{r['best_gap_percent']:.4f}"
            iters = str(r["iterations"])
            print(f"{inst:30} {cost:>12} {gap:>8} {iters:>6}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse challenge logs and solution coverage.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cov = sub.add_parser("coverage", help="Show which instances have / lack final solutions.")
    p_cov.set_defaults(func=cmd_coverage)

    p_sum = sub.add_parser("summary", help="Summarize log KPIs (best cost, gap, iterations).")
    p_sum.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        help="Optional CSV output path. If omitted, prints a text table.",
    )
    p_sum.set_defaults(func=cmd_summary)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

