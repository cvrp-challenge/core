#!/usr/bin/env python
"""
Pure design-oriented visualisations of CVRP solutions for the XL instances.

Logic is shared with `plot_solutions.py` (same parsing of instances and
solutions), but the styling is different:

- no axes, ticks or grid
- plain white or black background
- all routes drawn in a single colour
  - light mode: black routes on white background
  - dark mode: white routes on black background
- depot shown as a small red square
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = ROOT / "instances" / "challenge-instances"
SOLUTIONS_DIR = ROOT / "results" / "out" / "solutions"
OUTPUT_DIR = ROOT / "results" / "out" / "design_plots"


Route = List[int]
Coord = Tuple[float, float]


def _parse_instance_coords(path: Path) -> Dict[int, Coord]:
    """Parse NODE_COORD_SECTION of a CVRPLib .vrp instance."""

    coords: Dict[int, Coord] = {}
    in_coords = False

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("NODE_COORD_SECTION"):
                in_coords = True
                continue

            if not in_coords:
                continue

            if line.startswith("DEMAND_SECTION") or line.startswith("DEPOT_SECTION"):
                break

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                idx = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
            except ValueError:
                continue

            coords[idx] = (x, y)

    if not coords:
        raise ValueError(f"No coordinates parsed from {path}")

    return coords


ROUTE_LINE_RE = re.compile(r"Route #\d+:\s*(.*)")


def _parse_solution_routes(path: Path) -> List[Route]:
    """Parse all `Route #i:` lines from a solution file."""

    routes: List[Route] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = ROUTE_LINE_RE.match(line.strip())
            if not m:
                continue

            tail = m.group(1).strip()
            if not tail:
                continue

            route: List[int] = []
            for tok in tail.split():
                try:
                    node = int(tok)
                except ValueError:
                    continue
                # Solution indexing: 0 = depot, 1..n-1 = customers.
                if node == 0:
                    continue
                route.append(node)

            if route:
                routes.append(route)

    if not routes:
        raise ValueError(f"No routes found in solution {path}")

    return routes


def _plot_design_instance(
    ax: Axes,
    coords: Mapping[int, Coord],
    routes: Sequence[Route],
    *,
    depot_index: int = 1,
    dark: bool = False,
) -> None:
    """Render routes in a minimalist, design-focused style."""

    bg = "black" if dark else "white"
    fg = "white" if dark else "black"

    ax.set_facecolor(bg)

    all_x: List[float] = []
    all_y: List[float] = []

    for route in routes:
        xs: List[float] = []
        ys: List[float] = []
        for sol_node in route:
            coord_idx = sol_node + 1  # convert solution index to instance index
            if coord_idx not in coords:
                continue
            x, y = coords[coord_idx]
            xs.append(x)
            ys.append(y)
            all_x.append(x)
            all_y.append(y)

        if not xs:
            continue

        ax.plot(xs, ys, "-", linewidth=0.4, color=fg, alpha=0.9)
        ax.scatter(xs, ys, s=3, color=fg, alpha=0.9)

    # Depot as a small red square (works on both backgrounds).
    if depot_index in coords:
        dx, dy = coords[depot_index]
        ax.scatter(
            [dx],
            [dy],
            s=30,
            marker="s",
            edgecolor="none",
            facecolor="red",
            zorder=5,
        )

    # Fixed coordinate system for consistency across instances.
    if all_x and all_y:
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)

    ax.set_aspect("equal", adjustable="box")

    # Remove all axes, ticks, and spines for a clean look.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)


def visualise_design_instance(instance_name: str, show: bool = False) -> Tuple[Path, Path]:
    """
    Create light and dark design plots for a single instance.

    Returns (light_path, dark_path).
    """

    instance_path = INSTANCES_DIR / f"{instance_name}.vrp"
    sol_path = SOLUTIONS_DIR / f"{instance_name}_probabilistic.sol"

    if not instance_path.is_file():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    if not sol_path.is_file():
        raise FileNotFoundError(f"Solution file not found: {sol_path}")

    coords = _parse_instance_coords(instance_path)
    routes = _parse_solution_routes(sol_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    light_path = OUTPUT_DIR / f"{instance_name}_design_light.png"
    dark_path = OUTPUT_DIR / f"{instance_name}_design_dark.png"

    # Light version.
    fig_l: Figure
    ax_l: Axes
    fig_l, ax_l = plt.subplots(figsize=(6, 6), dpi=300)
    fig_l.patch.set_facecolor("white")
    _plot_design_instance(ax_l, coords, routes, depot_index=1, dark=False)
    fig_l.tight_layout(pad=0.05)
    fig_l.savefig(light_path, facecolor=fig_l.get_facecolor(), bbox_inches="tight")

    # Dark version.
    fig_d: Figure
    ax_d: Axes
    fig_d, ax_d = plt.subplots(figsize=(6, 6), dpi=300)
    fig_d.patch.set_facecolor("black")
    _plot_design_instance(ax_d, coords, routes, depot_index=1, dark=True)
    fig_d.tight_layout(pad=0.05)
    fig_d.savefig(dark_path, facecolor=fig_d.get_facecolor(), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig_l)
        plt.close(fig_d)

    return light_path, dark_path


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Design-focused visualisations of XL CVRP solutions."
    )
    parser.add_argument(
        "instances",
        nargs="*",
        help=(
            "Instance base names (e.g. XL-n2634-k17). "
            "If omitted, defaults to XL-n2634-k17."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving PNG files.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.instances:
        to_plot = list(dict.fromkeys(args.instances))
    else:
        to_plot = ["XL-n2634-k17"]

    print(f"Saving design plots to {OUTPUT_DIR}")
    for name in to_plot:
        light_path, dark_path = visualise_design_instance(name, show=args.show)
        print(f"  - {name} (light): {light_path}")
        print(f"  - {name} (dark):  {dark_path}")


if __name__ == "__main__":
    main()

