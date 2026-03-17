#!/usr/bin/env python
"""
Visualize CVRP solutions for the XL challenge instances.

Uses the original instance geometry from `instances/challenge-instances`
and the final solutions from `results/out/solutions`.

For each instance, creates a PNG image where:
- background is white
- each route has a distinct colour
- the legs from and to the depot are omitted
- the depot is highlighted explicitly
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import hsv_to_rgb


ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = ROOT / "instances" / "challenge-instances"
SOLUTIONS_DIR = ROOT / "results" / "out" / "solutions"
OUTPUT_DIR = ROOT / "results" / "out" / "plots"


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

            # End of section.
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
                # In the solution files, node 0 is the depot and 1..n-1 are
                # customers. We keep customers and drop depot visits so that
                # we only draw customer-to-customer legs.
                if node == 0:
                    continue
                route.append(node)
            if route:
                routes.append(route)

    if not routes:
        raise ValueError(f"No routes found in solution {path}")

    return routes


def _generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """
    Generate `n` visually distinct RGB colours.

    We combine:
    - a fixed qualitative palette with browns, greys, and pastel tones
    - additional evenly spaced HSV hues (also in pastel-ish saturation/value)
    """

    if n <= 0:
        return []

    # Base palette mixing vivid "primary" colours, pastels, browns and greys.
    base_palette: List[Tuple[float, float, float]] = [
        # Strong / basic colours
        (0.90, 0.10, 0.10),  # red
        (0.10, 0.40, 0.95),  # blue
        (0.10, 0.75, 0.10),  # green
        (0.95, 0.75, 0.10),  # yellow
        (0.95, 0.50, 0.05),  # orange
        (0.60, 0.20, 0.80),  # purple
        (0.10, 0.80, 0.80),  # cyan
        (0.95, 0.10, 0.60),  # magenta
        # Pastel-ish bright tones
        (0.90, 0.30, 0.30),  # soft red
        (0.30, 0.60, 0.95),  # sky blue
        (0.30, 0.80, 0.50),  # mint
        (0.95, 0.75, 0.35),  # warm yellow
        (0.80, 0.40, 0.90),  # lavender
        (0.40, 0.85, 0.80),  # aqua
        (0.95, 0.55, 0.55),  # coral
        (0.55, 0.75, 0.40),  # olive green
        (0.95, 0.80, 0.55),  # sand
        (0.75, 0.60, 0.95),  # lilac
        # Browns and earth tones
        (0.65, 0.45, 0.30),  # brown
        (0.55, 0.35, 0.20),  # dark brown
        (0.80, 0.60, 0.45),  # light brown
        (0.70, 0.55, 0.35),  # ochre
        # Greys
        (0.35, 0.35, 0.35),  # dark grey
        (0.55, 0.55, 0.55),  # medium grey
        (0.75, 0.75, 0.75),  # light grey
        (0.60, 0.65, 0.70),  # bluish grey
    ]

    colors: List[Tuple[float, float, float]] = []

    # Interleave base palette and HSV-generated colours so similar tones
    # are spread out instead of clustered.
    # We guarantee deterministic order based on index `i`.
    total_needed = n
    hsv_needed = max(0, total_needed - len(base_palette))

    def hsv_color(idx: int, count: int) -> Tuple[float, float, float]:
        if count <= 0:
            # Fallback vivid hue wheel similar to the original behaviour.
            h = (0.11 + idx / max(1, total_needed)) % 1.0
            s = 0.85
            v = 0.95
        else:
            h = (0.07 + idx / count) % 1.0
            s = 0.60
            v = 0.96
        r, g, b = hsv_to_rgb([h, s, v])
        return float(r), float(g), float(b)

    base_len = len(base_palette)
    for i in range(total_needed):
        if i % 2 == 0:
            # Even indices prefer base palette when available.
            base_idx = i // 2
            if base_idx < base_len:
                colors.append(base_palette[base_idx])
            else:
                hsv_idx = base_idx - base_len
                colors.append(hsv_color(hsv_idx, hsv_needed))
        else:
            # Odd indices use HSV colours to mix between base entries.
            hsv_idx = i // 2
            colors.append(hsv_color(hsv_idx, hsv_needed))

    return colors[:n]


def _plot_instance(
    ax: Axes,
    coords: Mapping[int, Coord],
    routes: Sequence[Route],
    depot_index: int = 1,
) -> None:
    """Render all routes for a single instance on the given axes."""

    # Background
    ax.set_facecolor("white")

    # Plot each route with its own colour.
    colors = _generate_distinct_colors(len(routes))
    all_x: List[float] = []
    all_y: List[float] = []

    for route, color in zip(routes, colors):
        if len(route) == 1:
            x, y = coords[route[0]]
            ax.scatter(x, y, s=8, color=color, alpha=0.9)
            all_x.append(x)
            all_y.append(y)
            continue

        xs: List[float] = []
        ys: List[float] = []
        for sol_node in route:
            # Solution indices: depot=0, customers=1..n-1
            # Instance coordinates use 1-based indexing with depot at 1.
            coord_idx = sol_node + 1
            if coord_idx not in coords:
                continue
            x, y = coords[coord_idx]
            xs.append(x)
            ys.append(y)
            all_x.append(x)
            all_y.append(y)

        ax.plot(xs, ys, "-", linewidth=0.8, color=color, alpha=0.9)
        ax.scatter(xs, ys, s=6, color=color, alpha=0.9)

    # Highlight depot explicitly.
    if depot_index in coords:
        dx, dy = coords[depot_index]
        ax.scatter(
            [dx],
            [dy],
            s=80,
            marker="s",
            edgecolor="black",
            facecolor="yellow",
            linewidth=1.0,
            zorder=5,
        )

    # Layout: keep aspect ratio and show full [0, 1000] coordinate system
    # like in the reference figure.
    if all_x and all_y:
        # Most XL instances live roughly in [0, 1000] x [0, 1000]; we fix
        # the axes to this box so that plots across instances are comparable.
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)

    ax.set_aspect("equal", adjustable="box")

    # Add coordinate ticks and a grid for readability (every 200 units).
    ticks = list(range(0, 1001, 200))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(
        which="both",
        linestyle="--",
        linewidth=0.4,
        color="lightgray",
        alpha=0.9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")


def visualise_instance(instance_name: str, show: bool = False) -> Path:
    """
    Create a plot for a single instance.

    `instance_name` should be the base name, e.g. 'XL-n1094-k157'
    (without extension or '_probabilistic.sol').
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
    out_path = OUTPUT_DIR / f"{instance_name}.png"

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    fig.patch.set_facecolor("white")

    _plot_instance(ax, coords, routes, depot_index=1)

    fig.tight_layout(pad=0.05)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def _list_available_instances() -> List[str]:
    """Base names of instances that have a `_probabilistic.sol` solution."""
    names: List[str] = []
    for p in SOLUTIONS_DIR.glob("XL-*_*probabilistic.sol"):
        if not p.is_file():
            continue
        name = p.name
        if name.endswith("_probabilistic.sol"):
            names.append(name[: -len("_probabilistic.sol")])
    return sorted(names)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualise XL challenge CVRP solutions."
    )
    parser.add_argument(
        "instances",
        nargs="*",
        help=(
            "Instance base names (e.g. XL-n1094-k157). "
            "If omitted, the script will visualise three example instances."
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
        # If no specific instances are requested, visualise *all* instances
        # for which a `_probabilistic.sol` file exists.
        to_plot = _list_available_instances()

    if not to_plot:
        raise SystemExit("No instances to visualise.")

    print(f"Saving plots to {OUTPUT_DIR}")
    for name in to_plot:
        out_path = visualise_instance(name, show=args.show)
        print(f"  - {name}: {out_path}")


if __name__ == "__main__":
    main()

