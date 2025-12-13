# viz/plot_viz_instance.py
"""
Plot the depot and all customer nodes of a CVRP instance.
This module shows only the instance layout — no routes or clustering lines.
"""

import matplotlib.pyplot as plt
from master.utils.loader import load_instance


def plot_instance(instance_name: str, show_labels: bool = False):
    """
    Plots the depot and all customers of a CVRP instance.

    Args:
        instance_name (str): name of the .vrp instance file
        show_labels (bool): if True, display node IDs next to each point
    """
    instance = load_instance(instance_name)
    coords = instance["node_coord"]
    depot = coords[0]

    plt.figure(figsize=(8, 8))
    plt.title(instance.get("name", instance_name))

    # --- Plot depot ---
    plt.scatter(
        depot[0], depot[1],
        c="yellow", edgecolors="black", marker="s", s=150,
        label="Depot"
    )

    # --- Plot customers ---
    plt.scatter(
        coords[1:, 0], coords[1:, 1],
        c="blue", s=30,
        label=f"Customers (n={len(coords)-1})"
    )

    # --- Optional: label nodes ---
    if show_labels:
        for i, (x, y) in enumerate(coords):
            label = "D" if i == 0 else str(i)
            plt.text(x + 5, y + 5, label, fontsize=8)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_instance("X-n101-k25.vrp", show_labels=False)
