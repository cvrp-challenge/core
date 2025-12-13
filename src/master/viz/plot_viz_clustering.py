# viz/plot_viz_clustering.py
"""
Visualize CVRP clustering results.
Each cluster is shown in a distinct color, including the depot and optional medoids/centroids.
Supports both:
    - medoids: node IDs (int)
    - centroids: coordinate pairs (x, y)
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from master.utils.loader import load_instance
import matplotlib as mpl


def plot_clustering(
    instance_name: str,
    clusters: dict,
    medoids: dict = None,
    show_labels: bool = False,
):
    """
    Plots clustered customers of a CVRP instance.

    Args:
        instance_name (str): name of the .vrp instance file
        clusters (dict): mapping {cluster_id: [node_ids], ...}
        medoids (dict): optional mapping {cluster_id: medoid_node or (x, y) centroid}
        show_labels (bool): if True, display customer IDs
    """
    instance = load_instance(instance_name)
    coords = instance["node_coord"]
    depot = coords[0]

    plt.figure(figsize=(8, 8))
    plt.title(f"Cluster Visualization — {instance.get('name', instance_name)}")

    # --- Plot depot ---
    plt.scatter(
        depot[0], depot[1],
        c="yellow", edgecolors="black", marker="s", s=150,
        label="Depot"
    )


    # --- Color setup ---
    n_clusters = len(clusters)
    cmap = mpl.colormaps.get_cmap("tab10")

    # --- Plot each cluster ---
    for idx, (cid, members) in enumerate(clusters.items()):
        color = cmap(idx / max(1, n_clusters - 1))
        member_coords = [coords[i - 1] for i in members]
        xs, ys = zip(*member_coords)
        plt.scatter(xs, ys, c=[color], s=30, label=f"Cluster {cid} (n={len(members)})")


        # --- Highlight medoid or centroid ---
        if medoids and cid in medoids:
            medoid = medoids[cid]

            # Handle both node ID and coordinate tuple
            if isinstance(medoid, int):
                mx, my = coords[medoid - 1]
            elif isinstance(medoid, (tuple, list)) and len(medoid) == 2:
                mx, my = medoid
            else:
                continue  # skip invalid

            plt.scatter(mx, my, c=[color], edgecolors="black", marker="X", s=120, linewidths=1.2)
            plt.text(mx + 5, my + 5, f"m{cid}", fontsize=8, fontweight="bold")

    # --- Optional: label all nodes ---
    if show_labels:
        for i, (x, y) in enumerate(coords):
            label = "D" if i == 0 else str(i)
            plt.text(x + 4, y + 4, label, fontsize=7)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # --- Example 1: Medoid-based clustering (like k-medoids or AC) ---
    clusters_medoid = {
        1: [2, 3, 4, 5, 6],
        2: [7, 8, 9, 10, 11],
        3: [12, 13, 14, 15, 16],
    }
    medoids_medoid = {1: 3, 2: 8, 3: 13}

    print("→ Plotting medoid-based clustering visualization...")
    plot_clustering("X-n101-k25.vrp", clusters_medoid, medoids_medoid)

    # --- Example 2: Centroid-based clustering (like k-means) ---
    clusters_centroid = {
        1: [2, 3, 4, 5, 6],
        2: [7, 8, 9, 10, 11],
        3: [12, 13, 14, 15, 16],
    }
    # Artificial centroids (x, y) — not tied to specific nodes
    medoids_centroid = {
        1: (400, 600),
        2: (700, 800),
        3: (300, 300),
    }

    print("→ Plotting centroid-based clustering visualization...")
    plot_clustering("X-n101-k25.vrp", clusters_centroid, medoids_centroid)
