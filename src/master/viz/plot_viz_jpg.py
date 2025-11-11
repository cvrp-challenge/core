import os
import matplotlib.pyplot as plt

# Automatically find the "xl" folder inside "test-instances"
BASE_DIR = os.path.dirname(__file__)
instance_folderXL = os.path.join(BASE_DIR, "test-instances", "xl")
instance_folderX = os.path.join(BASE_DIR, "test-instances", "x")

def read_instance_coords(filepath):
    """
    Reads the coordinates from a .vrp instance file.
    Returns a list of (x, y) tuples.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    coords_section = False
    coords = []
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            coords_section = True
            continue
        if "DEMAND_SECTION" in line or "DEPOT_SECTION" in line:
            coords_section = False
        if coords_section:
            parts = line.strip().split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    return coords

def plot_instance(filepath):
    """
    Creates a scatter plot of the VRP instance and saves it as a .png file.
    """
    coords = read_instance_coords(filepath)
    if not coords:
        print(f"⚠️ No coordinates found in {os.path.basename(filepath)}, skipping.")
        return

    depot = coords[0]
    customers = coords[1:]

    plt.figure(figsize=(10, 10))
    plt.scatter(
        [c[0] for c in customers],
        [c[1] for c in customers],
        color='blue',
        s=30,
        label=f'Customers (n={len(customers)})'  # 👈 added node count
    )
    plt.scatter(
        depot[0],
        depot[1],
        color='yellow',
        edgecolor='black',
        s=150,
        marker='s',
        label='Depot'
    )

    plt.title(os.path.basename(filepath))
    plt.legend(loc="upper right")
    plt.tight_layout()

    output_path = filepath.replace(".vrp", ".png")
    if os.path.exists(output_path):
        print(f"⏭️ Skipping existing {os.path.basename(output_path)}")
        plt.close()
        return

    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved: {os.path.basename(output_path)}")

# ---- Main execution ----
if __name__ == "__main__":
    print(f"📂 Reading instances from: {instance_folderX}")
    if not os.path.exists(instance_folderX):
        print("❌ Error: instance folder not found!")
        exit(1)

    vrp_files = [f for f in os.listdir(instance_folderX) if f.endswith(".vrp")]
    if not vrp_files:
        print("❌ No .vrp files found in folder.")
        exit(0)

    for filename in vrp_files:
        plot_instance(os.path.join(instance_folderX, filename))
    # plot_instance(os.path.join(instance_folder, "XLTEST-n4951-k225.vrp"))   # single Instance visualization


    print("\n🎉 All available instance visualizations created!")