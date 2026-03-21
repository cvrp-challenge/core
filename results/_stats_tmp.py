import csv, statistics
from pathlib import Path

p = Path(__file__).resolve().parent / "summary.csv"
cols = [
    "gap_to_paper_filo_best_percent",
    "gap_to_paper_filo2_best_percent",
    "gap_to_paper_hgs_cvrp_best_percent",
]
data = {c: [] for c in cols}
with p.open(encoding="utf-8") as f:
    for row in csv.DictReader(f):
        for c in cols:
            data[c].append(float(row[c]))
for c in cols:
    xs = data[c]
    print(f"{c}\tmean\t{statistics.mean(xs)}\tmedian\t{statistics.median(xs)}")
