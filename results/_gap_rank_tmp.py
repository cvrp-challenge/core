import csv
from pathlib import Path

rows = []
with Path(__file__).with_name("summary.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append((row["instance"], float(row["gap_to_bks_percent"])))
rows.sort(key=lambda x: x[1])
print("BOTTOM")
for inst, g in rows[:10]:
    print(inst, g)
print("TOP")
for inst, g in rows[-10:][::-1]:
    print(inst, g)
