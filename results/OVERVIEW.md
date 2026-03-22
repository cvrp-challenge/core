## Results summary

- **Instances solved**: 100 / 100 challenge instances have final `_probabilistic.sol` solutions in `results/out/solutions`.
- **Average gap to BKS**: 0.6198% (mean over all 100 instances, using costs from solution files vs. `challenge-bks.json`).
- **Median gap to BKS**: 0.5721%.
- **Average gap to initial BKS**: 0.5339% (mean over all 100 instances, using costs from solution files vs. `initial-bks.json`).
- **Median gap to initial BKS**: 0.4837%.

### Gaps vs. paper Table 2 baselines (best FILO / FILO2 / HGS-CVRP per instance, 100 instances)

- **Average gap vs. FILO (paper best)**: 0.3820%.
- **Median gap vs. FILO (paper best)**: 0.3391%.
- **Average gap vs. FILO2 (paper best)**: 0.4157%.
- **Median gap vs. FILO2 (paper best)**: 0.3759%.
- **Average gap vs. HGS-CVRP (paper best)**: -0.5667%.
- **Median gap vs. HGS-CVRP (paper best)**: -0.3840%.

### Gaps vs. paper Table 2 baselines (mean FILO / FILO2 / HGS-CVRP per instance, 100 instances)

- **Average gap vs. FILO (paper mean)**: 0.2850%.
- **Median gap vs. FILO (paper mean)**: 0.2496%.
- **Average gap vs. FILO2 (paper mean)**: 0.3248%.
- **Median gap vs. FILO2 (paper mean)**: 0.2741%.
- **Average gap vs. HGS-CVRP (paper mean)**: -0.8066%.
- **Median gap vs. HGS-CVRP (paper mean)**: -0.6533%.

### Gap distribution (vs. BKS, 100 instances)

- **gap ≤ 0.10%**:         9
- **0.10% < gap ≤ 0.20%**: 3
- **0.20% < gap ≤ 0.30%**: 5
- **0.30% < gap ≤ 0.40%**: 7
- **0.40% < gap ≤ 0.50%**: 13
- **0.50% < gap ≤ 0.60%**: 18
- **0.60% < gap ≤ 0.70%**: 13
- **0.70% < gap ≤ 0.80%**: 8
- **0.80% < gap ≤ 0.90%**: 6
- **0.90% < gap ≤ 1.00%**: 8
- **1.00% < gap ≤ 1.25%**: 3
- **1.25% < gap ≤ 1.50%**: 6
- **1.50% < gap ≤ 1.75%**: 0
- **1.75% < gap ≤ 2.00%**: 0
- **2.00% < gap ≤ 2.25%**: 0
- **gap > 2.25%**:         1

### Extremes by gap to BKS (`gap_to_bks_percent`, 100 instances)

Sorted from `results/summary.csv` (same definition as the gap bullets above: solution cost vs. `challenge-bks.json`). **Bottom** = smallest gap (closest to BKS); **top** = largest gap (farthest from BKS).

**Bottom 10 (lowest gap)**

| Rank | Instance | gap_to_bks_percent |
|-----:|----------|-------------------:|
| 1 | XL-n4535-k1134 | 0.0197% |
| 2 | XL-n2727-k546 | 0.0262% |
| 3 | XL-n7353-k1471 | 0.0319% |
| 4 | XL-n3484-k436 | 0.0502% |
| 5 | XL-n1094-k157 | 0.0605% |
| 6 | XL-n1794-k163 | 0.0670% |
| 7 | XL-n1561-k75 | 0.0798% |
| 8 | XL-n5526-k553 | 0.0852% |
| 9 | XL-n2214-k131 | 0.0912% |
| 10 | XL-n2541-k121 | 0.1490% |

**Top 10 (highest gap)**

| Rank | Instance | gap_to_bks_percent |
|-----:|----------|-------------------:|
| 1 | XL-n9571-k55 | 2.3593% |
| 2 | XL-n5174-k55 | 1.4807% |
| 3 | XL-n2634-k17 | 1.4548% |
| 4 | XL-n3888-k1010 | 1.4356% |
| 5 | XL-n8207-k108 | 1.3469% |
| 6 | XL-n5288-k1246 | 1.2962% |
| 7 | XL-n6034-k61 | 1.2934% |
| 8 | XL-n6884-k148 | 1.2418% |
| 9 | XL-n2028-k617 | 1.2136% |
| 10 | XL-n9160-k379 | 1.1147% |

### Gap distribution (vs. initial BKS, 100 instances)

- **gap ≤ 0.10%**:         9
- **0.10% < gap ≤ 0.20%**: 3
- **0.20% < gap ≤ 0.30%**: 10
- **0.30% < gap ≤ 0.40%**: 13
- **0.40% < gap ≤ 0.50%**: 18
- **0.50% < gap ≤ 0.60%**: 17
- **0.60% < gap ≤ 0.70%**: 6
- **0.70% < gap ≤ 0.80%**: 4
- **0.80% < gap ≤ 0.90%**: 10
- **0.90% < gap ≤ 1.00%**: 2
- **1.00% < gap ≤ 1.25%**: 4
- **1.25% < gap ≤ 1.50%**: 3
- **1.50% < gap ≤ 1.75%**: 0
- **1.75% < gap ≤ 2.00%**: 1
- **2.00% < gap ≤ 2.25%**: 0
- **gap > 2.25%**:         0

### Runtime summary by stage (all 100 instances)

| Stage | Total runtime [s] | Share of total runtime [%] | Average share per instance with stage [%] |
|-------|-------------------:|---------------------------:|-------------------------------------------:|
| cluster | 139765.605 | 3.84 | 3.79 |
| ls | 269786.674 | 7.41 | 7.33 |
| routing | 2726751.388 | 74.94 | 73.43 |
| scp | 670664.472 | 18.43 | 15.43 |

### Trajectory improvements (new best events)

- **Average iterations (max iter with a new best)**: 18.57
- **Average #improvements**: 5.46
- **Average #improvements via routing**: 2.71
- **Average #improvements via scp**: 2.75

- **Share of all improvements from routing/scp**: 49.63% routing, 50.37% scp
- **Average iteration index of improvements**: routing 6.75, scp 14.45
