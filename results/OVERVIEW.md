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
