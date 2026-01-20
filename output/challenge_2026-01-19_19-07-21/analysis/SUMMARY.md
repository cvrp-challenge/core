# Challenge Test Run Analysis Summary

**Date:** 2026-01-19  
**Duration:** ~10 hours  
**Instances:** 26

## Task 1: Route Pool Statistics

### Per-Instance Summary

Each instance's final route pool was analyzed by:
- **Mode**: VB (Vehicle-Based), RB (Route-Based), SCP (Set Cover Problem)
- **Method**: Clustering method used (sk_kmeans, sk_ac_min, sk_ac_avg, sk_ac_complete, fcm, k_medoids_pyclustering, gurobi_mip)
- **Solver**: Routing solver used (filo1, filo2, pyvrp)
- **Stage**: Processing stage (post_ls, scp_post_ls, final_scp_post_ls)

### Aggregate Statistics (All 26 Instances)

#### By Mode
- **VB (Vehicle-Based)**: 89,023 routes
- **RB (Route-Based)**: 56,778 routes  
- **SCP (Set Cover Problem)**: 8,293 routes

#### By Method
- **sk_kmeans**: 40,844 routes
- **sk_ac_min**: 28,630 routes
- **sk_ac_complete**: 28,511 routes
- **sk_ac_avg**: 25,359 routes
- **fcm**: 12,227 routes
- **k_medoids_pyclustering**: 10,230 routes
- **gurobi_mip**: 8,293 routes (SCP routes)

#### By Solver
- **filo2**: 62,650 routes
- **filo1**: 51,302 routes
- **pyvrp**: 31,849 routes
- **UNKNOWN**: 8,293 routes (SCP routes)

#### By Stage
- **post_ls**: 145,801 routes
- **scp_post_ls**: 7,319 routes
- **final_scp_post_ls**: 974 routes

### Top Route Pool Combinations

The most common combinations of (mode, method, solver, stage):

1. (VB, sk_kmeans, filo2, post_ls): 16,785 routes
2. (RB, sk_kmeans, filo2, post_ls): 7,664 routes
3. (SCP, gurobi_mip, UNKNOWN, scp_post_ls): 7,319 routes
4. (VB, sk_ac_min, filo1, post_ls): 6,919 routes
5. (VB, sk_ac_min, filo2, post_ls): 6,881 routes

## Task 2: Improvement Analysis

### Improvement Summary

- **Total DRI (Direct Route Improvement) improvements**: 80
- **Total SCP (Set Cover Problem) improvements**: 91

### Convergence Graphs

Convergence graphs have been generated for all 26 instances, showing:
- Cost timeline over the entire run
- DRI improvements marked in green (circles)
- SCP improvements marked in red (squares)

Graphs are saved in: `analysis/convergence_graphs/`

### Key Observations

1. **Route Pool Composition**: 
   - Vehicle-based (VB) clustering generated the most routes (89K)
   - sk_kmeans was the most used clustering method
   - filo2 solver was used most frequently

2. **Improvements**:
   - SCP phases contributed slightly more improvements (91) than DRI phases (80)
   - Most instances showed a mix of both improvement types
   - Some instances (e.g., XL-n10001-k1570, XL-n7353-k1471) had only DRI improvements

3. **Processing Stages**:
   - The vast majority of routes (145K) were generated in the post_ls stage
   - SCP stages added a smaller but significant number of routes (8K)

## Files Generated

1. **route_pool_statistics.json**: Detailed statistics in JSON format
2. **convergence_graphs/**: 26 PNG files, one per instance
3. **SUMMARY.md**: This summary document

## Notes

- Instance XL-n9784-k2774 had no route pool summary in its log file
- All other 25 instances were successfully analyzed
- The analysis script can be re-run with: `python3 analyze_results.py`
