# Depth to Groundwater (DTW) – Execution Plan

**Source spec:** Section 5.2 in `Water Design Scope doc.md`  
**Goal:** Break the DTW modelling spec into bite-sized, testable PRs that Copilot can execute.

---

## Sequenced PRs (testable chunks)

| PR | Scope | Primary outputs | How to verify |
|----|-------|-----------------|---------------|
| 1. Data audit & schema hardening | Snapshot wells/observations; enforce required cols and CRS/null checks. | • Field doc update<br>• Validation script + summary CSV | • Zero blockers (no missing required cols or CRS mismatches)<br>• CSV of flagged rows with row id + issue type |
| 2. Target computation (P10/P50/P90) | Seasonal target calculator + Koch sinusoidal backfill with quality flags. | • `targets.py` + tests for dense/sparse wells | • `pytest tests/test_targets.py` green<br>• Edge cases (single obs, coastal amplitude) covered |
| 3. Hydrogeologic class assignment | Permeability / condition / proximity classification (27 combos) with defaults. | • Classifier + lookup<br>• CLI to tag wells | • CLI outputs counts per class<br>• Tests cover fallbacks and proximity calc |
| 4. Covariate assembly | Build feature matrix (DEM, distance-to-coast/streams, landcover, geology) with caching. | • `covariates.py` loaders<br>• Cached artifacts listed | • Reprojection/nodata tests pass<br>• Sample run writes feather/parquet with expected cols |
| 5. Train/validate baseline model | Zone-aware splits; baseline RF/QRF + kriging comparators; metrics & plots. | • Training script<br>• Metrics JSON + plots in `reports/` | • `pytest tests/test_splits.py` green<br>• Training run writes metrics/plots meeting targets (e.g., interval coverage within ±5% of nominal; RMSE improves on prior baseline) |
| 6. Residual kriging & uncertainty | Fit residual kriging; propagate p10/p50/p90; handle artesian flags. | • Residual kriging module<br>• Predictor returning mean + bands | • Residual unit test<br>• Integration test emits artesian flags when DTW < 0 |
| 7. Rasterization & tiling | NZTM2000 rasters → z/x/y tiles with metadata and nodata masks. | • Tile generation script<br>• Metadata JSON beside tiles | • Tiles for 10 km × 10 km sample<br>• Grid/nodata alignment check + visual spot check |
| 8. API + docs | DTW query endpoint and operator/runbook updates. | • FastAPI route (or documented interface)<br>• Docs linking DTW outputs | • Local call returns DTW_low/high/med + uncertainty + metadata<br>• Docs reviewed |

---

## Execution notes

1. **Order matters:** Each PR is independent but assumes outputs from the previous one (e.g., targets → classes → covariates → models → tiles).
2. **Test-first:** Each PR includes unit/integration tests scoped to the new module; avoid touching unrelated code.
3. **Data paths:** Keep raw vs derived data separated (`data/raw`, `data/processed`, `data/tiles`); never mutate raw sources.
4. **Quality flags:** Carry quality flags from targets through to predictions and tiles; include in metadata/CLI output.
5. **Acceptance:** Each PR should land with green tests and a short README snippet describing how to re-run the new step.
