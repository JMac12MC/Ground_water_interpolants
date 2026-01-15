# Depth to Groundwater (DTW) – Execution Plan

**Source spec:** Section 5.2 in `Water design Scope doc.md`  
**Goal:** Break the DTW modelling spec into bite-sized, testable PRs that Copilot can execute.

---

## Sequenced PRs (testable chunks)

| PR | Scope | Primary outputs | How to verify |
|----|-------|-----------------|---------------|
| 1. Data audit & schema hardening | Snapshot well/observation tables; enforce required columns (e.g., well_id, x/y, ground_surface_elev, static_water_level, obs_date). Add null/CRS checks and quality flags. | `docs` update describing fields; data validation script with summary CSV. | Run validation script; expect zero critical errors and CSV of flagged rows generated. |
| 2. Target computation (P10/P50/P90) | Implement seasonal target calculator and Koch sinusoidal backfill (Section 5.2.2.3). Include quality flags based on observation counts/spans. | `targets.py` (or similar) with unit tests covering dense and sparse wells. | `pytest tests/test_targets.py` – all cases (dense series, single obs, coastal class amplitude) pass. |
| 3. Hydrogeologic class assignment | Encode permeability / aquifer condition / proximity classification (27 combinations) with default/unknown handling. Persist class on wells for reuse. | Classifier module + lookup table; CLI to annotate wells. | CLI run produces counts per class; unit tests cover fallbacks and CRS-safe proximity calculations. |
| 4. Covariate assembly | Build feature matrix (DEM, distance-to-coast/streams, landcover, geology). Cache rasters/vectors; add unit tests for reprojection and nodata handling. | `covariates.py` with reusable loaders; cached artifacts listed in README. | Unit tests for reprojection & nodata; sample run writes feather/parquet with expected columns. |
| 5. Train/validate baseline model | Implement zone-aware train/validation split (no spatial leakage), baseline model (e.g., RandomForest or QRF) plus Ordinary/Ridge Kriging comparators. Compute RMSE/MAE, bias, coverage of prediction intervals. | Training script + metrics JSON; plots saved to `reports/`. | `pytest tests/test_splits.py`; training run completes and writes metrics/plots; coverage ≥ expected thresholds from spec. |
| 6. Residual kriging & uncertainty | Fit residual kriging on model errors; propagate prediction intervals (p10/p50/p90). Add artesian flag handling (DTW < 0). | Residual kriging module; combined predictor returning mean + bands. | Unit test mocking residual surface; integration test ensures artesian flags emitted when DTW < 0. |
| 7. Rasterization & tiling | Convert predictions to NZTM2000 rasters and z/x/y tiles; include metadata (version, timestamp, model hash). Handle nodata masks in exclusion zones. | Tile generation script; metadata JSON alongside tiles. | Generate tiles for 10 km × 10 km sample; visual spot check; automated check that tiles align with grid and nodata masks respected. |
| 8. API + docs | Expose DTW query endpoint (point and tile metadata); add operator runbook and decision log. | FastAPI route (if backend exists) or documented interface; updated docs linking DTW outputs. | Local call returns DTW_low/high/med + uncertainty and source metadata; documentation reviewed. |

---

## Execution notes

1. **Order matters:** Each PR is independent but assumes outputs from the previous one (e.g., targets → classes → covariates → models → tiles).
2. **Test-first:** Each PR includes unit/integration tests scoped to the new module; avoid touching unrelated code.
3. **Data paths:** Keep raw vs derived data separated (`data/raw`, `data/processed`, `data/tiles`); never mutate raw sources.
4. **Quality flags:** Carry quality flags from targets through to predictions and tiles; include in metadata/CLI output.
5. **Acceptance:** Each PR should land with green tests and a short README snippet describing how to re-run the new step.
