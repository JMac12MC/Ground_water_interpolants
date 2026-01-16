# DTW PR1: Wells audit & schema checks

This PR introduces a repeatable validation pass for the wells snapshot to harden the DTW schema before modelling.

## Canonical columns & source mapping

| Canonical field | Purpose | Source column (sample wells CSV) |
|-----------------|---------|----------------------------------|
| `well_id` | Stable identifier per well | `WELL_NO` |
| `nztm_x` | Projected X (NZTM2000) | `NZTMX` |
| `nztm_y` | Projected Y (NZTM2000) | `NZTMY` |
| `ground_surface_elev_m` | Ground surface elevation (m) | `GROUND_RL` |
| `static_water_level_elev_m` | Static water level elevation (m) | `INITIAL_SWL` |
| `measurement_date` | Observation date (used as warning-only in PR1) | `DATE_DRILLED` |
| `well_depth_m` | Total drilled depth (warning-only in PR1) | `DEPTH` |

## Checks performed

- Missing required columns (schema blockers)
- Missing required values per row (`well_id`, `nztm_x`, `nztm_y`, `ground_surface_elev_m`, `static_water_level_elev_m`)
- NZTM2000 coordinate sanity bounds:
  - `nztm_x` in `[1,000,000, 3,000,000]`
  - `nztm_y` in `[4,700,000, 6,500,000]`
- Warning-only for optional fields (`measurement_date`, `well_depth_m`)

## How to run

```bash
cd /home/runner/work/Ground_water_interpolants/Ground_water_interpolants
python wells_validation.py \
  --input "data/sample/Wells/Wells_and_Bores_-_All (2).csv" \
  --output-dir reports/dtw_pr1
```

Outputs:
- `reports/dtw_pr1/wells_validation_summary.csv` – dataset-level metrics and resolved column mapping
- `reports/dtw_pr1/wells_validation_issues.csv` – row-level issues with `row_number`, `well_id`, `issue_type`, `severity`, and detail

### Current sample run (data/sample/Wells/Wells_and_Bores_-_All (2).csv)

- 551 rows evaluated
- 0 missing required columns
- 281 missing required values (predominantly `static_water_level_elev_m`)
- 286 warning-only missing optional fields
