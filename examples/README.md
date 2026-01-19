# Examples: High-Resolution WTD Mapping

This directory contains example scripts for generating high-resolution water table depth (WTD) maps using the Random Forest approach, similar to [HydroFrame-ML/high-res-WTD-static](https://github.com/HydroFrame-ML/high-res-WTD-static).

## Scripts

### 1. `train_wtd_model.py`
Train a Random Forest model for WTD prediction from well observations and environmental covariates.

**Example Usage:**
```bash
python train_wtd_model.py \
    --input ../data/wells_with_covariates.csv \
    --covariates elevation slope aspect twi dist_to_stream soil_perm precip pet \
    --target ground_water_level \
    --output wtd_rf_model.pkl \
    --test-split 0.2 \
    --n-estimators 100 \
    --plots-dir model_outputs
```

**Outputs:**
- `wtd_rf_model.pkl` - Trained Random Forest model
- `wtd_rf_model.metrics.json` - Training/test performance metrics
- `model_outputs/model_diagnostics.png` - Observed vs predicted plots
- `model_outputs/feature_importance.png` - Feature importance ranking

### 2. `generate_wtd_map.py`
Generate high-resolution WTD maps using a trained model and covariate rasters.

**Example Usage:**
```bash
python generate_wtd_map.py \
    --model wtd_rf_model.pkl \
    --covariates covariate_config.json \
    --extent 170.0 -45.0 175.0 -42.0 \
    --resolution 30 \
    --crs-epsg 2193 \
    --output wtd_map_30m.tif \
    --uncertainty wtd_uncertainty_30m.tif \
    --tile-size 1000
```

**Outputs:**
- `wtd_map_30m.tif` - WTD predictions at 30m resolution
- `wtd_uncertainty_30m.tif` - Prediction uncertainty (std across trees)

## Complete Workflow

### Step 1: Prepare Well Data with Covariates

First, you need well observation data with environmental covariates extracted at each location.

**Input CSV format** (`wells_with_covariates.csv`):
```csv
well_id,latitude,longitude,ground_water_level,elevation,slope,aspect,twi,dist_to_stream,soil_perm,precip,pet
W001,-43.5320,172.6306,15.5,350,5.2,180,8.5,500,50,800,600
W002,-43.5421,172.6415,22.3,420,8.1,90,7.2,1200,35,750,620
...
```

See [USAGE_FOR_WTD_MAPPING.md](../USAGE_FOR_WTD_MAPPING.md) for detailed instructions on preparing covariate data.

### Step 2: Train Random Forest Model

```bash
# Train model
python train_wtd_model.py \
    --input wells_with_covariates.csv \
    --covariates elevation slope aspect twi dist_to_stream soil_perm precip pet \
    --target ground_water_level \
    --output wtd_rf_model.pkl \
    --n-estimators 100 \
    --test-split 0.2

# Check outputs
ls model_outputs/
# model_diagnostics.png  - Visual validation
# feature_importance.png - Which features matter most
```

Review the diagnostic plots to ensure the model is performing well:
- **R² > 0.5** indicates reasonable performance
- **RMSE** should be acceptable for your application
- Check residual plots for bias or patterns

### Step 3: Create Covariate Configuration

Create a JSON file (`covariate_config.json`) mapping covariate names to raster file paths:

```json
{
  "elevation": "/path/to/elevation_30m.tif",
  "slope": "/path/to/slope_30m.tif",
  "aspect": "/path/to/aspect_30m.tif",
  "twi": "/path/to/twi_30m.tif",
  "dist_to_stream": "/path/to/distance_to_stream_30m.tif",
  "soil_perm": "/path/to/soil_permeability_30m.tif",
  "precip": "/path/to/mean_annual_precip_30m.tif",
  "pet": "/path/to/potential_et_30m.tif"
}
```

**Important:** 
- Covariate names must match those used during training
- All rasters must be at the same resolution (30m in this example)
- All rasters must cover your target extent
- CRS should be metric (e.g., NZTM2000, Albers Equal Area)

### Step 4: Generate WTD Map

```bash
# Generate 30m resolution WTD map for Canterbury region
python generate_wtd_map.py \
    --model wtd_rf_model.pkl \
    --covariates covariate_config.json \
    --extent 170.0 -45.0 175.0 -42.0 \
    --resolution 30 \
    --crs-epsg 2193 \
    --output wtd_map_30m.tif \
    --uncertainty wtd_uncertainty_30m.tif

# For large areas, adjust tile size to manage memory
python generate_wtd_map.py \
    --model wtd_rf_model.pkl \
    --covariates covariate_config.json \
    --extent 165.0 -47.0 179.0 -34.0 \
    --resolution 100 \
    --tile-size 500 \
    --output wtd_map_nz_100m.tif
```

### Step 5: Visualize Results

You can visualize the output GeoTIFF in QGIS, ArcGIS, or Python:

```python
import rasterio
import matplotlib.pyplot as plt

# Load WTD map
with rasterio.open('wtd_map_30m.tif') as src:
    wtd = src.read(1)
    wtd[wtd == src.nodata] = np.nan

# Plot
plt.figure(figsize=(12, 8))
plt.imshow(wtd, cmap='viridis_r', interpolation='nearest')
plt.colorbar(label='Water Table Depth (m)')
plt.title('High-Resolution WTD Map')
plt.tight_layout()
plt.savefig('wtd_visualization.png', dpi=150)
```

## Performance Considerations

### Resolution vs. Computation Time

| Resolution | Pixels (100km²) | Memory | Time (100 trees) |
|------------|----------------|---------|------------------|
| 30m        | ~11M           | ~44 MB  | ~10-20 min      |
| 100m       | ~1M            | ~4 MB   | ~1-2 min        |
| 1km        | ~10k           | ~40 KB  | ~seconds        |

### Memory Management

The scripts use tile-based processing to handle large areas:
- Default tile size: 1000×1000 pixels
- For limited RAM, reduce `--tile-size` to 500 or 250
- Each tile is processed independently

### Parallelization

Random Forest prediction uses all CPU cores by default (`n_jobs=-1`). For even faster processing:
- Process different regions in parallel (multiple script instances)
- Use GPU-accelerated Random Forest (cuML/RAPIDS)
- Pre-filter areas with no groundwater (ocean, high elevation)

## Adapting for Your Region

### 1. Update CRS

Change `--crs-epsg` to your region's metric coordinate system:
- **CONUS**: 5070 (Albers Equal Area)
- **Europe**: 3035 (ETRS89-extended)
- **Australia**: 3577 (GDA94 Australian Albers)
- **New Zealand**: 2193 (NZTM2000)

### 2. Adjust Extent

Specify your region's bounding box in WGS84 (lon/lat):
```bash
--extent MIN_LON MIN_LAT MAX_LON MAX_LAT
```

### 3. Select Appropriate Resolution

- **Local studies**: 10-30m
- **Regional**: 30-100m
- **National**: 100-1000m
- **Continental**: 1000m+

## Troubleshooting

### Issue: Out of Memory

**Solution:**
- Reduce `--tile-size` (e.g., 500 or 250)
- Increase system swap space
- Process smaller sub-regions separately

### Issue: Covariate Rasters Don't Align

**Error:** "Rasters have different CRS/resolution/extent"

**Solution:**
```bash
# Reproject and align all rasters using GDAL
gdalwarp -t_srs EPSG:2193 -tr 30 30 -r bilinear \
    input.tif output_aligned.tif

# Or use rasterio in Python
import rasterio
from rasterio.warp import reproject, Resampling
# ... reproject code ...
```

### Issue: Model Predicts Unrealistic Values

**Possible causes:**
1. Covariate values outside training range → Add more training data
2. Poor model performance → Improve model or add features
3. Wrong covariate order → Check covariate names match training

**Solution:**
```python
# Add prediction constraints
predictions = np.clip(predictions, 0, 500)  # WTD between 0-500m
```

### Issue: Slow Processing

**Solutions:**
- Reduce number of trees (`--n-estimators 50`)
- Increase resolution (e.g., 100m instead of 30m)
- Use smaller extent or tile-based batch processing
- Enable progress monitoring to track speed

## Validation

After generating maps, validate predictions:

```python
# Compare with held-out test wells
test_wells = pd.read_csv('test_wells.csv')

# Extract predicted values at well locations
predicted = []
with rasterio.open('wtd_map_30m.tif') as src:
    for _, well in test_wells.iterrows():
        row, col = src.index(well.longitude, well.latitude)
        predicted.append(src.read(1, window=((row, row+1), (col, col+1)))[0, 0])

# Calculate metrics
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(test_wells.ground_water_level, predicted))
r2 = r2_score(test_wells.ground_water_level, predicted)
print(f"Validation RMSE: {rmse:.2f} m")
print(f"Validation R²: {r2:.3f}")
```

## Additional Resources

- **Main Documentation**: [README.md](../README.md)
- **WTD Mapping Guide**: [USAGE_FOR_WTD_MAPPING.md](../USAGE_FOR_WTD_MAPPING.md)
- **HydroFrame-ML Paper**: [Ma et al. (2025)](https://doi.org/10.1038/s43247-025-03094-3)
- **Random Forest**: [Scikit-learn docs](https://scikit-learn.org/stable/modules/ensemble.html#forest)

## Support

For questions or issues:
1. Check the troubleshooting section above
2. Review [USAGE_FOR_WTD_MAPPING.md](../USAGE_FOR_WTD_MAPPING.md)
3. Open an issue on GitHub
