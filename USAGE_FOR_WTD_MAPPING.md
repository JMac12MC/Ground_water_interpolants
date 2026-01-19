# Using This Repository for High-Resolution Water Table Depth (WTD) Mapping

This guide explains how to adapt this groundwater interpolation system to build high-resolution water table depth maps similar to the [HydroFrame-ML/high-res-WTD-static](https://github.com/HydroFrame-ML/high-res-WTD-static) approach.

## Table of Contents
1. [Overview](#overview)
2. [Methodology Comparison](#methodology-comparison)
3. [Data Preparation](#data-preparation)
4. [Model Training Workflow](#model-training-workflow)
5. [Map Generation](#map-generation)
6. [Example Scripts](#example-scripts)
7. [Adapting to Your Region](#adapting-to-your-region)

## Overview

### What is Water Table Depth (WTD)?

Water Table Depth is the distance from the ground surface to the top of the saturated zone (water table). Accurate WTD maps are crucial for:
- Groundwater resource management
- Agriculture and irrigation planning
- Ecosystem health assessment
- Infrastructure development
- Contamination risk assessment

### Approach

This repository supports two primary approaches for WTD mapping:

1. **Geostatistical (Kriging)**
   - Best for: Dense observation networks, local areas
   - Advantages: Statistical rigor, uncertainty quantification
   - Limitations: Computational cost at large scales

2. **Machine Learning (Random Forest)**
   - Best for: Sparse observations, large areas, covariate-rich datasets
   - Advantages: Scales well, handles non-linear relationships
   - Limitations: Requires good covariate data

The **HydroFrame-ML approach uses Random Forest**, which this repository fully supports.

## Methodology Comparison

### HydroFrame-ML Workflow
```
Well Observations → Feature Engineering → Random Forest Training → 
  High-Res Prediction → GeoTIFF Export → Validation
```

### This Repository's Equivalent
```
Well Data (data_loader.py) → Covariate Processing (covariate_processing.py) → 
  RF Training (interpolation.py) → Grid Generation (interpolation.py) → 
  GeoTIFF Export (rasterio) → Visualization (app.py)
```

### Component Mapping

| HydroFrame-ML Component | This Repository Equivalent |
|------------------------|---------------------------|
| `random_forest_sklearn.py` | `interpolation.py` (RandomForestRegressor) |
| `1_model_training.ipynb` | See `examples/train_wtd_model.py` |
| `2_high_res_WTD_map_generation.py` | See `examples/generate_wtd_map.py` |
| `3_convert_to_TIFF.py` | Built into `generate_smooth_raster_overlay()` |

## Data Preparation

### Step 1: Well Observation Data

Your well data should be in CSV format with these columns:

```csv
well_id,latitude,longitude,ground_water_level,measurement_date,aquifer_type
W001,-43.5320,172.6306,15.5,2023-06-15,unconfined
W002,-43.5421,172.6415,22.3,2023-06-15,unconfined
...
```

**Required columns:**
- `latitude`, `longitude` - WGS84 coordinates
- `ground_water_level` - Depth to water table in meters (positive = depth below surface)

**Optional but recommended:**
- `measurement_date` - For temporal analysis
- `well_id` - Unique identifier
- `aquifer_type` - For stratified analysis
- `yield_rate` - For quality control

**Data Quality Checks:**
```python
import pandas as pd
from data_loader import load_sample_data

# Load your data
wells_df = pd.read_csv('your_wells.csv')

# Basic validation
print(f"Total wells: {len(wells_df)}")
print(f"Missing coordinates: {wells_df[['latitude', 'longitude']].isna().sum()}")
print(f"Missing WTD: {wells_df['ground_water_level'].isna().sum()}")

# Check coordinate ranges (adjust for your region)
print(f"Lat range: {wells_df['latitude'].min()} to {wells_df['latitude'].max()}")
print(f"Lon range: {wells_df['longitude'].min()} to {wells_df['longitude'].max()}")

# Statistical summary
print(wells_df['ground_water_level'].describe())
```

### Step 2: Covariate Data

For Random Forest WTD mapping, you need environmental covariates at your target resolution (e.g., 30m or 100m). The HydroFrame-ML paper used:

#### Topographic Covariates
- **Elevation** - DEM from SRTM, ASTER, or national sources
- **Slope** - Derived from DEM
- **Aspect** - Derived from DEM  
- **Topographic Wetness Index (TWI)** - Captures water accumulation
- **Terrain Ruggedness Index (TRI)** - Surface roughness

#### Hydrological Covariates
- **Distance to nearest stream** - From hydrography datasets
- **Drainage density** - Stream network density
- **Flow accumulation** - From DEM analysis
- **Catchment area** - Upstream contributing area

#### Soil Covariates
- **Soil permeability** - From soil surveys
- **Soil texture** - Sand/silt/clay fractions
- **Hydraulic conductivity** - If available
- **Depth to bedrock** - From geological surveys

#### Climate Covariates
- **Mean annual precipitation** - From climate models/stations
- **Potential evapotranspiration (PET)** - From climate data
- **Aridity index** - Precipitation / PET
- **Temperature** - Mean annual temperature

#### Geological Covariates
- **Lithology** - Rock type from geological maps
- **Aquifer type** - Confined vs unconfined
- **Permeability** - From hydrogeological maps

### Step 3: Covariate Processing

Use `covariate_processing.py` to extract covariate values at well locations:

```python
import geopandas as gpd
import rasterio
from covariate_processing import extract_raster_values

# Load well locations
wells_gdf = gpd.GeoDataFrame(
    wells_df,
    geometry=gpd.points_from_xy(wells_df.longitude, wells_df.latitude),
    crs='EPSG:4326'
)

# Extract covariate values at well locations
with rasterio.open('elevation_30m.tif') as src:
    wells_df['elevation'] = extract_raster_values(src, wells_gdf)

with rasterio.open('slope_30m.tif') as src:
    wells_df['slope'] = extract_raster_values(src, wells_gdf)

# Repeat for all covariates...

# Save processed data
wells_df.to_csv('wells_with_covariates.csv', index=False)
```

## Model Training Workflow

### Option 1: Using Built-in Random Forest

This repository includes Random Forest in the interpolation pipeline:

```python
from interpolation import generate_geo_json_grid
import pandas as pd

# Load well data with covariates
wells_df = pd.read_csv('wells_with_covariates.csv')

# Define covariate columns
covariate_cols = [
    'elevation', 'slope', 'aspect', 'twi',
    'dist_to_stream', 'soil_permeability',
    'mean_annual_precip', 'pet'
]

# Generate interpolated grid using Random Forest
geojson = generate_geo_json_grid(
    wells_df=wells_df,
    center_lat=wells_df['latitude'].mean(),
    center_lon=wells_df['longitude'].mean(),
    radius_km=50,
    grid_size=200,
    method='random_forest',  # or 'quantile_forest'
    covariate_columns=covariate_cols
)
```

### Option 2: Custom Random Forest Training

For more control (similar to HydroFrame-ML approach):

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Prepare training data
X = wells_df[covariate_cols].values
y = wells_df['ground_water_level'].values

# Remove rows with missing data
mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
X = X[mask]
y = y[mask]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',  # or 1/3 of features
    max_samples=1.0,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.2f} meters")
print(f"Test R²: {r2:.3f}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': covariate_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Save model
import joblib
joblib.dump(rf_model, 'wtd_rf_model.pkl')
```

## Map Generation

### Step 1: Create High-Resolution Grid

Define your prediction grid at the target resolution:

```python
import numpy as np
from interpolation import to_nztm2000, to_wgs84

# Define extent (adjust CRS for your region)
min_lon, max_lon = 170.0, 175.0
min_lat, max_lat = -45.0, -42.0

# Create grid at 30m resolution (adjust CRS to your region's metric system)
# Example for NZTM2000 (New Zealand)
from pyproj import Transformer

# Transform bounds to metric CRS
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
min_x, min_y = transformer.transform(min_lon, min_lat)
max_x, max_y = transformer.transform(max_lon, max_lat)

# Create grid at 30m resolution
resolution = 30  # meters
x_coords = np.arange(min_x, max_x, resolution)
y_coords = np.arange(min_y, max_y, resolution)
X_grid, Y_grid = np.meshgrid(x_coords, y_coords)

print(f"Grid size: {X_grid.shape}")
print(f"Total pixels: {X_grid.size:,}")
```

### Step 2: Extract Covariates for Grid

For each grid cell, extract covariate values:

```python
import rasterio
from rasterio.transform import xy

# Prepare covariate arrays
covariate_arrays = {}

covariate_files = {
    'elevation': 'elevation_30m.tif',
    'slope': 'slope_30m.tif',
    'aspect': 'aspect_30m.tif',
    'twi': 'twi_30m.tif',
    # ... add all covariate files
}

for name, filepath in covariate_files.items():
    with rasterio.open(filepath) as src:
        # Read the raster data
        data = src.read(1)
        covariate_arrays[name] = data

# Stack into feature matrix
n_pixels = X_grid.size
n_features = len(covariate_arrays)
X_predict = np.zeros((n_pixels, n_features))

for i, (name, array) in enumerate(covariate_arrays.items()):
    X_predict[:, i] = array.flatten()

# Remove pixels with missing data
valid_mask = ~np.isnan(X_predict).any(axis=1)
X_predict_clean = X_predict[valid_mask]
```

### Step 3: Generate Predictions

```python
# Load trained model
import joblib
rf_model = joblib.load('wtd_rf_model.pkl')

# Make predictions
wtd_predictions = np.full(n_pixels, np.nan)
wtd_predictions[valid_mask] = rf_model.predict(X_predict_clean)

# Reshape to grid
wtd_grid = wtd_predictions.reshape(X_grid.shape)

# Optional: Estimate uncertainty using quantile forest or RF variance
# For prediction intervals, use quantile-forest
from quantile_forest import RandomForestQuantileRegressor

qrf_model = RandomForestQuantileRegressor(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
qrf_model.fit(X_train, y_train)

# Predict median and 90% prediction interval
y_pred_median = qrf_model.predict(X_predict_clean, quantiles=0.5)
y_pred_lower = qrf_model.predict(X_predict_clean, quantiles=0.05)
y_pred_upper = qrf_model.predict(X_predict_clean, quantiles=0.95)

uncertainty = y_pred_upper - y_pred_lower
```

### Step 4: Export to GeoTIFF

```python
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

# Define transform (adjust for your CRS and resolution)
transform = from_origin(
    west=min_x,
    north=max_y,
    xsize=resolution,
    ysize=resolution
)

# Export WTD map
with rasterio.open(
    'wtd_map_30m.tif',
    'w',
    driver='GTiff',
    height=wtd_grid.shape[0],
    width=wtd_grid.shape[1],
    count=1,
    dtype=wtd_grid.dtype,
    crs=CRS.from_epsg(2193),  # Adjust to your CRS
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(wtd_grid, 1)

print("WTD map exported to wtd_map_30m.tif")

# Export uncertainty map
uncertainty_grid = uncertainty.reshape(X_grid.shape)
with rasterio.open(
    'wtd_uncertainty_30m.tif',
    'w',
    driver='GTiff',
    height=uncertainty_grid.shape[0],
    width=uncertainty_grid.shape[1],
    count=1,
    dtype=uncertainty_grid.dtype,
    crs=CRS.from_epsg(2193),
    transform=transform,
    compress='lzw'
) as dst:
    dst.write(uncertainty_grid, 1)

print("Uncertainty map exported to wtd_uncertainty_30m.tif")
```

## Example Scripts

### Complete WTD Mapping Script

A complete example is provided in `examples/generate_wtd_map.py`:

```bash
python examples/generate_wtd_map.py \
    --wells wells_with_covariates.csv \
    --covariates covariate_config.json \
    --extent 170.0 -45.0 175.0 -42.0 \
    --resolution 30 \
    --output wtd_map_30m.tif \
    --uncertainty wtd_uncertainty_30m.tif
```

### Training Script

See `examples/train_wtd_model.py`:

```bash
python examples/train_wtd_model.py \
    --input wells_with_covariates.csv \
    --covariates elevation slope aspect twi dist_to_stream \
    --target ground_water_level \
    --output wtd_rf_model.pkl \
    --test-split 0.2
```

## Adapting to Your Region

### 1. Change Coordinate Reference System (CRS)

Update the CRS in `interpolation.py`:

```python
# Current (New Zealand NZTM2000)
def get_transformers():
    transformer_to_metric = pyproj.Transformer.from_crs(
        "EPSG:4326",  # WGS84
        "EPSG:2193",  # NZTM2000
        always_xy=True
    )
    transformer_to_wgs84 = pyproj.Transformer.from_crs(
        "EPSG:2193",  # NZTM2000
        "EPSG:4326",  # WGS84
        always_xy=True
    )
    return transformer_to_metric, transformer_to_wgs84

# For your region, change EPSG:2193 to your metric CRS
# Examples:
# - CONUS: EPSG:5070 (Albers Equal Area)
# - Europe: EPSG:3035 (ETRS89-extended)
# - Australia: EPSG:3577 (GDA94 Australian Albers)
```

### 2. Update Data Sources

Modify `data_loader.py` to load from your data sources:

```python
def load_regional_wells(region='conus'):
    """Load well data for specified region"""
    if region == 'conus':
        # Load USGS groundwater data
        df = load_usgs_wells()
    elif region == 'nz':
        # Load NZ Environment Canterbury data
        df = load_nz_govt_data()
    # Add your region...
    return df
```

### 3. Adjust Spatial Extent

Update default extents in the Streamlit app (`app.py`):

```python
# Current (Canterbury, NZ)
DEFAULT_CENTER_LAT = -43.5320
DEFAULT_CENTER_LON = 172.6306

# For CONUS example
DEFAULT_CENTER_LAT = 39.8283  # USA center
DEFAULT_CENTER_LON = -98.5795
```

### 4. Configure Resolution

Choose appropriate grid resolution for your region:

```python
# Local/regional studies: 30m - 100m
GRID_RESOLUTION = 30  # meters

# Continental studies: 100m - 1km  
GRID_RESOLUTION = 100  # meters

# Global studies: 1km+
GRID_RESOLUTION = 1000  # meters
```

## Best Practices

### Data Quality
- **Minimum well density**: 1 well per 25 km² for reasonable accuracy
- **Temporal consistency**: Use measurements from similar time periods
- **Quality control**: Remove outliers and erroneous measurements
- **Stratification**: Consider aquifer type and hydrogeological setting

### Model Training
- **Feature selection**: Use domain knowledge and feature importance
- **Cross-validation**: Use spatial cross-validation (not random)
- **Hyperparameter tuning**: Optimize n_estimators, max_depth, max_features
- **Ensemble methods**: Consider averaging multiple models

### Validation
- **Hold-out test set**: At least 20% of data
- **Spatial validation**: Test on geographically separate areas
- **Metrics**: RMSE, MAE, R², bias
- **Uncertainty calibration**: Check prediction interval coverage

### Computational Efficiency
- **Tile-based processing**: Process large areas in tiles
- **Memory management**: Use dask for large arrays
- **Parallel processing**: Utilize multiple cores
- **Progressive refinement**: Start coarse, refine high-interest areas

## Troubleshooting

### Common Issues

**Issue**: Out of memory when processing large areas
- **Solution**: Use tile-based processing, reduce resolution, or use dask

**Issue**: Poor model performance (high RMSE)
- **Solution**: Check covariate quality, add more features, increase training data

**Issue**: Unrealistic predictions (negative WTD, extreme values)
- **Solution**: Add constraints, improve training data quality, check covariate alignment

**Issue**: Slow prediction time
- **Solution**: Reduce n_estimators, use model compression, pre-compute tiles

## Additional Resources

- **Original paper**: Ma et al. (2025) - https://doi.org/10.1038/s43247-025-03094-3
- **HydroFrame-ML code**: https://github.com/HydroFrame-ML/high-res-WTD-static
- **PyKrige documentation**: https://geostat-framework.readthedocs.io/projects/pykrige/
- **Scikit-learn RF**: https://scikit-learn.org/stable/modules/ensemble.html#forest
- **Quantile Forest**: https://github.com/zillow/quantile-forest

## Citation

If you use this code for WTD mapping research, please cite:

1. This repository: [Add citation]
2. HydroFrame-ML: Ma, Y., et al. (2025). High Resolution Static Water Table Depth Estimation over the Contiguous United States. *Nature Communications Earth & Environment*. https://doi.org/10.1038/s43247-025-03094-3

## Support

For questions or issues:
1. Check this documentation
2. Review example scripts
3. Open a GitHub issue
4. Contact repository maintainers
