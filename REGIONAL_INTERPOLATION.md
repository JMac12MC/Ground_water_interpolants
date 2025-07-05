# Canterbury Regional Groundwater Level Interpolation

This system generates a pre-computed, region-wide groundwater level interpolation for the Canterbury region using a grid-based approach with overlapping sub-regions.

## Overview

The regional interpolation system addresses the limitation of processing 34,000+ data points by:

1. **Grid-based Processing**: Divides Canterbury into a grid with 10km spacing
2. **Local Interpolations**: Creates 15km radius kriging interpolations at each grid point
3. **Overlap Management**: Uses 10km spacing with 15km radius for smooth transitions
4. **Parallel Processing**: Uses multiple workers to speed up computation
5. **Data Merging**: Combines all sub-regions into a single seamless surface

## Usage

### 1. Generate Regional Data (One-time Setup)

```bash
python generate_regional_data.py
```

This creates a pre-computed interpolation file that can be loaded instantly by the main application.

**Processing Time**: 30-60 minutes (depending on your system)
**Output**: `sample_data/canterbury_gwl_interpolation.json`

### 2. Check Regional Data Status

```bash
python check_regional_data.py
```

This shows information about the regional data file and validates it can be loaded.

### 3. Use in Main Application

In the main app, when selecting "Ground Water Level" interpolation:
- Check "Use pre-computed Canterbury region" for instant loading
- Uncheck for local interpolation around your selected point

## Technical Details

### Grid Parameters
- **Region**: Canterbury bounds (-42.5° to -45.0° lat, 170.5° to 173.5° lon)
- **Interpolation Radius**: 15km around each grid point
- **Grid Spacing**: 10km between grid points (15km radius - 5km overlap)
- **Overlap**: 5km overlap between adjacent interpolations
- **Resolution**: 50 points per interpolation

### Performance Optimizations
- **Parallel Processing**: Uses 4 workers by default
- **Resumable Processing**: Each sub-interpolation stored immediately in database
- **Progress Tracking**: Can resume from any interruption point
- **Fault Tolerance**: Failed grid points can be retried individually

### Data Quality
- **Input Wells**: 34,000+ wells filtered to valid groundwater level data
- **Kriging Method**: Spherical variogram with auto-fitting
- **Soil Filtering**: Optional filtering using soil drainage polygons
- **Value Filtering**: Excludes very small/zero values

## File Structure

```
├── generate_regional_gwl.py          # Core regional interpolation functions
├── generate_regional_data.py         # Script to generate regional data
├── check_regional_data.py           # Utility to check data status
├── sample_data/
│   └── canterbury_gwl_interpolation.json  # Pre-computed regional data
└── REGIONAL_INTERPOLATION.md       # This documentation
```

## Customization

To modify parameters, edit `generate_regional_gwl.py`:

```python
# Change grid spacing (smaller = higher resolution, longer processing)
grid_spacing_km = 10

# Change interpolation radius (larger = smoother, slower)
radius_km = 15

# Change number of parallel workers
max_workers = 4

# Change variogram model
variogram_model = 'spherical'  # or 'linear', 'gaussian', 'exponential'
```

## Memory Requirements

- **Generation**: ~2-4GB RAM during processing
- **Storage**: ~10-50MB for final GeoJSON file
- **Loading**: ~100-500MB RAM when loaded in app

## Troubleshooting

### Long Processing Time
- Reduce `max_workers` if system becomes unresponsive
- Increase `grid_spacing_km` for faster but lower resolution
- Reduce `radius_km` for faster processing

### Memory Issues
- Close other applications during generation
- Reduce `max_workers` parameter
- Consider processing smaller sub-regions

### Data Quality Issues
- Check input data has valid 'ground water level' column
- Ensure wells are within Canterbury region bounds
- Verify soil polygon files are available if using filtering