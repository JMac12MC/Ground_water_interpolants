# Groundwater Interpolation System

## Overview

This repository provides a comprehensive groundwater depth interpolation and visualization system with support for multiple interpolation methods including Ordinary Kriging, Indicator Kriging, Regression Kriging, and Random Forest machine learning. The system is designed for high-resolution water table depth (WTD) mapping and can be adapted for different geographic regions.

## Key Features

- **Multiple Interpolation Methods**
  - Ordinary Kriging with automatic variogram fitting
  - Indicator Kriging for probability mapping
  - Regression Kriging with covariates
  - Random Forest (Quantile Random Forest) for uncertainty estimation

- **Spatial Data Processing**
  - Coordinate transformation (WGS84 ↔ local metric CRS)
  - Grid generation at configurable resolutions
  - Polygon clipping and exclusion zones
  - GeoJSON and GeoTIFF export

- **Interactive Visualization**
  - Streamlit-based web interface
  - Interactive Folium maps
  - Heatmap generation and display
  - Well data exploration

- **Database Integration**
  - PostgreSQL + PostGIS for spatial data
  - Pre-computed heatmap storage
  - Well data management

## Compatibility with HydroFrame-ML High-Resolution WTD Mapping

This repository **can be used** to build high-resolution water table depth maps similar to the [HydroFrame-ML/high-res-WTD-static](https://github.com/HydroFrame-ML/high-res-WTD-static) approach. Both systems share:

### Common Capabilities
1. **Random Forest Implementation**: Both use scikit-learn Random Forest for WTD prediction
2. **Covariate-Based Modeling**: Support for terrain, soil, climate, and other environmental covariates
3. **High-Resolution Grid Generation**: Generate predictions at fine spatial resolutions (30m - 100m)
4. **Uncertainty Quantification**: Random Forest variance and Kriging variance estimates
5. **Large-Scale Processing**: Batch processing capabilities for regional to continental scales
6. **GeoTIFF Export**: Output maps in standard geospatial formats

### Key Differences
| Feature | This Repository | HydroFrame-ML |
|---------|----------------|---------------|
| **Primary Focus** | New Zealand (Canterbury region) | CONUS (Continental US) |
| **Coordinate System** | NZTM2000 (EPSG:2193) | Custom/WGS84 |
| **UI** | Streamlit interactive app | Scripts + notebooks |
| **Kriging Methods** | Multiple (OK, IK, RK) | Not featured |
| **Data Sources** | NZ government APIs | US hydrological datasets |

### Adapting for Your Region

See [USAGE_FOR_WTD_MAPPING.md](USAGE_FOR_WTD_MAPPING.md) for detailed instructions on:
- Preparing your well observation data
- Processing covariate datasets
- Training Random Forest models
- Generating high-resolution WTD maps
- Exporting to GeoTIFF format

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JMac12MC/Ground_water_interpolants.git
cd Ground_water_interpolants

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# OR using uv (faster)
uv pip install -r pyproject.toml
```

### Running the Streamlit App

```bash
streamlit run app.py
```

### Basic Usage Example

```python
from interpolation import generate_geo_json_grid
from data_loader import load_sample_data
import pandas as pd

# Load well data
wells_df = load_sample_data()

# Generate interpolated grid
geojson = generate_geo_json_grid(
    wells_df=wells_df,
    center_lat=-43.5320,
    center_lon=172.6306,
    radius_km=20,
    grid_size=100,
    method='kriging'
)

# Export to file
with open('output.geojson', 'w') as f:
    json.dump(geojson, f)
```

## Repository Structure

```
├── app.py                          # Main Streamlit application
├── interpolation.py                # Core interpolation algorithms
├── data_loader.py                  # Data ingestion and preprocessing
├── database.py                     # Database operations
├── covariate_processing.py         # Covariate data handling
├── automated_heatmap_generator.py  # Batch heatmap generation
├── precompute_heatmaps.py         # Pre-computation utilities
├── sequential_heatmap.py          # Sequential processing
├── utils.py                        # Utility functions
├── docs/                          # Documentation
│   └── scope/                     # Architecture and planning docs
├── data/                          # Data storage
├── sample_data/                   # Example datasets
└── pyproject.toml                 # Dependencies
```

## Core Components

### Interpolation Engine (`interpolation.py`)
- Implements Ordinary Kriging using PyKrige
- Random Forest regression with scikit-learn
- Quantile Random Forest for uncertainty
- Coordinate transformation utilities
- Grid generation and clipping

### Data Processing (`data_loader.py`)
- Well data loading from multiple sources
- Data validation and cleaning
- Coordinate transformation
- Well categorization for different analyses

### Covariate Processing (`covariate_processing.py`)
- Terrain data extraction (elevation, slope, aspect)
- Soil property integration
- Distance calculations
- Feature engineering for ML models

### Batch Processing (`automated_heatmap_generator.py`)
- Large-scale interpolation workflows
- Tile-based processing for memory efficiency
- Parallel processing support
- Progress tracking and logging

## Dependencies

Key Python packages:
- `streamlit` - Web interface
- `folium` - Interactive mapping
- `pykrige` - Geostatistical interpolation
- `scikit-learn` - Machine learning (Random Forest)
- `quantile-forest` - Quantile Random Forest
- `geopandas` - Spatial data manipulation
- `rasterio` - Raster data I/O
- `pyproj` - Coordinate transformations
- `pandas`, `numpy`, `scipy` - Data processing

See `pyproject.toml` for complete dependency list.

## Data Requirements

### Well Observation Data
Required fields:
- `latitude`, `longitude` - Location coordinates
- `ground water level` or `depth` - Water table depth/level
- Optional: `yield_rate`, `status`, `well_id`

### Covariate Data (for Random Forest)
Suggested covariates for WTD mapping:
- **Terrain**: Elevation, slope, aspect, curvature
- **Hydrology**: Distance to streams, drainage density
- **Soil**: Permeability, texture, drainage class
- **Climate**: Precipitation, temperature, evapotranspiration
- **Geology**: Lithology, aquifer properties

See [USAGE_FOR_WTD_MAPPING.md](USAGE_FOR_WTD_MAPPING.md) for detailed covariate preparation instructions.

## Use Cases

1. **Regional Groundwater Assessment**: Map water table depths across regions
2. **Well Viability Analysis**: Predict locations suitable for new wells
3. **Drought Monitoring**: Track temporal changes in groundwater levels
4. **Land Use Planning**: Inform development decisions with groundwater data
5. **High-Resolution WTD Mapping**: Generate continental-scale water table depth maps
6. **Hydrological Research**: Analyze spatial patterns in groundwater systems

## Performance Considerations

- **Small areas** (<10km radius): Real-time kriging feasible
- **Medium areas** (10-50km): Use pre-computed heatmaps
- **Large areas** (>50km): Batch processing with tiling recommended
- **Grid resolution**: Balance between detail and computation time
  - 100m grid: Good for regional analysis
  - 30m grid: High-resolution (slower computation)

## Advanced Usage

### Custom Variogram Models

```python
from interpolation import krige_on_grid

# Manual variogram parameters
variogram_params = {
    'range': 1500,  # meters
    'sill': 0.25,
    'nugget': 0.1
}

Z, SS, OK = krige_on_grid(
    wells_x_m, wells_y_m, values,
    x_vals_m, y_vals_m,
    variogram_model='spherical',
    variogram_parameters=variogram_params
)
```

### Batch WTD Map Generation

See `examples/generate_wtd_map.py` for a complete workflow example similar to HydroFrame-ML approach.

## Documentation

### Quick Start
- [Quick Start Guide](docs/scope/quick_start.md) - Get started quickly
- [Installation Guide](#installation) - Setup instructions

### WTD Mapping
- **[Detailed WTD Mapping Guide](USAGE_FOR_WTD_MAPPING.md)** - Complete workflow for high-resolution water table depth mapping
- **[Comparison with HydroFrame-ML](COMPARISON.md)** - Detailed comparison and compatibility analysis
- [Example Scripts](examples/README.md) - Ready-to-use training and map generation scripts

### Architecture & Development
- [Architecture Overview](docs/scope/architecture_summary.md) - System design
- [Development Plan](docs/scope/README.md) - Roadmap and planning

## Contributing

Contributions are welcome! Areas for improvement:
- Additional interpolation methods
- Performance optimizations
- Support for more data formats
- Enhanced uncertainty quantification
- Multi-temporal analysis

## References

If using this code for research, consider citing:
- Ma, Y., et al. (2025). High Resolution Static Water Table Depth Estimation. *Nature Communications Earth & Environment*. https://doi.org/10.1038/s43247-025-03094-3

## License

[Add appropriate license information]

## Contact

For questions about adapting this system for high-resolution WTD mapping, see the usage guide or open an issue on GitHub.
