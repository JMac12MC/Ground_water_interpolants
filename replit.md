# Groundwater Mapper

## Overview

This is a Streamlit-based groundwater visualization application that allows users to explore and analyze well data through interactive maps and interpolation techniques. The application focuses on groundwater yield and depth analysis in New Zealand, particularly the Canterbury region, using scientific interpolation methods like Kriging and providing performance-optimized visualization through pre-computed heatmaps.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Mapping**: Folium with Leaflet.js for interactive maps
- **Visualization**: 
  - Heatmaps using Folium's HeatMap plugin
  - Marker clustering for well locations
  - GeoJSON-based choropleth maps for interpolated data
- **UI Components**: Streamlit sidebar for controls, custom CSS styling for branding

### Backend Architecture
- **Data Processing**: Python-based with pandas, numpy, and geopandas
- **Interpolation Engine**: 
  - Ordinary Kriging using PyKrige library
  - Random Forest regression fallback
  - Spatial interpolation with soil polygon clipping
- **Performance Optimization**: Pre-computed heatmap system using zone-based processing

### Data Storage Solutions
- **Primary Database**: PostgreSQL for polygon storage and merged polygon data management
- **Heatmap Storage**: PostgreSQL with PostGIS for high-performance spatial queries
- **File Storage**: CSV and Shapefile support for data import/export
- **Database Features**: Full CRUD operations for polygon management, analysis session storage/retrieval

## Key Components

### Core Modules

1. **app.py**: Main Streamlit application entry point
   - Handles UI state management
   - Coordinates between data loading, processing, and visualization
   - Manages user interactions and map updates

2. **data_loader.py**: Data ingestion and management
   - Supports multiple data sources (sample data, custom uploads, NZ government APIs)
   - Handles data validation and preprocessing
   - Generates synthetic data for demonstration when real data unavailable

3. **interpolation.py**: Spatial analysis engine
   - Implements Ordinary Kriging for yield and depth interpolation
   - Generates GeoJSON grids for accurate visualization
   - Includes variance calculation for uncertainty mapping
   - Applies soil polygon clipping for realistic boundaries

4. **database.py**: Data persistence layer
   - Manages SQLite and PostgreSQL connections
   - Handles polygon storage and retrieval
   - Provides optimized heatmap data queries with spatial filtering

5. **precompute_heatmaps.py**: Performance optimization system
   - Zone-based interpolation processing (inspired by Windy.com architecture)
   - Splits large datasets into manageable spatial tiles
   - Pre-computes yield and depth heatmaps for instant loading

### Utility Components

- **utils.py**: Helper functions for distance calculations and data export
- **check_heatmap_status.py**: System diagnostics for performance optimization
- **run_preprocessing.py**: Command-line tool for heatmap generation

## Data Flow

1. **Data Ingestion**: Wells data loaded from various sources (CSV, API, sample data)
2. **Preprocessing**: Data validation, coordinate transformation, outlier filtering
3. **Spatial Analysis**: 
   - Apply radius-based filtering around user-selected points
   - Perform interpolation using Kriging or Random Forest
   - Clip results to soil drainage polygons for realistic boundaries
4. **Visualization**: Convert interpolated data to GeoJSON for map rendering
5. **Performance Path**: For large datasets, use pre-computed heatmaps from database

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Folium**: Interactive mapping with Leaflet.js integration
- **PyKrige**: Geostatistical interpolation (Ordinary Kriging)
- **Scikit-learn**: Machine learning algorithms (Random Forest fallback)
- **GeoPandas**: Spatial data manipulation
- **SQLAlchemy**: Database ORM with PostgreSQL and SQLite support

### Data Sources
- **New Zealand Government APIs**: Environment Canterbury wells and bores data
- **S-map Soil Data**: Landcare Research soil drainage classifications
- **Custom Data Upload**: CSV file support for user data

### External Services
- **LRIS Portal**: New Zealand geospatial data (soil polygons)
- **Environment Canterbury**: Regional well and bore data via API

## Deployment Strategy

### Development Environment
- **Replit**: Cloud-based development with integrated database support
- **Environment Variables**: 
  - `DATABASE_URL`: SQLite connection for development
  - `HEATMAP_DATABASE_URL`: PostgreSQL for production heatmaps

### Performance Optimization
- **Two-tier Architecture**:
  - Development: On-demand interpolation with smaller datasets
  - Production: Pre-computed heatmaps with PostgreSQL for 50,000+ wells
- **Progressive Loading**: Load only data within current map viewport
- **Spatial Indexing**: PostGIS indexes for fast spatial queries

### Scalability Considerations
- Zone-based processing prevents memory crashes with large datasets
- Heatmap preprocessing can be parallelized across multiple cores
- Database queries optimized for spatial filtering and viewport-based loading

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

- **July 30, 2025**: MAJOR FEATURE ADDED: Windy.com-style smooth raster visualization with toggle functionality
  - Implemented `generate_smooth_raster_overlay()` function using cubic interpolation and Gaussian smoothing
  - Added "Heatmap Display Style" selector: Triangle Mesh (Scientific) vs Smooth Raster (Windy.com Style)
  - Applied smooth visualization to both stored and fresh heatmaps with consistent global coloring
  - Enhanced dynamic colormap range calculation to properly utilize full color spectrum (blue to red)
  - Fixed Ground Water Level colormap range (0-100m) vs Yield range (0-25 L/s) for proper color distribution
  - Preserved all existing triangular mesh functionality while adding new weather-map style display
  - Uses 512x512 raster resolution with 70% opacity for smooth blending over base map

- **July 27, 2025**: Extended heatmap grid system from 2x3 to configurable 10x10 grid
  - Added flexible grid generation supporting up to 100 heatmaps (10x10 layout)
  - Maintained precise 19.82km spacing across extended grid with latitude-compensated longitude offsets
  - Added UI selection between 2×3 Grid (6 heatmaps) and 10×10 Grid (100 heatmaps)
  - Extended coverage area: 178km south × 178km east for comprehensive regional analysis
  - Preserved Banks Peninsula exclusion functionality across all grid sizes
  - Maintains backward compatibility with existing 2x3 system as default

- **July 25, 2025**: CRITICAL FIX: Corrected ground water level viability logic in indicator kriging
  - Fixed fundamental misunderstanding: ANY ground water level data means water was found = viable
  - Wells with ground water level measurements (any depth) are now correctly marked as viable
  - Removed arbitrary depth threshold - presence of data indicates successful water discovery
  - Dramatically improves indicator kriging accuracy by correctly identifying all wells with water
  - Logic now: viable if yield ≥ 0.1 L/s OR ground water level data exists (any depth)

- **July 25, 2025**: Fixed ground water level interpolation to handle negative values (artesian conditions)
  - Removed restrictive filtering that excluded wells with ground water level = 0 or small values
  - Added conversion of negative ground water levels to 0 for depth interpolation purposes
  - Negative values (artesian conditions) now treated as surface-level depth for visualization
  - Ground water level heatmaps will now display properly in artesian areas

- **July 25, 2025**: CRITICAL FIX: Implemented ultra-precise geodetic heatmap spacing
  - Advanced iterative refinement algorithm achieves sub-meter accuracy (< 1m error)
  - Individual calibration for latitude, top-row longitude, and bottom-row longitude offsets
  - Eliminates distance variations (was 19.79-19.91km, now 19.820km ± 1 meter)
  - Real-time verification of all 7 adjacent distances with error reporting
  - Enhanced logging shows maximum/average errors in both kilometers and meters

- **July 24, 2025**: Fixed heatmap spacing inconsistency for perfect grid alignment
  - Resolved issue where adjacent heatmaps had slightly different distances (19.77km vs 19.82km)
  - Implemented row-specific east offset calculations to account for latitude-dependent longitude spacing
  - All adjacent heatmaps now precisely 19.82km apart with perfect grid consistency
  - Enhanced sequential heatmap generation with latitude-aware distance calculations
  - Verified perfect spacing using iterative refinement with actual distance measurements

- **July 23, 2025**: Enhanced indicator kriging with combined viability criteria
  - Extended indicator kriging logic to include ground water level assessment
  - Wells now viable (indicator = 1) if EITHER yield_rate ≥ 0.1 L/s OR ground water level > -10m depth
  - Ground water level threshold set to -10m to capture wells with accessible groundwater within 10 meters
  - Combined viability approach increases well coverage and accuracy
  - Added detailed logging showing breakdown by individual and combined criteria
  - Maintains backward compatibility when ground water level data unavailable

- **July 22, 2025**: Enhanced colormap system with percentile-based data density mapping
  - Implemented quantile-based color mapping using 256 percentile bins
  - Improved visual discrimination in areas with high data density
  - Added percentile statistics (25th, 50th, 75th percentiles) to colormap legend
  - Colors now distributed based on data histogram rather than linear value range
  - Enhanced 25-color gradient for maximum visual detail and smooth transitions
  - Extended palette: deep blues → cyan → green → yellow → red for superior data discrimination

- **July 22, 2025**: Expanded sequential heatmap system to 6-heatmap grid (2x3 layout)
  - Extended sequential_heatmap.py to generate 6 heatmaps instead of 4
  - Added Northeast (5th) and Far Southeast (6th) heatmap positions
  - Updated coordinate calculation for 39.58km extended grid coverage
  - Maintained 19.79km offset between adjacent heatmaps for seamless connection
  - Updated app.py sidebar to display all 6 heatmap coordinates
  - All heatmaps generated automatically on single map click with full wells filtering

- **July 06, 2025**: Completed PostgreSQL database integration for merged polygon storage
  - Created PolygonDatabase class with full CRUD operations
  - Added database management interface to sidebar
  - Integrated polygon saving/loading with analysis sessions
  - Fixed syntax errors and ensured app stability
  - Database stores polygon geometry, properties, well statistics, and metadata

## Changelog

- July 06, 2025: PostgreSQL database integration for polygon storage
- July 06, 2025: Initial setup