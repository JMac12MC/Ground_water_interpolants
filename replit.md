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

- **July 23, 2025**: Enhanced indicator kriging with combined viability criteria
  - Extended indicator kriging logic to include ground water level assessment
  - Wells now viable (indicator = 1) if EITHER yield_rate ≥ 0.1 L/s OR ground water level > -3m depth
  - Ground water level threshold set to -3m to capture wells with accessible groundwater within 3 meters
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