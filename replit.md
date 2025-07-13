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

- **July 13, 2025**: Implemented heatmap clipping functionality and buffer zone optimization
  - Added `clip_stored_heatmap()` method to database class for post-generation clipping
  - Integrated buffer zone interpolation (20% expansion) to minimize boundary effects
  - Added heatmap clipping UI controls in sidebar for reducing stored heatmap areas
  - Enhanced interpolation functions with configurable buffer_factor parameter
  - Automatic clipping of interpolated results back to original search radius
- **July 06, 2025**: Completed PostgreSQL database integration for merged polygon storage
  - Created PolygonDatabase class with full CRUD operations
  - Added database management interface to sidebar
  - Integrated polygon saving/loading with analysis sessions
  - Fixed syntax errors and ensured app stability
  - Database stores polygon geometry, properties, well statistics, and metadata

## Changelog

- July 06, 2025: PostgreSQL database integration for polygon storage
- July 06, 2025: Initial setup