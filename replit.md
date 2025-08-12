# Groundwater Mapper

## Overview
This project is a Streamlit-based groundwater visualization application designed for exploring and analyzing well data through interactive maps and advanced interpolation techniques. Its primary purpose is to provide insights into groundwater yield and depth, specifically focusing on the Canterbury region of New Zealand. The application utilizes scientific interpolation methods like Kriging and optimizes visualization performance through pre-computed heatmaps, aiming to enhance groundwater resource management and understanding.

## User Preferences
Preferred communication style: Simple, everyday language.

**Default Settings**:
- Default colormap: mako (Black→Blue→Green)
- Default visualization method: Ground Water Level (Spherical Kriging)
- Soil Drainage Areas: Unchecked by default
- NEW Clipping Polygon: Unchecked by default
- User prefers these as the application defaults for new sessions

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for the web interface.
- **Mapping**: Folium with Leaflet.js for interactive maps.
- **Visualization**: Heatmaps, marker clustering, and GeoJSON-based choropleth maps for interpolated data.
- **UI Components**: Streamlit sidebar for controls and custom CSS for branding.

### Backend Architecture
- **Data Processing**: Python-based using pandas, numpy, and geopandas.
- **Interpolation Engine**: Implements Ordinary Kriging (PyKrige) with a Random Forest regression fallback, incorporating spatial interpolation with soil polygon clipping.
- **Performance Optimization**: Features a pre-computed heatmap system utilizing zone-based processing to enhance loading times.

### Data Storage Solutions
- **Primary Database**: PostgreSQL with PostGIS for polygon storage, merged polygon data management, and high-performance spatial queries for heatmaps.
- **File Storage**: Supports CSV and Shapefile for data import/export.
- **Database Features**: Supports CRUD operations for polygon management and analysis session storage.

### Key Components
- **app.py**: Main Streamlit entry point, coordinating UI, data, and visualization.
- **data_loader.py**: Manages data ingestion from multiple sources, validation, and preprocessing.
- **interpolation.py**: Core spatial analysis engine for Kriging, generating GeoJSON grids, and handling uncertainty mapping and soil polygon clipping.
- **database.py**: Persistence layer managing database connections and optimized data queries.
- **precompute_heatmaps.py**: System for performance optimization, enabling zone-based processing and pre-computation of heatmaps.

### Data Flow
1. **Data Ingestion**: Wells data from various sources (CSV, API, sample data).
2. **Preprocessing**: Validation, coordinate transformation, outlier filtering.
3. **Spatial Analysis**: Radius-based filtering, interpolation (Kriging/Random Forest), and clipping results to soil polygons.
4. **Visualization**: Conversion of interpolated data to GeoJSON for map rendering.
5. **Performance Path**: Utilization of pre-computed heatmaps from the database for large datasets.

## External Dependencies

### Core Libraries
- **Streamlit**: For the web application.
- **Folium**: For interactive mapping.
- **PyKrige**: For geostatistical interpolation.
- **Scikit-learn**: For machine learning algorithms (Random Forest).
- **GeoPandas**: For spatial data manipulation.
- **SQLAlchemy**: For database ORM (PostgreSQL and SQLite).

### Data Sources
- **New Zealand Government APIs**: Specifically Environment Canterbury for wells and bores data.
- **S-map Soil Data**: From Landcare Research for soil drainage classifications.
- **Custom Data Upload**: Via CSV files.

### External Services
- **LRIS Portal**: For New Zealand geospatial data, including soil polygons.
- **Environment Canterbury**: Provides regional well and bore data via API.
```