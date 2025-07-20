import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import os
import base64
import requests
import geopandas as gpd
from utils import get_distance, download_as_csv
from data_loader import load_sample_data, load_nz_govt_data, load_api_data
from interpolation import generate_heat_map_data, generate_geo_json_grid, calculate_kriging_variance, generate_indicator_kriging_mask, create_indicator_polygon_geometry, get_prediction_at_point, create_map_with_interpolated_data
from database import PolygonDatabase
import time
# Regional heatmap removed per user request

# Set page configuration with stability settings
st.set_page_config(
    page_title="Groundwater Mapper",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean title header without image
def add_banner():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h1 style="font-size: 2.5rem; font-weight: bold; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">Groundwater Finder</h1>
            <p style="font-size: 1.2rem; margin-bottom: 0; opacity: 0.9;">Clean interface for local groundwater analysis</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Initialize ALL session state variables first - order matters for Streamlit stability
session_defaults = {
    'selected_point': None,
    'selected_point_east': None,
    'zoom_level': 10,
    'wells_data': None,
    'filtered_wells': None,
    'filtered_wells_east': None,
    'heat_map_visibility': True,
    'well_markers_visibility': True,
    'search_radius': 20,
    'soil_polygons': None,
    'show_soil_polygons': True,
    'stored_heatmaps': [],
    'interpolation_method': 'kriging',
    'show_kriging_variance': False,
    'auto_fit_variogram': False,
    'variogram_model': 'spherical',
    'geojson_data': None,
    'fresh_heatmap_displayed': False,
    'new_heatmap_added': False,
    'colormap_updated': False
}

# Initialize all session state variables
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Initialize database connection after other session state
if 'polygon_db' not in st.session_state:
    try:
        st.session_state.polygon_db = PolygonDatabase()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.session_state.polygon_db = None

# Add banner
add_banner()

# Sidebar - for options and filters
with st.sidebar:
    st.header("Data Options")

    # Load Canterbury wells data with caching
    if st.session_state.wells_data is None:
        with st.spinner("Loading Canterbury wells data..."):
            try:
                st.session_state.wells_data = load_nz_govt_data()
                if st.session_state.wells_data is None or len(st.session_state.wells_data) == 0:
                    st.error("Failed to load wells data. Please refresh the page.")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading wells data: {e}")
                st.stop()

    # Load soil drainage polygons from database
    if st.session_state.soil_polygons is None and st.session_state.polygon_db is not None:
        try:
            with st.spinner("Loading merged soil drainage polygons from database..."):
                # Get stored merged polygons from database
                stored_polygons = st.session_state.polygon_db.get_all_polygons()

                if stored_polygons:
                    # Convert stored polygons back to GeoDataFrame
                    from shapely import wkt
                    geometries = []
                    properties = []

                    for polygon in stored_polygons:
                        # Parse WKT geometry
                        geom = wkt.loads(polygon['geometry_wkt'])
                        geometries.append(geom)

                        # Combine polygon info with properties
                        prop_dict = polygon['properties'] if polygon['properties'] else {}
                        prop_dict.update({
                            'polygon_name': polygon['polygon_name'],
                            'area_km2': polygon['area_km2'],
                            'well_count': polygon['well_count'],
                            'avg_yield': polygon['avg_yield']
                        })
                        properties.append(prop_dict)

                    # Create GeoDataFrame
                    st.session_state.soil_polygons = gpd.GeoDataFrame(
                        properties, 
                        geometry=geometries, 
                        crs='EPSG:4326'
                    )

                    st.success(f"Loaded {len(st.session_state.soil_polygons)} merged soil drainage polygons from database")
                else:
                    st.warning("No merged soil polygons found in database.")
                    st.info("Run the polygon processing script first to merge and store soil polygons.")
                    st.session_state.soil_polygons = None

        except Exception as e:
            st.warning(f"Could not load soil polygons from database: {str(e)}")
            # Try to reconnect to database
            try:
                st.session_state.polygon_db = PolygonDatabase()
                stored_polygons = st.session_state.polygon_db.get_all_polygons()
                if stored_polygons:
                    st.success("Database reconnected successfully")
                    # Removed automatic rerun to prevent restart loops
            except:
                st.info("Fallback: Run the polygon processing script to merge and store soil polygons.")
                st.session_state.soil_polygons = None
    elif st.session_state.polygon_db is None:
        st.warning("Database connection not available. Cannot load soil polygons.")
        st.session_state.soil_polygons = None





    st.header("Filters")

    # Manual refresh button for when hot reload isn't working
    if st.button("🔄 Refresh App", help="Click if the app doesn't update automatically"):
        # Clear session state to force fresh start
        for key in list(st.session_state.keys()):
            if key.startswith('regional_heatmap'):
                del st.session_state[key]
        # Removed automatic rerun to prevent restart loops

    # Radius filter (now used for local context when pre-computed heatmaps are available)
    st.session_state.search_radius = st.slider(
        "Search Area Size (km)",
        min_value=1,
        max_value=50,
        value=st.session_state.search_radius,
        step=1,
        help="Creates a square search area. For example, 10km creates a 20km × 20km square for showing nearby wells and local analysis when you click on the map"
    )

    # Informational note
    if st.session_state.polygon_db and st.session_state.polygon_db.pg_engine:
        try:
            yield_data = st.session_state.polygon_db.get_heatmap_data('yield', bounds={'north': -40, 'south': -50, 'east': 175, 'west': 165})
            if yield_data:
                st.info("📍 **High-Performance Mode**: Full regional heatmap displayed. Click anywhere to see local well details.")
            else:
                st.write("**Standard Mode**: Click on map to generate local interpolation within search radius.")
        except:
            st.write("**Standard Mode**: Click on map to generate local interpolation within search radius.")
    else:
        st.write("**Standard Mode**: Click on map to generate local interpolation within search radius.")



    # Visualization method selection - single dropdown for all options
    st.header("Analysis Options")
    visualization_method = st.selectbox(
        "Map Visualization Type",
        options=[
            "Standard Kriging (Yield)", 
            "Yield Kriging (Spherical)",
            "Specific Capacity Kriging (Spherical)",
            "Depth to Groundwater (Standard Kriging)",
            "Depth to Groundwater (Auto-Fitted Spherical)",
            "Ground Water Level (Spherical Kriging)",
            "Indicator Kriging (Yield Suitability)"
        ],
        index=0,
        help="Choose the visualization type: yield estimation, depth analysis, groundwater level, or yield suitability probability",
        key="visualization_method_selector"
    )

    # Map visualization selection to internal parameters
    if 'interpolation_method' not in st.session_state:
        st.session_state.interpolation_method = 'kriging'
    if 'show_kriging_variance' not in st.session_state:
        st.session_state.show_kriging_variance = False
    if 'auto_fit_variogram' not in st.session_state:
        st.session_state.auto_fit_variogram = False
    if 'variogram_model' not in st.session_state:
        st.session_state.variogram_model = 'spherical'





    # Update session state based on selection
    if visualization_method == "Standard Kriging (Yield)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = False
    elif visualization_method == "Yield Kriging (Spherical)":
        st.session_state.interpolation_method = 'yield_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
    elif visualization_method == "Specific Capacity Kriging (Spherical)":
        st.session_state.interpolation_method = 'specific_capacity_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
    elif visualization_method == "Depth to Groundwater (Standard Kriging)":
        st.session_state.interpolation_method = 'depth_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = False
    elif visualization_method == "Depth to Groundwater (Auto-Fitted Spherical)":
        st.session_state.interpolation_method = 'depth_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
    elif visualization_method == "Ground Water Level (Spherical Kriging)":
        st.session_state.interpolation_method = 'ground_water_level_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
    elif visualization_method == "Indicator Kriging (Yield Suitability)":
        st.session_state.interpolation_method = 'indicator_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'linear'

    # Display options
    st.header("Display Options")
    st.session_state.heat_map_visibility = st.checkbox("Show Heat Map", value=st.session_state.heat_map_visibility)
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=False)
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")

    # Stored Heatmaps Management Section
    st.markdown("---")
    st.subheader("🗺️ Stored Heatmaps")

    # Always ensure stored heatmaps are loaded from database for sidebar display
    if st.session_state.polygon_db and not st.session_state.stored_heatmaps:
        try:
            st.session_state.stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
        except Exception as e:
            print(f"Failed to load stored heatmaps for sidebar: {e}")
            st.session_state.stored_heatmaps = []

    if st.session_state.stored_heatmaps:
        st.write(f"**{len(st.session_state.stored_heatmaps)} stored heatmaps available**")

        # Refresh and Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh List", help="Reload stored heatmaps from database"):
                if st.session_state.polygon_db:
                    try:
                        stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
                        st.session_state.stored_heatmaps = stored_heatmaps
                        st.success(f"Refreshed: {len(stored_heatmaps)} heatmaps found")
                    except Exception as e:
                        st.error(f"Error refreshing: {e}")
                        st.session_state.stored_heatmaps = []
                else:
                    st.error("Database not available")

        with col2:
            if st.button("🗑️ Clear All", type="secondary"):
                if st.session_state.polygon_db:
                    try:
                        count = st.session_state.polygon_db.delete_all_stored_heatmaps()
                        st.session_state.stored_heatmaps = []
                        st.success(f"Cleared {count} stored heatmaps")
                    except Exception as e:
                        st.error(f"Error clearing heatmaps: {e}")
                else:
                    st.error("Database not available")

        # Display each stored heatmap with details
        for heatmap in st.session_state.stored_heatmaps:
            with st.expander(f"📍 {heatmap['heatmap_name']}"):
                st.write(f"**Method:** {heatmap['interpolation_method']}")
                st.write(f"**Location:** {heatmap['center_lat']:.4f}, {heatmap['center_lon']:.4f}")
                st.write(f"**Radius:** {heatmap['radius_km']} km")
                st.write(f"**Wells:** {heatmap['well_count']}")
                st.write(f"**Created:** {heatmap['created_at']}")

                if st.button(f"🗑️ Delete", key=f"delete_{heatmap['id']}"):
                    print(f"\n🗑️ DELETE BUTTON CLICKED: User wants to delete heatmap ID {heatmap['id']} ('{heatmap['heatmap_name']}')")

                    # Log current session state before deletion
                    current_session_count = len(st.session_state.stored_heatmaps)
                    current_session_ids = [h['id'] for h in st.session_state.stored_heatmaps]
                    print(f"📊 PRE-DELETE SESSION STATE: {current_session_count} heatmaps in session: {current_session_ids}")

                    try:
                        # Attempt deletion from database
                        print(f"🗑️ CALLING DATABASE DELETE: Attempting to delete heatmap ID {heatmap['id']}")
                        deletion_success = st.session_state.polygon_db.delete_stored_heatmap(heatmap['id'])
                        print(f"📊 DATABASE DELETE RESULT: Success = {deletion_success}")

                        if deletion_success:
                            print(f"✅ DATABASE DELETE SUCCESS: Heatmap ID {heatmap['id']} deleted from database")

                            # Remove from session state immediately (optimistic update)
                            print(f"🔄 UPDATING SESSION STATE: Removing heatmap ID {heatmap['id']} from session")
                            original_count = len(st.session_state.stored_heatmaps)
                            st.session_state.stored_heatmaps = [h for h in st.session_state.stored_heatmaps if h['id'] != heatmap['id']]
                            new_count = len(st.session_state.stored_heatmaps)
                            print(f"📊 SESSION UPDATE RESULT: {original_count} → {new_count} heatmaps in session")
                            print(f"📋 NEW SESSION IDS: {[h['id'] for h in st.session_state.stored_heatmaps]}")

                            st.success(f"Deleted heatmap: {heatmap['heatmap_name']}")
                        else:
                            print(f"❌ DATABASE DELETE FAILED: Could not delete heatmap ID {heatmap['id']}")
                            st.error(f"Failed to delete heatmap: {heatmap['heatmap_name']}")

                        # Update session state
                        st.session_state.stored_heatmaps = [h for h in st.session_state.stored_heatmaps if h['id'] != heatmap['id']]

                    except Exception as e:
                        print(f"❌ DELETE OPERATION ERROR: Unexpected error during deletion: {e}")
                        import traceback
                        print(f"📍 STACK TRACE: {traceback.format_exc()}")
                        st.error(f"Error deleting heatmap: {e}")

                        # Simple error handling without rerun
                        print(f"Error in delete operation: {e}")
                        st.session_state.stored_heatmaps = [h for h in st.session_state.stored_heatmaps if h['id'] != heatmap['id']]
    else:
        st.write("*No stored heatmaps available*")
        st.write("Generate a heatmap and use the 'Save Heatmap' button to store it permanently.")



# Main content area
main_col1, main_col2 = st.columns([3, 1])

with main_col1:
    # Default location (New Zealand as example)
    default_location = [-43.5320, 172.6306]  # Christchurch, New Zealand

    # Create map centered at default location
    if st.session_state.selected_point:
        center_location = st.session_state.selected_point
    else:
        center_location = default_location

    m = folium.Map(location=center_location, zoom_start=st.session_state.zoom_level, 
                  tiles="OpenStreetMap")

    # Add soil drainage polygons if available and enabled
    if st.session_state.soil_polygons is not None and st.session_state.show_soil_polygons:
        # Convert to GeoJSON and add to map
        folium.GeoJson(
            st.session_state.soil_polygons.__geo_interface__,
            name="Soil Drainage Areas",
            style_function=lambda feature: {
                'fillColor': 'transparent',
                'color': 'blue',
                'weight': 1,
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['DRAINAGE'] if 'DRAINAGE' in st.session_state.soil_polygons.columns else [],
                aliases=['Drainage:'] if 'DRAINAGE' in st.session_state.soil_polygons.columns else [],
                labels=True,
                sticky=False
            )
        ).add_to(m)

    # UNIFIED COLORMAP PROCESSING: Calculate global min/max across all heatmaps for consistent coloring
    global_min_value = float('inf')
    global_max_value = float('-inf')
    all_heatmap_values = []

    # Collect values from stored heatmaps for global color scaling
    if st.session_state.stored_heatmaps:
        for stored_heatmap in st.session_state.stored_heatmaps:
            geojson_data = stored_heatmap.get('geojson_data')
            if geojson_data and 'features' in geojson_data:
                for feature in geojson_data['features']:
                    value = feature['properties'].get('yield', feature['properties'].get('value', 0))
                    if value > 0:  # Only consider meaningful values
                        all_heatmap_values.append(value)
                        global_min_value = min(global_min_value, value)
                        global_max_value = max(global_max_value, value)

    # Include current fresh heatmap values if available
    if 'geojson_data' in st.session_state and st.session_state.geojson_data:
        for feature in st.session_state.geojson_data.get('features', []):
            value = feature['properties'].get('yield', feature['properties'].get('value', 0))
            if value > 0:
                all_heatmap_values.append(value)
                global_min_value = min(global_min_value, value)
                global_max_value = max(global_max_value, value)

    # Fallback to reasonable defaults if no data
    if global_min_value == float('inf'):
        global_min_value = 0
        global_max_value = 25

    print(f"UNIFIED COLORMAP: Global value range {global_min_value:.2f} to {global_max_value:.2f} across all displayed heatmaps")

    # DEFINE GLOBAL UNIFIED COLOR FUNCTION OUTSIDE THE LOOP
    def get_global_unified_color(value, method='kriging'):
        """Global unified color function for consistent visualization across ALL heatmaps"""
        if method == 'indicator_kriging':
            # Three-tier classification: red (poor), orange (moderate), green (good)
            if value <= 0.4:
                return '#FF0000'    # Red for poor
            elif value <= 0.7:
                return '#FF8000'    # Orange for moderate
            else:
                return '#00FF00'    # Green for good
        else:
            # Use GLOBAL min/max for consistent color scaling across ALL heatmaps
            if global_max_value <= global_min_value:
                return '#000080'  # Default blue if no range

            # Normalize value to 0-1 range using global min/max
            normalized_value = (value - global_min_value) / (global_max_value - global_min_value)
            normalized_value = max(0, min(1, normalized_value))  # Clamp to 0-1

            # Use 15-band color system with normalized value
            colors = [
                '#000080',  # Band 1: Dark blue
                '#0000B3',  # Band 2: Blue
                '#0000E6',  # Band 3: Bright blue
                '#0033FF',  # Band 4: Blue-cyan
                '#0066FF',  # Band 5: Light blue
                '#0099FF',  # Band 6: Sky blue
                '#00CCFF',  # Band 7: Cyan
                '#00FFCC',  # Band 8: Cyan-green
                '#00FF99',  # Band 9: Aqua green
                '#00FF66',  # Band 10: Green-yellow
                '#33FF33',  # Band 11: Green
                '#99FF00',  # Band 12: Yellow-green
                '#FFFF00',  # Band 13: Yellow
                '#FF9900',  # Band 14: Orange
                '#FF0000'   # Band 15: Red
            ]
            # Determine which band the normalized value falls into
            band_index = min(14, int(normalized_value * 15))
            return colors[band_index]

    # Stored heatmaps will be displayed after fresh heatmap generation and global range calculation

    # Load and display pre-computed heatmaps if available
    heatmap_data = None
    if st.session_state.polygon_db and st.session_state.polygon_db.pg_engine:
        try:
            # Try to load pre-computed heatmap data
            if visualization_method in ["Standard Kriging (Yield)", "Yield Kriging (Spherical)"]:
                heatmap_data = st.session_state.polygon_db.get_heatmap_data('yield')
            elif visualization_method == "Specific Capacity Kriging (Spherical)":
                heatmap_data = st.session_state.polygon_db.get_heatmap_data('specific_capacity')
            elif "Depth" in visualization_method:
                heatmap_data = st.session_state.polygon_db.get_heatmap_data('depth')
            elif visualization_method == "Ground Water Level (Spherical Kriging)":
                heatmap_data = st.session_state.polygon_db.get_heatmap_data('ground_water_level')

                # Debug ground water level data availability
                if st.session_state.wells_data is not None:
                    gwl_column_exists = 'ground water level' in st.session_state.wells_data.columns
                    if gwl_column_exists:
                        gwl_data = st.session_state.wells_data['ground water level']
                        non_null_count = gwl_data.notna().sum()
                        non_zero_count = (gwl_data != 0).sum() if non_null_count > 0 else 0
                        valid_count = ((gwl_data.notna()) & (gwl_data != 0) & (gwl_data.abs() > 0.1)).sum()
                        st.info(f"Ground Water Level Data: {non_null_count} non-null, {non_zero_count} non-zero, {valid_count} valid values")
                    else:
                        st.warning("'ground water level' column not found in dataset")

            if heatmap_data:
                st.info(f"🚀 Using pre-computed heatmap with {len(heatmap_data):,} data points")
        except Exception as e:
            st.warning(f"Pre-computed heatmaps not available: {e}")
            heatmap_data = None

    # Process wells data if available
    if st.session_state.wells_data is not None:
        wells_df = st.session_state.wells_data

        # If we have selected points, show local context for both
        if st.session_state.selected_point:
            from utils import is_within_square

            # Process original location
            center_lat, center_lon = st.session_state.selected_point
            wells_df['within_square'] = wells_df.apply(
                lambda row: is_within_square(
                    row['latitude'], 
                    row['longitude'],
                    center_lat,
                    center_lon,
                    st.session_state.search_radius
                ), 
                axis=1
            )

            # Calculate distances for display purposes (still useful for tooltips)
            wells_df['distance'] = wells_df.apply(
                lambda row: get_distance(
                    center_lat, 
                    center_lon, 
                    row['latitude'], 
                    row['longitude']
                ), 
                axis=1
            )

            # Filter wells for local display using square bounds
            filtered_wells = wells_df[wells_df['within_square']]
            st.session_state.filtered_wells = filtered_wells.copy()

            # Process east location if it exists
            if st.session_state.selected_point_east:
                center_lat_east, center_lon_east = st.session_state.selected_point_east
                wells_df['within_square_east'] = wells_df.apply(
                    lambda row: is_within_square(
                        row['latitude'], 
                        row['longitude'],
                        center_lat_east,
                        center_lon_east,
                        st.session_state.search_radius
                    ), 
                    axis=1
                )

                # Filter wells for east location
                filtered_wells_east = wells_df[wells_df['within_square_east']]
                st.session_state.filtered_wells_east = filtered_wells_east.copy()

            # Create marker for selected point
            folium.Marker(
                location=st.session_state.selected_point,
                popup="Selected Location (Original)",
                icon=folium.Icon(color='red', icon='crosshairs', prefix='fa'),
                tooltip="Your Selected Point (Original)"
            ).add_to(m)

            # Create marker for east point if it exists
            if st.session_state.selected_point_east:
                folium.Marker(
                    location=st.session_state.selected_point_east,
                    popup="Selected Location (10km East)",
                    icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa'),
                    tooltip=f"Auto-Generated Point ({st.session_state.search_radius * 2.0:.0f}km East - Seamless Join)"
                ).add_to(m)

            # Draw square for original search area
            center_lat, center_lon = st.session_state.selected_point
            radius_km = st.session_state.search_radius

            # Calculate square bounds for original location
            lat_radius_deg = radius_km / 111.0  # ~111km per degree latitude
            lon_radius_deg = radius_km / (111.0 * np.cos(np.radians(center_lat)))  # adjust for longitude

            # Create square coordinates for original location
            square_bounds = [
                [center_lat - lat_radius_deg, center_lon - lon_radius_deg],  # SW corner
                [center_lat - lat_radius_deg, center_lon + lon_radius_deg],  # SE corner
                [center_lat + lat_radius_deg, center_lon + lon_radius_deg],  # NE corner
                [center_lat + lat_radius_deg, center_lon - lon_radius_deg],  # NW corner
                [center_lat - lat_radius_deg, center_lon - lon_radius_deg]   # Close square
            ]

            # Draw square for original location
            folium.Polygon(
                locations=square_bounds,
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
                fill_opacity=0.1,
                weight=2,
                popup="Original Search Area"
            ).add_to(m)

            # Draw square for east location if it exists
            if st.session_state.selected_point_east:
                center_lat_east, center_lon_east = st.session_state.selected_point_east
                
                # Calculate square bounds for east location
                lat_radius_deg_east = radius_km / 111.0
                lon_radius_deg_east = radius_km / (111.0 * np.cos(np.radians(center_lat_east)))

                # Create square coordinates for east location
                square_bounds_east = [
                    [center_lat_east - lat_radius_deg_east, center_lon_east - lon_radius_deg_east],  # SW corner
                    [center_lat_east - lat_radius_deg_east, center_lon_east + lon_radius_deg_east],  # SE corner
                    [center_lat_east + lat_radius_deg_east, center_lon_east + lon_radius_deg_east],  # NE corner
                    [center_lat_east + lat_radius_deg_east, center_lon_east - lon_radius_deg_east],  # NW corner
                    [center_lat_east - lat_radius_deg_east, center_lon_east - lon_radius_deg_east]   # Close square
                ]

                # Draw square for east location
                folium.Polygon(
                    locations=square_bounds_east,
                    color="#cc8631",  # Different color for east area
                    fill=True,
                    fill_color="#cc8631",
                    fill_opacity=0.1,
                    weight=2,
                    popup=f"East Search Area ({st.session_state.search_radius * 2.0:.0f}km East - Seamless)"
                ).add_to(m)

        # Display heatmap - use pre-computed if available, otherwise generate on-demand
        if st.session_state.heat_map_visibility:
            if heatmap_data:
                # Display pre-computed heatmap
                st.success("⚡ Displaying pre-computed heatmap - instant loading!")

                # Convert pre-computed data to GeoJSON for display
                geojson_data = {"type": "FeatureCollection", "features": []}

                # Determine the value field based on heatmap type
                if visualization_method in ["Standard Kriging (Yield)", "Yield Kriging (Spherical)"]:
                    value_field = 'yield_value'
                    display_name = 'yield'
                elif visualization_method == "Specific Capacity Kriging (Spherical)":
                    value_field = 'specific_capacity_value'
                    display_name = 'yield'  # Keep for compatibility
                else:
                    value_field = 'depth_value'
                    display_name = 'yield'  # Keep for compatibility

                # Create triangulated surface from pre-computed points
                if len(heatmap_data) > 3:
                    from scipy.spatial import Delaunay
                    import numpy as np

                    # Extract coordinates and values
                    points_2d = np.array([[point['longitude'], point['latitude']] for point in heatmap_data])
                    values = np.array([point[value_field] for point in heatmap_data])

                    # Create Delaunay triangulation
                    tri = Delaunay(points_2d)

                    # Create triangular polygons
                    for simplex in tri.simplices:
                        vertices = points_2d[simplex]
                        vertex_values = values[simplex]
                        avg_value = float(np.mean(vertex_values))

                        if avg_value > 0.01:  # Only show meaningful values
                            poly = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[
                                        [float(vertices[0,0]), float(vertices[0,1])],
                                        [float(vertices[1,0]), float(vertices[1,1])],
                                        [float(vertices[2,0]), float(vertices[2,1])],
                                        [float(vertices[0,0]), float(vertices[0,1])]
                                    ]]
                                },
                                "properties": {
                                    display_name: avg_value
                                }
                            }
                            geojson_data["features"].append(poly)
            else:
                # Fallback to on-demand generation if no pre-computed data
                has_selected_point = st.session_state.selected_point is not None
                has_filtered_wells = 'filtered_wells' in st.session_state and st.session_state.filtered_wells is not None
                wells_count = len(st.session_state.filtered_wells) if has_filtered_wells else 0
                
                print(f"HEATMAP GENERATION CHECK: selected_point={has_selected_point}, filtered_wells={has_filtered_wells}, wells_count={wells_count}")
                
                if has_selected_point and has_filtered_wells and wells_count > 0:
                    with st.spinner("🔄 Generating dual interpolation (original + 10km east)..."):
                        # Generate heatmap for original location
                        indicator_mask = None
                        methods_requiring_mask = [
                            'kriging', 'yield_kriging_spherical', 'specific_capacity_kriging', 
                            'depth_kriging', 'depth_kriging_auto', 'ground_water_level_kriging'
                        ]

                        if st.session_state.interpolation_method in methods_requiring_mask:
                            # Generate indicator kriging mask for high-probability zones (≥0.7)
                            indicator_mask = generate_indicator_kriging_mask(
                                st.session_state.filtered_wells.copy(),
                                st.session_state.selected_point,
                                st.session_state.search_radius,
                                resolution=100,
                                soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None,
                                threshold=0.7
                            )

                        print(f"App.py: Generating original heatmap with method='{st.session_state.interpolation_method}'")
                        print(f"App.py: indicator_mask is {'provided' if indicator_mask is not None else 'None'}")

                        # Generate interpolation for original location
                        geojson_data = generate_geo_json_grid(
                            st.session_state.filtered_wells.copy(), 
                            st.session_state.selected_point, 
                            st.session_state.search_radius,
                            resolution=100,
                            method=st.session_state.interpolation_method,
                            show_variance=False,
                            auto_fit_variogram=st.session_state.get('auto_fit_variogram', False),
                            variogram_model=st.session_state.get('variogram_model', 'spherical'),
                            soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None,
                            indicator_mask=indicator_mask
                        )

                        # Generate heatmap for east location if it exists
                        geojson_data_east = None
                        if st.session_state.selected_point_east and 'filtered_wells_east' in st.session_state and st.session_state.filtered_wells_east is not None:
                            print(f"App.py: Generating east heatmap (10km east)")
                            
                            # Generate indicator mask for east location if needed
                            indicator_mask_east = None
                            if st.session_state.interpolation_method in methods_requiring_mask:
                                indicator_mask_east = generate_indicator_kriging_mask(
                                    st.session_state.filtered_wells_east.copy(),
                                    st.session_state.selected_point_east,
                                    st.session_state.search_radius,
                                    resolution=100,
                                    soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None,
                                    threshold=0.7
                                )

                            # Generate interpolation for east location
                            geojson_data_east = generate_geo_json_grid(
                                st.session_state.filtered_wells_east.copy(), 
                                st.session_state.selected_point_east, 
                                st.session_state.search_radius,
                                resolution=100,
                                method=st.session_state.interpolation_method,
                                show_variance=False,
                                auto_fit_variogram=st.session_state.get('auto_fit_variogram', False),
                                variogram_model=st.session_state.get('variogram_model', 'spherical'),
                                soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None,
                                indicator_mask=indicator_mask_east
                            )
                else:
                    geojson_data = {"type": "FeatureCollection", "features": []}

                print(f"DEBUG: geojson_data exists: {bool(geojson_data)}")
                if geojson_data:
                    print(f"DEBUG: geojson_data features count: {len(geojson_data.get('features', []))}")

                if geojson_data and len(geojson_data['features']) > 0:
                    # Calculate max value for setting the color scale
                    max_value = 0
                    value_field = 'variance' if st.session_state.interpolation_method == 'kriging_variance' else 'yield'

                    for feature in geojson_data['features']:
                        if value_field in feature['properties']:
                            max_value = max(max_value, feature['properties'][value_field])

                    # Ensure reasonable minimum for visualization
                    if st.session_state.interpolation_method == 'kriging_variance':
                        max_value = max(max_value, 1.0)  # Minimum variance value
                    else:
                        max_value = max(max_value, 20.0)  # Minimum yield value

                    # Add the new heatmap to the map (in addition to stored heatmaps)
                    # This ensures both stored and newly generated heatmaps display together

                    # Instead of choropleth, use direct GeoJSON styling for more control
                    # This allows us to precisely map values to colors

                    # Use the same global unified color function for fresh heatmaps
                    def get_color(value):
                        return get_global_unified_color(value, st.session_state.interpolation_method)

                    # Style function that uses our color mapping
                    def style_feature(feature):
                        yield_value = feature['properties']['yield']
                        return {
                            'fillColor': get_color(yield_value),
                            'color': 'none',
                            'weight': 0,
                            'fillOpacity': 0.7
                        }

                    # Add the GeoJSON with our custom styling
                    display_field = 'variance' if st.session_state.interpolation_method == 'kriging_variance' else 'yield'

                    # Create a unique name for the new heatmap based on location
                    new_heatmap_name = f"New: {st.session_state.interpolation_method.replace('_', ' ').title()}"
                    if st.session_state.selected_point:
                        lat, lon = st.session_state.selected_point
                        new_heatmap_name += f" ({lat:.3f}, {lon:.3f})"

                    # Add the fresh heatmap to the map
                    fresh_geojson = folium.GeoJson(
                        data=geojson_data,
                        name=new_heatmap_name,
                        style_function=lambda feature: {
                            'fillColor': get_global_unified_color(feature['properties'][display_field], st.session_state.interpolation_method),
                            'color': 'none',
                            'weight': 0,
                            'fillOpacity': 0.7
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=[display_field],
                            aliases=[f'{display_field.title()}:'],
                            labels=True,
                            sticky=False
                        )
                    )
                    fresh_geojson.add_to(m)

                    # Add the east heatmap to the map if it exists
                    if geojson_data_east and len(geojson_data_east['features']) > 0:
                        east_heatmap_name = f"East: {st.session_state.interpolation_method.replace('_', ' ').title()}"
                        if st.session_state.selected_point_east:
                            lat_east, lon_east = st.session_state.selected_point_east
                            east_heatmap_name += f" ({lat_east:.3f}, {lon_east:.3f})"

                        fresh_geojson_east = folium.GeoJson(
                            data=geojson_data_east,
                            name=east_heatmap_name,
                            style_function=lambda feature: {
                                'fillColor': get_global_unified_color(feature['properties'][display_field], st.session_state.interpolation_method),
                                'color': 'none',
                                'weight': 0,
                                'fillOpacity': 0.7
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=[display_field],
                                aliases=[f'{display_field.title()}:'],
                                labels=True,
                                sticky=False
                            )
                        )
                        fresh_geojson_east.add_to(m)
                        print(f"EAST HEATMAP ADDED TO MAP: {east_heatmap_name} with {len(geojson_data_east.get('features', []))} features")

                    # Mark that we have a fresh heatmap displayed
                    st.session_state.fresh_heatmap_displayed = False  # Will be handled by stored heatmaps

                    print(f"FRESH HEATMAP ADDED TO MAP: {new_heatmap_name} with {len(geojson_data.get('features', []))} features")

                    # AUTO-STORE: Automatically save both generated heatmaps
                    if st.session_state.polygon_db and st.session_state.selected_point:
                        try:
                            # Store original heatmap
                            center_lat, center_lon = st.session_state.selected_point
                            heatmap_name = f"{st.session_state.interpolation_method}_{center_lat:.3f}_{center_lon:.3f}"

                            # Convert GeoJSON to simple heatmap data for storage
                            heatmap_data = []
                            for feature in geojson_data.get('features', []):
                                if 'geometry' in feature and 'properties' in feature:
                                    geom = feature['geometry']
                                    if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                                        # Get centroid of polygon
                                        coords = geom['coordinates'][0]
                                        if len(coords) >= 3:
                                            lat = sum(coord[1] for coord in coords) / len(coords)
                                            lon = sum(coord[0] for coord in coords) / len(coords)
                                            value = feature['properties'].get('yield', 0)
                                            heatmap_data.append([lat, lon, value])

                            # Store original heatmap in database
                            stored_heatmap_id = st.session_state.polygon_db.store_heatmap(
                                heatmap_name=heatmap_name,
                                center_lat=center_lat,
                                center_lon=center_lon,
                                radius_km=st.session_state.search_radius,
                                interpolation_method=st.session_state.interpolation_method,
                                heatmap_data=heatmap_data,
                                geojson_data=geojson_data,
                                well_count=len(st.session_state.filtered_wells) if st.session_state.filtered_wells is not None else 0
                            )

                            # Store east heatmap if it exists
                            if geojson_data_east and st.session_state.selected_point_east:
                                center_lat_east, center_lon_east = st.session_state.selected_point_east
                                # Ensure coordinates are float, not numpy types
                                center_lat_east = float(center_lat_east)
                                center_lon_east = float(center_lon_east)
                                heatmap_name_east = f"{st.session_state.interpolation_method}_east_{center_lat_east:.3f}_{center_lon_east:.3f}"

                                # Convert east GeoJSON to simple heatmap data for storage
                                heatmap_data_east = []
                                for feature in geojson_data_east.get('features', []):
                                    if 'geometry' in feature and 'properties' in feature:
                                        geom = feature['geometry']
                                        if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                                            # Get centroid of polygon
                                            coords = geom['coordinates'][0]
                                            if len(coords) >= 3:
                                                lat = sum(coord[1] for coord in coords) / len(coords)
                                                lon = sum(coord[0] for coord in coords) / len(coords)
                                                value = feature['properties'].get('yield', 0)
                                                heatmap_data_east.append([lat, lon, value])

                                # Store east heatmap in database
                                stored_heatmap_id_east = st.session_state.polygon_db.store_heatmap(
                                    heatmap_name=heatmap_name_east,
                                    center_lat=center_lat_east,
                                    center_lon=center_lon_east,
                                    radius_km=st.session_state.search_radius,
                                    interpolation_method=st.session_state.interpolation_method,
                                    heatmap_data=heatmap_data_east,
                                    geojson_data=geojson_data_east,
                                    well_count=len(st.session_state.filtered_wells_east) if st.session_state.filtered_wells_east is not None else 0
                                )

                            # Always reload stored heatmaps to ensure fresh heatmaps are included
                            st.session_state.stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()

                            # Mark that new heatmaps were added
                            st.session_state.new_heatmap_added = True
                            print(f"AUTO-STORED DUAL HEATMAPS: {heatmap_name} and {heatmap_name_east if geojson_data_east else 'east failed'}")
                            # Let natural Streamlit flow handle the update
                        except Exception as e:
                            print(f"Error auto-storing dual heatmaps: {e}")

                    # Add UNIFIED colormap legend using global min/max values
                    if st.session_state.interpolation_method == 'indicator_kriging':
                        # Three-tier indicator kriging legend
                        colormap = folium.StepColormap(
                            colors=['#FF0000', '#FF8000', '#00FF00'],  # Red, Orange, Green
                            vmin=0,
                            vmax=1.0,
                            index=[0, 0.4, 0.7, 1.0],  # Three-tier thresholds
                            caption='Well Yield Quality: Red = Poor (0-0.4), Orange = Moderate (0.4-0.7), Green = Good (0.7-1.0)'
                        )
                    else:
                        # UNIFIED legend using GLOBAL min/max for all interpolation types
                        colormap = folium.LinearColormap(
                            colors=['#000080', '#0000B3', '#0000E6', '#0033FF', '#0066FF', 
                                    '#0099FF', '#00CCFF', '#00FFCC', '#00FF99', '#00FF66', 
                                    '#33FF33', '#99FF00', '#FFFF00', '#FF9900', '#FF0000'],
                            vmin=float(global_min_value),
                            vmax=float(global_max_value),
                            caption=f'UNIFIED Scale: {global_min_value:.1f} to {global_max_value:.1f} L/s (All Heatmaps)'
                        )
                    colormap.add_to(m)

                    # Analysis complete

                    # Add tooltips to show appropriate values on hover
                    style_function = lambda x: {'fillColor': 'transparent', 'color': 'transparent'}
                    highlight_function = lambda x: {'fillOpacity': 0.8}

                    # Determine tooltip label based on visualization type
                    if st.session_state.interpolation_method == 'depth_kriging':
                        tooltip_field = 'yield'
                        tooltip_label = 'Depth (m):'
                    elif st.session_state.interpolation_method == 'ground_water_level_kriging':
                        tooltip_field = 'yield'
                        tooltip_label = 'Ground Water Level (m):'
                    elif st.session_state.interpolation_method == 'indicator_kriging':
                        tooltip_field = 'yield'
                        tooltip_label = 'Probability:'
                    else:
                        tooltip_field = 'yield'
                        tooltip_label = 'Yield (L/s):'

                    # Add GeoJSON overlay for tooltips
                    folium.GeoJson(
                        geojson_data,
                        style_function=style_function,
                        tooltip=folium.GeoJsonTooltip(
                            fields=[tooltip_field],
                            aliases=[tooltip_label],
                            labels=True,
                            sticky=False
                        )
                    ).add_to(m)

                    # RECALCULATE GLOBAL RANGE including fresh heatmap values
                    # Now that we have both stored and fresh heatmap data, recalculate the unified range
                    updated_global_min = float('inf')
                    updated_global_max = float('-inf')

                    # Include fresh heatmap values
                    if geojson_data and 'features' in geojson_data:
                        for feature in geojson_data['features']:
                            value = feature['properties'].get('yield', feature['properties'].get('value', 0))
                            if value > 0:
                                updated_global_min = min(updated_global_min, value)
                                updated_global_max = max(updated_global_max, value)

                    # Include stored heatmap values
                    if st.session_state.stored_heatmaps:
                        for stored_heatmap in st.session_state.stored_heatmaps:
                            stored_geojson = stored_heatmap.get('geojson_data')
                            if stored_geojson and 'features' in stored_geojson:
                                for feature in stored_geojson['features']:
                                    value = feature['properties'].get('yield', feature['properties'].get('value', 0))
                                    if value > 0:
                                        updated_global_min = min(updated_global_min, value)
                                        updated_global_max = max(updated_global_max, value)

                    # Update global variables for consistent coloring ONLY if a new heatmap was added
                    if updated_global_min != float('inf') and st.session_state.get('new_heatmap_added', False):
                        global_min_value = updated_global_min
                        global_max_value = updated_global_max
                        print(f"UPDATED UNIFIED COLORMAP: Global range {global_min_value:.2f} to {global_max_value:.2f} (including fresh heatmap)")

                        # Mark that colormap has been updated for this session
                        st.session_state.colormap_updated = True
                        st.session_state.new_heatmap_added = False  # Reset the flag
                        print("Colormap range updated - will apply to all displayed heatmaps")

    # NOW DISPLAY ALL STORED HEATMAPS with the UPDATED unified colormap
    # But skip stored heatmaps that match the current fresh heatmap location
    stored_heatmap_count = 0
    fresh_heatmap_name = None
    if st.session_state.selected_point:
        center_lat, center_lon = st.session_state.selected_point
        fresh_heatmap_name = f"{st.session_state.interpolation_method}_{center_lat:.3f}_{center_lon:.3f}"

    if st.session_state.stored_heatmaps and len(st.session_state.stored_heatmaps) > 0:
        print(f"Attempting to display {len(st.session_state.stored_heatmaps)} stored heatmaps with UPDATED unified colormap")
        print(f"Fresh heatmap name to skip: {fresh_heatmap_name}")
        for i, stored_heatmap in enumerate(st.session_state.stored_heatmaps):
            try:
                # Don't skip the current fresh heatmap - let it display as a stored heatmap too
                # This ensures continuity when the page re-renders
                if fresh_heatmap_name and stored_heatmap.get('heatmap_name') == fresh_heatmap_name:
                    print(f"DISPLAYING stored version of fresh heatmap: {stored_heatmap['heatmap_name']}")
                # All stored heatmaps should display

                # Prefer GeoJSON data for triangular mesh visualization
                geojson_data = stored_heatmap.get('geojson_data')
                heatmap_data = stored_heatmap.get('heatmap_data', [])

                if geojson_data and geojson_data.get('features'):
                    print(f"Adding stored GeoJSON heatmap {i+1}: {stored_heatmap['heatmap_name']} with {len(geojson_data['features'])} triangular features")

                    # Fix compatibility: ensure stored data has both 'value' and 'yield' properties
                    for feature in geojson_data['features']:
                        if 'properties' in feature:
                            # If 'value' doesn't exist but 'yield' does, copy it
                            if 'value' not in feature['properties'] and 'yield' in feature['properties']:
                                feature['properties']['value'] = feature['properties']['yield']
                            # If 'yield' doesn't exist but 'value' does, copy it
                            elif 'yield' not in feature['properties'] and 'value' in feature['properties']:
                                feature['properties']['yield'] = feature['properties']['value']

                    # Use the UPDATED global unified color function with method info
                    method = stored_heatmap.get('interpolation_method', 'kriging')

                    # Add GeoJSON layer for triangular mesh visualization with UPDATED UNIFIED coloring
                    folium.GeoJson(
                        geojson_data,
                        name=f"Stored: {stored_heatmap['heatmap_name']}",
                        style_function=lambda feature, method=method: {
                            'fillColor': get_global_unified_color(feature['properties'].get('yield', 0), method),
                            'color': 'none',
                            'weight': 0,
                            'fillOpacity': 0.7
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['yield'],  # Use 'yield' since that's what's reliably in stored data
                            aliases=['Value:'],
                            localize=True
                        )
                    ).add_to(m)
                    stored_heatmap_count += 1

                elif heatmap_data and len(heatmap_data) > 0:
                    print(f"Adding stored point heatmap {i+1}: {stored_heatmap['heatmap_name']} with {len(heatmap_data)} data points")

                    # Fallback to HeatMap if no GeoJSON
                    HeatMap(heatmap_data, 
                           radius=20, 
                           blur=10, 
                           name=f"Stored: {stored_heatmap['heatmap_name']}",
                           overlay=True,
                           control=True,
                           max_zoom=1).add_to(m)
                    stored_heatmap_count += 1
                else:
                    print(f"Stored heatmap {stored_heatmap['heatmap_name']} has no data")

                # Add a marker showing the center point of the stored heatmap
                folium.Marker(
                    location=[stored_heatmap['center_lat'], stored_heatmap['center_lon']],
                    popup=f"<b>{stored_heatmap['heatmap_name']}</b><br>"
                          f"Method: {stored_heatmap['interpolation_method']}<br>"
                          f"Radius: {stored_heatmap['radius_km']} km<br>"
                          f"Wells: {stored_heatmap['well_count']}<br>"
                          f"Created: {stored_heatmap['created_at']}",
                    icon=folium.Icon(color='purple', icon='info-sign')
                ).add_to(m)

            except Exception as e:
                print(f"Error displaying stored heatmap {stored_heatmap.get('heatmap_name', 'unknown')}: {e}")

        print(f"Successfully displayed {stored_heatmap_count} stored heatmaps with UPDATED unified colormap")
    else:
        print("No stored heatmaps to display - list is empty or cleared")

    # Show summary of displayed heatmaps
    total_displayed = stored_heatmap_count
    print(f"TOTAL HEATMAPS ON MAP: {total_displayed} (All via stored heatmaps)")

    # Show wells within the radius when a point is selected (for local context)
    if st.session_state.well_markers_visibility and st.session_state.selected_point:
        if 'filtered_wells' in st.session_state and st.session_state.filtered_wells is not None:
            radius_wells_layer = folium.FeatureGroup(name="Local Wells").add_to(m)

            # Filter out geotechnical/geological investigation wells and wells with no depth value from well markers
            display_wells = st.session_state.filtered_wells.copy()
            if 'well_use' in display_wells.columns:
                geotechnical_mask = display_wells['well_use'].str.contains(
                    'Geotechnical.*Investigation|Geological.*Investigation', 
                    case=False, 
                    na=False, 
                    regex=True
                )
                display_wells = display_wells[~geotechnical_mask]

            # Filter out wells with no depth value (NaN or empty depth)
            if 'depth' in display_wells.columns:
                display_wells = display_wells[display_wells['depth'].notna() & (display_wells['depth'] > 0)]

            # Note: Active well status filtering removed to show all wells with depth data
            # This ensures wells with valid depth and yield data are displayed

            # Create small dot markers for wells within the radius (excluding geotechnical wells)
            for idx, row in display_wells.iterrows():
                folium.CircleMarker(
                    location=(float(row['latitude']), float(row['longitude'])),
                    radius=3,
                    color='gray',
                    fill=True,
                    fill_color='darkblue',
                    fill_opacity=0.7,
                    tooltip=f"Well {row['well_id']} - {row['yield_rate']} L/s - Groundwater: {row['depth']:.1f}m{'(Dry)' if row.get('is_dry_well', False) else ''}"
                ).add_to(radius_wells_layer)

    # Add click event to capture coordinates (only need this once)
    folium.LatLngPopup().add_to(m)

    # Add a simple click handler that manually tracks clicks
    folium.LayerControl().add_to(m)

    # Create a custom click handler
    from folium.plugins import MousePosition
    MousePosition().add_to(m)

    # Display the map
    st.subheader("Interactive Map")
    st.caption("Click on the map to select a location and find nearby wells")
    st.caption("🔧 App Version: Updated 2025-01-06")

    # Use st_folium instead of folium_static to capture clicks
    from streamlit_folium import st_folium

    # Make sure we disable folium_static's existing click handlers
    m.add_child(folium.Element("""
    <script>
    // Clear any existing click handlers
    </script>
    """))

    # Layer control removed - all heatmaps display simultaneously

    # Use st_folium with stability optimizations
    try:
        map_data = st_folium(
            m,
            use_container_width=True,
            height=600,
            key="main_map",
            returned_objects=["last_clicked"]
        )
    except Exception as e:
        print(f"Map rendering error: {e}")
        map_data = None

    # Process clicks from the map with better stability and error handling
    try:
        if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
            # Get the coordinates from the click
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lng = map_data["last_clicked"]["lng"]

            print(f"RAW CLICK DETECTED: lat={clicked_lat:.6f}, lng={clicked_lng:.6f}")

            # Allow all clicks to generate heatmaps (reduce threshold for more sensitive detection)
            current_point = st.session_state.get('selected_point')
            coordinate_threshold = 0.0001  # Very small threshold - almost any new click will generate heatmap

            print(f"Current stored point: {current_point}")
            print(f"Coordinate threshold: {coordinate_threshold}")

            # More robust coordinate comparison
            is_new_location = True
            if current_point and len(current_point) >= 2:
                lat_diff = abs(current_point[0] - clicked_lat)
                lng_diff = abs(current_point[1] - clicked_lng)
                print(f"Coordinate differences: lat={lat_diff:.6f}, lng={lng_diff:.6f}")
                if lat_diff <= coordinate_threshold and lng_diff <= coordinate_threshold:
                    is_new_location = False
                    print(f"MAP CLICK: Same location detected - skipping duplicate (lat_diff={lat_diff:.6f}, lng_diff={lng_diff:.6f})")
                else:
                    print(f"MAP CLICK: Different location detected - will process (lat_diff={lat_diff:.6f}, lng_diff={lng_diff:.6f})")
            else:
                print("MAP CLICK: No previous point or invalid previous point - will process")

            if is_new_location:
                print(f"MAP CLICK: New location detected - updating coordinates to ({clicked_lat:.6f}, {clicked_lng:.6f})")
                if current_point:
                    print(f"Previous coordinates: ({current_point[0]:.6f}, {current_point[1]:.6f})")
                    lat_diff = abs(current_point[0] - clicked_lat)
                    lng_diff = abs(current_point[1] - clicked_lng)
                    print(f"Coordinate differences: lat={lat_diff:.6f}, lng={lng_diff:.6f}")
                else:
                    print("No previous coordinates - this is the first click")

                # Calculate seamless east point for joining heatmaps
                # Position the east heatmap so it connects seamlessly with the original
                # Use the search radius to determine the offset (search radius * 2 for seamless join)
                km_per_degree_lon = 111.0 * np.cos(np.radians(clicked_lat))
                east_offset_degrees = (st.session_state.search_radius * 2.0) / km_per_degree_lon  # Seamless join
                
                clicked_east_lat = clicked_lat
                clicked_east_lng = clicked_lng + east_offset_degrees
                
                print(f"DUAL HEATMAP: Original point ({clicked_lat:.6f}, {clicked_lng:.6f})")
                print(f"DUAL HEATMAP: East point ({clicked_east_lat:.6f}, {clicked_east_lng:.6f}) - {st.session_state.search_radius * 2.0:.1f}km east (seamless)")

                # Store both points for dual heatmap generation
                st.session_state.selected_point = [clicked_lat, clicked_lng]
                st.session_state.selected_point_east = [clicked_east_lat, clicked_east_lng]

                # Immediately regenerate filtered wells for both locations
                from utils import is_within_square
                wells_df = st.session_state.wells_data
                
                # Filter wells for original location
                wells_df['within_square'] = wells_df.apply(
                    lambda row: is_within_square(
                        row['latitude'], 
                        row['longitude'],
                        clicked_lat,
                        clicked_lng,
                        st.session_state.search_radius
                    ), 
                    axis=1
                )
                
                # Calculate distances for display purposes
                wells_df['distance'] = wells_df.apply(
                    lambda row: get_distance(
                        clicked_lat, 
                        clicked_lng, 
                        row['latitude'], 
                        row['longitude']
                    ), 
                    axis=1
                )
                
                # Store filtered wells for original location
                filtered_wells = wells_df[wells_df['within_square']]
                st.session_state.filtered_wells = filtered_wells.copy()
                
                # Filter wells for east location
                wells_df['within_square_east'] = wells_df.apply(
                    lambda row: is_within_square(
                        row['latitude'], 
                        row['longitude'],
                        clicked_east_lat,
                        clicked_east_lng,
                        st.session_state.search_radius
                    ), 
                    axis=1
                )
                
                # Store filtered wells for east location
                filtered_wells_east = wells_df[wells_df['within_square_east']]
                st.session_state.filtered_wells_east = filtered_wells_east.copy()
                
                print(f"COORDINATES UPDATED: Original wells={len(filtered_wells)}, East wells={len(filtered_wells_east)}")
                print("WELLS FILTERED: Ready for heatmap generation")
                
                # Force session state flag to ensure heatmap generation triggers
                st.session_state.force_heatmap_generation = True
            else:
                if 'lat_diff' in locals() and 'lng_diff' in locals():
                    print(f"SKIPPING CLICK: Coordinate difference too small: lat={lat_diff:.6f}, lng={lng_diff:.6f} (threshold: {coordinate_threshold})")
                else:
                    print("SKIPPING CLICK: Location comparison failed")
        else:
            print("NO CLICK DATA: map_data or last_clicked is missing or empty")
    except Exception as e:
        print(f"ERROR in click processing: {e}")
        # Don't rerun on error to prevent crash loops

    # Add cache clearing and reset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Results", use_container_width=True):
            print("🗑️ CLEAR RESULTS: User clicked Clear Results button")
            
            # Clear stored heatmaps from database
            if st.session_state.polygon_db:
                try:
                    deleted_count = st.session_state.polygon_db.delete_all_stored_heatmaps()
                    print(f"🗑️ DATABASE CLEAR: Deleted {deleted_count} stored heatmaps from database")
                    st.success(f"Cleared {deleted_count} stored heatmaps from database")
                except Exception as e:
                    print(f"❌ DATABASE CLEAR ERROR: {e}")
                    st.error(f"Error clearing stored heatmaps: {e}")
            
            # Reset session state for map and heatmaps
            keys_to_clear = [
                'selected_point', 'selected_point_east', 
                'filtered_wells', 'filtered_wells_east',
                'stored_heatmaps', 'geojson_data',
                'fresh_heatmap_displayed', 'new_heatmap_added'
            ]
            
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
                    print(f"🔄 SESSION CLEAR: Deleted {key}")
            
            st.success("Cleared all results and stored heatmaps")

    with col2:
        if st.button("🔄 Refresh App", use_container_width=True):
            # Clear all caches and reset session state
            st.cache_data.clear()
            st.cache_resource.clear()
            # Only clear non-essential keys to prevent restart loops
            essential_keys = ['wells_data', 'soil_polygons', 'polygon_db']
            for key in list(st.session_state.keys()):
                if key not in essential_keys:
                    del st.session_state[key]
            print("🔄 REFRESH: App refreshed, session state reset")
            # Removed rerun to prevent restart loops - page will refresh automatically

with main_col2:
    st.subheader("Analysis Results")

    if 'filtered_wells' in st.session_state and st.session_state.filtered_wells is not None and len(st.session_state.filtered_wells) > 0:
        # Show information about dual heatmap generation
        st.subheader("Dual Heatmap Analysis")
        if st.session_state.selected_point_east:
            st.success("✅ Dual heatmaps generated: Original location + 20km East")
            st.write(f"**Original:** {st.session_state.selected_point[0]:.4f}, {st.session_state.selected_point[1]:.4f}")
            st.write(f"**East (20km):** {st.session_state.selected_point_east[0]:.4f}, {st.session_state.selected_point_east[1]:.4f}")
        
        # Add export data option
        st.subheader("Export Data")
        if st.button("Download Wells Data (Original)"):
            csv_data = download_as_csv(st.session_state.filtered_wells)
            st.download_button(
                label="Download CSV (Original)",
                data=csv_data,
                file_name="nearby_wells_original.csv",
                mime="text/csv"
            )
        
        # Export east wells data if available
        if 'filtered_wells_east' in st.session_state and st.session_state.filtered_wells_east is not None and len(st.session_state.filtered_wells_east) > 0:
            if st.button("Download Wells Data (East)"):
                csv_data_east = download_as_csv(st.session_state.filtered_wells_east)
                st.download_button(
                    label="Download CSV (East)",
                    data=csv_data_east,
                    file_name="nearby_wells_east.csv",
                    mime="text/csv"
                )
    elif st.session_state.get('selected_point'):
        st.info("Location selected. View the dual interpolated heatmaps on the left.")
    else:
        st.info("Click on the map to generate dual heatmaps: one at your location and one seamlessly connected to the east")

    # Heatmaps are automatically saved - no manual action needed

    # Add information about water well drilling - always display
    st.subheader("Finding Groundwater")
    st.write("""
    Traditional methods like water divining lack scientific basis. Our tool uses actual well data 
    to help you make informed decisions about where to drill based on:

    * Proximity to existing successful wells
    * Aquifer yield patterns in your area
    * Depth trends for accessing groundwater
    """)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p>© 2023 Groundwater Finder |Data sourced from public well databases</p>
    <p>This tool is designed to assist farmers in locating potential groundwater sources. Results are based on existing data and interpolation techniques.</p>
</div>
""", unsafe_allow_html=True)