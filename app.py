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
from data_loader import load_sample_data, load_custom_data, load_nz_govt_data, load_api_data
from interpolation import generate_heat_map_data, generate_geo_json_grid, calculate_kriging_variance, generate_indicator_kriging_mask, create_indicator_polygon_geometry, get_prediction_at_point, create_map_with_interpolated_data
from database import PolygonDatabase
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

# Initialize session state variables
if 'selected_point' not in st.session_state:
    st.session_state.selected_point = None
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 10
if 'wells_data' not in st.session_state:
    st.session_state.wells_data = None
if 'filtered_wells' not in st.session_state:
    st.session_state.filtered_wells = None
# No more yield filtering - removed

if 'heat_map_visibility' not in st.session_state:
    st.session_state.heat_map_visibility = True
if 'well_markers_visibility' not in st.session_state:
    st.session_state.well_markers_visibility = True
if 'search_radius' not in st.session_state:
    st.session_state.search_radius = 20

if 'soil_polygons' not in st.session_state:
    st.session_state.soil_polygons = None
if 'show_soil_polygons' not in st.session_state:
    st.session_state.show_soil_polygons = True
if 'polygon_db' not in st.session_state:
    try:
        st.session_state.polygon_db = PolygonDatabase()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        st.session_state.polygon_db = None

if 'stored_heatmaps' not in st.session_state:
    st.session_state.stored_heatmaps = []

# Load stored heatmaps from database on every app refresh
if st.session_state.polygon_db:
    try:
        st.session_state.stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
        print(f"Loaded {len(st.session_state.stored_heatmaps)} stored heatmaps from database")
    except Exception as e:
        print(f"Error loading stored heatmaps: {e}")
        st.session_state.stored_heatmaps = []
# Regional heatmap session state removed per user request

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
                    st.rerun()
            except:
                st.info("Fallback: Run the polygon processing script to merge and store soil polygons.")
                st.session_state.soil_polygons = None
    elif st.session_state.polygon_db is None:
        st.warning("Database connection not available. Cannot load soil polygons.")
        st.session_state.soil_polygons = None



    # Advanced option for uploading custom data (hidden in expander)
    with st.expander("Upload Custom Data (Optional)"):
        uploaded_file = st.file_uploader("Upload a CSV file with well data", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Loading custom data..."):
                st.session_state.wells_data = load_custom_data(uploaded_file)

    st.header("Filters")
    
    # Manual refresh button for when hot reload isn't working
    if st.button("🔄 Refresh App", help="Click if the app doesn't update automatically"):
        # Clear session state to force fresh start
        for key in list(st.session_state.keys()):
            if key.startswith('regional_heatmap'):
                del st.session_state[key]
        st.rerun()

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
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=st.session_state.well_markers_visibility)
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")
    
    # Stored Heatmaps Management Section
    st.markdown("---")
    st.subheader("🗺️ Stored Heatmaps")
    
    # Load stored heatmaps from database
    if st.session_state.polygon_db:
        try:
            stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
            st.session_state.stored_heatmaps = stored_heatmaps
            print(f"Loaded {len(stored_heatmaps)} stored heatmaps from database")
        except Exception as e:
            print(f"Error loading stored heatmaps: {e}")
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
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error refreshing: {e}")
        
        with col2:
            if st.button("🗑️ Clear All", type="secondary"):
                if st.session_state.polygon_db:
                    try:
                        count = st.session_state.polygon_db.delete_all_stored_heatmaps()
                        st.session_state.stored_heatmaps = []
                        st.success(f"Cleared {count} stored heatmaps")
                        st.rerun()
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
                    if st.session_state.polygon_db.delete_stored_heatmap(heatmap['id']):
                        st.session_state.stored_heatmaps = [h for h in st.session_state.stored_heatmaps if h['id'] != heatmap['id']]
                        st.success(f"Deleted heatmap: {heatmap['heatmap_name']}")
                        st.rerun()
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

    # Display all stored heatmaps on the map
    stored_heatmap_count = 0
    if st.session_state.stored_heatmaps:
        print(f"Attempting to display {len(st.session_state.stored_heatmaps)} stored heatmaps")
        for i, stored_heatmap in enumerate(st.session_state.stored_heatmaps):
            try:
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
                    
                    # Define dynamic color function for stored heatmaps (same as fresh ones)
                    def get_stored_color(value):
                        """Get color for stored heatmap based on method type"""
                        method = stored_heatmap.get('interpolation_method', 'kriging')
                        if method == 'indicator_kriging':
                            # Three-tier indicator kriging colors
                            if value < 0.4:
                                return '#FF0000'  # Red for poor zones
                            elif value < 0.7:
                                return '#FF8000'  # Orange for moderate zones  
                            else:
                                return '#00FF00'  # Green for good zones
                        else:
                            # Standard yield colors
                            if value <= 0.1:
                                return '#000080'
                            elif value <= 1:
                                return '#0033FF'
                            elif value <= 5:
                                return '#00CCFF'
                            elif value <= 10:
                                return '#00FF66'
                            elif value <= 20:
                                return '#FFFF00'
                            else:
                                return '#FF0000'

                    # Add GeoJSON layer for triangular mesh visualization (preferred)
                    folium.GeoJson(
                        geojson_data,
                        name=f"Stored: {stored_heatmap['heatmap_name']}",
                        style_function=lambda feature: {
                            'fillColor': get_stored_color(feature['properties'].get('yield', 0)),
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
        
        print(f"Successfully displayed {stored_heatmap_count} stored heatmaps on the map")
    else:
        print("No stored heatmaps to display")

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

        # If we have a selected point, show local context
        if st.session_state.selected_point:
            from utils import is_within_square
            
            # Filter wells using square bounds instead of circular distance
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

            # Keep yield values as-is - NaN values should remain NaN for proper filtering
            # Do not convert NaN to 0 as this creates false dry wells

            st.session_state.filtered_wells = filtered_wells.copy()

            # Create marker for selected point
            folium.Marker(
                location=st.session_state.selected_point,
                popup="Selected Location",
                icon=folium.Icon(color='red', icon='crosshairs', prefix='fa'),
                tooltip="Your Selected Point"
            ).add_to(m)

            # Draw square for search area (now just for local context)
            center_lat, center_lon = st.session_state.selected_point
            radius_km = st.session_state.search_radius
            
            # Calculate square bounds
            lat_radius_deg = radius_km / 111.0  # ~111km per degree latitude
            lon_radius_deg = radius_km / (111.0 * np.cos(np.radians(center_lat)))  # adjust for longitude
            
            # Create square coordinates
            square_bounds = [
                [center_lat - lat_radius_deg, center_lon - lon_radius_deg],  # SW corner
                [center_lat - lat_radius_deg, center_lon + lon_radius_deg],  # SE corner
                [center_lat + lat_radius_deg, center_lon + lon_radius_deg],  # NE corner
                [center_lat + lat_radius_deg, center_lon - lon_radius_deg],  # NW corner
                [center_lat - lat_radius_deg, center_lon - lon_radius_deg]   # Close square
            ]
            
            # Draw square
            folium.Polygon(
                locations=square_bounds,
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
                fill_opacity=0.1,
                weight=2
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
                if st.session_state.selected_point and st.session_state.filtered_wells is not None:
                    with st.spinner("🔄 Generating interpolation (consider running preprocessing for faster performance)..."):
                        pass


                        # Check if we need to generate indicator kriging mask for clipping
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

                        # Debug: Check if mask is being passed
                        print(f"App.py: About to call generate_geo_json_grid with method='{st.session_state.interpolation_method}'")
                        print(f"App.py: indicator_mask is {'provided' if indicator_mask is not None else 'None'}")
                        
                        # Generate regular interpolation visualization
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
                else:
                    geojson_data = {"type": "FeatureCollection", "features": []}

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

                    # Define colors based on what we're displaying
                    def get_color(value):
                        # Create 15-band color scale
                        step = max_value / 15.0

                        if st.session_state.interpolation_method == 'depth_kriging':
                            # Depth colors: green (shallow) to red (deep)
                            colors = [
                                '#00ff00',  # Green (shallow depth)
                                '#33ff00',
                                '#66ff00',
                                '#99ff00',
                                '#ccff00',
                                '#ffff00',  # Yellow
                                '#ffcc00',
                                '#ff9900',
                                '#ff6600',
                                '#ff3300',
                                '#ff0000',  # Red (deep depth)
                                '#cc0000',
                                '#990000',
                                '#660000',
                                '#330000'   # Dark red (very deep)
                            ]
                        elif st.session_state.interpolation_method == 'ground_water_level_kriging':
                            # Ground water level colors: blue (low level) to brown (high level)
                            colors = [
                                '#000080',  # Dark blue (low level)
                                '#0033CC',
                                '#0066FF',
                                '#0099FF',
                                '#00CCFF',
                                '#00FFFF',  # Cyan (medium-low)
                                '#66FFCC',
                                '#99FF99',
                                '#CCFF66',
                                '#FFFF33',  # Yellow (medium)
                                '#FFCC00',
                                '#FF9900',
                                '#FF6600',
                                '#CC3300',
                                '#993300'   # Brown (high level)
                            ]
                        elif st.session_state.interpolation_method == 'indicator_kriging':
                            # Three-tier indicator colors: red (poor), orange (moderate), green (good)
                            colors = ['#FF0000', '#FF8000', '#00FF00']  # Red, Orange, Green
                        else:
                            # Yield colors: blue (low yield) to red (high yield)
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

                        if st.session_state.interpolation_method == 'indicator_kriging':
                            # Three-tier classification: red (poor), orange (moderate), green (good)
                            if value <= 0.4:
                                return '#FF0000'    # Red for poor (0.25)
                            elif value <= 0.7:
                                return '#FF8000'    # Orange for moderate (0.625)
                            else:
                                return '#00FF00'    # Green for good (0.875)
                        else:
                            # Determine which band the value falls into
                            band_index = min(14, int(value / step))
                            return colors[band_index]

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
                    
                    folium.GeoJson(
                        data=geojson_data,
                        name=new_heatmap_name,
                        style_function=lambda feature: {
                            'fillColor': get_color(feature['properties'][display_field]),
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
                    ).add_to(m)

                    # AUTO-STORE: Automatically save every generated heatmap
                    if st.session_state.polygon_db and st.session_state.selected_point:
                        try:
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
                            
                            # Store in database
                            st.session_state.polygon_db.store_heatmap(
                                heatmap_name=heatmap_name,
                                center_lat=center_lat,
                                center_lon=center_lon,
                                radius_km=st.session_state.search_radius,
                                interpolation_method=st.session_state.interpolation_method,
                                heatmap_data=heatmap_data,
                                geojson_data=geojson_data,
                                well_count=len(st.session_state.filtered_wells) if st.session_state.filtered_wells is not None else 0
                            )
                            
                            # Immediately reload stored heatmaps to include the new one
                            st.session_state.stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
                            print(f"AUTO-STORED: {heatmap_name} with {len(heatmap_data)} points and {len(geojson_data.get('features', []))} features")
                        except Exception as e:
                            print(f"Error auto-storing heatmap: {e}")

                    # Add 15-band colormap legend to match the visualization
                    if st.session_state.interpolation_method == 'depth_kriging':
                        # Depth legend with depth-appropriate colors
                        colormap = folium.LinearColormap(
                            colors=['#00ff00', '#33ff00', '#66ff00', '#99ff00', '#ccff00', 
                                    '#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff3300', 
                                    '#ff0000', '#cc0000', '#990000', '#660000', '#330000'],
                            vmin=0,
                            vmax=float(max_value),
                            caption='Depth to Groundwater (m) - 15 Bands'
                        )
                    elif st.session_state.interpolation_method == 'ground_water_level_kriging':
                        # Ground water level legend
                        colormap = folium.LinearColormap(
                            colors=['#000080', '#0033CC', '#0066FF', '#0099FF', '#00CCFF', 
                                    '#00FFFF', '#66FFCC', '#99FF99', '#CCFF66', '#FFFF33', 
                                    '#FFCC00', '#FF9900', '#FF6600', '#CC3300', '#993300'],
                            vmin=0,
                            vmax=float(max_value),
                            caption='Ground Water Level (m) - 15 Bands'
                        )
                    elif st.session_state.interpolation_method == 'indicator_kriging':
                        # Three-tier indicator kriging legend
                        colormap = folium.StepColormap(
                            colors=['#FF0000', '#FF8000', '#00FF00'],  # Red, Orange, Green
                            vmin=0,
                            vmax=1.0,
                            index=[0, 0.4, 0.7, 1.0],  # Three-tier thresholds
                            caption='Well Yield Quality: Red = Poor (0-0.5 L/s), Orange = Moderate (0.5-0.75 L/s), Green = Good (≥0.75 L/s)'
                        )
                    else:
                        # Yield legend with original colors
                        colormap = folium.LinearColormap(
                            colors=['#000080', '#0000B3', '#0000E6', '#0033FF', '#0066FF', 
                                    '#0099FF', '#00CCFF', '#00FFCC', '#00FF99', '#00FF66', 
                                    '#33FF33', '#99FF00', '#FFFF00', '#FF9900', '#FF0000'],
                            vmin=0,
                            vmax=float(max_value),
                            caption='Estimated Water Yield (L/s) - 15 Bands'
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

    # Add layer control to toggle between different heatmaps
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    # Use st_folium with stable key and minimal returned objects
    map_data = st_folium(
        m,
        use_container_width=True,
        height=600,
        key="main_map",
        returned_objects=["last_clicked"]
    )

    # Process clicks from the map with better stability
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        # Get the coordinates from the click
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]

        # Only update if this is a genuinely new location (larger threshold for stability)
        current_point = st.session_state.selected_point
        if not current_point or (abs(current_point[0] - clicked_lat) > 0.001 or abs(current_point[1] - clicked_lng) > 0.001):
            print(f"MAP CLICK: New location detected - updating coordinates to ({clicked_lat:.3f}, {clicked_lng:.3f})")
            # Update session state with the new coordinates
            st.session_state.selected_point = [clicked_lat, clicked_lng]
            # Clear filtered wells to trigger recalculation
            st.session_state.filtered_wells = None
            # Clear any cached interpolation data that might use old coordinates
            if 'geojson_data' in st.session_state:
                del st.session_state['geojson_data']
            # Use experimental_rerun to reduce instability
            st.rerun()

    # Add cache clearing and reset buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Results", use_container_width=True):
            # Reset the session state safely
            for key in ['selected_point', 'filtered_wells']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh App", use_container_width=True):
            # Clear all caches and reset session state
            st.cache_data.clear()
            st.cache_resource.clear()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

with main_col2:
    st.subheader("Analysis Results")

    if st.session_state.filtered_wells is not None and len(st.session_state.filtered_wells) > 0:
        # Add export data option
        st.subheader("Export Data")
        if st.button("Download Wells Data"):
            csv_data = download_as_csv(st.session_state.filtered_wells)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="nearby_wells.csv",
                mime="text/csv"
            )
    elif st.session_state.selected_point:
        st.info("Location selected. View the interpolated heatmap on the left.")
    else:
        st.info("Click on the map to select a location and view nearby wells")

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
    <p>© 2023 Groundwater Finder | Data sourced from public well databases</p>
    <p>This tool is designed to assist farmers in locating potential groundwater sources. Results are based on existing data and interpolation techniques.</p>
</div>
""", unsafe_allow_html=True)