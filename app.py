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
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=False)
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")
    
    # Stored Heatmaps Management Section
    st.markdown("---")
    st.subheader("🗺️ Stored Heatmaps")
    
    # Load stored heatmaps from database (only if not already loaded or cleared)
    if not hasattr(st.session_state, 'stored_heatmaps'):
        if st.session_state.polygon_db:
            try:
                stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
                st.session_state.stored_heatmaps = stored_heatmaps
                print(f"Loaded {len(stored_heatmaps)} stored heatmaps from database")
            except Exception as e:
                print(f"Error loading stored heatmaps: {e}")
                st.session_state.stored_heatmaps = []
        else:
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
                        # Reset all flags and cache
                        st.session_state.new_heatmap_added = False
                        st.session_state.colormap_updated = False
                        st.session_state.fresh_heatmap_displayed = False
                        # Clear all session data to force complete refresh
                        for key in ['selected_point', 'filtered_wells', 'geojson_data', 'heat_map_data', 'indicator_mask']:
                            if key in st.session_state:
                                del st.session_state[key]
                        print(f"CLEARED ALL: Deleted {count} stored heatmaps from database and cleared all session data")
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

                print(f"DEBUG: geojson_data exists: {bool(geojson_data)}")
                if geojson_data:
                    print(f"DEBUG: geojson_data features count: {len(geojson_data.get('features', []))}")
                
                # OLD HEATMAP DISPLAY CODE REMOVED - using simplified approach instead
                pass

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

    # Use st_folium with stable key and minimal returned objects
    map_data = st_folium(
        m,
        use_container_width=True,
        height=600,
        key="main_map",
        returned_objects=["last_clicked"]
    )

    # Simple click processing - every click creates a heatmap
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        # Get the coordinates from the click
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        
        print(f"MAP CLICK DETECTED: lat={clicked_lat:.6f}, lng={clicked_lng:.6f}")
        
        # Set clicked location
        st.session_state.selected_point = [clicked_lat, clicked_lng]
        
        # Generate heatmap immediately and store it
        search_radius = getattr(st.session_state, 'search_radius', 20)
        interpolation_method = getattr(st.session_state, 'interpolation_method', 'kriging')
        
        wells_df = load_nz_govt_data(search_center=(clicked_lat, clicked_lng), search_radius_km=search_radius)
        
        if not wells_df.empty:
            geojson_data = generate_geo_json_grid(
                wells_df, 
                (clicked_lat, clicked_lng), 
                search_radius,
                resolution=50,  # Fixed resolution
                method=interpolation_method
            )
            
            if geojson_data:
                # Store immediately in database
                heatmap_name = f"{interpolation_method}_{clicked_lat:.3f}_{clicked_lng:.3f}"
                
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
                
                st.session_state.polygon_db.store_heatmap(
                    heatmap_name=heatmap_name,
                    center_lat=clicked_lat,
                    center_lon=clicked_lng,
                    radius_km=search_radius,
                    interpolation_method=interpolation_method,
                    heatmap_data=heatmap_data,
                    geojson_data=geojson_data,
                    well_count=len(wells_df)
                )
                print(f"STORED NEW HEATMAP: {heatmap_name}")
        
        # Trigger rerun to refresh map with all stored heatmaps
        st.rerun()
    
    # Always display ALL stored heatmaps from database (simple approach)
    all_stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
    if all_stored_heatmaps:
        print(f"DISPLAYING {len(all_stored_heatmaps)} stored heatmaps")
        for i, heatmap in enumerate(all_stored_heatmaps):
            geojson_data = heatmap.get('geojson_data')
            if geojson_data and geojson_data.get('features'):
                print(f"Adding heatmap {i+1}: {heatmap['heatmap_name']}")
                
                # Fix the lambda function scope issue by using a closure
                def create_style_function(method):
                    def style_function(feature):
                        value = feature['properties'].get('yield', 0)
                        if method == 'indicator_kriging':
                            # Three-tier indicator kriging colors
                            if value < 0.4:
                                color = '#FF0000'  # Red
                            elif value < 0.7:
                                color = '#FF8000'  # Orange  
                            else:
                                color = '#00FF00'  # Green
                        else:
                            # Standard continuous color mapping for other methods
                            color = get_global_unified_color(value, method)
                        
                        return {
                            'fillColor': color,
                            'color': 'none',
                            'weight': 0,
                            'fillOpacity': 0.6  # Slightly lower opacity for better layering
                        }
                    return style_function
                
                folium.GeoJson(
                    geojson_data,
                    name=f"Heatmap {i+1}: {heatmap['heatmap_name']}",
                    style_function=create_style_function(heatmap.get('interpolation_method', 'indicator_kriging')),
                    tooltip=folium.GeoJsonTooltip(
                        fields=['yield'],
                        aliases=['Probability:'],
                        localize=True
                    )
                ).add_to(m)
    
    # Add layer control to let users toggle individual heatmaps
    folium.LayerControl().add_to(m)
    
    # Add appropriate colormap legend based on interpolation method
    if all_stored_heatmaps:
        # Check if any stored heatmaps use indicator kriging
        has_indicator = any(hm.get('interpolation_method') == 'indicator_kriging' for hm in all_stored_heatmaps)
        
        if has_indicator:
            # Three-tier indicator kriging legend
            colormap = folium.StepColormap(
                colors=['#FF0000', '#FF8000', '#00FF00'],  # Red, Orange, Green
                vmin=0,
                vmax=1.0,
                index=[0, 0.4, 0.7, 1.0],  # Three-tier thresholds
                caption='Well Yield Quality: Red = Poor (0-0.4), Orange = Moderate (0.4-0.7), Green = Good (0.7-1.0)'
            )
        else:
            # Standard continuous colormap for other methods
            colormap = folium.LinearColormap(
                colors=['#000080', '#0000B3', '#0000E6', '#0033FF', '#0066FF', 
                        '#0099FF', '#00CCFF', '#00FFCC', '#00FF99', '#00FF66', 
                        '#33FF33', '#99FF00', '#FFFF00', '#FF9900', '#FF0000'],
                vmin=0,
                vmax=50,  # Reasonable max for yield
                caption='Well Yield (L/s): Continuous Scale'
            )
        colormap.add_to(m)

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