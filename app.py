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
from interpolation import generate_heat_map_data, generate_geo_json_grid, calculate_kriging_variance, generate_indicator_kriging_mask, create_indicator_polygon_geometry, get_prediction_at_point, create_map_with_interpolated_data, generate_smooth_raster_overlay
from database import PolygonDatabase
from polygon_display import parse_coordinates_file, add_polygon_to_map
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
    'colormap_updated': False,
    'show_banks_peninsula': True,
    'banks_peninsula_coords': None,
    'heatmap_visualization_mode': 'triangular_mesh'  # 'triangular_mesh' or 'smooth_raster'
}

# Initialize all session state variables
for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# Initialize database connection only once with caching
@st.cache_resource
def get_database_connection():
    """Cached database connection to prevent repeated initializations"""
    try:
        return PolygonDatabase()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Load Banks Peninsula coordinates once and cache them
@st.cache_data
def load_banks_peninsula_coords():
    """Load and cache Banks Peninsula polygon coordinates"""
    try:
        file_path = "attached_assets/banks peninsula_1753603323297.txt"
        coordinates = parse_coordinates_file(file_path)
        if coordinates:
            print(f"Loaded {len(coordinates)} Banks Peninsula coordinates")
            return coordinates
        else:
            print("No valid coordinates found in Banks Peninsula file")
            return None
    except Exception as e:
        print(f"Error loading Banks Peninsula coordinates: {e}")
        return None

if 'polygon_db' not in st.session_state:
    st.session_state.polygon_db = get_database_connection()

# Load Banks Peninsula coordinates if not already loaded
if st.session_state.banks_peninsula_coords is None:
    st.session_state.banks_peninsula_coords = load_banks_peninsula_coords()

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
    
    # Heatmap visualization style selection
    heatmap_style = st.selectbox(
        "Heatmap Display Style",
        options=["Triangle Mesh (Scientific)", "Smooth Raster (Windy.com Style)"],
        index=0,  # Default to Triangle Mesh
        help="Choose how interpolated data is visualized: Triangle Mesh shows precise triangular interpolation, Smooth Raster provides a weather-map style visualization",
        key="heatmap_style_selector"
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

    # Grid size selection for heatmap generation
    st.subheader("Heatmap Grid Options")
    grid_option = st.selectbox(
        "Heatmap Grid Size",
        options=["2×3 Grid (6 heatmaps)", "10×10 Grid (100 heatmaps)"],
        index=0,  # Default to 2x3
        help="Choose the grid size for automatic heatmap generation. 10×10 creates comprehensive regional coverage but takes longer to generate."
    )
    
    # Convert selection to grid_size tuple and store in session state
    if "10×10" in grid_option:
        st.session_state.grid_size = (10, 10)
        st.info("📊 **Extended Mode**: Will generate 100 heatmaps covering 178km south × 178km east area")
    else:
        st.session_state.grid_size = (2, 3)
        st.info("📊 **Standard Mode**: Will generate 6 heatmaps in compact 2×3 layout")

    # Display options
    st.header("Display Options")
    
    # Heatmap visualization mode toggle
    st.subheader("Heatmap Visualization Style")
    heatmap_mode = st.radio(
        "Choose heatmap rendering style:",
        options=["Triangular Mesh (Current)", "Smooth Raster (Windy.com style)"],
        index=0 if st.session_state.heatmap_visualization_mode == 'triangular_mesh' else 1,
        help="Triangular Mesh: Current triangular interpolation with sharp boundaries. Smooth Raster: Windy.com-style smooth gradient visualization without triangle artifacts."
    )
    
    # Update session state based on selection
    if "Smooth Raster" in heatmap_mode:
        st.session_state.heatmap_visualization_mode = 'smooth_raster'
        st.info("🌊 **Smooth Raster Mode**: Heatmaps will display with smooth gradients like Windy.com weather maps")
    else:
        st.session_state.heatmap_visualization_mode = 'triangular_mesh'
        st.info("🔺 **Triangular Mesh Mode**: Heatmaps display current triangular interpolation boundaries")
    
    # Colormap Selection
    st.subheader("Colormap Selection")
    colormap_option = st.selectbox(
        "Choose colormap style:",
        options=[
            "turbo (Blue→Green→Yellow→Red)",
            "viridis (Purple→Blue→Green→Yellow)", 
            "plasma (Purple→Pink→Yellow)",
            "inferno (Black→Red→Yellow)",
            "magma (Black→Purple→White)",
            "rainbow (Full Spectrum)",
            "spectral (Blue→Green→Yellow→Red)",
            "winter (Blue→Green)",
            "icefire (Blue→White→Orange)",
            "flare (Orange→Red→Pink)",
            "rocket (Black→Red→Orange)",
            "mako (Black→Blue→Green)",
            "viag (Purple→Green)",
            "crest (Blue→Teal→Yellow)"
        ],
        index=0,
        help="Choose from professional scientific colormaps. Each offers different visual emphasis for your data."
    )
    
    # Extract colormap name 
    st.session_state.selected_colormap = colormap_option.split(" (")[0]
    
    # Color Distribution Method
    st.subheader("Color Distribution Method")
    color_method = st.selectbox(
        "How should colors be distributed?",
        options=[
            "Linear Distribution (Current)", 
            "Equal Count Bins (Quantile)", 
            "Data-Density Optimized"
        ],
        index=0,
        help="Linear: Even spacing across value range. Equal Count: Each color represents same number of data points. Data-Density: Emphasizes areas with most data variation."
    )
    st.session_state.color_distribution_method = color_method.split(" (")[0].lower().replace(" ", "_").replace("-", "_")
    
    # Transparency Control
    st.subheader("Transparency Settings")
    opacity = st.slider(
        "Heatmap Transparency",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust how transparent the heatmaps appear over the base map. Lower values = more transparent, higher values = more opaque."
    )
    st.session_state.heatmap_opacity = opacity
    
    st.session_state.heat_map_visibility = st.checkbox("Show Heat Map", value=st.session_state.heat_map_visibility)
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=False)
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")
    
    # Banks Peninsula polygon display option
    st.session_state.show_banks_peninsula = st.checkbox("Show Banks Peninsula Boundary", value=st.session_state.show_banks_peninsula, help="Display the Banks Peninsula coastline boundary")

    # Stored Heatmaps Management Section
    st.markdown("---")
    st.subheader("🗺️ Stored Heatmaps")

    # Load stored heatmaps only if not already loaded
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

    # Add Banks Peninsula polygon if enabled and coordinates are available
    if st.session_state.show_banks_peninsula and st.session_state.banks_peninsula_coords:
        try:
            # Add the polygon to the map with distinctive styling
            m, peninsula_center = add_polygon_to_map(
                m, 
                st.session_state.banks_peninsula_coords,
                name="Banks Peninsula",
                color="#FF6B35",  # Distinctive orange-red color
                weight=3,
                opacity=0.8,
                fill_opacity=0.15
            )
            print(f"Banks Peninsula polygon added to map (center: {peninsula_center})")
        except Exception as e:
            print(f"Error adding Banks Peninsula to map: {e}")

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
                'fillOpacity': 0.1  # Keep low opacity for polygon outlines
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['DRAINAGE'] if 'DRAINAGE' in st.session_state.soil_polygons.columns else [],
                aliases=['Drainage:'] if 'DRAINAGE' in st.session_state.soil_polygons.columns else [],
                labels=True,
                sticky=False
            )
        ).add_to(m)

    # UNIFIED COLORMAP PROCESSING: Use stored colormap metadata for consistent coloring
    global_min_value = float('inf')
    global_max_value = float('-inf')
    all_heatmap_values = []
    colormap_source = "calculated"

    # First, check if we have stored colormap metadata from generated heatmaps
    stored_colormap_metadata = None
    if st.session_state.stored_heatmaps:
        print(f"🎨 COLORMAP CONSISTENCY CHECK: Analyzing {len(st.session_state.stored_heatmaps)} stored heatmaps")
        
        # Look for colormap metadata in stored heatmaps (all should have the same range)
        for stored_heatmap in st.session_state.stored_heatmaps:
            colormap_metadata = stored_heatmap.get('colormap_metadata')
            if colormap_metadata and isinstance(colormap_metadata, dict):
                if 'global_min' in colormap_metadata and 'global_max' in colormap_metadata:
                    stored_colormap_metadata = colormap_metadata
                    global_min_value = colormap_metadata['global_min']
                    global_max_value = colormap_metadata['global_max']
                    colormap_source = "stored_metadata"
                    print(f"🎨 USING STORED COLORMAP RANGE: {global_min_value:.2f} to {global_max_value:.2f} (from metadata)")
                    break
        
        # If no stored metadata found, this means heatmaps were generated with old system
        if stored_colormap_metadata is None:
            print(f"🎨 NO STORED METADATA FOUND: Old heatmaps detected - need regeneration for consistency")
            # Don't recalculate here - this creates inconsistency
            # Instead, use fallback and suggest regeneration
            colormap_source = "needs_regeneration"

    # Fallback to reasonable defaults if no data or inconsistent data
    if global_min_value == float('inf') or colormap_source == "needs_regeneration":
        if st.session_state.interpolation_method == 'indicator_kriging':
            global_min_value, global_max_value = 0.0, 1.0
        elif st.session_state.interpolation_method == 'ground_water_level_kriging':
            global_min_value, global_max_value = 0.0, 100.0  # Ground water level is typically 0-100+ meters
        elif 'depth' in st.session_state.interpolation_method:
            global_min_value, global_max_value = 0.0, 50.0   # Depth to groundwater is typically 0-50 meters  
        else:
            global_min_value, global_max_value = 0.0, 25.0   # Yield data is typically 0-25 L/s
        if colormap_source == "needs_regeneration":
            print(f"🎨 OLD HEATMAPS DETECTED: Using fallback range {global_min_value:.2f} to {global_max_value:.2f} - regenerate for consistency")
        else:
            print(f"🎨 USING FALLBACK COLORMAP RANGE: {global_min_value:.2f} to {global_max_value:.2f}")
        colormap_source = "fallback_defaults"

    # DYNAMIC RANGE ADJUSTMENT: Calculate actual data range from currently displayed heatmaps
    # FORCE this calculation for ALL stored heatmaps to fix color distribution
    if st.session_state.stored_heatmaps:
        print("🎨 CALCULATING DYNAMIC RANGE from current stored heatmaps...")
        actual_min = float('inf')
        actual_max = float('-inf')
        total_values = 0
        
        for stored_heatmap in st.session_state.stored_heatmaps:
            geojson_data = stored_heatmap.get('geojson_data')
            if geojson_data and geojson_data.get('features'):
                for feature in geojson_data['features']:
                    value = feature['properties'].get('yield', feature['properties'].get('value', 0))
                    if value > 0:  # Only count positive values
                        actual_min = min(actual_min, value)
                        actual_max = max(actual_max, value)
                        total_values += 1
        
        if actual_min != float('inf') and total_values > 0:
            # Calculate percentile-based range for better color distribution
            all_values = []
            for stored_heatmap in st.session_state.stored_heatmaps:
                geojson_data = stored_heatmap.get('geojson_data')
                if geojson_data and geojson_data.get('features'):
                    for feature in geojson_data['features']:
                        value = feature['properties'].get('yield', feature['properties'].get('value', 0))
                        if value > 0:
                            all_values.append(value)
            
            if all_values:
                all_values.sort()
                # Use 5th to 95th percentile range to avoid extreme outliers
                p5_index = int(len(all_values) * 0.05)
                p95_index = int(len(all_values) * 0.95)
                global_min_value = all_values[p5_index] if p5_index < len(all_values) else actual_min
                global_max_value = all_values[p95_index] if p95_index < len(all_values) else actual_max
                
                # Ensure we don't lose too much data at the extremes
                if global_max_value - global_min_value < (actual_max - actual_min) * 0.5:
                    # If the 5-95% range is too narrow, use 10-90% range
                    p10_index = int(len(all_values) * 0.10)
                    p90_index = int(len(all_values) * 0.90)
                    global_min_value = all_values[p10_index] if p10_index < len(all_values) else actual_min
                    global_max_value = all_values[p90_index] if p90_index < len(all_values) else actual_max
                
                colormap_source = "percentile_based_calculation"
                
                # DETAILED DATA DISTRIBUTION ANALYSIS
                all_values_sorted = sorted(all_values)
                p10 = all_values_sorted[int(len(all_values_sorted) * 0.10)]
                p25 = all_values_sorted[int(len(all_values_sorted) * 0.25)]
                p50 = all_values_sorted[int(len(all_values_sorted) * 0.50)]
                p75 = all_values_sorted[int(len(all_values_sorted) * 0.75)]
                p90 = all_values_sorted[int(len(all_values_sorted) * 0.90)]
                
                # Count values in different color ranges for transparency
                blue_range = global_min_value + (global_max_value - global_min_value) * 0.3  # 0-30%
                green_range = global_min_value + (global_max_value - global_min_value) * 0.7  # 30-70%
                orange_range = global_min_value + (global_max_value - global_min_value) * 0.9  # 70-90%
                # 90-100% is red range
                
                blue_count = sum(1 for v in all_values if v <= blue_range)
                green_count = sum(1 for v in all_values if blue_range < v <= green_range)
                yellow_count = sum(1 for v in all_values if green_range < v <= orange_range)
                red_count = sum(1 for v in all_values if v > orange_range)
                
                print(f"🎨 PERCENTILE-BASED RANGE: {global_min_value:.2f} to {global_max_value:.2f} (5-95% of {total_values} values, excludes outliers)")
                print(f"🎨 ORIGINAL FULL RANGE WAS: {actual_min:.2f} to {actual_max:.2f} - now using percentile range for better color distribution")
                print(f"🎨 DATA DISTRIBUTION: p10={p10:.1f}, p25={p25:.1f}, p50={p50:.1f}, p75={p75:.1f}, p90={p90:.1f}")
                print(f"🎨 COLOR DISTRIBUTION: Blue(0-30%)={blue_count} ({blue_count/total_values*100:.1f}%), Green(30-70%)={green_count} ({green_count/total_values*100:.1f}%), Yellow(70-90%)={yellow_count} ({yellow_count/total_values*100:.1f}%), Red(90%+)={red_count} ({red_count/total_values*100:.1f}%)")
                
                # Calculate quantile breakpoints for equal count bins
                global quantile_breakpoints
                num_bins = 40  # Match the number of colors
                quantile_breakpoints = []
                for i in range(num_bins + 1):
                    percentile = i / num_bins
                    idx = int(percentile * (len(all_values_sorted) - 1))
                    quantile_breakpoints.append(all_values_sorted[idx])
                print(f"🎨 QUANTILE BREAKPOINTS CALCULATED: {num_bins} equal-count bins ready")
            else:
                global_min_value = actual_min
                global_max_value = actual_max
                colormap_source = "dynamic_calculation"
                print(f"🎨 DYNAMIC RANGE CALCULATED: {global_min_value:.2f} to {global_max_value:.2f} (from {total_values} data points)")
    
    print(f"🎨 FINAL COLORMAP RANGE: {global_min_value:.2f} to {global_max_value:.2f} (source: {colormap_source})")

    # PERCENTILE-BASED COLOR ENHANCEMENT: Now handled during generation for consistency
    # Extract percentile information from stored metadata if available
    percentile_info = ""
    if stored_colormap_metadata and 'percentiles' in stored_colormap_metadata:
        percentiles = stored_colormap_metadata['percentiles']
        if percentiles:
            p25 = percentiles.get('25th', 'N/A')
            p50 = percentiles.get('50th', 'N/A') 
            p75 = percentiles.get('75th', 'N/A')
            total_values = stored_colormap_metadata.get('total_values', 0)
            percentile_info = f" | Percentiles: 25th={p25:.2f}, 50th={p50:.2f}, 75th={p75:.2f} (from {total_values} values)"
            print(f"🎨 PERCENTILE DATA AVAILABLE: {percentile_info}")
    
    print(f"🎨 COLORMAP READY: Range {global_min_value:.2f} to {global_max_value:.2f}{percentile_info}")

    # DEFINE GLOBAL UNIFIED COLOR FUNCTION 
    def get_global_unified_color(value, method='kriging'):
        """Global unified color function using stored global range for consistency"""
        if method == 'indicator_kriging':
            # Three-tier classification: red (poor), orange (moderate), green (good)
            if value <= 0.4:
                return '#FF0000'    # Red for poor
            elif value <= 0.7:
                return '#FF8000'    # Orange for moderate
            else:
                return '#00FF00'    # Green for good
        else:
            # Apply selected color distribution method
            color_dist_method = getattr(st.session_state, 'color_distribution_method', 'linear_distribution')
            
            if color_dist_method == 'equal_count_bins' and 'quantile_breakpoints' in globals():
                # Use quantile-based color mapping for equal representation
                for i, breakpoint in enumerate(quantile_breakpoints):
                    if value <= breakpoint:
                        normalized_value = i / (len(quantile_breakpoints) - 1)
                        break
                else:
                    normalized_value = 1.0
            elif color_dist_method == 'data_density_optimized':
                # Use percentile-based mapping to spread colors across data distribution
                if global_max_value > global_min_value:
                    import math
                    # Apply square root transformation to spread out lower values
                    min_val = max(0.0, global_min_value)  # Use 0.0 as minimum
                    max_val = global_max_value
                    val = max(min_val, min(max_val, value))  # Clamp value to range
                    
                    # Safe square root normalization - avoid negative values
                    try:
                        sqrt_input = max(0.0, val - min_val + 1.0)  # Ensure positive input
                        sqrt_val = math.sqrt(sqrt_input)
                        sqrt_range = math.sqrt(max_val - min_val + 1.0)
                        normalized_value = sqrt_val / sqrt_range if sqrt_range > 0 else 0.5
                    except (ValueError, ZeroDivisionError):
                        # Fallback to linear normalization if sqrt fails
                        normalized_value = (val - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else 0.5
                else:
                    normalized_value = 0.5
            else:
                # Default linear distribution but with enhanced range utilization
                if global_max_value > global_min_value:
                    import math
                    # Clamp value to valid range
                    val = max(global_min_value, min(global_max_value, value))
                    normalized_value = (val - global_min_value) / (global_max_value - global_min_value)
                    # Apply curve to better utilize full colormap range - safe power function
                    try:
                        normalized_value = math.pow(max(0.0, min(1.0, normalized_value)), 0.6)
                    except (ValueError, OverflowError):
                        # Fallback to simple linear if power fails
                        pass
                else:
                    normalized_value = 0.5
            
            normalized_value = max(0.0, min(1.0, normalized_value))
            
            # Get selected colormap from session state
            selected_colormap = getattr(st.session_state, 'selected_colormap', 'turbo')
            
            # Professional scientific colormaps (40 colors each for smooth gradients)
            colormap_palettes = {
                'turbo': [
                    '#30123b', '#311542', '#321649', '#341750', '#351857', '#36195e', '#371a65', '#381b6c',
                    '#3a1c73', '#3b1d7a', '#3c1e81', '#3d1f88', '#3e208f', '#402196', '#41229d', '#4223a4',
                    '#4324ab', '#4425b2', '#4626b9', '#4727c0', '#4828c7', '#4929ce', '#4a2ad5', '#4c2bdc',
                    '#4d2ce3', '#4e2dea', '#4f2ef1', '#502ff8', '#5230ff', '#5331ff', '#5432ff', '#5533ff',
                    '#5634ff', '#5735ff', '#5836ff', '#5937ff', '#5a38ff', '#5b39ff', '#5c3aff', '#5d3bff'
                ],
                'viridis': [
                    '#440154', '#460a5d', '#481467', '#481c6e', '#482576', '#472d7b', '#453581', '#433d84',
                    '#404588', '#3d4d8a', '#3a538b', '#375b8d', '#34618d', '#31688e', '#2e6e8e', '#2b748e',
                    '#297b8e', '#26818e', '#24878e', '#228d8d', '#20938c', '#1f998a', '#1fa088', '#21a585',
                    '#25ac82', '#2db27d', '#35b779', '#40bd72', '#4cc26c', '#5ac864', '#67cc5c', '#77d153',
                    '#89d548', '#98d83e', '#aadc32', '#bade28', '#cde11d', '#dde318', '#efe51c', '#fde725'
                ],
                'plasma': [
                    '#0d0887', '#100a8a', '#130c8c', '#160e8f', '#191091', '#1c1294', '#1f1496', '#221699',
                    '#25189b', '#281a9e', '#2b1ca0', '#2e1ea3', '#3120a5', '#3422a8', '#3724aa', '#3a26ad',
                    '#3d28af', '#402ab2', '#432cb4', '#462eb7', '#4930b9', '#4c32bc', '#4f34be', '#5236c1',
                    '#5538c3', '#583ac6', '#5b3cc8', '#5e3ecb', '#6140cd', '#6442d0', '#6744d2', '#6a46d5',
                    '#6d48d7', '#704ada', '#734cdc', '#764edf', '#7950e1', '#7c52e4', '#7f54e6', '#8256e9'
                ],
                'inferno': [
                    '#000004', '#020109', '#04020e', '#060312', '#080417', '#0a051c', '#0c0621', '#0e0726',
                    '#10082b', '#120930', '#140a35', '#160b3a', '#180c3f', '#1a0d44', '#1c0e49', '#1e0f4e',
                    '#201053', '#221158', '#24125d', '#261362', '#281467', '#2a156c', '#2c1671', '#2e1776',
                    '#30187b', '#321980', '#341a85', '#361b8a', '#381c8f', '#3a1d94', '#3c1e99', '#3e1f9e',
                    '#4020a3', '#4221a8', '#4422ad', '#4623b2', '#4824b7', '#4a25bc', '#4c26c1', '#4e27c6'
                ],
                'magma': [
                    '#000004', '#020109', '#04020e', '#060312', '#080417', '#0a051c', '#0c0621', '#0e0726',
                    '#10082b', '#120930', '#140a35', '#160b3a', '#180c3f', '#1a0d44', '#1c0e49', '#1e0f4e',
                    '#201053', '#221158', '#24125d', '#261362', '#281467', '#2a156c', '#2c1671', '#2e1776',
                    '#30187b', '#321980', '#341a85', '#361b8a', '#381c8f', '#3a1d94', '#3c1e99', '#3e1f9e',
                    '#4020a3', '#4221a8', '#4422ad', '#4623b2', '#4824b7', '#4a25bc', '#4c26c1', '#4e27c6'
                ],
                'rainbow': [
                    '#ff0000', '#ff1100', '#ff2200', '#ff3300', '#ff4400', '#ff5500', '#ff6600', '#ff7700',
                    '#ff8800', '#ff9900', '#ffaa00', '#ffbb00', '#ffcc00', '#ffdd00', '#ffee00', '#ffff00',
                    '#eeff00', '#ddff00', '#ccff00', '#bbff00', '#aaff00', '#99ff00', '#88ff00', '#77ff00',
                    '#66ff00', '#55ff00', '#44ff00', '#33ff00', '#22ff00', '#11ff00', '#00ff00', '#00ff11',
                    '#00ff22', '#00ff33', '#00ff44', '#00ff55', '#00ff66', '#00ff77', '#00ff88', '#00ff99'
                ],
                'spectral': [
                    '#9e0142', '#a61f4d', '#ae3c58', '#b65963', '#be766e', '#c69379', '#ceb084', '#d6cd8f',
                    '#deea9a', '#e6ffa5', '#eeffaa', '#e8ff9f', '#e2ff94', '#dcff89', '#d6ff7e', '#d0ff73',
                    '#caff68', '#c4ff5d', '#beff52', '#b8ff47', '#b2ff3c', '#acff31', '#a6ff26', '#a0ff1b',
                    '#9aff10', '#94ff05', '#8eff00', '#88ff06', '#82ff0c', '#7cff12', '#76ff18', '#70ff1e',
                    '#6aff24', '#64ff2a', '#5eff30', '#58ff36', '#52ff3c', '#4cff42', '#46ff48', '#40ff4e'
                ],
                'winter': [
                    '#0000ff', '#0011ff', '#0022ff', '#0033ff', '#0044ff', '#0055ff', '#0066ff', '#0077ff',
                    '#0088ff', '#0099ff', '#00aaff', '#00bbff', '#00ccff', '#00ddff', '#00eeff', '#00ffff',
                    '#00ffee', '#00ffdd', '#00ffcc', '#00ffbb', '#00ffaa', '#00ff99', '#00ff88', '#00ff77',
                    '#00ff66', '#00ff55', '#00ff44', '#00ff33', '#00ff22', '#00ff11', '#00ff00', '#11ff00',
                    '#22ff00', '#33ff00', '#44ff00', '#55ff00', '#66ff00', '#77ff00', '#88ff00', '#99ff00'
                ],
                'icefire': [
                    '#040613', '#0a0b2c', '#101045', '#16155e', '#1c1a77', '#221f90', '#2824a9', '#2e29c2',
                    '#342edb', '#3a33f4', '#4038ff', '#463dff', '#4c42ff', '#5247ff', '#584cff', '#5e51ff',
                    '#6456ff', '#6a5bff', '#7060ff', '#7665ff', '#7c6aff', '#826fff', '#8874ff', '#8e79ff',
                    '#947eff', '#9a83ff', '#a088ff', '#a68dff', '#ac92ff', '#b297ff', '#b89cff', '#bea1ff',
                    '#c4a6ff', '#caabff', '#d0b0ff', '#d6b5ff', '#dcbaff', '#e2bfff', '#e8c4ff', '#eec9ff'
                ],
                'flare': [
                    '#ffa600', '#ff9e00', '#ff9600', '#ff8e00', '#ff8600', '#ff7e00', '#ff7600', '#ff6e00',
                    '#ff6600', '#ff5e00', '#ff5600', '#ff4e00', '#ff4600', '#ff3e00', '#ff3600', '#ff2e00',
                    '#ff2600', '#ff1e00', '#ff1600', '#ff0e00', '#ff0600', '#fe0008', '#f60010', '#ee0018',
                    '#e60020', '#de0028', '#d60030', '#ce0038', '#c60040', '#be0048', '#b60050', '#ae0058',
                    '#a60060', '#9e0068', '#960070', '#8e0078', '#860080', '#7e0088', '#760090', '#6e0098'
                ],
                'rocket': [
                    '#03051a', '#0a0722', '#110929', '#180b31', '#1f0d38', '#260f40', '#2d1147', '#34134f',
                    '#3b1556', '#42175e', '#491965', '#501b6d', '#571d74', '#5e1f7c', '#652183', '#6c238b',
                    '#732592', '#7a279a', '#8129a1', '#882ba9', '#8f2db0', '#962fb8', '#9d31bf', '#a433c7',
                    '#ab35ce', '#b237d6', '#b939dd', '#c03be5', '#c73dec', '#ce3ff4', '#d541fb', '#dc43ff',
                    '#e345ff', '#ea47ff', '#f149ff', '#f84bff', '#ff4dff', '#ff4ff6', '#ff51ed', '#ff53e4'
                ],
                'mako': [
                    '#12080d', '#180d16', '#1e111f', '#241628', '#2a1b33', '#2f1f3d', '#342447', '#372851', 
                    '#3b2e5d', '#3e3367', '#403872', '#413e7d', '#414488', '#3f4b90', '#3d5296', '#3a599a', 
                    '#38629d', '#36699f', '#3670a0', '#3576a2', '#357ea4', '#3485a5', '#348ca7', '#3492a8', 
                    '#349aaa', '#35a1ab', '#37a8ac', '#3aaead', '#3fb6ad', '#45bdad', '#4cc3ad', '#55caad', 
                    '#65d0ad', '#76d5ae', '#88d9b1', '#99ddb6', '#abe2be', '#b9e6c7', '#c6ebd1', '#d2f0db'
                ],
                'viag': [
                    '#440154', '#441a55', '#442156', '#442757', '#432d58', '#433359', '#42395a', '#423f5b',
                    '#41455c', '#404b5d', '#3f515e', '#3e575f', '#3d5d60', '#3c6361', '#3b6962', '#3a6f63',
                    '#397564', '#387b65', '#378166', '#368767', '#358d68', '#349369', '#33996a', '#329f6b',
                    '#31a56c', '#30ab6d', '#2fb16e', '#2eb76f', '#2dbd70', '#2cc371', '#2bc972', '#2acf73',
                    '#29d574', '#28db75', '#27e176', '#26e777', '#25ed78', '#24f379', '#23f97a', '#22ff7b'
                ],
                'crest': [
                    '#a5cd90', '#9eca91', '#96c691', '#90c391', '#88bf91', '#81bc91', '#79b891', '#73b590',
                    '#6cb190', '#65ad90', '#60aa90', '#5aa690', '#55a290', '#509e90', '#4b9b90', '#47978f',
                    '#42928f', '#3e8f8e', '#3a8b8e', '#36878d', '#31838d', '#2d808c', '#297b8c', '#25788c',
                    '#22748b', '#1f6f8a', '#1d6c8a', '#1c6789', '#1c6388', '#1d5f87', '#1f5b86', '#205684',
                    '#225182', '#244d80', '#26487e', '#28447b', '#2a3f78', '#2b3b76', '#2c3574', '#2c3172'
                ]
            }
            
            # Get the selected palette or default to turbo
            color_palette = colormap_palettes.get(selected_colormap, colormap_palettes['turbo'])
            
            # Calculate exact color index using normalized value
            # This ensures the FULL spectrum is used across the selected colormap
            color_index = int(normalized_value * (len(color_palette) - 1))
            color_index = min(color_index, len(color_palette) - 1)
            
            return color_palette[color_index]

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
                    tooltip=f"Auto-Generated Point ({st.session_state.search_radius:.0f}km East - Seamless Join)"
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
                    popup=f"East Search Area ({st.session_state.search_radius:.0f}km East - Seamless)"
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
                    try:
                        print(f"AUTOMATIC SEQUENTIAL GENERATION: Triggering quad heatmap generation on click")
                        
                        # Use the dedicated sequential processing module for automatic generation
                        from sequential_heatmap import generate_quad_heatmaps_sequential
                        
                        # Generate heatmaps sequentially with Banks Peninsula exclusion and selected grid size
                        success_count, stored_heatmap_ids, error_messages = generate_quad_heatmaps_sequential(
                            wells_data=st.session_state.wells_data,
                            click_point=st.session_state.selected_point,
                            search_radius=st.session_state.search_radius,
                            interpolation_method=st.session_state.interpolation_method,
                            polygon_db=st.session_state.polygon_db,
                            soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None,
                            banks_peninsula_coords=st.session_state.banks_peninsula_coords,
                            grid_size=st.session_state.get('grid_size', (2, 3))
                        )
                        
                        print(f"AUTOMATIC GENERATION COMPLETE: {success_count} heatmaps successful")
                        
                        if success_count > 0:
                            # Reload stored heatmaps to display the new ones
                            st.session_state.stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
                            st.session_state.new_heatmap_added = True
                            st.session_state.fresh_heatmap_displayed = False
                            
                            # For display purposes, get the first generated heatmap
                            if stored_heatmap_ids:
                                primary_heatmap = st.session_state.stored_heatmaps[0] if st.session_state.stored_heatmaps else None
                                if primary_heatmap and primary_heatmap.get('geojson_data'):
                                    geojson_data = primary_heatmap['geojson_data']
                                    print(f"AUTOMATIC GENERATION: Using stored heatmap for display")
                        
                        # Sequential processing and storage handled by the dedicated module
                    except Exception as e:
                        print(f"CRITICAL ERROR in heatmap generation: {e}")
                        st.error(f"Error generating heatmaps: {e}")
                        geojson_data = {"type": "FeatureCollection", "features": []}
                        geojson_data_east = None
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
                            'fillOpacity': st.session_state.get('heatmap_opacity', 0.7)
                        }

                    # Add the GeoJSON with our custom styling
                    display_field = 'variance' if st.session_state.interpolation_method == 'kriging_variance' else 'yield'

                    # Create a unique name for the new heatmap based on location
                    new_heatmap_name = f"New: {st.session_state.interpolation_method.replace('_', ' ').title()}"
                    if st.session_state.selected_point:
                        lat, lon = st.session_state.selected_point
                        new_heatmap_name += f" ({lat:.3f}, {lon:.3f})"

                    # Add the fresh heatmap to the map based on selected style
                    if heatmap_style == "Smooth Raster (Windy.com Style)":
                        # Generate smooth raster overlay for fresh heatmap
                        print(f"Generating smooth raster overlay for fresh heatmap: {new_heatmap_name}")
                        
                        # Calculate bounds for the fresh heatmap
                        all_coords = []
                        for feature in geojson_data['features']:
                            if feature['geometry']['type'] == 'Polygon':
                                coords = feature['geometry']['coordinates'][0]
                                all_coords.extend(coords)
                        
                        if all_coords:
                            lons = [coord[0] for coord in all_coords]
                            lats = [coord[1] for coord in all_coords]
                            bounds = {
                                'north': max(lats),
                                'south': min(lats),
                                'east': max(lons),
                                'west': min(lons)
                            }
                            
                            # Generate smooth raster with global colormap function and configurable opacity
                            raster_overlay = generate_smooth_raster_overlay(
                                geojson_data, 
                                bounds, 
                                raster_size=(512, 512), 
                                global_colormap_func=lambda value: get_global_unified_color(value, st.session_state.interpolation_method),
                                opacity=st.session_state.get('heatmap_opacity', 0.7)
                            )
                            
                            if raster_overlay:
                                # Add raster overlay to map
                                folium.raster_layers.ImageOverlay(
                                    image=f"data:image/png;base64,{raster_overlay['image_base64']}",
                                    bounds=raster_overlay['bounds'],
                                    opacity=raster_overlay['opacity'],
                                    name=f"Fresh Smooth: {new_heatmap_name}"
                                ).add_to(m)
                                print(f"Added smooth raster overlay for fresh heatmap: {new_heatmap_name}")
                            else:
                                print("Failed to generate smooth raster for fresh heatmap, falling back to triangle mesh")
                                # Fallback to triangle mesh
                                fresh_geojson = folium.GeoJson(
                                    data=geojson_data,
                                    name=new_heatmap_name,
                                    style_function=lambda feature: {
                                        'fillColor': get_global_unified_color(feature['properties'][display_field], st.session_state.interpolation_method),
                                        'color': 'none',
                                        'weight': 0,
                                        'fillOpacity': st.session_state.get('heatmap_opacity', 0.7)
                                    },
                                    tooltip=folium.GeoJsonTooltip(
                                        fields=[display_field],
                                        aliases=[f'{display_field.title()}:'],
                                        labels=True,
                                        sticky=False
                                    )
                                )
                                fresh_geojson.add_to(m)
                        else:
                            print("No valid coordinates found for fresh smooth raster, using triangle mesh")
                            # Fallback to triangle mesh
                            fresh_geojson = folium.GeoJson(
                                data=geojson_data,
                                name=new_heatmap_name,
                                style_function=lambda feature: {
                                    'fillColor': get_global_unified_color(feature['properties'][display_field], st.session_state.interpolation_method),
                                    'color': 'none',
                                    'weight': 0,
                                    'fillOpacity': st.session_state.get('heatmap_opacity', 0.7)
                                },
                                tooltip=folium.GeoJsonTooltip(
                                    fields=[display_field],
                                    aliases=[f'{display_field.title()}:'],
                                    labels=True,
                                    sticky=False
                                )
                            )
                            fresh_geojson.add_to(m)
                    else:
                        # Default: Triangle Mesh (Scientific) visualization
                        fresh_geojson = folium.GeoJson(
                            data=geojson_data,
                            name=new_heatmap_name,
                            style_function=lambda feature: {
                                'fillColor': get_global_unified_color(feature['properties'][display_field], st.session_state.interpolation_method),
                                'color': 'none',
                                'weight': 0,
                                'fillOpacity': st.session_state.get('heatmap_opacity', 0.7)
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=[display_field],
                                aliases=[f'{display_field.title()}:'],
                                labels=True,
                                sticky=False
                            )
                        )
                        fresh_geojson.add_to(m)

                    # Mark that we have a fresh heatmap displayed
                    st.session_state.fresh_heatmap_displayed = False  # Will be handled by stored heatmaps
                    st.session_state.new_heatmap_added = True
                    
                    print(f"FRESH HEATMAP ADDED TO MAP: {new_heatmap_name} with {len(geojson_data.get('features', []))} features")

                    # Add UNIFIED colormap legend using global min/max values - only once for fresh heatmaps
                    # Note: For stored heatmaps, the legend will be added separately after all heatmaps are processed

                    # Analysis complete

                    # Add tooltips to show appropriate values on hover
                    style_function = lambda x: {'fillColor': 'transparent', 'color': 'transparent'}
                    highlight_function = lambda x: {'fillOpacity': min(1.0, st.session_state.get('heatmap_opacity', 0.7) + 0.1)}

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
                    
                    # Debug individual heatmap color mapping
                    sample_values = []
                    for feature in geojson_data['features'][:5]:  # Sample first 5 features
                        value = feature['properties'].get('yield', 0)
                        color = get_global_unified_color(value, method)
                        sample_values.append(f"{value:.2f}→{color}")
                    # Debug the actual value distribution in this heatmap
                    all_values_in_heatmap = [feature['properties'].get('yield', 0) for feature in geojson_data['features']]
                    if all_values_in_heatmap:
                        heatmap_min = min(all_values_in_heatmap)
                        heatmap_max = max(all_values_in_heatmap)
                        heatmap_mean = sum(all_values_in_heatmap) / len(all_values_in_heatmap)
                        print(f"  HEATMAP DATA RANGE for {stored_heatmap['heatmap_name']}: min={heatmap_min:.2f}, max={heatmap_max:.2f}, mean={heatmap_mean:.2f}, global_range=({global_min_value:.2f}-{global_max_value:.2f})")
                    print(f"  COLORMAP SAMPLE for {stored_heatmap['heatmap_name']}: {', '.join(sample_values)}")

                    # Choose visualization style based on user selection
                    if heatmap_style == "Smooth Raster (Windy.com Style)":
                        # Generate smooth raster overlay
                        print(f"  Generating smooth raster overlay for {stored_heatmap['heatmap_name']}")
                        
                        # Calculate bounds for the heatmap
                        all_coords = []
                        for feature in geojson_data['features']:
                            if feature['geometry']['type'] == 'Polygon':
                                coords = feature['geometry']['coordinates'][0]
                                all_coords.extend(coords)
                        
                        if all_coords:
                            lons = [coord[0] for coord in all_coords]
                            lats = [coord[1] for coord in all_coords]
                            bounds = {
                                'north': max(lats),
                                'south': min(lats),
                                'east': max(lons),
                                'west': min(lons)
                            }
                            
                            # Generate smooth raster with global colormap function and configurable opacity
                            raster_overlay = generate_smooth_raster_overlay(
                                geojson_data, 
                                bounds, 
                                raster_size=(512, 512), 
                                global_colormap_func=lambda value: get_global_unified_color(value, method),
                                opacity=st.session_state.get('heatmap_opacity', 0.7)
                            )
                            
                            if raster_overlay:
                                # Add raster overlay to map
                                folium.raster_layers.ImageOverlay(
                                    image=f"data:image/png;base64,{raster_overlay['image_base64']}",
                                    bounds=raster_overlay['bounds'],
                                    opacity=raster_overlay['opacity'],
                                    name=f"Smooth: {stored_heatmap['heatmap_name']}"
                                ).add_to(m)
                                stored_heatmap_count += 1
                                print(f"  Added smooth raster overlay for {stored_heatmap['heatmap_name']}")
                            else:
                                print(f"  Failed to generate smooth raster, falling back to triangle mesh")
                                # Fallback to triangle mesh
                                folium.GeoJson(
                                    geojson_data,
                                    name=f"Stored: {stored_heatmap['heatmap_name']}",
                                    style_function=lambda feature, method=method: {
                                        'fillColor': get_global_unified_color(feature['properties'].get('yield', 0), method),
                                        'color': 'none',
                                        'weight': 0,
                                        'fillOpacity': st.session_state.get('heatmap_opacity', 0.7)
                                    },
                                    tooltip=folium.GeoJsonTooltip(
                                        fields=['yield'],
                                        aliases=['Value:'],
                                        localize=True
                                    )
                                ).add_to(m)
                                stored_heatmap_count += 1
                        else:
                            print(f"  No valid coordinates found for smooth raster, using triangle mesh")
                            # Fallback to triangle mesh
                            folium.GeoJson(
                                geojson_data,
                                name=f"Stored: {stored_heatmap['heatmap_name']}",
                                style_function=lambda feature, method=method: {
                                    'fillColor': get_global_unified_color(feature['properties'].get('yield', 0), method),
                                    'color': 'none',
                                    'weight': 0,
                                    'fillOpacity': st.session_state.get('heatmap_opacity', 0.7)
                                },
                                tooltip=folium.GeoJsonTooltip(
                                    fields=['yield'],
                                    aliases=['Value:'],
                                    localize=True
                                )
                            ).add_to(m)
                            stored_heatmap_count += 1
                    else:
                        # Default: Triangle Mesh (Scientific) visualization
                        folium.GeoJson(
                            geojson_data,
                            name=f"Stored: {stored_heatmap['heatmap_name']}",
                            style_function=lambda feature, method=method: {
                                'fillColor': get_global_unified_color(feature['properties'].get('yield', 0), method),
                                'color': 'none',
                                'weight': 0,
                                'fillOpacity': st.session_state.get('heatmap_opacity', 0.7)
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

                # Removed centroid marker as per user request - no purple "i" icons needed

            except Exception as e:
                print(f"Error displaying stored heatmap {stored_heatmap.get('heatmap_name', 'unknown')}: {e}")

        print(f"Successfully displayed {stored_heatmap_count} stored heatmaps with UPDATED unified colormap")
        
        # Add colormap legend AFTER all heatmaps are processed
        if stored_heatmap_count > 0:
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
                # Determine appropriate caption based on method
                if st.session_state.interpolation_method == 'ground_water_level_kriging':
                    caption_text = f'Ground Water Level: {global_min_value:.1f} to {global_max_value:.1f} meters depth'
                    unit_label = 'm'
                elif 'depth' in st.session_state.interpolation_method:
                    caption_text = f'Depth to Groundwater: {global_min_value:.1f} to {global_max_value:.1f} meters'
                    unit_label = 'm'
                else:
                    caption_text = f'Groundwater Yield: {global_min_value:.1f} to {global_max_value:.1f} L/s'
                    unit_label = 'L/s'
                
                # Enhanced PERCENTILE-based colormap legend using stored metadata
                if stored_colormap_metadata and 'percentiles' in stored_colormap_metadata:
                    percentiles = stored_colormap_metadata['percentiles']
                    if percentiles:
                        p25 = percentiles.get('25th', 0)
                        p50 = percentiles.get('50th', 0)
                        p75 = percentiles.get('75th', 0)
                        caption_text = f'Data-Density Optimized: {global_min_value:.1f} → {p25:.1f} (25%) → {p50:.1f} (50%) → {p75:.1f} (75%) → {global_max_value:.1f} {unit_label}'
                
                colormap = folium.LinearColormap(
                    colors=['#000033', '#000066', '#000099', '#0000CC', '#0000FF',
                            '#0033FF', '#0066FF', '#0099FF', '#00CCFF', '#00FFFF',
                            '#00FFCC', '#00FF99', '#00FF66', '#00FF33', '#00FF00',
                            '#33FF00', '#66FF00', '#99FF00', '#CCFF00', '#FFFF00',
                            '#FFCC00', '#FF9900', '#FF6600', '#FF3300', '#FF0000'],
                    vmin=float(global_min_value),
                    vmax=float(global_max_value),
                    caption=caption_text
                )
            colormap.add_to(m)
            print(f"Added colormap legend: {caption_text}")
    else:
        print("No stored heatmaps to display - list is empty or cleared")

    # Show summary of displayed heatmaps
    total_displayed = stored_heatmap_count
    print(f"TOTAL HEATMAPS ON MAP: {total_displayed} (All via stored heatmaps)")

    # Show wells that overlap with displayed heatmap areas
    if st.session_state.well_markers_visibility:
        wells_layer = folium.FeatureGroup(name="Heatmap Area Wells").add_to(m)
        
        # Collect all wells from stored heatmaps
        all_heatmap_wells = []
        
        # Get wells from stored heatmaps
        if st.session_state.stored_heatmaps:
            for stored_heatmap in st.session_state.stored_heatmaps:
                # Extract center coordinates and radius from stored heatmap
                center_lat = stored_heatmap.get('center_lat')
                center_lon = stored_heatmap.get('center_lon') 
                radius_km = stored_heatmap.get('radius_km', 20)
                
                if center_lat and center_lon:
                    # Use existing wells data from session state
                    try:
                        wells_df = st.session_state.wells_data
                        
                        if wells_df is not None and not wells_df.empty:
                            # Filter wells within this heatmap's radius
                            from utils import get_distance
                            wells_in_area = []
                            
                            for idx, well in wells_df.iterrows():
                                distance = get_distance(
                                    center_lat, center_lon,
                                    well['latitude'], well['longitude']
                                )
                                if distance <= radius_km:
                                    wells_in_area.append(well.to_dict())
                            
                            all_heatmap_wells.extend(wells_in_area)
                            print(f"Found {len(wells_in_area)} wells in heatmap area: {stored_heatmap['heatmap_name']}")
                    except Exception as e:
                        print(f"Error loading wells for heatmap area: {e}")
        
        # Get wells from current filtered wells if available
        if 'filtered_wells' in st.session_state and st.session_state.filtered_wells is not None:
            current_wells = st.session_state.filtered_wells.to_dict('records')
            all_heatmap_wells.extend(current_wells)
        
        # Remove duplicates based on well_id and create display wells
        if all_heatmap_wells:
            import pandas as pd
            display_wells_df = pd.DataFrame(all_heatmap_wells)
            
            # Remove duplicates by well_id if column exists
            if 'well_id' in display_wells_df.columns:
                display_wells_df = display_wells_df.drop_duplicates(subset=['well_id'])
            else:
                # Fallback: remove duplicates by coordinates
                display_wells_df = display_wells_df.drop_duplicates(subset=['latitude', 'longitude'])
            
            # Filter out geotechnical/geological investigation wells
            if 'well_use' in display_wells_df.columns:
                geotechnical_mask = display_wells_df['well_use'].str.contains(
                    'Geotechnical.*Investigation|Geological.*Investigation', 
                    case=False, 
                    na=False, 
                    regex=True
                )
                display_wells_df = display_wells_df[~geotechnical_mask]

            # Filter out wells with no depth value (NaN or empty depth)
            if 'depth' in display_wells_df.columns:
                display_wells_df = display_wells_df[display_wells_df['depth'].notna() & (display_wells_df['depth'] > 0)]

            # Create well markers for all wells in heatmap areas
            print(f"Displaying {len(display_wells_df)} wells in heatmap areas")
            for idx, row in display_wells_df.iterrows():
                try:
                    folium.CircleMarker(
                        location=(float(row['latitude']), float(row['longitude'])),
                        radius=3,
                        color='gray',
                        fill=True,
                        fill_color='darkblue',
                        fill_opacity=0.7,
                        tooltip=f"Well {row.get('well_id', 'Unknown')} - {row.get('yield_rate', 'N/A')} L/s - Depth: {row.get('depth', 'N/A'):.1f}m"
                    ).add_to(wells_layer)
                except Exception as e:
                    print(f"Error creating marker for well: {e}")
        else:
            print("No wells found in heatmap areas")

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

                # Calculate east point for dual heatmaps (corrected based on actual gap measurement)
                # Gap measurement showed 9.3km overlap, so increase offset to eliminate it
                km_per_degree_lon = 111.0 * np.cos(np.radians(clicked_lat))
                
                # Latest measurement shows 0.211km gap - need to reduce offset slightly
                # Previous: 20km offset gave 0.211km gap
                # Reduce offset by gap amount to achieve seamless connection
                gap_to_close_km = 0.21  # Measured gap from logs
                base_offset_km = 20.0  # Current offset
                east_offset_km = base_offset_km - gap_to_close_km  # 19.79km for seamless join
                east_offset_degrees = east_offset_km / km_per_degree_lon
                
                clicked_east_lat = clicked_lat
                clicked_east_lng = clicked_lng + east_offset_degrees
                
                # Calculate south and southeast positions for 2x2 grid
                south_offset_km = east_offset_km  # Same distance (19.79km) south
                km_per_degree_lat = 111.0  # Latitude degrees are constant
                south_offset_degrees = south_offset_km / km_per_degree_lat
                
                clicked_south_lat = clicked_lat - south_offset_degrees
                clicked_south_lng = clicked_lng
                
                clicked_southeast_lat = clicked_south_lat
                clicked_southeast_lng = clicked_south_lng + east_offset_degrees
                
                # Detailed logging to understand quad heatmap positioning
                search_radius_km = st.session_state.search_radius
                clipped_width_km = search_radius_km * 0.5  # Each heatmap is clipped to 50% of search radius
                
                # Calculate additional positions for 6-heatmap grid
                clicked_northeast_lat = clicked_lat
                clicked_northeast_lng = clicked_lng + (2 * east_offset_degrees)
                clicked_far_southeast_lat = clicked_south_lat
                clicked_far_southeast_lng = clicked_south_lng + (2 * east_offset_degrees)
                
                print(f"6-HEATMAP GRID POSITIONING ANALYSIS:")
                print(f"  Original center: ({clicked_lat:.6f}, {clicked_lng:.6f})")
                print(f"  East center: ({clicked_east_lat:.6f}, {clicked_east_lng:.6f})")
                print(f"  Northeast center: ({clicked_northeast_lat:.6f}, {clicked_northeast_lng:.6f})")
                print(f"  South center: ({clicked_south_lat:.6f}, {clicked_south_lng:.6f})")
                print(f"  Southeast center: ({clicked_southeast_lat:.6f}, {clicked_southeast_lng:.6f})")
                print(f"  Far Southeast center: ({clicked_far_southeast_lat:.6f}, {clicked_far_southeast_lng:.6f})")
                print(f"  Distance between centers: {east_offset_km:.2f}km (seamless connection)")
                print(f"  Search radius: {search_radius_km:.1f}km")
                print(f"  Clipped heatmap width: {clipped_width_km:.1f}km")

                # Store all six points for 6-heatmap grid generation
                st.session_state.selected_point = [clicked_lat, clicked_lng]
                st.session_state.selected_point_east = [clicked_east_lat, clicked_east_lng]
                st.session_state.selected_point_northeast = [clicked_northeast_lat, clicked_northeast_lng]
                st.session_state.selected_point_south = [clicked_south_lat, clicked_south_lng]
                st.session_state.selected_point_southeast = [clicked_southeast_lat, clicked_southeast_lng]
                st.session_state.selected_point_far_southeast = [clicked_far_southeast_lat, clicked_far_southeast_lng]

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

                # Filter wells for south location
                wells_df['within_square_south'] = wells_df.apply(
                    lambda row: is_within_square(
                        row['latitude'], 
                        row['longitude'],
                        clicked_south_lat,
                        clicked_south_lng,
                        st.session_state.search_radius
                    ), 
                    axis=1
                )
                filtered_wells_south = wells_df[wells_df['within_square_south']]
                st.session_state.filtered_wells_south = filtered_wells_south.copy()

                # Filter wells for southeast location
                wells_df['within_square_southeast'] = wells_df.apply(
                    lambda row: is_within_square(
                        row['latitude'], 
                        row['longitude'],
                        clicked_southeast_lat,
                        clicked_southeast_lng,
                        st.session_state.search_radius
                    ), 
                    axis=1
                )
                filtered_wells_southeast = wells_df[wells_df['within_square_southeast']]
                st.session_state.filtered_wells_southeast = filtered_wells_southeast.copy()

                # Filter wells for northeast location
                wells_df['within_square_northeast'] = wells_df.apply(
                    lambda row: is_within_square(
                        row['latitude'], 
                        row['longitude'],
                        clicked_northeast_lat,
                        clicked_northeast_lng,
                        st.session_state.search_radius
                    ), 
                    axis=1
                )
                filtered_wells_northeast = wells_df[wells_df['within_square_northeast']]
                st.session_state.filtered_wells_northeast = filtered_wells_northeast.copy()

                # Filter wells for far_southeast location
                wells_df['within_square_far_southeast'] = wells_df.apply(
                    lambda row: is_within_square(
                        row['latitude'], 
                        row['longitude'],
                        clicked_far_southeast_lat,
                        clicked_far_southeast_lng,
                        st.session_state.search_radius
                    ), 
                    axis=1
                )
                filtered_wells_far_southeast = wells_df[wells_df['within_square_far_southeast']]
                st.session_state.filtered_wells_far_southeast = filtered_wells_far_southeast.copy()
                
                print(f"COORDINATES UPDATED: Original wells={len(filtered_wells)}, East wells={len(filtered_wells_east)}, Northeast wells={len(filtered_wells_northeast)}, South wells={len(filtered_wells_south)}, Southeast wells={len(filtered_wells_southeast)}, Far Southeast wells={len(filtered_wells_far_southeast)}")
                print("WELLS FILTERED: Ready for 6-heatmap grid generation")
                
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
                'selected_point', 'selected_point_east', 'selected_point_northeast', 'selected_point_south', 'selected_point_southeast', 'selected_point_far_southeast',
                'filtered_wells', 'filtered_wells_east', 'filtered_wells_northeast', 'filtered_wells_south', 'filtered_wells_southeast', 'filtered_wells_far_southeast',
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
        # Show information about 6-heatmap generation
        st.subheader("Heatmap Grid Analysis (2x3 Grid)")
        
        # Show all six heatmap locations
        grid_count = 1  # Always have original
        if st.session_state.selected_point:
            st.write(f"**Original:** {st.session_state.selected_point[0]:.4f}, {st.session_state.selected_point[1]:.4f}")
        
        if st.session_state.selected_point_east:
            grid_count += 1
            st.write(f"**East (19.79km):** {st.session_state.selected_point_east[0]:.4f}, {st.session_state.selected_point_east[1]:.4f}")
            
        # Calculate northeast position (would be generated automatically)
        if st.session_state.selected_point:
            import numpy as np
            clicked_lat, clicked_lng = st.session_state.selected_point
            km_per_degree_lon = 111.0 * np.cos(np.radians(clicked_lat))
            east_offset_km = 19.79
            east_offset_degrees = east_offset_km / km_per_degree_lon
            northeast_lat = clicked_lat
            northeast_lng = clicked_lng + (2 * east_offset_degrees)
            grid_count += 1
            st.write(f"**Northeast (39.58km):** {northeast_lat:.4f}, {northeast_lng:.4f}")
            
        if st.session_state.selected_point_south:
            grid_count += 1
            st.write(f"**South (19.79km):** {st.session_state.selected_point_south[0]:.4f}, {st.session_state.selected_point_south[1]:.4f}")
            
        if st.session_state.selected_point_southeast:
            grid_count += 1
            st.write(f"**Southeast (19.79km E of S):** {st.session_state.selected_point_southeast[0]:.4f}, {st.session_state.selected_point_southeast[1]:.4f}")
        
        # Calculate far_southeast position (would be generated automatically)
        if st.session_state.selected_point_south:
            import numpy as np
            clicked_lat, clicked_lng = st.session_state.selected_point
            km_per_degree_lon = 111.0 * np.cos(np.radians(clicked_lat))
            east_offset_km = 19.79
            south_offset_degrees = east_offset_km / 111.0
            east_offset_degrees = east_offset_km / km_per_degree_lon
            far_southeast_lat = clicked_lat - south_offset_degrees
            far_southeast_lng = clicked_lng + (2 * east_offset_degrees)
            grid_count += 1
            st.write(f"**Far Southeast (39.58km E):** {far_southeast_lat:.4f}, {far_southeast_lng:.4f}")
        
        st.success(f"✅ {grid_count} heatmaps generated in seamless 2x3 grid")
        
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