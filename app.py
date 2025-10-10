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

def classify_well_viability(row):
    """
    Classify a well as viable (1) or not viable (0) using the same 3-rule system as indicator kriging.
    
    A well is viable if ANY of these conditions are met:
    - yield_rate ≥ 0.1 L/s
    - ground water level data exists (any valid depth means water was found)
    - status = "Active (exist, present)"
    
    Returns:
    --------
    int: 1 if viable, 0 if not viable
    """
    yield_threshold = 0.1
    
    # Rule 1: Check yield rate
    yield_rate = row.get('yield_rate', np.nan)
    if pd.notna(yield_rate) and yield_rate >= yield_threshold:
        return 1
    
    # Rule 2: Check ground water level data (any valid data means water was found)
    gwl_value = row.get('ground water level', np.nan)
    if pd.notna(gwl_value):
        return 1
    
    # Rule 3: Check active status
    status = row.get('status', '')
    if status == "Active (exist, present)":
        return 1
    
    # Not viable if none of the conditions are met
    return 0

# Set page configuration with stability settings
st.set_page_config(
    page_title="Groundwater Mapper",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
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
    'show_soil_polygons': False,
    'stored_heatmaps': [],
    'interpolation_method': 'kriging',
    'show_kriging_variance': False,
    'auto_fit_variogram': False,
    'variogram_model': 'spherical',
    'geojson_data': None,
    'fresh_heatmap_displayed': False,
    'new_heatmap_added': False,
    'colormap_updated': False,

    'new_clipping_polygon': None,
    'show_new_clipping_polygon': False,
    'show_well_bounds': False,
    'show_convex_hull': False,
    'show_grid_points': False,
    'heatmap_visualization_mode': 'smooth_raster'  # 'triangular_mesh' or 'smooth_raster'
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

# Load comprehensive clipping polygon
@st.cache_data
def load_new_clipping_polygon():
    """Load comprehensive polygon with holes from ring-structured JSON data"""
    try:
        import os
        import json
        from shapely.geometry import Polygon
        
        # Try the processed GeoJSON first
        geojson_path = "all_polygons_complete_latest.geojson"
        if os.path.exists(geojson_path):
            gdf = gpd.read_file(geojson_path)
            if gdf is not None and not gdf.empty:
                print(f"✅ COMPREHENSIVE POLYGON DATA LOADED: {len(gdf)} separate polygon features from GeoJSON")
                print(f"📍 Coordinate bounds: {gdf.total_bounds}")
                print(f"🗺️ Individual polygons include main Canterbury Plains area plus {len(gdf)-1} separate drainage areas")
                return gdf
        
        # Fallback: Process ring-structured JSON using containment detection
        json_path = "attached_assets/big_1754735961105.json"
        if os.path.exists(json_path):
            print(f"🔄 Processing comprehensive polygon using containment detection...")
            
            # Check if processed file already exists
            processed_path = "comprehensive_polygons_processed.geojson"
            if os.path.exists(processed_path):
                print(f"📁 Using existing processed comprehensive polygons...")
                gdf = gpd.read_file(processed_path)
                print(f"✅ Loaded {len(gdf)} processed polygon features")
                return gdf
        
        print("No comprehensive polygon file found")
        return None
        
    except Exception as e:
        print(f"Error loading new clipping polygon: {e}")
        return None



if 'polygon_db' not in st.session_state:
    st.session_state.polygon_db = get_database_connection()

# Load existing red/orange zone boundary on startup (once per session)
if not hasattr(st.session_state, 'loaded_existing_boundary'):
    if st.session_state.polygon_db:
        try:
            from green_zone_extractor import get_stored_red_orange_polygon
            existing_boundary = get_stored_red_orange_polygon(st.session_state.polygon_db)
            
            if existing_boundary:
                st.session_state.green_zone_boundary = existing_boundary
                st.session_state.show_green_zone_boundary = True
                print("🔄 STARTUP: Loaded existing red/orange zone boundary from database")
        except Exception as e:
            print(f"❌ Error loading existing boundary: {e}")
            
    st.session_state.loaded_existing_boundary = True

# Force complete reload and clear ALL caches for comprehensive polygon data
print("🔄 CLEARING ALL CACHED POLYGON DATA...")
st.cache_data.clear()  # Clear all cached data including load_new_clipping_polygon cache
st.session_state.new_clipping_polygon = None  # Clear session state

# Now reload fresh comprehensive polygon data
print("📥 LOADING FRESH COMPREHENSIVE POLYGON DATA...")
st.session_state.new_clipping_polygon = load_new_clipping_polygon()

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
            "Indicator Kriging (Yield Suitability)",
            "Indicator Kriging (Spherical)",
            "Indicator Kriging (Spherical Continuous)"
        ],
        index=5,  # Default to Ground Water Level (Spherical Kriging)
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
        st.session_state.interpolation_method = 'depth_kriging_auto'
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
    elif visualization_method == "Indicator Kriging (Spherical)":
        st.session_state.interpolation_method = 'indicator_kriging_spherical'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
    elif visualization_method == "Indicator Kriging (Spherical Continuous)":
        st.session_state.interpolation_method = 'indicator_kriging_spherical_continuous'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
        # Log confirmation that enhanced functionality is active
        print(f"🎯 ENHANCED CONTINUOUS INDICATOR: Selected method with 2x well search radius functionality")

    # Indicator Kriging Variogram Settings (only for indicator methods)
    is_indicator_method = st.session_state.interpolation_method in [
        'indicator_kriging', 
        'indicator_kriging_spherical', 
        'indicator_kriging_spherical_continuous'
    ]
    
    if is_indicator_method:
        st.subheader("Indicator Kriging Variogram Settings")
        
        # Initialize session state defaults for indicator kriging variogram
        if 'indicator_auto_fit' not in st.session_state:
            st.session_state.indicator_auto_fit = False
        if 'indicator_range' not in st.session_state:
            st.session_state.indicator_range = 1500.0
        if 'indicator_sill' not in st.session_state:
            st.session_state.indicator_sill = 0.25
        if 'indicator_nugget' not in st.session_state:
            st.session_state.indicator_nugget = 0.1
        
        # Auto-fit checkbox
        st.session_state.indicator_auto_fit = st.checkbox(
            "Auto-fit variogram (experimental)",
            value=st.session_state.indicator_auto_fit,
            help="⚠️ Auto-fit often fails with sparse data, producing flat green areas. Manual parameters (below) usually work better for indicator kriging.",
            key="indicator_auto_fit_checkbox"
        )
        
        if st.session_state.indicator_auto_fit:
            st.warning("⚠️ Auto-fit may produce flat results (all green) if wells are sparse. If you see solid green areas, uncheck this and use manual parameters instead.")
        
        # Manual parameters - always show for reference
        param_label = "Manual Override Parameters" if st.session_state.indicator_auto_fit else "Manual Variogram Parameters"
        st.markdown(f"**{param_label}**")
        
        st.session_state.indicator_range = st.number_input(
            "Range (meters)",
            min_value=100.0,
            max_value=50000.0,
            value=st.session_state.indicator_range,
            step=100.0,
            help="Spatial influence distance. Wells beyond this range have minimal impact. Typical: 1500m (1.5km)" + (" (used as fallback if auto-fit fails)" if st.session_state.indicator_auto_fit else ""),
            key="indicator_range_input",
            disabled=st.session_state.indicator_auto_fit
        )
        
        st.session_state.indicator_sill = st.number_input(
            "Sill",
            min_value=0.01,
            max_value=1.0,
            value=st.session_state.indicator_sill,
            step=0.01,
            help="Maximum variance for binary (0/1) indicator data. Theoretical value for binary data is 0.25" + (" (used as fallback if auto-fit fails)" if st.session_state.indicator_auto_fit else ""),
            key="indicator_sill_input",
            disabled=st.session_state.indicator_auto_fit
        )
        
        st.session_state.indicator_nugget = st.number_input(
            "Nugget",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.indicator_nugget,
            step=0.01,
            help="Measurement error and micro-scale variability. Higher values reduce smoothing. Typical: 0.01-0.1" + (" (used as fallback if auto-fit fails)" if st.session_state.indicator_auto_fit else ""),
            key="indicator_nugget_input",
            disabled=st.session_state.indicator_auto_fit
        )
        
        if not st.session_state.indicator_auto_fit:
            st.info(f"📐 Using Manual Settings: Range={st.session_state.indicator_range}m, Sill={st.session_state.indicator_sill}, Nugget={st.session_state.indicator_nugget}")

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
    
    # Update session state and create heatmap_style variable based on selection
    if "Smooth Raster" in heatmap_mode:
        st.session_state.heatmap_visualization_mode = 'smooth_raster'
        heatmap_style = "Smooth Raster (Windy.com Style)"
        st.info("🌊 **Smooth Raster Mode**: Heatmaps will display with smooth gradients like Windy.com weather maps")
    else:
        st.session_state.heatmap_visualization_mode = 'triangular_mesh'
        heatmap_style = "Triangle Mesh (Scientific)"
    
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
        index=11,  # Default to mako colormap
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
    st.session_state.show_well_bounds = st.checkbox("Show Well Data Bounds", value=getattr(st.session_state, 'show_well_bounds', False), help="Show the rectangular boundary of all well data used for automated generation")
    st.session_state.show_convex_hull = st.checkbox("Show Convex Hull Boundary", value=getattr(st.session_state, 'show_convex_hull', False), help="Show the efficient convex hull boundary calculated from ALL wells (62% more efficient than rectangular bounds)")
    st.session_state.show_grid_points = st.checkbox("Show 19.82km Grid Points", value=getattr(st.session_state, 'show_grid_points', False), help="Show grid of potential heatmap centers at 19.82km spacing within the convex hull boundary")
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")
    
    # NEW CLIPPING POLYGON display option (takes priority)
    if st.session_state.new_clipping_polygon is not None:
        st.session_state.show_new_clipping_polygon = st.checkbox(
            "🟢 Show NEW Clipping Polygon", 
            value=st.session_state.show_new_clipping_polygon, 
            help="Display the comprehensive clipping polygon for Canterbury Plains drainage areas"
        )
    


    # Automated Heatmap Generation Section
    st.markdown("---")
    st.subheader("🤖 Automated Heatmap Generation")
    
    st.write("Generate heatmaps automatically covering all available well data without manual clicking.")
    
    max_tiles_full = st.number_input("Max tiles for full generation", min_value=10, max_value=1000, value=100, step=10,
                                    help="Automated generation continues until reaching actual well data bounds (up to this limit)")
    
    # Initialize persistent generation flag
    if 'auto_generation_in_progress' not in st.session_state:
        st.session_state.auto_generation_in_progress = False
    
    # Button sets persistent flag instead of directly running generation
    if st.button("🚀 Full Auto Generation", help="Generate heatmaps for all well data (limited by max tiles)"):
        if st.session_state.wells_data is not None and st.session_state.polygon_db is not None:
            # Set flag and store parameters
            st.session_state.auto_generation_in_progress = True
            st.session_state.generation_params = {
                'max_tiles': max_tiles_full,
                'interpolation_method': st.session_state.interpolation_method,
                'search_radius_km': st.session_state.search_radius,
                'show_soil_polygons': st.session_state.show_soil_polygons,
                'indicator_auto_fit': st.session_state.get('indicator_auto_fit', False),
                'indicator_range': st.session_state.get('indicator_range', 1500.0),
                'indicator_sill': st.session_state.get('indicator_sill', 0.25),
                'indicator_nugget': st.session_state.get('indicator_nugget', 0.1)
            }
            st.rerun()
        else:
            st.error("Wells data or database not available")
    
    # Execute generation when flag is True (survives reruns)
    if st.session_state.auto_generation_in_progress:
        params = st.session_state.generation_params
        
        # CRITICAL FIX: Check if generation already completed to prevent infinite loops
        existing_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
        existing_count = len(existing_heatmaps)
        
        if existing_count >= params['max_tiles']:
            # Generation already complete - clear flag and show success
            st.session_state.auto_generation_in_progress = False
            st.success(f"✅ Auto-generation complete! {existing_count} heatmaps already exist.")
            st.info("Refresh the page to view all heatmaps.")
            st.stop()
        
        st.info(f"⏳ Auto-generation in progress ({existing_count}/{params['max_tiles']} heatmaps)... Check console for progress.")
        
        try:
            from automated_heatmap_generator import generate_automated_heatmaps
            
            success_count, stored_ids, errors = generate_automated_heatmaps(
                wells_data=st.session_state.wells_data,
                interpolation_method=params['interpolation_method'],
                polygon_db=st.session_state.polygon_db,
                soil_polygons=st.session_state.soil_polygons if params['show_soil_polygons'] else None,
                new_clipping_polygon=st.session_state.new_clipping_polygon,
                search_radius_km=params['search_radius_km'],
                max_tiles=params['max_tiles'],
                indicator_auto_fit=params['indicator_auto_fit'],
                indicator_range=params['indicator_range'],
                indicator_sill=params['indicator_sill'],
                indicator_nugget=params['indicator_nugget']
            )
            
            # Clear flag after successful completion
            st.session_state.auto_generation_in_progress = False
            
            if success_count > 0:
                st.success(f"✅ Generated {success_count} heatmaps successfully!")
                if errors:
                    st.warning(f"⚠️ {len(errors)} tiles had errors")
                # Don't reload stored_heatmaps here - it triggers rerun before flag clear takes effect
                # Map will auto-load them when it renders
            else:
                st.error("❌ No heatmaps generated. Check console for details.")
                
        except Exception as e:
            st.session_state.auto_generation_in_progress = False
            st.error(f"Generation error: {e}")
        
        # CRITICAL: Stop script execution to prevent map rendering during generation
        st.stop()
    
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

        # Red/Orange Zone Analysis section
        st.subheader("🔴 Red/Orange Zone Analysis")
        st.write("Extract and display boundary polygons around low/medium probability indicator zones (<0.7 threshold). Adjacent red and orange zones will be unified into continuous boundaries.")
        
        if st.button("🔴 Extract Red/Orange Zone Boundary", type="primary"):
            if st.session_state.polygon_db:
                with st.spinner("Extracting red/orange zones from indicator kriging heatmaps..."):
                    try:
                        from green_zone_extractor import extract_green_zones_from_indicator_heatmaps, store_green_zone_boundary
                        
                        # Extract red/orange zone boundary
                        boundary_geojson = extract_green_zones_from_indicator_heatmaps(st.session_state.polygon_db)
                        
                        if boundary_geojson:
                            # Store the boundary polygon
                            polygon_id = store_green_zone_boundary(st.session_state.polygon_db, boundary_geojson)
                            
                            if polygon_id:
                                st.success(f"✅ Red/orange zone boundary extracted and stored! Polygon ID: {polygon_id}")
                                
                                # Store boundary for map display
                                st.session_state.green_zone_boundary = boundary_geojson
                                st.session_state.show_green_zone_boundary = True
                                
                                # Show some stats
                                total_zones = len(boundary_geojson['features'])
                                st.info(f"📊 Created {total_zones} boundary polygon(s) around red/orange zones")
                            else:
                                st.error("❌ Failed to store boundary polygon")
                        else:
                            st.warning("⚠️ No red/orange zones found. Make sure indicator kriging heatmaps are available.")
                            
                    except Exception as e:
                        st.error(f"Error extracting red/orange zones: {e}")
                        print(f"Red/orange zone extraction error: {e}")
            else:
                st.error("Database not available")
        
        # Clear red/orange polygon button
        if st.button("🗑️ Clear Red/Orange Zone Boundary", help="Remove stored red/orange zone boundary from database"):
            if st.session_state.polygon_db:
                from green_zone_extractor import clear_stored_red_orange_polygon
                success = clear_stored_red_orange_polygon(st.session_state.polygon_db)
                
                if success:
                    st.success("✅ Red/orange zone boundary cleared from database!")
                    # Clear from session state too
                    if hasattr(st.session_state, 'green_zone_boundary'):
                        st.session_state.green_zone_boundary = None
                        st.session_state.show_green_zone_boundary = False
                else:
                    st.warning("⚠️ No red/orange zone boundary found to clear")
            else:
                st.error("Database not available")
        
        # Display option for red/orange zone boundary
        if hasattr(st.session_state, 'green_zone_boundary') and st.session_state.green_zone_boundary:
            st.session_state.show_green_zone_boundary = st.checkbox(
                "🔴 Show Red/Orange Zone Boundary", 
                value=getattr(st.session_state, 'show_green_zone_boundary', False),
                help="Display the boundary polygon around low/medium probability indicator zones"
            )

        # Tile Boundary Snapping section
        st.subheader("🔧 Tile Boundary Optimization")
        st.write("Fix gaps and overlaps between adjacent heatmap tiles by snapping only boundary vertices (within 300m). Internal triangle vertices remain unchanged.")
        
        if st.button("🎯 Snap Boundary Vertices Only", type="primary"):
            print("\n" + "="*50)
            print("🎯 SNAP TILE BOUNDARIES BUTTON CLICKED!")
            print("="*50)
            
            # Log current session state
            heatmap_count = len(st.session_state.stored_heatmaps) if st.session_state.stored_heatmaps else 0
            print(f"📊 BUTTON CLICK CONTEXT:")
            print(f"   Session state heatmaps: {heatmap_count}")
            print(f"   Database connection: {st.session_state.polygon_db is not None}")
            
            if st.session_state.stored_heatmaps:
                print(f"✅ STORED HEATMAPS AVAILABLE: {len(st.session_state.stored_heatmaps)} heatmaps found")
                
                # Log some heatmap details for debugging
                for i, heatmap in enumerate(st.session_state.stored_heatmaps[:3]):  # First 3 only
                    has_geojson = bool(heatmap.get('geojson_data'))
                    print(f"   Heatmap {i+1}: ID {heatmap['id']} - {heatmap['heatmap_name']} - GeoJSON: {has_geojson}")
                
                with st.spinner("Snapping tile boundaries to reduce gaps and overlaps..."):
                    print("🔄 ENTERING SPINNER CONTEXT")
                    try:
                        print("📦 IMPORTING boundary_only_snapping module...")
                        from boundary_only_snapping import run_boundary_only_snapping
                        print("✅ Module imported successfully")
                        
                        # Capture the snapping process output
                        import io
                        import sys
                        
                        print("🔧 SETTING UP OUTPUT CAPTURE...")
                        # Redirect stdout to capture print statements
                        old_stdout = sys.stdout
                        sys.stdout = captured_output = io.StringIO()
                        
                        print("🎯 CALLING run_boundary_only_snapping()...")
                        # Run the boundary-only snapping using existing database connection
                        run_boundary_only_snapping(st.session_state.polygon_db)
                        
                        # Restore stdout and get the output
                        sys.stdout = old_stdout
                        output = captured_output.getvalue()
                        
                        print("📋 SNAPPING FUNCTION COMPLETED")
                        print(f"📏 OUTPUT LENGTH: {len(output)} characters")
                        print(f"📄 OUTPUT PREVIEW: {output[:200]}..." if len(output) > 200 else output)
                        
                        # Display results
                        if "BOUNDARY-ONLY SNAPPING COMPLETE" in output or "BOUNDARY SNAPPING COMPLETE" in output:
                            print("✅ SUCCESS: Found 'BOUNDARY SNAPPING COMPLETE' in output")
                            st.success("Tile boundaries snapped successfully!")
                            
                            # Extract statistics from output
                            lines = output.split('\n')
                            stats_found = 0
                            for line in lines:
                                if "vertices snapped across" in line:
                                    st.info(f"📊 {line.strip()}")
                                    stats_found += 1
                                elif "Tile " in line and ": " in line and "vertices snapped" in line:
                                    st.write(f"  • {line.strip()}")
                                    stats_found += 1
                            
                            print(f"📊 DISPLAYED {stats_found} statistics lines")
                            
                            # Refresh stored heatmaps to show updated data
                            print("🔄 REFRESHING SESSION STATE...")
                            st.session_state.stored_heatmaps = st.session_state.polygon_db.get_all_stored_heatmaps()
                            print("🔄 CALLING st.rerun()...")
                            st.rerun()
                            
                        elif "NO BOUNDARY SNAPPING NEEDED" in output or "NO SNAPPING NEEDED" in output:
                            print("ℹ️ INFO: Found 'NO SNAPPING NEEDED' in output")
                            st.info("All tiles are already well-aligned (within 100m tolerance)")
                        else:
                            print("⚠️ WARNING: No recognized status message in output")
                            print(f"🔍 FULL OUTPUT FOR DEBUGGING:\n{output}")
                            st.warning("Boundary snapping completed but no clear status was reported")
                            st.code(output, language="text")  # Show raw output to user
                            
                    except Exception as e:
                        print(f"❌ EXCEPTION DURING SNAPPING: {e}")
                        print(f"🐛 EXCEPTION TYPE: {type(e).__name__}")
                        import traceback
                        traceback_str = traceback.format_exc()
                        print(f"🔍 FULL TRACEBACK:\n{traceback_str}")
                        
                        st.error(f"Error during boundary snapping: {e}")
                        st.code(traceback_str, language="text")  # Show traceback to user
                        
            else:
                print("⚠️ WARNING: No stored heatmaps available")
                print(f"📊 STORED_HEATMAPS VALUE: {st.session_state.stored_heatmaps}")
                st.warning("No stored heatmaps available for boundary snapping")
            
            print("🏁 BUTTON CLICK PROCESSING COMPLETE")
            print("="*50)

        st.divider()

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
# Map now takes full width
# CRITICAL: Skip map rendering during auto-generation to prevent re-entrant display logic
if st.session_state.auto_generation_in_progress:
    st.info("⏳ Auto-generation in progress... Map display temporarily disabled to prevent interruption.")
    st.stop()  # Stop script execution here during generation

# Default location (New Zealand as example)
default_location = [-43.5320, 172.6306]  # Christchurch, New Zealand

# Create map centered at default location
if st.session_state.selected_point:
    center_location = st.session_state.selected_point
else:
    center_location = default_location

m = folium.Map(location=center_location, zoom_start=st.session_state.zoom_level, 
              tiles="OpenStreetMap")

# Add well data bounds visualization if enabled
if st.session_state.show_well_bounds and st.session_state.wells_data is not None:
    try:
        wells_df = st.session_state.wells_data
        
        # Calculate bounds using the same logic as automated generation
        if 'latitude' in wells_df.columns and 'longitude' in wells_df.columns:
            valid_wells = wells_df.dropna(subset=['latitude', 'longitude'])
            
            if len(valid_wells) > 0:
                lat_coords = valid_wells['latitude'].astype(float)
                lon_coords = valid_wells['longitude'].astype(float)
                
                sw_lat, ne_lat = lat_coords.min(), lat_coords.max()
                sw_lon, ne_lon = lon_coords.min(), lon_coords.max()
                
                # Create rectangle coordinates for the bounds
                bounds_coords = [
                    [sw_lat, sw_lon],  # Southwest corner
                    [sw_lat, ne_lon],  # Southeast corner
                    [ne_lat, ne_lon],  # Northeast corner
                    [ne_lat, sw_lon],  # Northwest corner
                    [sw_lat, sw_lon]   # Close the rectangle
                ]
                
                # Add rectangle to map
                folium.PolyLine(
                    locations=bounds_coords,
                    color='red',
                    weight=3,
                    opacity=0.8,
                    popup=f"Well Data Bounds<br>SW: {sw_lat:.6f}, {sw_lon:.6f}<br>NE: {ne_lat:.6f}, {ne_lon:.6f}<br>Area: {len(valid_wells)} wells"
                ).add_to(m)
                
                # Add corner markers for better visibility
                folium.Marker(
                    [sw_lat, sw_lon],
                    popup=f"SW Corner: {sw_lat:.6f}, {sw_lon:.6f}",
                    icon=folium.Icon(color='red', icon='arrow-down', prefix='fa')
                ).add_to(m)
                
                folium.Marker(
                    [ne_lat, ne_lon],
                    popup=f"NE Corner: {ne_lat:.6f}, {ne_lon:.6f}",
                    icon=folium.Icon(color='red', icon='arrow-up', prefix='fa')
                ).add_to(m)
                
                # Calculate and show the grid that would be generated
                import numpy as np
                lat_span = ne_lat - sw_lat
                lon_span = ne_lon - sw_lon
                center_lat = (sw_lat + ne_lat) / 2
                
                # Convert to approximate km
                lat_km = lat_span * 111.0
                lon_km = lon_span * 111.0 * np.cos(np.radians(center_lat))
                
                # Calculate optimal grid size (19.82km spacing) with buffer
                grid_spacing_km = 19.82
                rows_needed = max(1, int(np.ceil(lat_km / grid_spacing_km)) + 2)
                cols_needed = max(1, int(np.ceil(lon_km / grid_spacing_km)) + 2)
                total_needed = rows_needed * cols_needed
                
                # Add center point marker
                center_lat_calc = (sw_lat + ne_lat) / 2
                center_lon_calc = (sw_lon + ne_lon) / 2
                folium.Marker(
                    [center_lat_calc, center_lon_calc],
                    popup=f"Grid Center<br>Calculated Grid: {rows_needed}×{cols_needed} = {total_needed} tiles<br>Data extent: {lat_km:.1f}km × {lon_km:.1f}km",
                    icon=folium.Icon(color='orange', icon='crosshairs', prefix='fa')
                ).add_to(m)
                
                print(f"WELL BOUNDS VISUALIZATION: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
                print(f"Covering area with {len(valid_wells)} wells")
                print(f"Calculated grid: {rows_needed}×{cols_needed} = {total_needed} tiles needed for full coverage")
                
    except Exception as e:
        print(f"Error adding well bounds visualization: {e}")

# Show convex hull boundary if requested
if st.session_state.show_convex_hull and st.session_state.wells_data is not None:
    try:
        wells_df = st.session_state.wells_data
        
        # Calculate convex hull using the same logic as automated generation
        if 'latitude' in wells_df.columns and 'longitude' in wells_df.columns:
            valid_wells = wells_df.dropna(subset=['latitude', 'longitude'])
            
            if len(valid_wells) > 0:
                from scipy.spatial import ConvexHull
                from pyproj import Transformer
                
                # Convert lat/lon to NZTM for accurate hull calculation - use ALL wells
                transformer_to_nztm = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
                transformer_to_latlon = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
                
                # Convert ALL wells to NZTM for accurate convex hull
                all_lons = valid_wells['longitude'].astype(float)
                all_lats = valid_wells['latitude'].astype(float)
                nztm_coords = [transformer_to_nztm.transform(lon, lat) for lat, lon in zip(all_lats, all_lons)]
                nztm_x = [coord[0] for coord in nztm_coords]
                nztm_y = [coord[1] for coord in nztm_coords]
                
                # Calculate convex hull
                points = np.column_stack([nztm_x, nztm_y])
                hull = ConvexHull(points)
                hull_area_km2 = hull.volume / 1e6  # ConvexHull.volume is area in 2D
                
                # Convert hull vertices back to lat/lon for display
                hull_vertices = points[hull.vertices]
                hull_coords = []
                for x, y in hull_vertices:
                    lon, lat = transformer_to_latlon.transform(x, y)
                    hull_coords.append([lat, lon])
                
                # Close the polygon
                hull_coords.append(hull_coords[0])
                
                # Calculate rectangular area for comparison
                lat_coords = valid_wells['latitude'].astype(float)
                lon_coords = valid_wells['longitude'].astype(float)
                sw_lat, ne_lat = lat_coords.min(), lat_coords.max()
                sw_lon, ne_lon = lon_coords.min(), lon_coords.max()
                center_lat = (sw_lat + ne_lat) / 2
                lat_km = (ne_lat - sw_lat) * 111.0
                lon_km = (ne_lon - sw_lon) * 111.0 * np.cos(np.radians(center_lat))
                rectangular_area_km2 = lat_km * lon_km
                
                reduction_percent = (1 - hull_area_km2/rectangular_area_km2) * 100
                
                # Add convex hull boundary to map
                folium.PolyLine(
                    locations=hull_coords,
                    color='blue',
                    weight=3,
                    opacity=0.8,
                    popup=f"Convex Hull Boundary<br>Efficient boundary following data distribution<br>Area: {hull_area_km2:.0f} km² ({reduction_percent:.1f}% smaller than rectangle)<br>Based on ALL {len(valid_wells)} wells"
                ).add_to(m)
                
                # Add markers for first few hull vertices
                for i, (lat, lon) in enumerate(hull_coords[:6]):  # Show first 6 vertices
                    folium.Marker(
                        [lat, lon],
                        popup=f"Hull Vertex {i+1}: {lat:.6f}, {lon:.6f}",
                        icon=folium.Icon(color='blue', icon='circle', prefix='fa')
                    ).add_to(m)
                
                # Add center marker showing the efficiency gain
                hull_center_lat = sum([coord[0] for coord in hull_coords[:-1]]) / len(hull_coords[:-1])
                hull_center_lon = sum([coord[1] for coord in hull_coords[:-1]]) / len(hull_coords[:-1])
                
                folium.Marker(
                    [hull_center_lat, hull_center_lon],
                    popup=f"Convex Hull Center<br>Efficient Coverage: {hull_area_km2:.0f} km²<br>Reduction: {reduction_percent:.1f}% vs rectangular bounds<br>Estimated tiles saved: ~{int((rectangular_area_km2 - hull_area_km2) / 100)} for 10km resolution",
                    icon=folium.Icon(color='darkblue', icon='star', prefix='fa')
                ).add_to(m)
                
                print(f"CONVEX HULL VISUALIZATION: {len(hull_coords)-1} vertices, {hull_area_km2:.0f} km² ({reduction_percent:.1f}% reduction)")
                
    except Exception as e:
        print(f"Error displaying convex hull: {e}")

# Show 19.82km grid points within convex hull if requested
if st.session_state.show_grid_points and st.session_state.wells_data is not None:
    try:
        wells_df = st.session_state.wells_data
        
        # Calculate convex hull and generate grid points within it
        if 'latitude' in wells_df.columns and 'longitude' in wells_df.columns:
            valid_wells = wells_df.dropna(subset=['latitude', 'longitude'])
            
            if len(valid_wells) > 0:
                from scipy.spatial import ConvexHull
                from pyproj import Transformer
                from shapely.geometry import Point, Polygon
                from utils import get_distance
                
                # Convert to NZTM for accurate grid generation
                transformer_to_nztm = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
                transformer_to_latlon = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
                
                # Get convex hull in NZTM coordinates
                all_lons = valid_wells['longitude'].astype(float)
                all_lats = valid_wells['latitude'].astype(float)
                nztm_coords = [transformer_to_nztm.transform(lon, lat) for lat, lon in zip(all_lats, all_lons)]
                nztm_x = [coord[0] for coord in nztm_coords]
                nztm_y = [coord[1] for coord in nztm_coords]
                
                # Calculate convex hull
                points = np.column_stack([nztm_x, nztm_y])
                hull = ConvexHull(points)
                hull_vertices_nztm = points[hull.vertices]
                
                # Create Shapely polygon from convex hull
                hull_polygon = Polygon(hull_vertices_nztm)
                
                # Get bounds for grid generation
                min_x, min_y, max_x, max_y = hull_polygon.bounds
                
                # Generate grid points at 19.82km spacing
                grid_spacing = 19820  # 19.82km in meters (NZTM units)
                
                # Calculate grid bounds with padding
                start_x = int(min_x // grid_spacing) * grid_spacing
                start_y = int(min_y // grid_spacing) * grid_spacing
                end_x = int(max_x // grid_spacing + 1) * grid_spacing
                end_y = int(max_y // grid_spacing + 1) * grid_spacing
                
                # Generate all grid points
                grid_points_nztm = []
                grid_points_latlon = []
                
                y = start_y
                while y <= end_y:
                    x = start_x
                    while x <= end_x:
                        point_nztm = Point(x, y)
                        # Check if point is within convex hull
                        if hull_polygon.contains(point_nztm):
                            grid_points_nztm.append((x, y))
                            # Convert to lat/lon for display
                            lon, lat = transformer_to_latlon.transform(x, y)
                            grid_points_latlon.append((lat, lon))
                        x += grid_spacing
                    y += grid_spacing
                
                print(f"GRID GENERATION: Created {len(grid_points_latlon)} grid points at 19.82km spacing within convex hull")
                
                # Add grid points to map
                for i, (lat, lon) in enumerate(grid_points_latlon):
                    # Calculate distance from center for reference
                    center_lat = sum([coord[0] for coord in grid_points_latlon]) / len(grid_points_latlon)
                    center_lon = sum([coord[1] for coord in grid_points_latlon]) / len(grid_points_latlon)
                    distance_from_center = get_distance(center_lat, center_lon, lat, lon)
                    
                    folium.Marker(
                        [lat, lon],
                        popup=f"Grid Point {i+1}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>Distance from center: {distance_from_center:.1f} km<br>Potential heatmap center",
                        icon=folium.Icon(color='green', icon='plus', prefix='fa'),
                        tooltip=f"Grid Point {i+1} (19.82km spacing)"
                    ).add_to(m)
                
                # Add grid info marker at convex hull center
                hull_center_nztm = hull_polygon.centroid
                hull_center_lon, hull_center_lat = transformer_to_latlon.transform(hull_center_nztm.x, hull_center_nztm.y)
                
                folium.Marker(
                    [hull_center_lat, hull_center_lon],
                    popup=f"19.82km Grid System<br>Total grid points: {len(grid_points_latlon)}<br>Spacing: 19.82km (North/South/East/West)<br>Coverage: Convex hull boundary only<br>These represent optimal heatmap centers",
                    icon=folium.Icon(color='darkgreen', icon='th', prefix='fa'),
                    tooltip="Grid System Center"
                ).add_to(m)
                
                print(f"GRID VISUALIZATION: Added {len(grid_points_latlon)} green markers at 19.82km spacing")
                
    except Exception as e:
        print(f"Error displaying grid points: {e}")

# Add comprehensive clipping polygon if available and enabled
if st.session_state.show_new_clipping_polygon and st.session_state.new_clipping_polygon is not None:
    try:
        # Convert to GeoJSON and add to map with distinctive styling
        folium.GeoJson(
            st.session_state.new_clipping_polygon.__geo_interface__,
            name="NEW Clipping Polygon",
            style_function=lambda feature: {
                'fillColor': '#32CD32',  # Lime green for new polygon
                'color': '#228B22',      # Forest green border
                'weight': 3,
                'fillOpacity': 0.2,
                'opacity': 0.9
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['area_deg2'] if 'area_deg2' in st.session_state.new_clipping_polygon.columns else [],
                aliases=['Area (sq deg):'] if 'area_deg2' in st.session_state.new_clipping_polygon.columns else [],
                labels=True,
                sticky=False
            )
        ).add_to(m)
        
        # Calculate and print center
        centroid = st.session_state.new_clipping_polygon.dissolve().centroid.iloc[0]
        polygon_center = (centroid.y, centroid.x)
        print(f"NEW CLIPPING POLYGON added to map (center: {polygon_center})")
        print(f"Area covered: {st.session_state.new_clipping_polygon.geometry.area.sum():.8f} square degrees")
        
    except Exception as e:
        print(f"Error adding new clipping polygon to map: {e}")

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

# Add green zone boundary if available and enabled
if (hasattr(st.session_state, 'show_green_zone_boundary') and 
    st.session_state.show_green_zone_boundary and 
    hasattr(st.session_state, 'green_zone_boundary') and 
    st.session_state.green_zone_boundary):
    
    try:
        from green_zone_extractor import display_green_zone_boundary_on_map
        
        success = display_green_zone_boundary_on_map(m, st.session_state.green_zone_boundary)
        if success:
            print("🔴 RED/ORANGE ZONE BOUNDARY: Added to map successfully")
        else:
            print("❌ RED/ORANGE ZONE BOUNDARY: Failed to add to map")
            
    except Exception as e:
        print(f"Error displaying red/orange zone boundary: {e}")

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
        raw_geojson_data = stored_heatmap.get('geojson_data')
        
        # Apply exclusion clipping ONLY to non-indicator methods for colormap calculation
        method = stored_heatmap.get('interpolation_method', 'kriging')
        indicator_methods = [
            'indicator_kriging', 
            'indicator_kriging_spherical', 
            'indicator_kriging_spherical_continuous'
        ]
        
        if raw_geojson_data:
            if method not in indicator_methods:
                from interpolation import apply_exclusion_clipping_to_stored_heatmap
                geojson_data = apply_exclusion_clipping_to_stored_heatmap(raw_geojson_data)
            else:
                geojson_data = raw_geojson_data
        else:
            geojson_data = raw_geojson_data
            
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
            raw_geojson_data = stored_heatmap.get('geojson_data')
            
            # Apply exclusion clipping ONLY to non-indicator methods for percentile calculation
            method = stored_heatmap.get('interpolation_method', 'kriging')
            indicator_methods = [
                'indicator_kriging', 
                'indicator_kriging_spherical', 
                'indicator_kriging_spherical_continuous'
            ]
            
            if raw_geojson_data:
                from interpolation import apply_exclusion_clipping_to_stored_heatmap
                geojson_data = apply_exclusion_clipping_to_stored_heatmap(raw_geojson_data, method_name=method)
            else:
                geojson_data = raw_geojson_data
                
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
    if method == 'indicator_kriging' or method == 'indicator_kriging_spherical' or method == 'indicator_kriging_spherical_continuous':
        # Four-tier classification: red (poor), orange (low-moderate), yellow (moderate), green (good)
        if value <= 0.4:
            return '#FF0000'    # Red for poor
        elif value <= 0.6:
            return '#FF8000'    # Orange for low-moderate
        elif value <= 0.7:
            return '#FFFF00'    # Yellow for moderate
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
                '#acdbd7', '#9bcfd3', '#8ac4d0', '#78b9ce', '#63adcd', '#53a2cd', '#4596ce', '#3a89cf', 
                '#377ad0', '#3e6ccb', '#465ebe', '#4952ad', '#474792', '#42407b', '#3b3866', '#333153', 
                '#2b2a40', '#252532', '#212028', '#1f1e21', '#221e1e', '#2a1e20', '#352024', '#412329', 
                '#522731', '#622937', '#722c3d', '#842d42', '#992e44', '#ab3043', '#bb363f', '#c93f3a', 
                '#d74e35', '#e15c33', '#e86d35', '#ed7e40', '#f29255', '#f5a36a', '#f8b380', '#fcc396'
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
                '#d2f0db', '#c6ebd1', '#b9e6c7', '#abe2be', '#99ddb6', '#88d9b1', '#76d5ae', '#65d0ad', 
                '#55caad', '#4cc3ad', '#45bdad', '#3fb6ad', '#3aaead', '#37a8ac', '#35a1ab', '#349aaa', 
                '#3492a8', '#348ca7', '#3485a5', '#357ea4', '#3576a2', '#3670a0', '#36699f', '#38629d', 
                '#3a599a', '#3d5296', '#3f4b90', '#414488', '#413e7d', '#403872', '#3e3367', '#3b2e5d', 
                '#372851', '#342447', '#2f1f3d', '#2a1b33', '#241628', '#1e111f', '#180d16', '#12080d'
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
    # Initialize geojson_data to prevent NameError
    geojson_data = {"type": "FeatureCollection", "features": []}
    
    if st.session_state.heat_map_visibility:
        if heatmap_data:
            # Display pre-computed heatmap
            st.success("⚡ Displaying pre-computed heatmap - instant loading!")

            # Convert pre-computed data to GeoJSON for display
            # geojson_data already initialized above

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
                    
                    # Generate heatmaps sequentially with comprehensive clipping polygon
                    indicator_auto_fit = st.session_state.get('indicator_auto_fit', False)
                    indicator_range = st.session_state.get('indicator_range', 1500.0)
                    indicator_sill = st.session_state.get('indicator_sill', 0.25)
                    indicator_nugget = st.session_state.get('indicator_nugget', 0.1)
                    
                    success_count, stored_heatmap_ids, error_messages = generate_quad_heatmaps_sequential(
                        wells_data=st.session_state.wells_data,
                        click_point=st.session_state.selected_point,
                        search_radius=st.session_state.search_radius,
                        interpolation_method=st.session_state.interpolation_method,
                        polygon_db=st.session_state.polygon_db,
                        soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None,
                        new_clipping_polygon=st.session_state.new_clipping_polygon,
                        grid_size=st.session_state.get('grid_size', (2, 3)),
                        indicator_auto_fit=indicator_auto_fit,
                        indicator_range=indicator_range,
                        indicator_sill=indicator_sill,
                        indicator_nugget=indicator_nugget
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
                        else:
                            # Ensure geojson_data is still defined if no stored heatmaps
                            geojson_data = {"type": "FeatureCollection", "features": []}
                    
                    # Sequential processing and storage handled by the dedicated module
                except Exception as e:
                    print(f"CRITICAL ERROR in heatmap generation: {e}")
                    st.error(f"Error generating heatmaps: {e}")
                    geojson_data = {"type": "FeatureCollection", "features": []}
                    geojson_data_east = None
            # else clause removed as geojson_data is already initialized

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
                        'color': '',
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
                                    'color': '',
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
                                'color': '',
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
                            'color': '',
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
                        raw_stored_geojson = stored_heatmap.get('geojson_data')
                        stored_method = stored_heatmap.get('interpolation_method', 'kriging')
                        # Apply exclusion clipping for global color range calculation
                        if raw_stored_geojson:
                            from interpolation import apply_exclusion_clipping_to_stored_heatmap
                            stored_geojson = apply_exclusion_clipping_to_stored_heatmap(raw_stored_geojson, method_name=stored_method)
                        else:
                            stored_geojson = raw_stored_geojson
                            
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
    
    # For smooth raster style, collect ALL triangulated data first for unified processing
    if heatmap_style == "Smooth Raster (Windy.com Style)":
        combined_geojson = {"type": "FeatureCollection", "features": []}
        overall_bounds = {'north': float('-inf'), 'south': float('inf'), 
                         'east': float('-inf'), 'west': float('inf')}
        valid_heatmaps_for_raster = []
        
        # Collect all triangulated data from stored heatmaps
        for i, stored_heatmap in enumerate(st.session_state.stored_heatmaps):
            # Skip fresh heatmap to avoid duplication
            if fresh_heatmap_name and stored_heatmap.get('heatmap_name') == fresh_heatmap_name:
                continue
                
            raw_geojson_data = stored_heatmap.get('geojson_data')
            
            # Apply exclusion clipping ONLY to non-indicator methods for smooth raster
            method = stored_heatmap.get('interpolation_method', 'kriging')
            indicator_methods = [
                'indicator_kriging', 
                'indicator_kriging_spherical', 
                'indicator_kriging_spherical_continuous'
            ]
            
            if raw_geojson_data:
                from interpolation import apply_exclusion_clipping_to_stored_heatmap
                geojson_data = apply_exclusion_clipping_to_stored_heatmap(raw_geojson_data, method_name=method)
            else:
                geojson_data = raw_geojson_data
                
            if geojson_data and geojson_data.get('features'):
                # Add features to combined dataset
                for feature in geojson_data['features']:
                    # Fix compatibility: ensure stored data has both 'value' and 'yield' properties
                    if 'value' not in feature['properties'] and 'yield' in feature['properties']:
                        feature['properties']['value'] = feature['properties']['yield']
                    elif 'yield' not in feature['properties'] and 'value' in feature['properties']:
                        feature['properties']['yield'] = feature['properties']['value']
                    combined_geojson['features'].append(feature)
                
                # Update overall bounds to cover all heatmaps
                feature_count = 0
                coord_samples = []
                for feature in geojson_data['features']:
                    if feature['geometry']['type'] == 'Polygon':
                        coords = feature['geometry']['coordinates'][0]
                        for coord in coords:
                            lon, lat = coord[0], coord[1]
                            overall_bounds['west'] = min(overall_bounds['west'], lon)
                            overall_bounds['east'] = max(overall_bounds['east'], lon)
                            overall_bounds['south'] = min(overall_bounds['south'], lat)
                            overall_bounds['north'] = max(overall_bounds['north'], lat)
                            
                            # Sample first few coordinates for debugging
                            if len(coord_samples) < 5:
                                coord_samples.append((lat, lon))
                        feature_count += 1
                
                # DEBUG: Log coordinate samples from this heatmap
                if coord_samples:
                    heatmap_name = stored_heatmap.get('heatmap_name', 'Unknown')
                    print(f"🔧 TRIANGULATED BOUNDS DEBUG: {heatmap_name}")
                    print(f"🔧   Features: {feature_count}, Sample coords: {coord_samples[:3]}")
                
                valid_heatmaps_for_raster.append(stored_heatmap['heatmap_name'])
        
        # Generate single unified smooth raster if we have combined data
        if combined_geojson['features']:
            print(f"🌬️  UNIFIED SMOOTH RASTER: Combining {len(combined_geojson['features'])} triangles from {len(valid_heatmaps_for_raster)} heatmaps")
            print(f"🌬️  Heatmaps included: {', '.join(valid_heatmaps_for_raster)}")
            print(f"🌬️  Overall bounds: N={overall_bounds['north']:.3f}, S={overall_bounds['south']:.3f}, E={overall_bounds['east']:.3f}, W={overall_bounds['west']:.3f}")
            
            # Use the stored heatmap's interpolation method for colormap consistency
            method = st.session_state.stored_heatmaps[0].get('interpolation_method', 'kriging') if st.session_state.stored_heatmaps else 'kriging'
            
            # Create combined clipping polygon: include soil areas AND exclude red/orange zones
            combined_clipping_polygon = st.session_state.new_clipping_polygon
            
            # For non-indicator methods, subtract red/orange exclusion zones from clipping polygon
            if method not in ['indicator_kriging', 'indicator_kriging_spherical', 'indicator_kriging_spherical_continuous']:
                try:
                    # Load red/orange exclusion zones
                    import geopandas as gpd
                    exclusion_file_path = "attached_assets/red_orange_zones_stored_2025-09-16_1758401039896.geojson"
                    
                    if os.path.exists(exclusion_file_path):
                        exclusion_gdf = gpd.read_file(exclusion_file_path)
                        print(f"🚫 SMOOTH RASTER: Loaded {len(exclusion_gdf)} red/orange exclusion polygons for clipping")
                        
                        # If we have a base clipping polygon, subtract exclusion zones from it
                        if combined_clipping_polygon is not None and len(combined_clipping_polygon) > 0:
                            # Get the unary union of the exclusion zones
                            exclusion_union = exclusion_gdf.geometry.unary_union
                            
                            # Get the clipping polygon geometry  
                            if hasattr(combined_clipping_polygon, 'geometry'):
                                base_clipping_geom = combined_clipping_polygon.geometry.unary_union
                            else:
                                base_clipping_geom = combined_clipping_polygon
                            
                            # Subtract exclusion zones from clipping polygon using difference
                            try:
                                combined_clipping_geom = base_clipping_geom.difference(exclusion_union)
                                # Convert back to GeoDataFrame for consistency
                                combined_clipping_polygon = gpd.GeoDataFrame([1], geometry=[combined_clipping_geom], crs='EPSG:4326')
                                print(f"🚫 SMOOTH RASTER: Successfully created combined clipping polygon (soil areas minus red/orange zones)")
                            except Exception as e:
                                print(f"🚫 SMOOTH RASTER: Failed to subtract exclusion zones: {e}, using original clipping")
                        else:
                            print(f"🚫 SMOOTH RASTER: No base clipping polygon, red/orange exclusion not applied to smooth raster")
                    else:
                        print(f"🚫 SMOOTH RASTER: Red/orange exclusion file not found: {exclusion_file_path}")
                except Exception as e:
                    print(f"🚫 SMOOTH RASTER: Error applying red/orange exclusion clipping: {e}")
            
            # Generate single unified smooth raster across ALL triangulated data
            raster_overlay = generate_smooth_raster_overlay(
                combined_geojson, 
                overall_bounds, 
                raster_size=(512, 512), 
                global_colormap_func=lambda value: get_global_unified_color(value, method),
                opacity=st.session_state.get('heatmap_opacity', 0.7),
                clipping_polygon=combined_clipping_polygon
            )
            
            if raster_overlay:
                # Add single unified raster overlay to map
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{raster_overlay['image_base64']}",
                    bounds=raster_overlay['bounds'],
                    opacity=raster_overlay['opacity'],
                    name=f"Unified Smooth Raster ({len(valid_heatmaps_for_raster)} tiles)"
                ).add_to(m)
                stored_heatmap_count = len(valid_heatmaps_for_raster)
                print(f"🌬️  SUCCESS: Added unified smooth raster covering {len(valid_heatmaps_for_raster)} heatmap areas")
                print(f"🌬️  Using snapped boundary vertices from stored triangulated data - no gaps or overlaps")
            else:
                print(f"🌬️  FAILED: Could not generate unified smooth raster, falling back to individual processing")
        
        # Skip individual loop if unified raster was successful
        if stored_heatmap_count > 0:
            print(f"🌬️  UNIFIED PROCESSING COMPLETE: Skipping individual raster generation to avoid overlaps")
        else:
            print(f"🌬️  UNIFIED PROCESSING FAILED: Falling back to individual triangle mesh display")
    
    # PERFORMANCE OPTIMIZATION 1: Viewport-based filtering
    def get_heatmap_bounds(geojson_data):
        """Extract bounding box from GeoJSON data"""
        if not geojson_data or not geojson_data.get('features'):
            return None
        
        all_coords = []
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                all_coords.extend(coords)
        
        if not all_coords:
            return None
            
        lons = [coord[0] for coord in all_coords]
        lats = [coord[1] for coord in all_coords]
        return {
            'north': max(lats), 'south': min(lats),
            'east': max(lons), 'west': min(lons)
        }
    
    def bounds_intersect(bounds1, bounds2):
        """Check if two bounding boxes intersect"""
        if not bounds1 or not bounds2:
            return True  # If we can't determine bounds, include it
        return not (bounds1['east'] < bounds2['west'] or 
                   bounds1['west'] > bounds2['east'] or
                   bounds1['north'] < bounds2['south'] or 
                   bounds1['south'] > bounds2['north'])
    
    # Get current viewport (use Canterbury region as default)
    if 'map_bounds' not in st.session_state:
        # Default to Canterbury region bounds
        st.session_state.map_bounds = {
            'north': -42.5, 'south': -45.0,
            'east': 174.0, 'west': 169.0
        }
    
    # PERFORMANCE OPTIMIZATION 2: Filter heatmaps by viewport intersection
    visible_heatmaps = []
    total_heatmaps = len(st.session_state.stored_heatmaps)
    
    for stored_heatmap in st.session_state.stored_heatmaps:
        raw_geojson_data = stored_heatmap.get('geojson_data')
        if raw_geojson_data:
            heatmap_bounds = get_heatmap_bounds(raw_geojson_data)
            if bounds_intersect(heatmap_bounds, st.session_state.map_bounds):
                visible_heatmaps.append(stored_heatmap)
    
    print(f"🔍 VIEWPORT FILTERING: {len(visible_heatmaps)}/{total_heatmaps} heatmaps visible in current viewport")
    
    # PERFORMANCE OPTIMIZATION 5: Progressive loading with user feedback
    if len(visible_heatmaps) > 0:
        with st.spinner(f"⚡ Loading {len(visible_heatmaps)} visible heatmaps (optimized)..."):
            pass  # Visual feedback for user
    
    # PERFORMANCE OPTIMIZATION 3: Precomputed preprocessing cache
    def get_preprocessed_heatmap(heatmap_id, raw_geojson_data, method):
        """Get or create preprocessed heatmap data"""
        cache_key = f"preprocessed_{heatmap_id}_{method}_v2"
        
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        # Apply exclusion clipping and preprocessing
        if method not in ['indicator_kriging', 'indicator_kriging_spherical', 'indicator_kriging_spherical_continuous']:
            from interpolation import apply_exclusion_clipping_to_stored_heatmap
            processed_data = apply_exclusion_clipping_to_stored_heatmap(raw_geojson_data, method_name=method, heatmap_id=heatmap_id)
        else:
            processed_data = raw_geojson_data
        
        # Cache the result
        st.session_state[cache_key] = processed_data
        return processed_data

    def create_unified_clipping_geometry():
        """Create unified clipping geometry combining red/orange exclusions and NEW clipping polygon"""
        try:
            from shapely.ops import unary_union
            from shapely.geometry import Point
            import geopandas as gpd
            
            clipping_parts = []
            
            # Get NEW clipping polygon (allowed areas) 
            if 'new_clipping_polygon' in st.session_state and st.session_state.new_clipping_polygon is not None:
                new_polygon = st.session_state.new_clipping_polygon
                if hasattr(new_polygon, 'geometry') and len(new_polygon) > 0:
                    # Union all NEW clipping polygon geometries
                    allowed_geoms = [geom for geom in new_polygon.geometry if geom.is_valid]
                    if allowed_geoms:
                        allowed_union = unary_union(allowed_geoms)
                        print(f"🎯 UNIFIED CLIPPING: NEW polygon with {len(allowed_geoms)} parts loaded")
                        
                        # Get red/orange exclusion polygons to subtract from allowed areas
                        from interpolation import get_prepared_exclusion_union
                        exclusion_data = get_prepared_exclusion_union()
                        
                        if exclusion_data is not None:
                            exclusion_union, _, _, _ = exclusion_data
                            # Create final allowed geometry: NEW_areas - red/orange_exclusions
                            final_allowed = allowed_union.difference(exclusion_union)
                            print(f"🎯 UNIFIED CLIPPING: Applied red/orange exclusions to NEW polygon")
                            return final_allowed
                        else:
                            print(f"🎯 UNIFIED CLIPPING: Using NEW polygon only (no exclusions)")
                            return allowed_union
            
            print(f"🎯 UNIFIED CLIPPING: No clipping geometry available")
            return None
            
        except Exception as e:
            print(f"❌ ERROR creating unified clipping geometry: {e}")
            return None

    # PERFORMANCE OPTIMIZATION 4: Single raster overlay for triangulated display
    def create_combined_raster_overlay(visible_heatmaps, style):
        """Combine multiple heatmaps into a single raster overlay"""
        if not visible_heatmaps or style == "Smooth Raster (Windy.com Style)":
            return None
        
        try:
            print(f"🎨 CREATING COMBINED RASTER: Processing {len(visible_heatmaps)} visible heatmaps")
            
            # Combine all visible heatmap features
            combined_features = []
            for heatmap in visible_heatmaps:
                raw_geojson_data = heatmap.get('geojson_data')
                method = heatmap.get('interpolation_method', 'kriging')
                
                if raw_geojson_data:
                    processed_data = get_preprocessed_heatmap(heatmap['heatmap_name'], raw_geojson_data, method)
                    if processed_data and processed_data.get('features'):
                        # Normalize feature properties
                        for feature in processed_data['features']:
                            if 'properties' in feature:
                                # Ensure value property exists
                                if 'value' not in feature['properties'] and 'yield' in feature['properties']:
                                    feature['properties']['value'] = feature['properties']['yield']
                                elif 'yield' not in feature['properties'] and 'value' in feature['properties']:
                                    feature['properties']['yield'] = feature['properties']['value']
                        combined_features.extend(processed_data['features'])
            
            if not combined_features:
                return None
            
            # Create combined GeoJSON
            combined_geojson = {
                'type': 'FeatureCollection',
                'features': combined_features
            }
            
            # Calculate overall bounds
            all_coords = []
            for feature in combined_features:
                if feature['geometry']['type'] == 'Polygon':
                    coords = feature['geometry']['coordinates'][0]
                    all_coords.extend(coords)
            
            if not all_coords:
                return None
            
            lons = [coord[0] for coord in all_coords]
            lats = [coord[1] for coord in all_coords]
            bounds = {
                'north': max(lats), 'south': min(lats),
                'east': max(lons), 'west': min(lons)
            }
            
            # Generate single raster overlay
            from interpolation import generate_smooth_raster_overlay
            raster_overlay = generate_smooth_raster_overlay(
                combined_geojson, 
                bounds, 
                raster_size=(1024, 1024),  # Higher resolution for combined overlay
                global_colormap_func=lambda value: get_global_unified_color(value, 'combined'),
                opacity=st.session_state.get('heatmap_opacity', 0.7),
                clipping_polygon=create_unified_clipping_geometry()  # FIXED: Apply proper polygon clipping
            )
            
            return raster_overlay
            
        except Exception as e:
            print(f"❌ COMBINED RASTER FAILED: {e}")
            return None
    
    # Try to create combined raster overlay for better performance (only for Smooth Raster mode)
    combined_raster = None
    # NOTE: Combined raster should only be used for Smooth Raster mode, not Triangle Mesh mode
    # Triangle Mesh mode should always show individual triangular polygons, not raster overlays
    if len(visible_heatmaps) > 5 and heatmap_style == "Smooth Raster (Windy.com Style)":
        combined_raster = create_combined_raster_overlay(visible_heatmaps, heatmap_style)
        
    if combined_raster:
        # Add single combined raster overlay
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{combined_raster['image_base64']}",
            bounds=combined_raster['bounds'],
            opacity=combined_raster['opacity'],
            name=f"Combined Heatmap Overlay ({len(visible_heatmaps)} tiles)"
        ).add_to(m)
        stored_heatmap_count = len(visible_heatmaps)
        print(f"🎨 SUCCESS: Added combined raster overlay for {len(visible_heatmaps)} heatmaps")
    
    # Only run individual loop if not using unified smooth raster, combined raster failed, or few heatmaps
    elif heatmap_style != "Smooth Raster (Windy.com Style)" or stored_heatmap_count == 0:
        # PERFORMANCE OPTIMIZATION 6: For Triangle Mesh mode, use reasonable batch sizes to prevent WebSocket timeouts
        # Process in batches to avoid overwhelming the browser and WebSocket connection
        max_individual_layers = 50  # Reasonable limit to prevent WebSocket errors while still showing many heatmaps
        heatmaps_to_process = visible_heatmaps[:max_individual_layers]
        
        if len(visible_heatmaps) > max_individual_layers:
            print(f"📐 TRIANGLE MESH BATCH: Processing {max_individual_layers}/{len(visible_heatmaps)} visible heatmaps to prevent WebSocket overload")
            print(f"📐 Use zoom/pan to see different heatmaps, or reduce heatmap count for full display")
        else:
            print(f"📐 TRIANGLE MESH: Processing all {len(visible_heatmaps)} visible heatmaps as individual triangular polygons")
        
        for i, stored_heatmap in enumerate(heatmaps_to_process):
            try:
                # Don't skip the current fresh heatmap - let it display as a stored heatmap too
                # This ensures continuity when the page re-renders
                if fresh_heatmap_name and stored_heatmap.get('heatmap_name') == fresh_heatmap_name:
                    print(f"DISPLAYING stored version of fresh heatmap: {stored_heatmap['heatmap_name']}")
                # All stored heatmaps should display

                # OPTIMIZATION: Use preprocessed data instead of reprocessing
                raw_geojson_data = stored_heatmap.get('geojson_data')
                heatmap_data = stored_heatmap.get('heatmap_data', [])
                method = stored_heatmap.get('interpolation_method', 'kriging')
                
                if raw_geojson_data:
                    # Use preprocessed data with caching
                    geojson_data = get_preprocessed_heatmap(stored_heatmap['heatmap_name'], raw_geojson_data, method)
                    print(f"🚀 OPTIMIZED: Using preprocessed data for {stored_heatmap['heatmap_name']}")
                else:
                    geojson_data = raw_geojson_data

                # Process the heatmap data
                
                if geojson_data and geojson_data.get('features'):
                    print(f"Adding stored GeoJSON heatmap {i+1}: {stored_heatmap['heatmap_name']} with {len(geojson_data['features'])} triangular features")

                    # Fix compatibility: ensure stored data has both 'value' and 'yield' properties
                    # Handle multiple possible property names for ground water level heatmaps
                    value_keys_to_check = ['value', 'yield', 'ground_water_level', 'ground water level', 'z', 'prediction', 'gwl_value', 'depth', 'level']
                    
                    for feature in geojson_data['features']:
                        if 'properties' in feature:
                            # Find the first available numeric value from possible property names
                            found_value = None
                            found_key = None
                            
                            for key in value_keys_to_check:
                                if key in feature['properties']:
                                    val = feature['properties'][key]
                                    if isinstance(val, (int, float)) and not (isinstance(val, bool)):
                                        found_value = val
                                        found_key = key
                                        break
                            
                            # If we found a value, ensure both 'value' and 'yield' exist
                            if found_value is not None:
                                feature['properties']['value'] = found_value
                                feature['properties']['yield'] = found_value
                                if found_key not in ['value', 'yield']:
                                    print(f"  PROPERTY NORMALIZATION: Found data in '{found_key}' property, normalized to 'value' and 'yield'")
                            else:
                                # Log what properties are available for debugging
                                available_props = list(feature['properties'].keys())
                                print(f"  WARNING: No numeric value found in feature properties: {available_props}")
                                # Set default values to prevent errors
                                feature['properties']['value'] = 0
                                feature['properties']['yield'] = 0

                    # Use the UPDATED global unified color function with method info
                    method = stored_heatmap.get('interpolation_method', 'kriging')
                    
                    # Apply global color mapping to heatmap

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
                                        'color': '',
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
                                    'color': '',
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
                                'color': '',
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
        if (st.session_state.interpolation_method == 'indicator_kriging' or 
            st.session_state.interpolation_method == 'indicator_kriging_spherical' or 
            st.session_state.interpolation_method == 'indicator_kriging_spherical_continuous'):
            # Four-tier indicator kriging legend
            caption_text = 'Well Yield Quality: Red = Poor (0-0.4), Orange = Low-Moderate (0.4-0.6), Yellow = Moderate (0.6-0.7), Green = Good (0.7-1.0)'
            colormap = folium.StepColormap(
                colors=['#FF0000', '#FF8000', '#FFFF00', '#00FF00'],  # Red, Orange, Yellow, Green
                vmin=0,
                vmax=1.0,
                index=[0, 0.4, 0.6, 0.7, 1.0],  # Four-tier thresholds
                caption=caption_text
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
                    tooltip=f"Well {row.get('well_id', 'Unknown')} - {row.get('yield_rate', 'N/A')} L/s - Depth: {row.get('depth', 'N/A'):.1f}m - Indicator: {classify_well_viability(row)}"
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

st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p>© 2023 Groundwater Finder |Data sourced from public well databases</p>
    <p>This tool is designed to assist farmers in locating potential groundwater sources. Results are based on existing data and interpolation techniques.</p>
</div>
""", unsafe_allow_html=True)