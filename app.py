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
from interpolation import generate_heat_map_data, generate_geo_json_grid, calculate_kriging_variance
from database import PolygonDatabase
from regional_heatmap import generate_default_regional_heatmap, RegionalHeatmapGenerator

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
            <h1 style="font-size: 2.5rem; font-weight: bold; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">Groundwater Mapper</h1>
            <p style="font-size: 1.2rem; margin-bottom: 0; opacity: 0.9;">Helping you understand water resources using scientific data</p>
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
    st.session_state.search_radius = 10

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
if 'regional_heatmap_data' not in st.session_state:
    st.session_state.regional_heatmap_data = None
if 'show_regional_heatmap' not in st.session_state:
    st.session_state.show_regional_heatmap = True
if 'regional_heatmap_generator' not in st.session_state:
    st.session_state.regional_heatmap_generator = RegionalHeatmapGenerator()

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
            "Ground Water Level (Spherical Kriging)"
        ],
        index=0,
        help="Choose the visualization type: yield estimation, depth analysis, or groundwater level"
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

    # Regional Heatmap Controls
    st.header("Regional Background")
    st.session_state.show_regional_heatmap = st.checkbox(
        "Show Regional Groundwater Depth Heatmap",
        value=st.session_state.show_regional_heatmap,
        help="Display a high-resolution background heatmap covering the entire Canterbury region using all available well data"
    )
    
    # Generate regional heatmap button
    if st.button("Generate Regional Heatmap"):
        if st.session_state.wells_data is not None:
            with st.spinner("Generating regional heatmap (this may take several minutes)..."):
                # Generate comprehensive regional heatmap
                st.session_state.regional_heatmap_data = generate_default_regional_heatmap(
                    st.session_state.wells_data, 
                    st.session_state.soil_polygons
                )
                if st.session_state.regional_heatmap_data:
                    st.success(f"Generated regional heatmap with {len(st.session_state.regional_heatmap_data)} data points")
                else:
                    st.error("Failed to generate regional heatmap")
        else:
            st.warning("No wells data available for regional heatmap generation")
    
    # Load existing regional heatmap if available
    if st.session_state.regional_heatmap_data is None and st.session_state.wells_data is not None:
        # Try to load cached regional heatmap
        cached_heatmap = st.session_state.regional_heatmap_generator.load_regional_heatmap()
        if cached_heatmap:
            # Apply soil polygon mask
            if st.session_state.soil_polygons is not None:
                cached_heatmap = st.session_state.regional_heatmap_generator.apply_soil_polygon_mask(
                    cached_heatmap, st.session_state.soil_polygons
                )
            st.session_state.regional_heatmap_data = cached_heatmap
            st.sidebar.success(f"Loaded cached regional heatmap ({len(cached_heatmap)} points)")
    
    # Show regional heatmap status
    if st.session_state.regional_heatmap_data:
        st.sidebar.info(f"✓ Regional heatmap ready: {len(st.session_state.regional_heatmap_data):,} data points")
    elif st.session_state.show_regional_heatmap:
        st.sidebar.warning("Regional heatmap not available. Click 'Generate Regional Heatmap' to create one.")



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

    # Display options
    st.header("Display Options")
    st.session_state.heat_map_visibility = st.checkbox("Show Heat Map", value=st.session_state.heat_map_visibility)
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=st.session_state.well_markers_visibility)
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")



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

    # Add regional groundwater depth heatmap as background layer
    if st.session_state.show_regional_heatmap and st.session_state.regional_heatmap_data:
        try:
            # Add regional heatmap as a background layer
            HeatMap(
                data=st.session_state.regional_heatmap_data,
                name="Regional Groundwater Depth",
                min_opacity=0.3,
                max_opacity=0.6,
                radius=8,
                blur=5,
                gradient={
                    0.0: '#000080',  # Deep blue (shallow)
                    0.2: '#0066FF',  # Blue
                    0.4: '#00CCFF',  # Light blue
                    0.6: '#FFFF00',  # Yellow (medium depth)
                    0.8: '#FF9900',  # Orange
                    1.0: '#FF0000'   # Red (deep)
                }
            ).add_to(m)
            
        except Exception as e:
            st.warning(f"Could not display regional heatmap: {e}")

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

            # Ensure all missing yield values are replaced with 0
            if 'yield_rate' in filtered_wells.columns:
                filtered_wells = filtered_wells.copy()
                filtered_wells['yield_rate'] = filtered_wells['yield_rate'].fillna(0)

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
                            soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None
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

                    folium.GeoJson(
                        data=geojson_data,
                        name='Interpolation Visualization',
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
            # Update session state with the new coordinates
            st.session_state.selected_point = [clicked_lat, clicked_lng]
            # Clear filtered wells to trigger recalculation
            st.session_state.filtered_wells = None
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