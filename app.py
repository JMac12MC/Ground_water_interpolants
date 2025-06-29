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
if 'total_wells_in_radius' not in st.session_state:
    st.session_state.total_wells_in_radius = 0
if 'heat_map_visibility' not in st.session_state:
    st.session_state.heat_map_visibility = True
if 'well_markers_visibility' not in st.session_state:
    st.session_state.well_markers_visibility = True
if 'search_radius' not in st.session_state:
    st.session_state.search_radius = 10
if 'selected_well' not in st.session_state:
    st.session_state.selected_well = None
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
        "Local Context Radius (km)",
        min_value=1,
        max_value=50,
        value=st.session_state.search_radius,
        step=1,
        help="Radius for showing nearby wells and local analysis when you click on the map"
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
            "Kriging Variance (Yield Uncertainty)",
            "Kriging Variance (Depth Uncertainty)",
            "Depth to Groundwater (Standard Kriging)",
            "Depth to Groundwater (Auto-Fitted Spherical)"
        ],
        index=0,
        help="Choose the visualization type: yield estimation, uncertainty analysis, or depth analysis"
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
    elif visualization_method == "Kriging Variance (Yield Uncertainty)":
        st.session_state.interpolation_method = 'kriging_variance'
        st.session_state.show_kriging_variance = True
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
        st.session_state.variance_type = 'yield'
    elif visualization_method == "Kriging Variance (Depth Uncertainty)":
        st.session_state.interpolation_method = 'kriging_variance'
        st.session_state.show_kriging_variance = True
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
        st.session_state.variance_type = 'depth'
    elif visualization_method == "Depth to Groundwater (Standard Kriging)":
        st.session_state.interpolation_method = 'depth_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = False
    elif visualization_method == "Depth to Groundwater (Auto-Fitted Spherical)":
        st.session_state.interpolation_method = 'depth_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'

    # Display options
    st.header("Display Options")
    st.session_state.heat_map_visibility = st.checkbox("Show Heat Map", value=st.session_state.heat_map_visibility)
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=st.session_state.well_markers_visibility)
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")

    # Add explanation for kriging variance
    if visualization_method in ["Kriging Variance (Yield Uncertainty)", "Kriging Variance (Depth Uncertainty)"]:
        variance_type = "yield" if "Yield" in visualization_method else "depth to groundwater"
        st.info(f"""
        **Kriging Variance Visualization ({variance_type.title()})**

        This shows the prediction uncertainty (variance) of the kriging interpolation for {variance_type}:
        - 🟢 **Green areas**: Low uncertainty - high confidence in predictions
        - 🟡 **Yellow areas**: Medium uncertainty - moderate confidence
        - 🔴 **Red areas**: High uncertainty - low confidence, more wells needed

        Use this to identify where additional wells would most improve prediction accuracy for {variance_type} estimates.
        """)

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

    # Load and display pre-computed heatmaps if available
    heatmap_data = None
    if st.session_state.polygon_db and st.session_state.polygon_db.pg_engine:
        try:
            # Try to load pre-computed heatmap data
            if visualization_method in ["Standard Kriging (Yield)", "Yield Kriging (Spherical)"]:
                heatmap_data = st.session_state.polygon_db.get_heatmap_data('yield')
            elif "Depth" in visualization_method:
                heatmap_data = st.session_state.polygon_db.get_heatmap_data('depth')

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
            # Calculate distances from selected point for local well display
            wells_df['distance'] = wells_df.apply(
                lambda row: get_distance(
                    st.session_state.selected_point[0], 
                    st.session_state.selected_point[1], 
                    row['latitude'], 
                    row['longitude']
                ), 
                axis=1
            )

            # Filter wells for local display (still show nearby wells for context)
            filtered_wells = wells_df[
                (wells_df['distance'] <= st.session_state.search_radius)
            ]

            # Ensure all missing yield values are replaced with 0
            if 'yield_rate' in filtered_wells.columns:
                filtered_wells = filtered_wells.copy()
                filtered_wells['yield_rate'] = filtered_wells['yield_rate'].fillna(0)

            st.session_state.filtered_wells = filtered_wells.copy()
            st.session_state.total_wells_in_radius = len(filtered_wells)

            # Create marker for selected point
            folium.Marker(
                location=st.session_state.selected_point,
                popup="Selected Location",
                icon=folium.Icon(color='red', icon='crosshairs', prefix='fa'),
                tooltip="Your Selected Point"
            ).add_to(m)

            # Draw circle for search radius (now just for local context)
            folium.Circle(
                location=st.session_state.selected_point,
                radius=st.session_state.search_radius * 1000,
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
                fill_opacity=0.1
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

                    # Check if we're showing kriging variance
                    if st.session_state.interpolation_method == 'kriging_variance':
                        # Variance calculations (unchanged)
                        variance_method = 'depth_kriging' if st.session_state.get('variance_type', 'yield') == 'depth' else 'kriging'

                        if st.session_state.get('variance_type', 'yield') == 'depth':
                            depth_wells = st.session_state.filtered_wells.copy()
                            if 'is_dry_well' in depth_wells.columns:
                                depth_wells = depth_wells[~depth_wells['is_dry_well']]
                            if 'depth_to_groundwater' in depth_wells.columns:
                                depth_wells = depth_wells[depth_wells['depth_to_groundwater'].notna() & (depth_wells['depth_to_groundwater'] > 0)]
                            else:
                                depth_wells = depth_wells[depth_wells['depth'].notna() & (depth_wells['depth'] > 0)]

                            variance_data = calculate_kriging_variance(
                                depth_wells,
                                st.session_state.selected_point,
                                st.session_state.search_radius,
                                resolution=100,
                                variogram_model=st.session_state.get('variogram_model', 'spherical'),
                                use_depth=True,
                                soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None
                            )
                        else:
                            variance_data = calculate_kriging_variance(
                                st.session_state.filtered_wells.copy(),
                                st.session_state.selected_point,
                                st.session_state.search_radius,
                                resolution=100,
                                variogram_model=st.session_state.get('variogram_model', 'spherical'),
                                use_depth=False,
                                soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None
                            )

                        # Convert variance data to GeoJSON format
                        geojson_data = {"type": "FeatureCollection", "features": []}
                        if variance_data:
                            from scipy.spatial import Delaunay
                            import numpy as np

                            points_2d = np.array([[point[1], point[0]] for point in variance_data])
                            variances = np.array([point[2] for point in variance_data])

                            if len(points_2d) > 3:
                                tri = Delaunay(points_2d)

                                for simplex in tri.simplices:
                                    vertices = points_2d[simplex]
                                    vertex_variances = variances[simplex]
                                    avg_variance = float(np.mean(vertex_variances))

                                    if avg_variance > 0.0001:
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
                                                "variance": avg_variance,
                                                "yield": avg_variance
                                            }
                                        }
                                        geojson_data["features"].append(poly)
                    else:
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

                        if st.session_state.interpolation_method == 'kriging_variance':
                            # Variance colors: green (low uncertainty) to red (high uncertainty)
                            colors = [
                                '#00ff00',  # Green (low uncertainty)
                                '#33ff00',
                                '#66ff00',
                                '#99ff00',
                                '#ccff00',                                '#ffff00',  # Yellow
                                '#ffcc00',
                                '#ff9900',
                                '#ff6600',
                                '#ff3300',
                                '#ff0000',  # Red (high uncertainty)
                                '#cc0000',
                                '#990000',
                                '#660000',
                                '#330000'   # Dark red (very high uncertainty)
                            ]
                        elif st.session_state.interpolation_method == 'depth_kriging':
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
                    if st.session_state.interpolation_method == 'kriging_variance':
                        # Kriging variance legend
                        variance_type_label = "Depth" if st.session_state.get('variance_type', 'yield') == 'depth' else "Yield"
                        colormap = folium.LinearColormap(
                            colors=['#00ff00', '#33ff00', '#66ff00', '#99ff00', '#ccff00', 
                                    '#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff3300', 
                                    '#ff0000', '#cc0000', '#990000', '#660000', '#330000'],
                            vmin=0,
                            vmax=float(max_value),
                            caption=f'Kriging Uncertainty ({variance_type_label}) - 15 Bands'
                        )
                    elif st.session_state.interpolation_method == 'depth_kriging':
                        # Depth legend with depth-appropriate colors
                        colormap = folium.LinearColormap(
                            colors=['#00ff00', '#33ff00', '#66ff00', '#99ff00', '#ccff00', 
                                    '#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff3300', 
                                    '#ff0000', '#cc0000', '#990000', '#660000', '#330000'],
                            vmin=0,
                            vmax=float(max_value),
                            caption='Depth to Groundwater (m) - 15 Bands'
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
                    if st.session_state.interpolation_method == 'kriging_variance':
                        tooltip_field = 'variance'
                        variance_type_label = "Depth" if st.session_state.get('variance_type', 'yield') == 'depth' else "Yield"
                        tooltip_label = f'{variance_type_label} Variance:'
                    elif st.session_state.interpolation_method == 'depth_kriging':
                        tooltip_field = 'yield'
                        tooltip_label = 'Depth (m):'
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

                    # Filter out geotechnical/geological investigation wells and wells with no depth

                    # Add well markers
                    for index, well in st.session_state.filtered_wells.iterrows():
                        if pd.notna(well['depth']):
                            # Add a regular marker
                            popup_text = f"""
                            <b>Well Details</b><br>
                            Well ID: {well['WELL_ID']}<br>
                            Depth: {well['depth']} m<br>
                            Yield: {well['yield_rate']} L/s<br>
                            Distance: {well['distance']:.2f} km
                            """

                            folium.Marker(
                                location=[well['latitude'], well['longitude']],
                                popup=folium.Popup(popup_text, max_width=300),
                                icon=folium.Icon(color='blue', icon='tint', prefix='fa'),
                                tooltip=f"Well ID: {well['WELL_ID']}"
                            ).add_to(radius_wells_layer)

    # Add click handler for map interaction
    m.add_child(folium.LatLngPopup())
    m.add_child(folium.ClickForMarker(popup="Selected Location"))

    # Add layer control
    folium.LayerControl().add_to(m)

    # Display map with error handling
    try:
        map_data = folium_static(m, width=None, height=600)

        # Debug information
        if st.session_state.selected_point:
            st.write(f"📍 Map centered at: {st.session_state.selected_point[0]:.4f}, {st.session_state.selected_point[1]:.4f}")
        else:
            st.write(f"📍 Map centered at: {default_location[0]:.4f}, {default_location[1]:.4f}")

    except Exception as e:
        st.error(f"Map loading error: {e}")
        st.info("If the map doesn't load, try refreshing the page or checking your internet connection.")