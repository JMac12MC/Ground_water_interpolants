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
from interpolation import generate_heat_map_data, generate_geo_json_grid

# Set page configuration
st.set_page_config(
    page_title="Groundwater Mapper",
    page_icon="💧",
    layout="wide",
)

# Clean title header without image
def add_banner():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h1 style="font-size: 2.5rem; font-weight: bold; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">Groundwater Finder</h1>
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

# Add banner
add_banner()

# Sidebar - for options and filters
with st.sidebar:
    st.header("Data Options")

    # Load Canterbury wells data
    if st.session_state.wells_data is None:
        with st.spinner("Loading Canterbury wells data..."):
            st.session_state.wells_data = load_nz_govt_data()
    
    # Load soil drainage polygons
    if st.session_state.soil_polygons is None:
        try:
            with st.spinner("Loading soil drainage polygons..."):
                # Set GDAL config to restore corrupted .shx files
                import os
                os.environ['SHAPE_RESTORE_SHX'] = 'YES'
                
                # Load the shapefile with error handling
                soil_gdf = gpd.read_file("attached_assets/s-map-soil-drainage-aug-2024_1749379069732.shp")
                
                # Convert to WGS84 if needed
                if soil_gdf.crs and soil_gdf.crs.to_string() != 'EPSG:4326':
                    soil_gdf = soil_gdf.to_crs('EPSG:4326')
                elif not soil_gdf.crs:
                    # Assume it's already in WGS84 if no CRS is defined
                    soil_gdf.crs = 'EPSG:4326'
                
                # Take a sample of polygons for performance (first 1100)
                st.session_state.soil_polygons = soil_gdf.head(1100)
                st.success(f"Loaded {len(st.session_state.soil_polygons)} soil drainage polygons")
        except Exception as e:
            st.warning(f"Could not load soil polygons: {str(e)}")
            st.warning("Shapefile may be corrupted. You can try re-uploading the shapefile data.")
            st.session_state.soil_polygons = None

    # Advanced option for uploading custom data (hidden in expander)
    with st.expander("Upload Custom Data (Optional)"):
        uploaded_file = st.file_uploader("Upload a CSV file with well data", type=["csv"])
        if uploaded_file is not None:
            with st.spinner("Loading custom data..."):
                st.session_state.wells_data = load_custom_data(uploaded_file)

    st.header("Filters")

    # Radius filter
    st.session_state.search_radius = st.slider(
        "Search Radius (km)",
        min_value=1,
        max_value=50,
        value=st.session_state.search_radius,
        step=1
    )

    # Informational note about wells with missing yield data
    st.write("**NOTE:** All wells within the search radius are displayed.")
    st.write("Wells with missing yield values are treated as dry wells.")

    # Visualization method selection - single dropdown for all options
    st.header("Analysis Options")
    visualization_method = st.selectbox(
        "Map Visualization Type",
        options=[
            "Standard Kriging (Yield)", 
            "Random Forest + Kriging (Yield)",
            "Kriging Uncertainty (Yield)",
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
    elif visualization_method == "Random Forest + Kriging (Yield)":
        st.session_state.interpolation_method = 'rf_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = False
    elif visualization_method == "Kriging Uncertainty (Yield)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = True
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

    # Display options
    st.header("Display Options")
    st.session_state.heat_map_visibility = st.checkbox("Show Heat Map", value=st.session_state.heat_map_visibility)
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=st.session_state.well_markers_visibility)
    if st.session_state.soil_polygons is not None:
        st.session_state.show_soil_polygons = st.checkbox("Show Soil Drainage Areas", value=st.session_state.show_soil_polygons, help="Shows areas suitable for groundwater")
    
    # Add explanation for kriging uncertainty
    if visualization_method == "Kriging Uncertainty (Yield)":
        st.info("""
        **Kriging Uncertainty Visualization**
        
        This shows the prediction uncertainty (standard deviation) of the kriging interpolation:
        - 🟢 **Green areas**: High confidence in yield predictions
        - 🟡 **Yellow areas**: Medium confidence 
        - 🔴 **Red areas**: Low confidence, more wells needed
        
        Use this to identify where additional wells would most improve prediction accuracy.
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

    # Process wells data if available
    if st.session_state.wells_data is not None:
        wells_df = st.session_state.wells_data

        # Filter based on yield if selected point exists
        if st.session_state.selected_point:
            # Calculate distances from selected point
            wells_df['distance'] = wells_df.apply(
                lambda row: get_distance(
                    st.session_state.selected_point[0], 
                    st.session_state.selected_point[1], 
                    row['latitude'], 
                    row['longitude']
                ), 
                axis=1
            )

            # ONLY filter by distance - show ALL wells in the search radius regardless of yield
            filtered_wells = wells_df[
                (wells_df['distance'] <= st.session_state.search_radius)
            ]

            # Ensure all missing yield values are replaced with 0
            # This treats missing yield data as 0 instead of filtering them out
            if 'yield_rate' in filtered_wells.columns:
                # Need to replace NaN yield values with 0 - using proper pandas approach
                filtered_wells = filtered_wells.copy()  # Make a copy to avoid SettingWithCopyWarning
                filtered_wells['yield_rate'] = filtered_wells['yield_rate'].fillna(0)

            # Store all wells in radius - NO yield filtering whatsoever
            # Clear the previous filtered_wells to force complete recalculation
            st.session_state.filtered_wells = filtered_wells.copy()  # Make a copy to ensure clean data

            # Store total count
            st.session_state.total_wells_in_radius = len(filtered_wells)

            # Create marker for selected point
            folium.Marker(
                location=st.session_state.selected_point,
                popup="Selected Location",
                icon=folium.Icon(color='red', icon='crosshairs', prefix='fa'),
                tooltip="Your Selected Point"
            ).add_to(m)

            # Draw circle for search radius
            folium.Circle(
                location=st.session_state.selected_point,
                radius=st.session_state.search_radius * 1000,  # Convert km to meters
                color="#3186cc",
                fill=True,
                fill_color="#3186cc",
                fill_opacity=0.1
            ).add_to(m)

            # Add heat map based on yield
            if st.session_state.heat_map_visibility and isinstance(filtered_wells, pd.DataFrame) and not filtered_wells.empty:
                # Show progress overlay during processing with better visibility
                progress_container = st.container()
                with progress_container:
                    st.info("🔄 **Analysis in Progress** - Please wait while we process the data...")

                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.write("📍 **Step 1/4:** Processing well locations...")
                    progress_bar.progress(25)

                # Generate proper GeoJSON grid with interpolated yield values
                geojson_data = generate_geo_json_grid(
                    filtered_wells.copy(), 
                    st.session_state.selected_point, 
                    st.session_state.search_radius,
                    resolution=100,  # Higher resolution for smoother appearance
                    method=st.session_state.interpolation_method,
                    show_variance=st.session_state.show_kriging_variance,
                    auto_fit_variogram=st.session_state.get('auto_fit_variogram', False),
                    variogram_model=st.session_state.get('variogram_model', 'spherical'),
                    soil_polygons=st.session_state.soil_polygons if st.session_state.show_soil_polygons else None
                )

                progress_bar.progress(75)

                if geojson_data and len(geojson_data['features']) > 0:
                    # Calculate max value for setting the color scale
                    max_value = 0
                    for feature in geojson_data['features']:
                        max_value = max(max_value, feature['properties']['yield'])

                    # Ensure reasonable minimum for visualization
                    max_value = max(max_value, 20.0)

                    # Instead of choropleth, use direct GeoJSON styling for more control
                    # This allows us to precisely map values to colors

                    # Define colors based on what we're displaying
                    def get_color(value):
                        # Create 15-band color scale
                        step = max_value / 15.0

                        if st.session_state.show_kriging_variance:
                            # Variance colors: blue (low uncertainty) to red (high uncertainty)
                            colors = [
                                '#0000ff',  # Blue (low uncertainty)
                                '#0033ff',
                                '#0066ff',
                                '#0099ff',
                                '#00ccff',
                                '#00ffff',  # Cyan
                                '#33ffcc',
                                '#66ff99',
                                '#99ff66',
                                '#ccff33',
                                '#ffff00',  # Yellow
                                '#ffcc00',
                                '#ff9900',
                                '#ff6600',
                                '#ff0000',  # Red (high uncertainty)
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
                    folium.GeoJson(
                        data=geojson_data,
                        name='Yield Interpolation',
                        style_function=lambda feature: {
                            'fillColor': get_color(feature['properties']['yield']),
                            'color': 'none',
                            'weight': 0,
                            'fillOpacity': 0.7
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['yield'],
                            aliases=['Yield (L/s):'],
                            labels=True,
                            sticky=False
                        )
                    ).add_to(m)

                    # Add 15-band colormap legend to match the visualization
                    if st.session_state.show_kriging_variance:
                        # Variance legend with different colors
                        colormap = folium.LinearColormap(
                            colors=['#0000ff', '#0033ff', '#0066ff', '#0099ff', '#00ccff', 
                                    '#00ffff', '#33ffcc', '#66ff99', '#99ff66', '#ccff33', 
                                    '#ffff00', '#ffcc00', '#ff9900', '#ff6600', '#ff0000'],
                            vmin=0,
                            vmax=float(max_value),
                            caption='Kriging Uncertainty (Variance) - 15 Bands'
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

                    progress_bar.progress(100)
                    status_text.text('Heatmap generation complete!')

                    # Clear progress indicators after a moment
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    # Add tooltips to show appropriate values on hover
                    style_function = lambda x: {'fillColor': 'transparent', 'color': 'transparent'}
                    highlight_function = lambda x: {'fillOpacity': 0.8}

                    # Determine tooltip label based on visualization type
                    if st.session_state.show_kriging_variance:
                        tooltip_field = 'yield'
                        tooltip_label = 'Variance:'
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


            # ONLY show wells within the search radius when a point is selected
            if st.session_state.well_markers_visibility:
                # Only add wells within the search radius (filtered_wells)
                radius_wells_layer = folium.FeatureGroup(name="Wells Within Radius").add_to(m)

                # Create small dot markers for ONLY wells within the radius
                for idx, row in filtered_wells.iterrows():
                    # Use a smaller CircleMarker for wells within radius
                    folium.CircleMarker(
                        location=(float(row['latitude']), float(row['longitude'])),
                        radius=3,  # Small dot
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

    # Use st_folium instead of folium_static to capture clicks
    from streamlit_folium import st_folium

    # Make sure we disable folium_static's existing click handlers
    m.add_child(folium.Element("""
    <script>
    // Clear any existing click handlers
    </script>
    """))

    # Use st_folium with return_clicked_latlon to get click coordinates
    map_data = st_folium(
        m,
        use_container_width=True,
        height=600,
        key="interactive_map",
        returned_objects=["last_clicked"]
    )

    # Process clicks from the map
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        # Get the coordinates from the click
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]

        # Only update if this is a new location
        current_point = st.session_state.selected_point
        if not current_point or (abs(current_point[0] - clicked_lat) > 0.0001 or abs(current_point[1] - clicked_lng) > 0.0001):
            # Update session state with the new coordinates
            st.session_state.selected_point = [clicked_lat, clicked_lng]
            # Clear filtered wells to trigger recalculation
            st.session_state.filtered_wells = None
            # Force a rerun to process the new location
            st.rerun()

    # Add comprehensive well data summary report
    if st.session_state.filtered_wells is not None and len(st.session_state.filtered_wells) > 0:
        st.subheader("📊 Well Data Summary Report")

        # Create summary statistics
        wells_data = st.session_state.filtered_wells

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Wells Found", 
                len(wells_data),
                help="Number of wells within the selected radius"
            )

            # Filter for active productive wells only (exclude dry wells)
            active_productive_wells = wells_data[
                (wells_data['yield_rate'] > 0) & 
                (wells_data['status'].str.contains('Active', case=False, na=False)) &
                (~wells_data.get('is_dry_well', False))
            ]
            
            # Use depth_to_groundwater if available, otherwise fall back to depth
            if 'depth_to_groundwater' in active_productive_wells.columns and not active_productive_wells['depth_to_groundwater'].isna().all():
                depth_column = 'depth_to_groundwater'
                active_depths = active_productive_wells[depth_column].dropna()
            else:
                depth_column = 'depth'
                active_depths = active_productive_wells[depth_column].dropna()
            
            if len(active_depths) > 0:
                avg_depth = active_depths.mean()
                min_depth = active_depths.min()
                max_depth = active_depths.max()
                
                st.metric(
                    "Average Depth to Groundwater", 
                    f"{avg_depth:.1f} m",
                    help="Mean depth to groundwater for active productive wells only"
                )
            else:
                st.metric(
                    "Average Depth to Groundwater", 
                    "No data",
                    help="No active productive wells with depth data found"
                )

        with col2:
            # Yield statistics  
            yields = wells_data['yield_rate'].fillna(0)
            avg_yield = yields.mean()
            productive_wells = len(wells_data[wells_data['yield_rate'] > 1])

            st.metric(
                "Average Yield Rate", 
                f"{avg_yield:.2f} L/s",
                help="Mean water yield across all wells"
            )

            st.metric(
                "Productive Wells", 
                f"{productive_wells}",
                help="Wells with yield > 1 L/s"
            )

        with col3:
            # Show active wells count and depth range for active productive wells
            active_wells_count = len(active_productive_wells)
            st.metric(
                "Active Wells", 
                f"{active_wells_count}",
                help="Active wells with yield > 0 L/s"
            )
            
            if len(active_depths) > 0:
                st.metric(
                    "Groundwater Depth Range", 
                    f"{min_depth:.1f} - {max_depth:.1f} m",
                    help="Depth to groundwater range for active productive wells"
                )
            else:
                st.metric(
                    "Groundwater Depth Range", 
                    "No data",
                    help="No active productive wells with depth data"
                )

            high_yield_wells = len(wells_data[wells_data['yield_rate'] > 5])

            st.metric(
                "High-Yield Wells", 
                f"{high_yield_wells}",
                help="Wells with yield > 5 L/s"
            )

        # Detailed data table
        st.subheader("📋 Detailed Well Information")

        # Display top wells by yield
        top_wells = wells_data.nlargest(10, 'yield_rate')[['well_id', 'yield_rate', 'depth', 'distance', 'status']]
        st.write("**Top 10 Wells by Yield Rate:**")
        st.dataframe(
            top_wells,
            column_config={
                "well_id": "Well ID",
                "yield_rate": st.column_config.NumberColumn("Yield (L/s)", format="%.2f"),
                "depth": st.column_config.NumberColumn("Depth (m)", format="%.1f"),
                "distance": st.column_config.NumberColumn("Distance (km)", format="%.2f"),
                "status": "Status"
            },
            hide_index=True
        )

        # Downloadable CSV
        csv_data = wells_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Well Data (CSV)",
            data=csv_data,
            file_name=f"well_data_{st.session_state.selected_point[0]:.4f}_{st.session_state.selected_point[1]:.4f}.csv",
            mime="text/csv"
        )

    # Add a clear button to reset the map
    if st.button("Clear Results", use_container_width=True):
        # Reset the session state
        st.session_state.selected_point = None
        st.session_state.filtered_wells = None
        st.session_state.selected_well = None
        st.rerun()

with main_col2:
    st.subheader("Analysis Results")

    if st.session_state.filtered_wells is not None and len(st.session_state.filtered_wells) > 0:
        # Show the total wells count in radius
        st.write(f"**Total wells in radius:** {st.session_state.total_wells_in_radius}")

        # Calculate statistics using ALL wells in the radius (no yield filtering)
        # Replace NaN with 0 for yield calculations
        yields = st.session_state.filtered_wells['yield_rate'].fillna(0)
        avg_yield = yields.mean() if len(yields) > 0 else 0
        max_yield = yields.max() if len(yields) > 0 else 0

        # Use all wells in radius for depth statistics
        avg_depth = st.session_state.filtered_wells['depth'].mean() if not st.session_state.filtered_wells.empty else 0

        st.write(f"**Average Yield:** {avg_yield:.2f} L/s")
        st.write(f"**Maximum Yield:** {max_yield:.2f} L/s")
        st.write(f"**Average Depth:** {avg_depth:.2f} m")

        # Show detailed well info if selected
        if st.session_state.selected_well:
            well_details = st.session_state.filtered_wells[
                st.session_state.filtered_wells['well_id'] == st.session_state.selected_well
            ]

            if isinstance(well_details, pd.DataFrame) and not well_details.empty:
                well = well_details.iloc[0]
                st.subheader(f"Well {well['well_id']} Details")

                details_col1, details_col2 = st.columns(2)

                with details_col1:
                    st.write(f"**Latitude:** {well['latitude']}")
                    st.write(f"**Depth:** {well['depth']} m")
                    st.write(f"**Status:** {well['status']}")

                with details_col2:
                    st.write(f"**Longitude:** {well['longitude']}")
                    st.write(f"**Yield Rate:** {well['yield_rate']} L/s")
                    st.write(f"**Distance:** {well['distance']:.2f} km")

                # Clear selection button
                if st.button("Close Well Details"):
                    st.session_state.selected_well = None
                    st.rerun()

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
        st.warning("No wells found in the selected area with current filters. Try increasing the search radius or adjusting yield filters.")
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