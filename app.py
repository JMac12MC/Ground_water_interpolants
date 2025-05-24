import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import os
import base64
from utils import get_distance, download_as_csv
from data_loader import load_sample_data, load_custom_data, load_nz_govt_data, load_api_data
from interpolation import generate_heat_map_data, generate_geo_json_grid

# Set page configuration
st.set_page_config(
    page_title="Groundwater Finder",
    page_icon="💧",
    layout="wide",
)

# Banner image with opacity overlay and text
def add_banner():
    banner_images = [
        "https://pixabay.com/get/gada6da50f8fe938c58e71fb0d6fc33a6632c13da22335a91dd19c2a5fd0ef2c5e979f40f7f3a8ab79ad042d24833ba939cb86bc84ff82ba4d9bd16d955f1eb7f_1280.jpg",
        "https://pixabay.com/get/g63e0006de12468c63fb460735538cca31a25230485a1a484ee5d4c7e23b10ed776f67120aa02d58917ac890ebc718d8cda8e94be7a8954a7a00d3195175771a6_1280.jpg",
        "https://pixabay.com/get/gd84f75e8a71451844bca377d1f6579972a80b3c67d0565fdb32dc7c3004de061fecb4dfae7ccd12baf3df11a5eed41b44d98c9bb28c11a5c078a6935c88d3448_1280.jpg"
    ]
    
    st.markdown(
        f"""
        <div style="position: relative; margin-bottom: 30px;">
            <img src="{banner_images[0]}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 10px;">
            <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.4); border-radius: 10px;"></div>
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; color: white;">
                <h1 style="font-size: 2.5rem; font-weight: bold; margin-bottom: 0;">Groundwater Finder</h1>
                <p style="font-size: 1.2rem;">Helping farmers locate water resources using scientific data</p>
            </div>
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

# Add banner
add_banner()

# Sidebar - for options and filters
with st.sidebar:
    st.header("Data Options")
    
    # Canterbury Wells Data Options
    use_full_dataset = st.checkbox("Load all 56,498 Canterbury wells", 
                                 value=False, 
                                 help="Check to load the complete Canterbury dataset (slower but more comprehensive)")
    
    # Load Canterbury wells data
    if st.session_state.wells_data is None:
        with st.spinner("Loading Canterbury wells data..."):
            st.session_state.wells_data = load_nz_govt_data(use_full_dataset=use_full_dataset)
    elif 'use_full_dataset' not in st.session_state or st.session_state.use_full_dataset != use_full_dataset:
        # Only reload if the dataset choice changed
        with st.spinner(f"Loading {'all 56,498' if use_full_dataset else 'sample of'} Canterbury wells..."):
            st.session_state.wells_data = load_nz_govt_data(use_full_dataset=use_full_dataset)
            st.session_state.use_full_dataset = use_full_dataset
    
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
    st.write("Wells with missing yield values are treated as having 0 yield for map interpolation.")
    
    # Visualization method selection - single dropdown for all options
    st.header("Analysis Options")
    visualization_method = st.selectbox(
        "Map Visualization Type",
        options=[
            "Standard Kriging (Yield)", 
            "Kriging (Auto-Fitted Spherical)",
            "Kriging (Auto-Fitted Gaussian)", 
            "Kriging (Auto-Fitted Exponential)",
            "Random Forest + Kriging (Yield)",
            "Kriging Uncertainty (Fixed Model)",
            "Kriging Uncertainty (Auto-Fitted Spherical)",
            "Kriging Uncertainty (Auto-Fitted Gaussian)",
            "Kriging Uncertainty (Auto-Fitted Exponential)"
        ],
        index=0,
        help="Choose the visualization type: yield estimation or uncertainty analysis"
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
    elif visualization_method == "Kriging (Auto-Fitted Spherical)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
    elif visualization_method == "Kriging (Auto-Fitted Gaussian)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'gaussian'
    elif visualization_method == "Kriging (Auto-Fitted Exponential)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'exponential'
    elif visualization_method == "Random Forest + Kriging (Yield)":
        st.session_state.interpolation_method = 'rf_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = False
    elif visualization_method == "Kriging Uncertainty (Fixed Model)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = True
        st.session_state.auto_fit_variogram = False
        st.session_state.variogram_model = 'linear'
    elif visualization_method == "Kriging Uncertainty (Auto-Fitted Spherical)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = True
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
    elif visualization_method == "Kriging Uncertainty (Auto-Fitted Gaussian)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = True
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'gaussian'
    elif visualization_method == "Kriging Uncertainty (Auto-Fitted Exponential)":
        st.session_state.interpolation_method = 'kriging'
        st.session_state.show_kriging_variance = True
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'exponential'
    
    # Display options
    st.header("Display Options")
    st.session_state.heat_map_visibility = st.checkbox("Show Heat Map", value=st.session_state.heat_map_visibility)
    st.session_state.well_markers_visibility = st.checkbox("Show Well Markers", value=st.session_state.well_markers_visibility)
    
    # Add some guidance info for farmers
    st.header("About This Tool")
    st.info("""
    This tool helps you find potential groundwater locations by:
    
    1. Showing existing wells near your location
    2. Creating a heat map of water yield
    3. Providing data on depth and flow rates
    
    Click anywhere on the map to analyze that location.
    """)
    
    # Add well drilling image
    st.image("https://pixabay.com/get/g3dd7957e8d30d47521b260f1654a0dcffa87f6fd6a8ebaa4f8ba72de270754f6b1ad015b8bc19b503cbd5c12dfe935d4ab5c547948cecf08e4ded91ba49dce79_1280.jpg", 
             caption="Water well drilling", use_column_width=True)

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
    
    # Add location search functionality
    st.subheader("Search Location")
    location_input = st.text_input("Enter address or coordinates (lat, lng)")
    search_button = st.button("Search")
    
    if search_button and location_input:
        try:
            # Check if input is coordinates
            if ',' in location_input:
                parts = location_input.split(',')
                if len(parts) == 2:
                    try:
                        lat = float(parts[0].strip())
                        lng = float(parts[1].strip())
                        st.session_state.selected_point = [lat, lng]
                        st.rerun()
                    except ValueError:
                        st.error("Invalid coordinate format. Please use 'latitude, longitude'")
            else:
                # This would normally use a geocoding service
                st.warning("Address search requires API integration (coming soon)")
        except Exception as e:
            st.error(f"Error searching location: {e}")
    
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
                # Show progress bar for heatmap generation
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text('Generating heatmap interpolation...')
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
                    variogram_model=st.session_state.get('variogram_model', 'spherical')
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
                    
                    # Add tooltips to show yield values on hover
                    style_function = lambda x: {'fillColor': 'transparent', 'color': 'transparent'}
                    highlight_function = lambda x: {'fillOpacity': 0.8}
                    
                    # Add GeoJSON overlay for tooltips
                    folium.GeoJson(
                        geojson_data,
                        style_function=style_function,
                        tooltip=folium.GeoJsonTooltip(
                            fields=['yield'],
                            aliases=['Yield (L/s):'],
                            labels=True,
                            sticky=False
                        )
                    ).add_to(m)
                    
                    # Add well markers to a cluster group for better performance and cleaner display
                    # Only add markers for wells with significant yield values to reduce clutter
                    if 'yield_rate' in filtered_wells.columns:
                        # Create a marker cluster group with custom settings
                        marker_cluster = MarkerCluster(
                            name="Well Markers",
                            overlay=True,
                            control=True,
                            icon_create_function="""
                                function(cluster) {
                                    return L.divIcon({
                                        html: '<div style="background-color: rgba(255, 255, 0, 0.6); border: 1px solid #888; border-radius: 50%; text-align: center; width: 20px; height: 20px; line-height: 20px;">' + cluster.getChildCount() + '</div>',
                                        className: 'marker-cluster',
                                        iconSize: L.point(20, 20)
                                    });
                                }
                            """
                        )
                        
                        # Add only wells with yield > 1 to reduce visual clutter
                        # And use a more subtle style that won't interfere with the heat map
                        for _, row in filtered_wells.iterrows():
                            yield_value = row['yield_rate']
                            if isinstance(yield_value, (int, float)) and yield_value > 3.0:
                                folium.CircleMarker(
                                    location=(float(row['latitude']), float(row['longitude'])),
                                    radius=min(10, float(yield_value)/6 + 1),  # Limit maximum size
                                    color=None,
                                    weight=1,  # Thinner border
                                    fill=True,
                                    fill_color='yellow',
                                    fill_opacity=0.5,  # More transparent
                                    tooltip=f"Well yield: {yield_value} L/s"
                                ).add_to(marker_cluster)
                        
                        # Add the cluster group to the map
                        marker_cluster.add_to(m)
            
            # ONLY show wells within the search radius when a point is selected
            if st.session_state.well_markers_visibility:
                # Only add wells within the search radius (filtered_wells)
                radius_wells_layer = folium.FeatureGroup(name="Wells Within Radius").add_to(m)
                
                # Create markers for ONLY wells within the radius (small markers)
                for idx, row in filtered_wells.iterrows():
                    # Use a smaller CircleMarker for wells within radius
                    folium.CircleMarker(
                        location=(float(row['latitude']), float(row['longitude'])),
                        radius=3,  # Small dot
                        color='gray',
                        fill=True,
                        fill_color='darkblue',
                        fill_opacity=0.7,
                        tooltip=f"Well {row['well_id']} - {row['yield_rate']} L/s - Depth: {row['depth']:.1f}m"
                    ).add_to(radius_wells_layer)
                
                # Add filtered wells with more details
                if isinstance(filtered_wells, pd.DataFrame) and not filtered_wells.empty:
                    marker_cluster = MarkerCluster(name="Filtered Wells").add_to(m)
                    
                    for idx, row in filtered_wells.iterrows():
                        # Create popup content with well information
                        popup_content = f"""
                        <b>Well ID:</b> {row['well_id']}<br>
                        <b>Depth:</b> {row['depth']} m<br>
                        <b>Yield Rate:</b> {row['yield_rate']} L/s<br>
                        <b>Distance:</b> {row['distance']:.2f} km<br>
                        <button onclick="
                            parent.postMessage({{
                                type: 'streamlit:setComponentValue', 
                                value: '{row['well_id']}'
                            }}, '*');
                        ">View Details</button>
                        """
                        
                        # Create marker with popup
                        folium.Marker(
                            location=(float(row['latitude']), float(row['longitude'])),
                            popup=folium.Popup(popup_content, max_width=300),
                            tooltip=f"Well {row['well_id']} - {row['yield_rate']} L/s",
                            icon=folium.Icon(color='blue', icon='tint', prefix='fa')
                        ).add_to(marker_cluster)
        
        # Add click event to capture coordinates (only need this once)
        folium.LatLngPopup().add_to(m)
        
        # Add a simple click handler that manually tracks clicks
        folium.LayerControl().add_to(m)
        
        # Create a custom click handler
        from folium.plugins import MousePosition
        MousePosition().add_to(m)
    
    # Display the map
    st.subheader("Interactive Map")
    st.caption("Click on the map to select a location or use the search box above")
    
    # Use st_folium instead of folium_static to capture clicks
    from streamlit_folium import st_folium
    
    # Make sure we disable folium_static's existing click handlers
    m.add_child(folium.Element("""
    <script>
    // Clear any existing click handlers
    </script>
    """))
    
    # Use st_folium with return_clicked_latlon to get click coordinates
    map_data = st_folium(m, width=800, zoom=st.session_state.zoom_level, 
                       key="interactive_map", returned_objects=["last_clicked"])
    
    # Process clicks from the map
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        # Get the coordinates from the click
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lng = map_data["last_clicked"]["lng"]
        
        # Only update and rerun if this is a new location
        current_point = st.session_state.selected_point
        if not current_point or (abs(current_point[0] - clicked_lat) > 0.0001 or abs(current_point[1] - clicked_lng) > 0.0001):
            # Update session state with the new coordinates
            st.session_state.selected_point = [clicked_lat, clicked_lng]
            
            # Clear previous filtered wells and heat map data to force recalculation
            st.session_state.filtered_wells = None
            
            # Force a complete rerun to rebuild the map with the new location
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
            
            # Depth statistics
            avg_depth = wells_data['depth'].mean()
            min_depth = wells_data['depth'].min()
            max_depth = wells_data['depth'].max()
            
            st.metric(
                "Average Depth to Groundwater", 
                f"{avg_depth:.1f} m",
                help="Mean depth to first well screen"
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
            st.metric(
                "Depth Range", 
                f"{min_depth:.1f} - {max_depth:.1f} m",
                help="Minimum to maximum depth range"
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
        
        # Add visualization image
        st.image("https://pixabay.com/get/gb0bac1d41113e673a752c7a7148ed0ce5da8bc08c2dc48d6b5885b642d028f37a3ca9e7d4f79e4a8f326bab54c3160f53ccb8cf20cb1c9fbc2b3bba86216a20f_1280.jpg", 
                 caption="Groundwater visualization", use_column_width=True)
        
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
        
        # Add information about water well drilling
        st.subheader("Finding Groundwater")
        st.write("""
        Traditional methods like water divining lack scientific basis. Our tool uses actual well data 
        to help you make informed decisions about where to drill based on:
        
        * Proximity to existing successful wells
        * Water yield patterns in your area
        * Depth trends for accessing groundwater
        
        The heat map shows areas where higher yields are likely based on interpolation of existing data.
        """)
        
        # Additional info image
        st.image("https://pixabay.com/get/g05c0207a49d5248437f5982142626c75f1162b7b70bae40f24f9443b8711ff5e6d01382ef3f33d395a3abb0c6ee2d19776e986dcb5bd1ea9ae5c913e2832ab7e_1280.jpg", 
                 caption="Water well drilling equipment", use_column_width=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p>© 2023 Groundwater Finder | Data sourced from public well databases</p>
    <p>This tool is designed to assist farmers in locating potential groundwater sources. Results are based on existing data and interpolation techniques.</p>
</div>
""", unsafe_allow_html=True)
