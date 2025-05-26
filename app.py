import streamlit as st
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import os
import base64
import requests
from utils import get_distance, download_as_csv
from data_loader import load_sample_data, load_custom_data, load_nz_govt_data, load_api_data
from interpolation import generate_heat_map_data, generate_geo_json_grid

# Set page configuration
st.set_page_config(
    page_title="Groundwater Finder",
    page_icon="💧",
    layout="wide",
)

# Clean title header without image
def add_banner():
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 30px; padding: 40px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
            <h1 style="font-size: 2.5rem; font-weight: bold; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">Groundwater Finder</h1>
            <p style="font-size: 1.2rem; margin-bottom: 0; opacity: 0.9;">Helping farmers locate water resources using scientific data</p>
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

    # Load Canterbury wells data
    if st.session_state.wells_data is None:
        with st.spinner("Loading Canterbury wells data..."):
            st.session_state.wells_data = load_nz_govt_data()

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
            "Kriging (Auto-Fitted Spherical)",
            "Kriging (Auto-Fitted Gaussian)", 
            "Kriging (Auto-Fitted Exponential)",
            "Depth to Groundwater (Standard Kriging)",
            "Depth to Groundwater (Auto-Fitted Spherical)",
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
    elif visualization_method == "Depth to Groundwater (Standard Kriging)":
        st.session_state.interpolation_method = 'depth_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = False
    elif visualization_method == "Depth to Groundwater (Auto-Fitted Spherical)":
        st.session_state.interpolation_method = 'depth_kriging'
        st.session_state.show_kriging_variance = False
        st.session_state.auto_fit_variogram = True
        st.session_state.variogram_model = 'spherical'
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
    
    # Create columns for search input and button
    search_col1, search_col2 = st.columns([4, 1])
    
    with search_col1:
        location_input = st.text_input(
            "Enter address or coordinates (lat, lng)",
            placeholder="e.g. Christchurch, New Zealand or -43.532, 172.636",
            help="Type an address for suggestions or enter coordinates as 'lat, lng'"
        )
    
    with search_col2:
        search_button = st.button("🔍 Search", use_container_width=True)

    # Address verification and selection system
    if 'address_suggestions' not in st.session_state:
        st.session_state.address_suggestions = []
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = False
    if 'search_performed' not in st.session_state:
        st.session_state.search_performed = False

    # Real-time address verification
    if location_input and len(location_input) > 3:
        # Check if this is a coordinate input
        if not (',' in location_input and location_input.count(',') == 1):
            try:
                import requests
                
                # Use Nominatim for address verification
                search_url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': location_input,
                    'format': 'json',
                    'limit': 10,  # Get more options for verification
                    'countrycodes': 'nz',  # Focus on New Zealand
                    'addressdetails': 1
                }
                
                headers = {
                    'User-Agent': 'GroundwaterFinder/1.0'
                }
                
                # Only make API call when user stops typing (every 3 characters)
                if len(location_input) % 3 == 0 or st.session_state.search_performed:
                    with st.spinner("Verifying address..."):
                        response = requests.get(search_url, params=params, headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        suggestions = response.json()
                        
                        if suggestions:
                            # Process and deduplicate suggestions
                            processed_suggestions = []
                            seen_locations = set()
                            
                            for suggestion in suggestions:
                                display_name = suggestion.get('display_name', '')
                                lat = float(suggestion.get('lat', 0))
                                lon = float(suggestion.get('lon', 0))
                                
                                # Create a location key for deduplication
                                location_key = f"{lat:.4f},{lon:.4f}"
                                
                                if location_key not in seen_locations and display_name:
                                    # Extract useful address components
                                    address = suggestion.get('address', {})
                                    house_number = address.get('house_number', '')
                                    road = address.get('road', '')
                                    suburb = address.get('suburb', '')
                                    city = address.get('city', address.get('town', address.get('village', '')))
                                    
                                    # Create a cleaner display format
                                    address_parts = []
                                    if house_number and road:
                                        address_parts.append(f"{house_number} {road}")
                                    elif road:
                                        address_parts.append(road)
                                    
                                    if suburb and suburb != city:
                                        address_parts.append(suburb)
                                    if city:
                                        address_parts.append(city)
                                    
                                    clean_address = ", ".join(address_parts) if address_parts else display_name
                                    
                                    processed_suggestions.append({
                                        'display': clean_address,
                                        'full_address': display_name,
                                        'lat': lat,
                                        'lon': lon,
                                        'type': suggestion.get('type', 'address')
                                    })
                                    seen_locations.add(location_key)
                            
                            st.session_state.address_suggestions = processed_suggestions[:8]  # Limit to 8 suggestions
                            st.session_state.show_suggestions = True
                        else:
                            st.session_state.address_suggestions = []
                            st.session_state.show_suggestions = False
                    else:
                        st.session_state.show_suggestions = False
                        
            except Exception as e:
                st.session_state.show_suggestions = False
    else:
        st.session_state.show_suggestions = False

    # Display address suggestions dropdown
    if st.session_state.show_suggestions and st.session_state.address_suggestions:
        st.markdown("**📍 Select your address:**")
        
        # Create a container with custom styling for the address list
        suggestions_container = st.container()
        
        with suggestions_container:
            for i, suggestion in enumerate(st.session_state.address_suggestions):
                # Create two columns for better layout
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # Show the clean address with type indicator
                    address_type = suggestion['type'].title() if suggestion['type'] != 'address' else ''
                    type_indicator = f" ({address_type})" if address_type and address_type != 'Address' else ''
                    
                    if st.button(
                        f"📍 {suggestion['display']}{type_indicator}",
                        key=f"address_select_{i}",
                        use_container_width=True,
                        help=f"Full address: {suggestion['full_address']}"
                    ):
                        # Set the selected location
                        st.session_state.selected_point = [suggestion['lat'], suggestion['lon']]
                        st.session_state.show_suggestions = False
                        st.session_state.address_suggestions = []
                        st.success(f"✅ Selected: {suggestion['display']}")
                        st.rerun()
                
                with col2:
                    # Show coordinates for verification
                    st.caption(f"{suggestion['lat']:.4f}, {suggestion['lon']:.4f}")
        
        # Add option to hide suggestions
        if st.button("❌ Clear suggestions", key="clear_suggestions"):
            st.session_state.show_suggestions = False
            st.session_state.address_suggestions = []
            st.rerun()
    
    elif location_input and len(location_input) > 3 and not st.session_state.show_suggestions:
        if not (',' in location_input and location_input.count(',') == 1):
            st.info("💡 Start typing an address to see suggestions, or use coordinates format: lat, lng")

    # Handle search button click
    if search_button and location_input:
        st.session_state.search_performed = True
        
        try:
            # Check if input is coordinates
            if ',' in location_input and location_input.count(',') == 1:
                parts = location_input.split(',')
                if len(parts) == 2:
                    try:
                        lat = float(parts[0].strip())
                        lng = float(parts[1].strip())
                        
                        # Validate coordinates are reasonable for New Zealand
                        if -48 <= lat <= -34 and 166 <= lng <= 179:
                            st.session_state.selected_point = [lat, lng]
                            st.session_state.show_suggestions = False
                            st.success(f"✅ Location set to: {lat:.4f}, {lng:.4f}")
                            st.rerun()
                        else:
                            st.error("Coordinates seem to be outside New Zealand. Please check your input.")
                    except ValueError:
                        st.error("Invalid coordinate format. Please use 'latitude, longitude'")
            else:
                # Trigger address verification and show suggestions
                if not st.session_state.show_suggestions:
                    # Force address lookup
                    import requests
                    
                    search_url = "https://nominatim.openstreetmap.org/search"
                    params = {
                        'q': location_input,
                        'format': 'json',
                        'limit': 10,
                        'countrycodes': 'nz',
                        'addressdetails': 1
                    }
                    
                    headers = {
                        'User-Agent': 'GroundwaterFinder/1.0'
                    }
                    
                    with st.spinner("🔍 Searching for address..."):
                        response = requests.get(search_url, params=params, headers=headers, timeout=5)
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        if results:
                            # Process results for suggestions
                            processed_suggestions = []
                            seen_locations = set()
                            
                            for result in results:
                                display_name = result.get('display_name', '')
                                lat = float(result.get('lat', 0))
                                lon = float(result.get('lon', 0))
                                
                                location_key = f"{lat:.4f},{lon:.4f}"
                                
                                if location_key not in seen_locations and display_name:
                                    address = result.get('address', {})
                                    house_number = address.get('house_number', '')
                                    road = address.get('road', '')
                                    suburb = address.get('suburb', '')
                                    city = address.get('city', address.get('town', address.get('village', '')))
                                    
                                    address_parts = []
                                    if house_number and road:
                                        address_parts.append(f"{house_number} {road}")
                                    elif road:
                                        address_parts.append(road)
                                    
                                    if suburb and suburb != city:
                                        address_parts.append(suburb)
                                    if city:
                                        address_parts.append(city)
                                    
                                    clean_address = ", ".join(address_parts) if address_parts else display_name
                                    
                                    processed_suggestions.append({
                                        'display': clean_address,
                                        'full_address': display_name,
                                        'lat': lat,
                                        'lon': lon,
                                        'type': result.get('type', 'address')
                                    })
                                    seen_locations.add(location_key)
                            
                            if processed_suggestions:
                                st.session_state.address_suggestions = processed_suggestions[:8]
                                st.session_state.show_suggestions = True
                                st.success(f"Found {len(processed_suggestions)} address matches. Please select one below:")
                                st.rerun()
                            else:
                                st.error("No addresses found matching your search.")
                        else:
                            st.error("No addresses found. Please try a different search term or use coordinates.")
                    else:
                        st.error("Search service temporarily unavailable. Please try coordinates instead.")
                else:
                    st.info("Please select an address from the suggestions above, or modify your search.")
                    
        except Exception as e:
            st.error(f"Error searching location: {str(e)}")
            st.info("Try using coordinates format: latitude, longitude (e.g. -43.532, 172.636)")

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
                # Clear processing flag to prevent overlay after data is ready
                if st.session_state.get('processing', False):
                    st.session_state.processing = False

                # Show progress overlay during processing with better visibility
                progress_container = st.container()
                with progress_container:
                    st.info("🔄 **Analysis in Progress** - Map view will be preserved")

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
                        tooltip=f"Well {row['well_id']} - {row['yield_rate']} L/s - Groundwater: {row['depth']:.1f}m{'(Dry)' if row.get('is_dry_well', False) else ''}"
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
    # Use consistent default values and avoid overriding user interactions
    if 'map_initialized' not in st.session_state:
        st.session_state.map_initialized = True
        st.session_state.map_center = [-43.5321, 172.6362]
        st.session_state.map_zoom = 8

    map_data = st_folium(m, width=800, 
                       key="interactive_map", 
                       returned_objects=["last_clicked"])

    # Only update session state if map data is available, don't force saved states
    if map_data and "center" in map_data and map_data["center"]:
        st.session_state.map_center = [map_data["center"]["lat"], map_data["center"]["lng"]]
    if map_data and "zoom" in map_data and map_data["zoom"]:
        st.session_state.map_zoom = map_data["zoom"]

    # Process clicks from the map
    new_location_clicked = False
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
            new_location_clicked = True

    # Process new location clicks without interfering with map state
    if new_location_clicked:
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

            # Depth statistics
            avg_depth = wells_data['depth'].mean()
            min_depth = wells_data['depth'].min()
            max_depth = wells_data['depth'].max()

            st.metric(
                "Average Depth to Groundwater", 
                f"{avg_depth:.1f} m",
                help="Mean depth to shallowest water screen (actual groundwater depth)"
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
            dry_wells = len(wells_data[wells_data.get('is_dry_well', False) == True]) if 'is_dry_well' in wells_data.columns else 0

            st.metric(
                "High-Yield Wells", 
                f"{high_yield_wells}",
                help="Wells with yield > 5 L/s"
            )

            if dry_wells > 0:
                st.metric(
                    "Dry Wells", 
                    f"{dry_wells}",
                    help="Wells with no water screen (likely dry)"
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
                 caption="Groundwater visualization", use_container_width=True)

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
        * Aquifer yield patterns in your area
        * Depth trends for accessing groundwater
        """)

        # Additional info image
        st.image("https://pixabay.com/get/g05c0207a49d5248437f5982142626c75f1162b7b70bae40f24f9443b8711ff5e6d01382ef3f33d395a3abb0c6ee2d19776e986dcb5bd1ea9ae5c913e2832ab7e_1280.jpg", 
                 caption="Water well drilling equipment", use_container_width=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p>© 2023 Groundwater Finder | Data sourced from public well databases</p>
    <p>This tool is designed to assist farmers in locating potential groundwater sources. Results are based on existing data and interpolation techniques.</p>
</div>
""", unsafe_allow_html=True)