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
from interpolation import generate_heat_map_data

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
if 'heat_map_visibility' not in st.session_state:
    st.session_state.heat_map_visibility = True
if 'well_markers_visibility' not in st.session_state:
    st.session_state.well_markers_visibility = True
if 'search_radius' not in st.session_state:
    st.session_state.search_radius = 10
if 'min_yield' not in st.session_state:
    st.session_state.min_yield = 0
if 'max_yield' not in st.session_state:
    st.session_state.max_yield = 1000
if 'selected_well' not in st.session_state:
    st.session_state.selected_well = None

# Add banner
add_banner()

# Sidebar - for options and filters
with st.sidebar:
    st.header("Data Options")
    
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["Sample Data", "NZ Government Data", "Custom Upload", "API Data"]
    )
    
    if data_source == "Sample Data":
        st.session_state.wells_data = load_sample_data()
    elif data_source == "NZ Government Data":
        # Load the New Zealand government well data
        st.session_state.wells_data = load_nz_govt_data()
    elif data_source == "Custom Upload":
        uploaded_file = st.file_uploader("Upload a CSV file with well data", type=["csv"])
        if uploaded_file is not None:
            st.session_state.wells_data = load_custom_data(uploaded_file)
    else:  # API Data
        api_url = st.text_input("Enter API URL or type 'NZ' for New Zealand data")
        api_key = st.text_input("API Key (if required)", type="password")
        
        if st.button("Fetch Data"):
            if api_url:
                st.session_state.wells_data = load_api_data(api_url, api_key)
            else:
                st.error("Please enter an API URL")
                st.session_state.wells_data = load_sample_data()
    
    st.header("Filters")
    
    # Radius filter
    st.session_state.search_radius = st.slider(
        "Search Radius (km)",
        min_value=1,
        max_value=50,
        value=st.session_state.search_radius,
        step=1
    )
    
    # Yield filter
    if st.session_state.wells_data is not None:
        # Calculate max yield from data if available
        max_data_yield = int(st.session_state.wells_data['yield_rate'].max()) + 100
        min_yield, max_yield = st.slider(
            "Yield Rate Range (L/s)",
            min_value=0,
            max_value=max_data_yield,
            value=(st.session_state.min_yield, min(st.session_state.max_yield, max_data_yield))
        )
        st.session_state.min_yield = min_yield
        st.session_state.max_yield = max_yield
    
    # Visibility options
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
            
            # Apply filters
            filtered_wells = wells_df[
                (wells_df['distance'] <= st.session_state.search_radius) & 
                (wells_df['yield_rate'] >= st.session_state.min_yield) & 
                (wells_df['yield_rate'] <= st.session_state.max_yield)
            ]
            
            st.session_state.filtered_wells = filtered_wells
            
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
            if st.session_state.heat_map_visibility and not filtered_wells.empty:
                heat_data = generate_heat_map_data(
                    filtered_wells, 
                    st.session_state.selected_point, 
                    st.session_state.search_radius
                )
                
                gradient = {
                    0.0: 'blue',
                    0.25: 'cyan',
                    0.5: 'lime',
                    0.75: 'yellow',
                    1.0: 'red'
                }
                
                HeatMap(
                    heat_data,
                    radius=15,
                    gradient=gradient,
                    blur=10,
                    max_zoom=10,
                    overlay=True,
                ).add_to(m)
                
                # Add visualization legend
                colormap = folium.LinearColormap(
                    colors=['blue', 'cyan', 'lime', 'yellow', 'red'],
                    vmin=st.session_state.min_yield,
                    vmax=st.session_state.max_yield,
                    caption='Estimated Water Yield (L/s)'
                )
                colormap.add_to(m)
            
            # Add ALL wells to the map as dots, not just filtered ones
            if st.session_state.well_markers_visibility:
                # Add ALL wells as small dots
                all_wells_layer = folium.FeatureGroup(name="All Wells").add_to(m)
                
                # Create markers for all wells (small markers)
                for idx, row in wells_df.iterrows():
                    # Use a smaller CircleMarker for all wells
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=3,  # Small dot
                        color='gray',
                        fill=True,
                        fill_color='darkblue',
                        fill_opacity=0.7,
                        tooltip=f"Well {row['well_id']} - {row['yield_rate']} L/s"
                    ).add_to(all_wells_layer)
                
                # Add filtered wells with more details
                if not filtered_wells.empty:
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
                            location=[row['latitude'], row['longitude']],
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
        
        # Update session state with the new coordinates
        st.session_state.selected_point = [clicked_lat, clicked_lng]
        st.rerun()
    
    # Add manual coordinate selection
    st.subheader("Manually Select Coordinates")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        latitude = st.number_input("Latitude", value=default_location[0], format="%.6f")
    with col2:
        longitude = st.number_input("Longitude", value=default_location[1], format="%.6f")
    
    button_col1, button_col2 = st.columns(2)
    with button_col1:
        if st.button("Set Location", use_container_width=True):
            st.session_state.selected_point = [latitude, longitude]
            st.rerun()
    
    with button_col2:
        if st.button("Clear Results", use_container_width=True):
            # Reset the session state
            st.session_state.selected_point = None
            st.session_state.filtered_wells = None
            st.session_state.selected_well = None
            st.rerun()

with main_col2:
    st.subheader("Analysis Results")
    
    if st.session_state.filtered_wells is not None and len(st.session_state.filtered_wells) > 0:
        # Show summary stats
        st.write(f"**Wells found:** {len(st.session_state.filtered_wells)}")
        
        avg_yield = st.session_state.filtered_wells['yield_rate'].mean()
        max_yield = st.session_state.filtered_wells['yield_rate'].max()
        avg_depth = st.session_state.filtered_wells['depth'].mean()
        
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
            
            if not well_details.empty:
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
