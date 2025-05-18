import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import base64
from utils import get_distance, download_as_csv
from data_loader import load_sample_data, load_custom_data, load_nz_govt_data, load_api_data
import kriging_interpolation

# Set page configuration
st.set_page_config(
    page_title="Groundwater Finder",
    page_icon="💧",
    layout="wide",
)

# Helper function to get color for yield value from gradient
def get_color_for_yield(yield_value, min_yield, max_yield, color_gradient):
    """Return a color from the gradient based on the yield value"""
    if min_yield == max_yield:
        norm_val = 0.5
    else:
        norm_val = (yield_value - min_yield) / (max_yield - min_yield)
        norm_val = max(0.0, min(1.0, norm_val))
    
    # Find the closest gradient key
    closest_key = min(color_gradient.keys(), key=lambda x: abs(x - norm_val))
    return color_gradient[closest_key]

# Banner image with opacity overlay and text
def add_banner():
    banner_images = [
        "https://pixabay.com/get/g05c0207a49d5248437f5982142626c75f1162b7b70bae40f24f9443b8711ff5e6d01382ef3f33d395a3abb0c6ee2d19776e986dcb5bd1ea9ae5c913e2832ab7e_1280.jpg",
        "https://pixabay.com/get/g3dd7957e8d30d47521b260f1654a0dcffa87f6fd6a8ebaa4f8ba72de270754f6b1ad015b8bc19b503cbd5c12dfe935d4ab5c547948cecf08e4ded91ba49dce79_1280.jpg",
        "https://pixabay.com/get/gb0bac1d41113e673a752c7a7148ed0ce5da8bc08c2dc48d6b5885b642d028f37a3ca9e7d4f79e4a8f326bab54c3160f53ccb8cf20cb1c9fbc2b3bba86216a20f_1280.jpg"
    ]
    
    # Use the first image for the banner
    banner_html = f"""
    <div style="position: relative; text-align: center; color: white;">
        <img src="{banner_images[0]}" style="width: 100%; height: 200px; object-fit: cover; border-radius: 10px;">
        <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.5); border-radius: 10px;"></div>
        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 80%;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">Groundwater Finder</h1>
            <p style="font-size: 1.2rem;">A data-driven approach to locating groundwater resources</p>
        </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

# Initialize session state variables
if 'selected_point' not in st.session_state:
    st.session_state.selected_point = None
if 'wells_data' not in st.session_state:
    st.session_state.wells_data = None
if 'filtered_wells' not in st.session_state:
    st.session_state.filtered_wells = None
if 'search_radius' not in st.session_state:
    st.session_state.search_radius = 5.0  # Default radius in km
if 'zoom_level' not in st.session_state:
    st.session_state.zoom_level = 10
if 'min_yield' not in st.session_state:
    st.session_state.min_yield = 0.0
if 'max_yield' not in st.session_state:
    st.session_state.max_yield = 100.0
if 'heat_map_visibility' not in st.session_state:
    st.session_state.heat_map_visibility = True
if 'show_isopach' not in st.session_state:
    st.session_state.show_isopach = True  # Show isopach map by default

# Handle when user clicks on the map
def handle_map_click(event_data):
    st.session_state.selected_point = [event_data["lat"], event_data["lng"]]
    st.rerun()

# Define color gradient for yield visualization
gradient = {
    0.0: 'blue',    # Lowest yield
    0.2: 'cyan',    # Low yield 
    0.4: 'green',   # Moderate yield
    0.6: 'yellow',  # Good yield
    0.8: 'orange',  # High yield
    1.0: 'red'      # Highest yield
}

# Main app
def main():
    add_banner()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data source selector
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["New Zealand Sample Data", "Custom CSV Upload", "API Data"]
    )
    
    # Load the appropriate data based on selected source
    if data_source == "New Zealand Sample Data":
        if st.session_state.wells_data is None:
            with st.sidebar.status("Loading sample data..."):
                st.session_state.wells_data = load_nz_govt_data()
    
    elif data_source == "Custom CSV Upload":
        uploaded_file = st.sidebar.file_uploader("Upload Well Data CSV", type=["csv"])
        if uploaded_file:
            try:
                st.session_state.wells_data = load_custom_data(uploaded_file)
                st.sidebar.success(f"Loaded {len(st.session_state.wells_data)} wells from CSV")
            except Exception as e:
                st.sidebar.error(f"Error loading CSV: {e}")
    
    elif data_source == "API Data":
        api_url = st.sidebar.text_input("API URL")
        api_key = st.sidebar.text_input("API Key (if required)", type="password")
        
        if api_url and st.sidebar.button("Fetch Data"):
            try:
                with st.sidebar.status("Fetching data from API..."):
                    st.session_state.wells_data = load_api_data(api_url, api_key)
                st.sidebar.success(f"Loaded {len(st.session_state.wells_data)} wells from API")
            except Exception as e:
                st.sidebar.error(f"Error fetching API data: {e}")
    
    # Yield range filter
    st.sidebar.subheader("Yield Filter")
    min_yield, max_yield = st.sidebar.slider(
        "Yield Range (L/s)",
        0.0, 100.0, (st.session_state.min_yield, st.session_state.max_yield)
    )
    st.session_state.min_yield = min_yield
    st.session_state.max_yield = max_yield
    
    # Search radius configuration
    st.sidebar.subheader("Search Radius")
    search_radius = st.sidebar.slider(
        "Search Radius (km)",
        1.0, 20.0, st.session_state.search_radius
    )
    st.session_state.search_radius = search_radius
    
    # Visualization options
    st.sidebar.subheader("Visualization")
    
    # Toggle for isopach visualization (one single control)
    show_isopach = st.sidebar.checkbox(
        "Show Yield Isopach Map", 
        value=st.session_state.show_isopach,
        help="Display interpolated yield contours across the entire area"
    )
    st.session_state.show_isopach = show_isopach
    st.session_state.heat_map_visibility = show_isopach  # Keep both in sync
    
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
        
        # Create map
        m = folium.Map(location=center_location, zoom_start=st.session_state.zoom_level, 
                      tiles="OpenStreetMap")
        
        # No need for a second toggle since we already have one in the sidebar
        
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
            
            # Create feature group for wells
            wells_group = folium.FeatureGroup(name="Wells")
            
            # Generate the global isopach map first (if enabled)
            if st.session_state.show_isopach and st.session_state.heat_map_visibility:
                # Create a feature group for the isopach visualization
                isopach_group = folium.FeatureGroup(name="Isopach Map")
                
                # Generate kriging-based contour map for the entire dataset
                contour_json = kriging_interpolation.create_kriging_contours(
                    wells_df,
                    center_point=None,  # Auto-center based on well locations
                    radius_km=None,     # Auto-radius to cover reasonable area
                    min_yield=st.session_state.min_yield,
                    max_yield=st.session_state.max_yield,
                    num_points=120      # Higher resolution for smoother contours
                )
                
                if contour_json:
                    # Style function for the contours based on yield value
                    style_function = lambda feature: {
                        'fillColor': get_color_for_yield(
                            feature['properties']['yield_value'], 
                            st.session_state.min_yield, 
                            st.session_state.max_yield, 
                            gradient
                        ),
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.65,
                    }
                    
                    # Add tooltip to show yield value on hover
                    tooltip = folium.GeoJsonTooltip(
                        fields=['yield_value'],
                        aliases=['Yield (L/s):'],
                        localize=True,
                        sticky=False,
                        labels=True,
                        style="""
                            background-color: #F0EFEF;
                            border: 1px solid black;
                            border-radius: 3px;
                            box-shadow: 3px 3px 3px rgba(0,0,0,0.25);
                        """
                    )
                    
                    # Add the GeoJSON contours to the map
                    folium.GeoJson(
                        data=contour_json,
                        name='Yield Contours',
                        style_function=style_function,
                        tooltip=tooltip,
                        overlay=True
                    ).add_to(isopach_group)
                    
                    # Add the isopach group to the map
                    isopach_group.add_to(m)
            
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
                
                # Calculate well statistics for the clicked location
                stats = kriging_interpolation.get_well_stats(
                    wells_df, 
                    st.session_state.selected_point[0],
                    st.session_state.selected_point[1],
                    st.session_state.search_radius
                )
                
                # Display statistics in the sidebar
                with main_col2:
                    st.subheader("Well Statistics")
                    st.write(f"**Wells Found:** {stats['well_count']}")
                    st.write(f"**Average Yield:** {stats['avg_yield']} L/s")
                    st.write(f"**Maximum Yield:** {stats['max_yield']} L/s")
                    st.write(f"**Average Depth:** {stats['avg_depth']} m")
                    st.write(f"**Predicted Yield:** {stats['predicted_yield']} L/s")
                    
                    # Colored indicator based on predicted yield
                    if stats['predicted_yield'] > 0:
                        norm_yield = (stats['predicted_yield'] - st.session_state.min_yield) / (st.session_state.max_yield - st.session_state.min_yield)
                        norm_yield = max(0.0, min(1.0, norm_yield))
                        
                        # Find color for this yield value
                        closest_key = min(gradient.keys(), key=lambda x: abs(x - norm_yield))
                        color = gradient[closest_key]
                        
                        # Show colored indicator
                        st.markdown(f"""
                        <div style="background-color: {color}; width: 100%; height: 20px; 
                                    border-radius: 5px; margin-top: 10px;"></div>
                        <p style="text-align: center; margin-top: 5px;">Yield Potential Indicator</p>
                        """, unsafe_allow_html=True)
                
                # Show filtered wells info
                if not filtered_wells.empty:
                    st.write(f"Found {len(filtered_wells)} wells within {st.session_state.search_radius} km.")
                    
                    # Offer download option for filtered wells
                    if st.button("Download Filtered Wells Data"):
                        csv_data = download_as_csv(filtered_wells)
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="filtered_wells.csv",
                            mime="text/csv"
                        )
                else:
                    st.write("No wells found in the selected area.")
            
            # Add all wells to the map (with different colors based on yield)
            for _, row in wells_df.iterrows():
                # Skip wells outside the yield filter range
                if row['yield_rate'] < st.session_state.min_yield or row['yield_rate'] > st.session_state.max_yield:
                    continue
                    
                # Normalize the yield value for color selection
                norm_yield = (row['yield_rate'] - st.session_state.min_yield) / (st.session_state.max_yield - st.session_state.min_yield)
                norm_yield = min(1.0, max(0.0, norm_yield))
                
                # Get the closest color from our gradient
                closest_key = min(gradient.keys(), key=lambda x: abs(x - norm_yield))
                color = gradient[closest_key]
                
                # Create circle marker for the well
                folium.CircleMarker(
                    location=[float(row['latitude']), float(row['longitude'])],
                    radius=5,
                    color='black',
                    weight=1,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    tooltip=f"Well ID: {row['well_id']}<br>Yield: {row['yield_rate']:.1f} L/s"
                ).add_to(wells_group)
            
            # Add wells group to the map
            wells_group.add_to(m)
            
            # Add layer control to toggle layers
            folium.LayerControl().add_to(m)
        
        # Add the click handler to update the selected point
        m.add_child(folium.LatLngPopup())
        
        # Display the map
        folium_static(m, width=800, height=600)
        
        # Handle map click (stores in session state for next render)
        map_data = st.session_state.get('last_clicked')
        if map_data:
            handle_map_click(map_data)
    
    # Right column for displaying well information
    with main_col2:
        if st.session_state.selected_point is None:
            st.info("Click on the map to select a location")
        
        if st.session_state.wells_data is not None and not st.session_state.wells_data.empty:
            st.subheader("All Wells Overview")
            st.write(f"Total wells in dataset: {len(st.session_state.wells_data)}")
            
            # Basic statistics
            yield_mean = st.session_state.wells_data['yield_rate'].mean()
            yield_max = st.session_state.wells_data['yield_rate'].max()
            
            st.write(f"Average yield: {yield_mean:.2f} L/s")
            st.write(f"Maximum yield: {yield_max:.2f} L/s")
            
            # Basic histogram of yield distribution
            if st.checkbox("Show Yield Distribution"):
                fig, ax = plt.subplots(figsize=(4, 3))
                st.session_state.wells_data['yield_rate'].hist(bins=20, ax=ax)
                ax.set_xlabel('Yield (L/s)')
                ax.set_ylabel('Number of Wells')
                st.pyplot(fig)

if __name__ == "__main__":
    main()