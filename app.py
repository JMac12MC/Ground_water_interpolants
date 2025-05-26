import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from data_loader import load_nz_govt_data
from interpolation import generate_geo_json_grid
from utils import get_distance
import time

# Page configuration
st.set_page_config(
    page_title="Groundwater Finder",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'wells_data' not in st.session_state:
    st.session_state.wells_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'selected_coords' not in st.session_state:
    st.session_state.selected_coords = None
if 'map_view' not in st.session_state:
    st.session_state.map_view = {'center': [-43.5320, 172.6306], 'zoom': 8}
if 'show_processing' not in st.session_state:
    st.session_state.show_processing = False

# Load data once
@st.cache_data
def load_wells_data():
    return load_nz_govt_data(use_full_dataset=True)

# Header
st.title("🌊 Groundwater Finder for Canterbury")
st.markdown("**Find the best locations for drilling water wells using scientific data analysis**")

# Load wells data
if st.session_state.wells_data is None:
    with st.spinner("Loading Canterbury wells database..."):
        st.session_state.wells_data = load_wells_data()

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🗺️ Interactive Map")
    
    # Create clean map
    map_center = st.session_state.map_view['center']
    map_zoom = st.session_state.map_view['zoom']
    
    m = folium.Map(
        location=map_center,
        zoom_start=map_zoom,
        tiles="OpenStreetMap"
    )
    
    # Add analysis results if available
    if st.session_state.analysis_complete and st.session_state.selected_coords:
        # Add selected point marker
        folium.Marker(
            location=st.session_state.selected_coords,
            popup="Analysis Location",
            icon=folium.Icon(color='red', icon='crosshairs', prefix='fa'),
            tooltip=f"Analysis Point: {st.session_state.selected_coords[0]:.4f}, {st.session_state.selected_coords[1]:.4f}"
        ).add_to(m)
        
        # Add search radius
        search_radius_km = st.session_state.get('search_radius', 5)
        folium.Circle(
            location=st.session_state.selected_coords,
            radius=search_radius_km * 1000,
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
            fill_opacity=0.2
        ).add_to(m)
        
        # Add sophisticated interpolation visualization
        if 'filtered_wells' in st.session_state and not st.session_state.filtered_wells.empty:
            # Get stored visualization method
            vis_method = st.session_state.get('visualization_method', 'Standard Kriging (Yield)')
            
            # Get interpolation parameters
            method = 'kriging'
            show_variance = False
            auto_fit_variogram = False
            variogram_model = 'spherical'
            
            # Set parameters based on visualization method
            if vis_method == "Kriging (Auto-Fitted Spherical)":
                auto_fit_variogram = True
                variogram_model = 'spherical'
            elif vis_method == "Kriging (Auto-Fitted Gaussian)":
                auto_fit_variogram = True
                variogram_model = 'gaussian'
            elif vis_method == "Kriging (Auto-Fitted Exponential)":
                auto_fit_variogram = True
                variogram_model = 'exponential'
            elif vis_method == "Depth to Groundwater (Standard Kriging)":
                method = 'depth_kriging'
            elif vis_method == "Depth to Groundwater (Auto-Fitted Spherical)":
                method = 'depth_kriging'
                auto_fit_variogram = True
                variogram_model = 'spherical'
            elif "Uncertainty" in vis_method:
                show_variance = True
                if "Auto-Fitted" in vis_method:
                    auto_fit_variogram = True
                    if "Spherical" in vis_method:
                        variogram_model = 'spherical'
                    elif "Gaussian" in vis_method:
                        variogram_model = 'gaussian'
                    elif "Exponential" in vis_method:
                        variogram_model = 'exponential'
            
            # Generate interpolation overlay
            try:
                geojson_data = generate_geo_json_grid(
                    st.session_state.filtered_wells.copy(),
                    st.session_state.selected_coords,
                    search_radius_km,
                    resolution=50,
                    method=method,
                    show_variance=show_variance,
                    auto_fit_variogram=auto_fit_variogram,
                    variogram_model=variogram_model
                )
                
                if geojson_data and 'features' in geojson_data:
                    # Add interpolation overlay to map
                    folium.GeoJson(
                        geojson_data,
                        style_function=lambda feature: {
                            'fillColor': feature['properties'].get('color', '#blue'),
                            'color': 'transparent',
                            'weight': 0,
                            'fillOpacity': 0.6
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['yield'],
                            aliases=['Value:'],
                            labels=True
                        )
                    ).add_to(m)
            except Exception as e:
                st.warning(f"Could not generate interpolation: {str(e)}")
        
        # Add well markers
        if 'filtered_wells' in st.session_state and not st.session_state.filtered_wells.empty:
            for _, well in st.session_state.filtered_wells.iterrows():
                folium.CircleMarker(
                    location=[well['latitude'], well['longitude']],
                    radius=4,
                    popup=f"Well {well['well_id']}<br>Yield: {well['yield_rate']} L/s<br>Depth: {well['depth']:.1f}m",
                    color='blue',
                    fill=True,
                    fillColor='darkblue'
                ).add_to(m)
    
    # Show processing overlay if needed
    if st.session_state.show_processing:
        st.info("🔄 **Processing Analysis** - Please wait while we analyze the groundwater data...")
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)
            time.sleep(0.01)
        st.session_state.show_processing = False
        st.rerun()
    
    # Display map and capture interactions
    map_data = st_folium(
        m,
        width=800,
        height=500,
        returned_objects=["last_clicked", "zoom", "center"]
    )
    
    # Save map view state
    if map_data:
        if "zoom" in map_data and map_data["zoom"]:
            st.session_state.map_view['zoom'] = map_data["zoom"]
        if "center" in map_data and map_data["center"]:
            st.session_state.map_view['center'] = [map_data["center"]["lat"], map_data["center"]["lng"]]
    
    # Handle clicks
    if map_data and "last_clicked" in map_data and map_data["last_clicked"]:
        new_coords = [map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]]
        
        # Check if this is a new location
        if (not st.session_state.selected_coords or 
            abs(st.session_state.selected_coords[0] - new_coords[0]) > 0.001 or
            abs(st.session_state.selected_coords[1] - new_coords[1]) > 0.001):
            
            # Start processing
            st.session_state.selected_coords = new_coords
            st.session_state.analysis_complete = False
            st.session_state.show_processing = True
            
            # Get current search radius from session state
            current_search_radius = st.session_state.get('search_radius_setting', 5)
            st.session_state.search_radius = current_search_radius
            
            # Process wells data
            if st.session_state.wells_data is not None:
                wells_df = st.session_state.wells_data.copy()
                
                # Calculate distances
                wells_df['distance'] = wells_df.apply(
                    lambda row: get_distance(
                        new_coords[0], new_coords[1],
                        row['latitude'], row['longitude']
                    ), axis=1
                )
                
                # Filter wells within search radius
                filtered_wells = wells_df[wells_df['distance'] <= current_search_radius].copy()
                
                # Handle missing yield values (treat as 0 for analysis)
                if 'yield_rate' in filtered_wells.columns:
                    filtered_wells['yield_rate'] = filtered_wells['yield_rate'].fillna(0)
                
                # Store results
                st.session_state.filtered_wells = filtered_wells
                st.session_state.analysis_complete = True
            
            st.rerun()

with col2:
    st.subheader("📊 Analysis Results")
    
    if not st.session_state.analysis_complete:
        st.info("👆 **Click anywhere on the map to start groundwater analysis**")
        st.markdown("""
        **How to use:**
        1. Navigate to your area of interest
        2. Click on the map where you want to analyze
        3. View the wells and groundwater data for that location
        """)
    else:
        st.success(f"✅ **Analysis Complete**")
        st.write(f"**Location:** {st.session_state.selected_coords[0]:.4f}, {st.session_state.selected_coords[1]:.4f}")
        
        if 'filtered_wells' in st.session_state and not st.session_state.filtered_wells.empty:
            wells_data = st.session_state.filtered_wells
            
            # Summary metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Wells", len(wells_data))
                st.metric("Avg Yield", f"{wells_data['yield_rate'].mean():.1f} L/s")
            with col_b:
                st.metric("Avg Depth", f"{wells_data['depth'].mean():.1f} m")
                productive_wells = len(wells_data[wells_data['yield_rate'] > 1])
                st.metric("Productive Wells", productive_wells)
            
            # Top wells table
            st.subheader("🏆 Top Wells by Yield")
            top_wells = wells_data.nlargest(5, 'yield_rate')[['well_id', 'yield_rate', 'depth', 'distance']]
            st.dataframe(
                top_wells,
                column_config={
                    "well_id": "Well ID",
                    "yield_rate": st.column_config.NumberColumn("Yield (L/s)", format="%.1f"),
                    "depth": st.column_config.NumberColumn("Depth (m)", format="%.1f"),
                    "distance": st.column_config.NumberColumn("Distance (km)", format="%.2f")
                },
                hide_index=True
            )
            
            # Download button
            csv_data = wells_data.to_csv(index=False)
            st.download_button(
                "📥 Download Well Data",
                data=csv_data,
                file_name=f"wells_{st.session_state.selected_coords[0]:.4f}_{st.session_state.selected_coords[1]:.4f}.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚠️ No wells found within 5km of this location")
    
    # Clear button
    if st.button("🗑️ Clear Analysis", use_container_width=True):
        st.session_state.analysis_complete = False
        st.session_state.selected_coords = None
        if 'filtered_wells' in st.session_state:
            del st.session_state.filtered_wells
        st.rerun()

# Sidebar with settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Analysis options
    st.subheader("Analysis Options")
    
    visualization_method = st.selectbox(
        "Visualization Type",
        [
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
        ]
    )
    
    search_radius = st.slider(
        "Search Radius (km)",
        min_value=1,
        max_value=10,
        value=5,
        help="Radius to search for wells around clicked location"
    )
    
    # Store settings in session state for use in map generation
    st.session_state.visualization_method = visualization_method
    st.session_state.search_radius_setting = search_radius
    
    st.subheader("📋 About")
    st.markdown("""
    **Groundwater Finder** uses real data from Canterbury's well database to help you:
    
    - 🎯 Find optimal drilling locations
    - 📊 Analyze nearby well performance
    - 💧 Estimate water yield potential
    - ⛏️ Understand groundwater depth
    
    **Data Source:** Canterbury Maps OpenData
    """)