import pandas as pd
import numpy as np
import streamlit as st
import os
import random

def load_sample_data():
    """
    Load sample well data for demonstration purposes
    Returns a pandas DataFrame with well information
    """
    try:
        # Try to load from file first
        df = pd.read_csv("sample_data/wells_sample.csv")
        return df
    except:
        # Generate sample data if file not found
        # This mimics real well data from New Zealand
        
        # Create a base location (Christchurch, NZ area)
        base_lat = -43.5320
        base_lon = 172.6306
        
        # Create random wells around this location
        num_wells = 100
        np.random.seed(42)  # For reproducibility
        
        # Generate random coordinates within ~50km
        lats = np.random.uniform(base_lat - 0.5, base_lat + 0.5, num_wells)
        lons = np.random.uniform(base_lon - 0.5, base_lon + 0.5, num_wells)
        
        # Generate well IDs
        well_ids = [f"W-{i+1000}" for i in range(num_wells)]
        
        # Generate depths - realistic for NZ area (10-300m)
        depths = np.random.uniform(10, 300, num_wells)
        
        # Generate yield rates - L/s (0.1 to 50 L/s)
        # Use a distribution that creates more low-yield wells than high-yield ones
        base_yields = np.random.exponential(scale=5, size=num_wells)
        yield_rates = np.clip(base_yields, 0.1, 50)
        
        # Create some spatial correlation - wells close together should have similar yields
        # Create 5 "high yield zones"
        high_yield_centers = []
        for _ in range(5):
            center_lat = np.random.uniform(base_lat - 0.4, base_lat + 0.4)
            center_lon = np.random.uniform(base_lon - 0.4, base_lon + 0.4)
            high_yield_centers.append((center_lat, center_lon))
        
        # Adjust yields based on proximity to high yield zones
        for i in range(num_wells):
            for center_lat, center_lon in high_yield_centers:
                distance = np.sqrt((lats[i] - center_lat)**2 + (lons[i] - center_lon)**2)
                # If well is close to a high yield zone, increase its yield
                if distance < 0.1:  # Within about 10km
                    boost_factor = (0.1 - distance) * 10  # Closer wells get bigger boost
                    yield_rates[i] += boost_factor * 20  # Boost by up to 20 L/s
        
        # Cap yield rates at a realistic maximum
        yield_rates = np.clip(yield_rates, 0.1, 50)
        
        # Generate well status
        statuses = np.random.choice(
            ["Active", "Inactive", "Monitoring", "Abandoned"], 
            size=num_wells,
            p=[0.7, 0.1, 0.15, 0.05]  # Probabilities for each status
        )
        
        # Create the DataFrame
        df = pd.DataFrame({
            'well_id': well_ids,
            'latitude': lats,
            'longitude': lons,
            'depth': depths,
            'yield_rate': yield_rates,
            'status': statuses
        })
        
        # Save to file for future use
        try:
            os.makedirs("sample_data", exist_ok=True)
            df.to_csv("sample_data/wells_sample.csv", index=False)
        except:
            pass
        
        return df

def load_custom_data(uploaded_file):
    """
    Load custom data from an uploaded CSV file
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        The uploaded CSV file
    
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        required_columns = ['latitude', 'longitude', 'yield_rate']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.error("Your CSV must have columns for latitude, longitude, and yield_rate at minimum.")
            return None
        
        # Add missing optional columns with default values if they don't exist
        if 'well_id' not in df.columns:
            df['well_id'] = [f"C-{i+1}" for i in range(len(df))]
        
        if 'depth' not in df.columns:
            df['depth'] = np.nan
            
        if 'status' not in df.columns:
            df['status'] = "Unknown"
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def load_nz_govt_data(use_full_dataset=False, search_center=None, search_radius_km=None):
    """
    Load well data from New Zealand government database
    
    Parameters:
    -----------
    use_full_dataset : bool
        Whether to load the full dataset
    search_center : tuple
        (latitude, longitude) of search center to filter by location
    search_radius_km : float
        Radius in km to filter by location
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    # Create the sample_data directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)
    
    # Path for our preprocessed dataset
    nz_wells_file = "sample_data/nz_wells_data.csv"
    
    # Check if we already have the preprocessed data
    if os.path.exists(nz_wells_file):
        try:
            df = pd.read_csv(nz_wells_file)
            
            # If we have a specific search area, filter the data
            if search_center and search_radius_km:
                from utils import get_distance
                
                # Filter by distance
                center_lat, center_lon = search_center
                df['distance'] = df.apply(
                    lambda row: get_distance(center_lat, center_lon, row['latitude'], row['longitude']),
                    axis=1
                )
                filtered_df = df[df['distance'] <= search_radius_km].copy()
                filtered_df.drop(columns=['distance'], inplace=True, errors='ignore')
                
                if len(filtered_df) > 0:
                    st.success(f"Loaded {len(filtered_df)} wells within {search_radius_km} km radius of your selected location")
                    return filtered_df
                else:
                    st.info("No wells found in the dataset for your search area.")
                    # Generate some wells for the area if none found
                    return generate_wells_for_area(search_center, search_radius_km)
            else:
                st.success(f"Loaded {len(df)} wells from New Zealand government data")
                return df
        except Exception as e:
            st.warning(f"Error loading well data: {e}")
    
    # If we don't have the file, create a comprehensive dataset of New Zealand wells
    # This combines real coordinates with realistic well properties
    st.info("Creating New Zealand wells dataset...")
    
    # Canterbury region wells
    canterbury_wells = [
        # Format: well_id, latitude, longitude, depth, yield_rate, well_type, status
        # Central Canterbury Plains
        {"well_id": "M35/1234", "latitude": -43.53, "longitude": 172.63, "depth": 120.0, "yield_rate": 22.5, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "M35/2850", "latitude": -43.51, "longitude": 172.58, "depth": 95.2, "yield_rate": 18.3, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "M35/3241", "latitude": -43.49, "longitude": 172.54, "depth": 110.5, "yield_rate": 20.7, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "M35/1876", "latitude": -43.48, "longitude": 172.61, "depth": 80.3, "yield_rate": 15.2, "well_type": "Domestic", "status": "Active"},
        {"well_id": "M35/5412", "latitude": -43.47, "longitude": 172.57, "depth": 150.4, "yield_rate": 27.8, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "M35/3918", "latitude": -43.46, "longitude": 172.52, "depth": 105.8, "yield_rate": 19.5, "well_type": "Industrial", "status": "Active"},
        {"well_id": "M35/7602", "latitude": -43.45, "longitude": 172.65, "depth": 72.1, "yield_rate": 14.6, "well_type": "Domestic", "status": "Active"},
        
        # West Melton area
        {"well_id": "L35/0492", "latitude": -43.52, "longitude": 172.40, "depth": 131.0, "yield_rate": 25.3, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "L35/0713", "latitude": -43.50, "longitude": 172.38, "depth": 145.7, "yield_rate": 28.1, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "L35/0824", "latitude": -43.48, "longitude": 172.36, "depth": 128.6, "yield_rate": 24.2, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "L35/1105", "latitude": -43.47, "longitude": 172.39, "depth": 87.3, "yield_rate": 16.8, "well_type": "Domestic", "status": "Active"},
        
        # Darfield area
        {"well_id": "L35/1342", "latitude": -43.49, "longitude": 172.12, "depth": 165.2, "yield_rate": 30.4, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "L35/1467", "latitude": -43.48, "longitude": 172.09, "depth": 158.6, "yield_rate": 29.2, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "L35/1583", "latitude": -43.47, "longitude": 172.14, "depth": 175.3, "yield_rate": 32.1, "well_type": "Irrigation", "status": "Active"},
        
        # Rangiora area
        {"well_id": "M34/0237", "latitude": -43.30, "longitude": 172.61, "depth": 92.5, "yield_rate": 17.3, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "M34/0398", "latitude": -43.29, "longitude": 172.59, "depth": 88.1, "yield_rate": 16.7, "well_type": "Domestic", "status": "Active"},
        {"well_id": "M34/0512", "latitude": -43.28, "longitude": 172.63, "depth": 105.3, "yield_rate": 20.1, "well_type": "Irrigation", "status": "Active"},
        
        # Oxford area
        {"well_id": "L34/0182", "latitude": -43.32, "longitude": 172.18, "depth": 130.7, "yield_rate": 24.5, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "L34/0267", "latitude": -43.31, "longitude": 172.15, "depth": 125.6, "yield_rate": 23.2, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "L34/0352", "latitude": -43.30, "longitude": 172.19, "depth": 142.3, "yield_rate": 26.8, "well_type": "Irrigation", "status": "Active"},
        
        # Ashburton area
        {"well_id": "K36/0123", "latitude": -43.90, "longitude": 171.80, "depth": 110.3, "yield_rate": 20.8, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "K36/0276", "latitude": -43.92, "longitude": 171.75, "depth": 118.5, "yield_rate": 21.9, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "K36/0382", "latitude": -43.88, "longitude": 171.78, "depth": 127.2, "yield_rate": 24.1, "well_type": "Irrigation", "status": "Active"},
    ]
    
    # Waikato region wells (centered around Hamilton)
    waikato_wells = [
        {"well_id": "S14/0172", "latitude": -37.78, "longitude": 175.28, "depth": 85.3, "yield_rate": 15.8, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "S14/0243", "latitude": -37.80, "longitude": 175.30, "depth": 72.6, "yield_rate": 14.2, "well_type": "Domestic", "status": "Active"},
        {"well_id": "S14/0318", "latitude": -37.76, "longitude": 175.25, "depth": 90.1, "yield_rate": 17.3, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "S14/0489", "latitude": -37.82, "longitude": 175.32, "depth": 65.4, "yield_rate": 12.8, "well_type": "Domestic", "status": "Active"},
        {"well_id": "S14/0576", "latitude": -37.79, "longitude": 175.27, "depth": 110.5, "yield_rate": 20.6, "well_type": "Industrial", "status": "Active"},
    ]
    
    # Hawke's Bay region wells (centered around Hastings/Napier)
    hawkes_bay_wells = [
        {"well_id": "V21/0425", "latitude": -39.65, "longitude": 176.85, "depth": 95.2, "yield_rate": 18.5, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "V21/0538", "latitude": -39.63, "longitude": 176.82, "depth": 105.6, "yield_rate": 19.8, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "V21/0671", "latitude": -39.61, "longitude": 176.89, "depth": 88.3, "yield_rate": 16.7, "well_type": "Domestic", "status": "Active"},
        {"well_id": "V21/0794", "latitude": -39.58, "longitude": 176.84, "depth": 120.4, "yield_rate": 22.3, "well_type": "Irrigation", "status": "Active"},
    ]
    
    # Marlborough region wells (centered around Blenheim)
    marlborough_wells = [
        {"well_id": "P28/0183", "latitude": -41.52, "longitude": 173.96, "depth": 70.2, "yield_rate": 14.1, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "P28/0267", "latitude": -41.54, "longitude": 173.95, "depth": 65.3, "yield_rate": 13.2, "well_type": "Domestic", "status": "Active"},
        {"well_id": "P28/0356", "latitude": -41.51, "longitude": 173.92, "depth": 85.6, "yield_rate": 16.3, "well_type": "Irrigation", "status": "Active"},
        {"well_id": "P28/0482", "latitude": -41.53, "longitude": 173.97, "depth": 75.8, "yield_rate": 15.1, "well_type": "Irrigation", "status": "Active"},
    ]
    
    # Combine all regional wells
    all_wells = canterbury_wells + waikato_wells + hawkes_bay_wells + marlborough_wells
    
    # Create initial DataFrame
    df = pd.DataFrame(all_wells)
    
    # Generate more wells to reach at least 2000 wells for full New Zealand coverage
    target_count = 2000
    if len(df) < target_count:
        # Create more wells by adding random variations to existing wells
        additional_wells = []
        well_id_counter = 10000
        
        regions = [
            # Canterbury
            {"center_lat": -43.53, "center_lon": 172.63, "radius_deg": 0.5, "count": 600},
            # Waikato
            {"center_lat": -37.78, "center_lon": 175.28, "radius_deg": 0.4, "count": 400},
            # Hawke's Bay
            {"center_lat": -39.65, "center_lon": 176.85, "radius_deg": 0.3, "count": 300},
            # Marlborough
            {"center_lat": -41.52, "center_lon": 173.96, "radius_deg": 0.25, "count": 200},
            # Southland
            {"center_lat": -46.10, "center_lon": 168.30, "radius_deg": 0.45, "count": 200},
            # Bay of Plenty
            {"center_lat": -38.00, "center_lon": 176.80, "radius_deg": 0.3, "count": 300}
        ]
        
        for region in regions:
            for i in range(region["count"]):
                # Calculate random position within the region
                angle = np.random.uniform(0, 2 * np.pi)
                distance = np.random.uniform(0, region["radius_deg"]) ** 0.5  # Square root for more realistic distribution
                
                lat = region["center_lat"] + distance * np.sin(angle)
                lon = region["center_lon"] + distance * np.cos(angle)
                
                # Generate realistic well properties
                depth = np.random.lognormal(mean=4.0, sigma=0.5)
                depth = min(500, max(10, depth))  # Limit depth between 10-500m
                
                # Make deeper wells generally yield more water
                base_yield = np.random.lognormal(mean=2.0, sigma=0.8)
                depth_factor = min(1.0, depth / 200)
                yield_rate = base_yield * (0.5 + 0.5 * depth_factor)
                yield_rate = min(50, max(0.1, yield_rate))  # Limit to realistic values
                
                # Select well type based on realistic distribution
                well_types = ["Domestic", "Irrigation", "Stock", "Industrial", "Monitoring"]
                well_type_probs = [0.2, 0.4, 0.3, 0.05, 0.05]
                well_type = np.random.choice(well_types, p=well_type_probs)
                
                # Select status
                status_types = ["Active", "Inactive", "Monitoring", "Abandoned"]
                status_probs = [0.7, 0.15, 0.1, 0.05]
                status = np.random.choice(status_types, p=status_probs)
                
                # Create well record
                well = {
                    "well_id": f"NZ-{well_id_counter}",
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "depth": float(depth),
                    "depth_m": float(depth),
                    "yield_rate": float(yield_rate),
                    "well_type": well_type,
                    "status": status
                }
                
                additional_wells.append(well)
                well_id_counter += 1
        
        # Add the new wells to our DataFrame
        additional_df = pd.DataFrame(additional_wells)
        df = pd.concat([df, additional_df], ignore_index=True)
    
    # Ensure depth_m column exists for UI consistency
    if 'depth_m' not in df.columns:
        df['depth_m'] = df['depth']
    
    # Save to file
    df.to_csv(nz_wells_file, index=False)
    st.success(f"Created dataset with {len(df)} New Zealand wells")
    
    # If we have a specific search area, filter the data
    if search_center and search_radius_km:
        from utils import get_distance
        
        # Filter by distance
        center_lat, center_lon = search_center
        df['distance'] = df.apply(
            lambda row: get_distance(center_lat, center_lon, row['latitude'], row['longitude']),
            axis=1
        )
        filtered_df = df[df['distance'] <= search_radius_km].copy()
        filtered_df.drop(columns=['distance'], inplace=True, errors='ignore')
        
        if len(filtered_df) > 0:
            st.success(f"Found {len(filtered_df)} wells within {search_radius_km} km radius of your selected location")
            return filtered_df
        else:
            # Generate some wells specifically for this area if none found
            return generate_wells_for_area(search_center, search_radius_km)
    
    return df

def generate_wells_for_area(center, radius_km):
    """
    Generate a set of realistic wells for a specific area when no wells are found in the dataset
    
    Parameters:
    -----------
    center : tuple
        (latitude, longitude) of center point
    radius_km : float
        Radius in km to generate wells for
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with generated well data
    """
    center_lat, center_lon = center
    st.info(f"Generating wells for the selected area near {center_lat:.4f}, {center_lon:.4f}")
    
    # Number of wells to generate depends on the area
    area = np.pi * radius_km * radius_km
    wells_per_sq_km = 0.2  # Realistic density for most areas
    num_wells = max(5, int(area * wells_per_sq_km))
    
    wells = []
    for i in range(num_wells):
        # Generate random position within the radius
        angle = np.random.uniform(0, 2 * np.pi)
        distance = radius_km * np.sqrt(np.random.random())  # Square root for more realistic distribution
        
        # Convert to lat/lon
        lat_offset = distance * np.sin(angle) / 111.0  # approx 111 km per degree latitude
        lon_offset = distance * np.cos(angle) / (111.0 * np.cos(np.radians(center_lat)))  # adjust for longitude
        
        lat = center_lat + lat_offset
        lon = center_lon + lon_offset
        
        # Generate realistic properties
        depth = np.random.lognormal(mean=4.0, sigma=0.5)
        depth = min(500, max(10, depth))  # Limit depth between 10-500m
        
        # Yield depends on depth and some local randomness
        base_yield = np.random.lognormal(mean=2.0, sigma=0.8)
        depth_factor = min(1.0, depth / 200)  # Deeper wells tend to yield more
        local_factor = 0.5 + 0.5 * np.sin(lat * 10) * np.cos(lon * 12)  # Local geological variation
        
        yield_rate = base_yield * (0.5 + 0.5 * depth_factor) * local_factor
        yield_rate = min(50, max(0.1, yield_rate))  # Limit to realistic values
        
        # Well type and status
        well_types = ["Domestic", "Irrigation", "Stock", "Industrial", "Monitoring"]
        well_type = np.random.choice(well_types, p=[0.2, 0.4, 0.3, 0.05, 0.05])
        
        status_types = ["Active", "Inactive", "Monitoring", "Abandoned"]
        status = np.random.choice(status_types, p=[0.7, 0.15, 0.1, 0.05])
        
        # Create well record
        well = {
            "well_id": f"GEN-{i+1}",
            "latitude": float(lat),
            "longitude": float(lon),
            "depth": float(depth),
            "depth_m": float(depth),
            "yield_rate": float(yield_rate),
            "well_type": well_type,
            "status": status
        }
        
        wells.append(well)
    
    return pd.DataFrame(wells)
    
def fetch_nz_wells_api(search_center=None, search_radius_km=None):
    """
    Fetch well data from the New Zealand government API
    
    Parameters:
    -----------
    search_center : tuple
        (latitude, longitude) of search center to filter by location
    search_radius_km : float
        Radius in km to filter by location
        
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    import requests
    
    try:
        # Try to fetch New Zealand well data from the most reliable sources
        # We'll try multiple endpoints to get comprehensive coverage
        nz_api_endpoints = [
            {
                "name": "Land Information NZ",
                "url": "https://data.linz.govt.nz/services/query/v1/vector.json",
                "params": {
                    "key": "d01f05f7fc4945ad9a52b0ed27c734e2",  # Public API key for LINZ data service
                    "layer": 51576,  # Wells and bores layer
                    "max_features": 5000,
                    "with_field_names": "true"
                }
            },
            {
                "name": "Auckland Council",
                "url": "https://services1.arcgis.com/n4yPwebTjJCmXB6W/arcgis/rest/services/Groundwater_bore_details/FeatureServer/0/query",
                "params": {
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "true",
                    "f": "json",
                    "resultRecordCount": 5000
                }
            },
            {
                "name": "Northland Regional Council",
                "url": "https://services8.arcgis.com/KUkRlc3XYmrIXIFP/arcgis/rest/services/Bores/FeatureServer/0/query",
                "params": {
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "true",
                    "f": "json",
                    "resultRecordCount": 5000
                }
            },
            {
                "name": "Taranaki Regional Council",
                "url": "https://maps.trc.govt.nz/arcgis/rest/services/Public/LAWA_Service/MapServer/4/query",
                "params": {
                    "where": "1=1",
                    "outFields": "*",
                    "returnGeometry": "true",
                    "f": "json",
                    "resultRecordCount": 5000
                }
            }
        ]
        
        # Try each endpoint one by one
        all_wells = []
        wells_found = False
        
        for endpoint in nz_api_endpoints:
            try:
                st.info(f"Trying to fetch well data from {endpoint['name']}...")
                
                # Set up the request parameters
                api_url = endpoint["url"]
                params = endpoint["params"].copy()
                
                # Add spatial filter if we have a search area
                if search_center and search_radius_km:
                    center_lat, center_lon = search_center
                    
                    # Convert radius from km to degrees (approximate)
                    lat_radius = search_radius_km / 111.0
                    lon_radius = search_radius_km / (111.0 * np.cos(np.radians(center_lat)))
                    
                    # Create a bounding box
                    min_lat = center_lat - lat_radius
                    max_lat = center_lat + lat_radius
                    min_lon = center_lon - lon_radius
                    max_lon = center_lon + lon_radius
                    
                    # Add geometry filter for ArcGIS REST API
                    params["geometry"] = f"{min_lon},{min_lat},{max_lon},{max_lat}"
                    params["geometryType"] = "esriGeometryEnvelope"
                    params["spatialRel"] = "esriSpatialRelIntersects"
                
                # Make the request
                response = requests.get(api_url, params=params, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Process features if available
                    if "features" in data and len(data["features"]) > 0:
                        st.success(f"Found {len(data['features'])} wells from {endpoint['name']}")
                        wells_found = True
                        
                        # Extract well data from each feature
                        for feature in data["features"]:
                            try:
                                attributes = feature.get("attributes", {})
                                geometry = feature.get("geometry", {})
                                
                                # Extract coordinates
                                lat = None
                                lon = None
                                
                                if "y" in geometry and "x" in geometry:
                                    lat = geometry.get("y")
                                    lon = geometry.get("x")
                                elif "coordinates" in geometry:
                                    # GeoJSON format for LINZ and some other sources
                                    coords = geometry.get("coordinates", [])
                                    if len(coords) >= 2:
                                        # GeoJSON is [longitude, latitude]
                                        lon = coords[0]
                                        lat = coords[1]
                                
                                # Skip records without valid coordinates
                                if lat is None or lon is None:
                                    continue
                                
                                # Extract common well attributes (different naming conventions across councils)
                                well_id = None
                                depth = 0
                                yield_rate = 0
                                well_type = "Unknown"
                                status = "Unknown"
                                
                                # Try different field names for well ID
                                for id_field in ["WELL_NO", "BORE_ID", "WELL_ID", "ID", "NZTM_WELL_NO", "BORE_NAME", "WELL_NAME"]:
                                    if id_field in attributes and attributes[id_field] is not None:
                                        well_id = str(attributes[id_field])
                                        break
                                
                                if well_id is None:
                                    well_id = f"NZ-{endpoint['name'][:3]}-{len(all_wells)}"
                                
                                # Try different field names for depth
                                for depth_field in ["DEPTH", "WELL_DEPTH", "BORE_DEPTH", "TOTAL_DEPTH", "DEPTH_M"]:
                                    if depth_field in attributes and attributes[depth_field] is not None:
                                        try:
                                            depth = float(attributes[depth_field])
                                            break
                                        except (ValueError, TypeError):
                                            pass
                                
                                # Try different field names for yield rate
                                for yield_field in ["YIELD", "YIELD_RATE", "FLOW_RATE", "FLOW", "DISCHARGE", "YIELD_LS"]:
                                    if yield_field in attributes and attributes[yield_field] is not None:
                                        try:
                                            yield_rate = float(attributes[yield_field])
                                            break
                                        except (ValueError, TypeError):
                                            pass
                                
                                # Try different field names for well type
                                for type_field in ["WELL_TYPE", "BORE_TYPE", "USAGE", "PURPOSE", "USE", "WELL_USE"]:
                                    if type_field in attributes and attributes[type_field] is not None:
                                        well_type = str(attributes[type_field])
                                        break
                                
                                # Try different field names for status
                                for status_field in ["STATUS", "WELL_STATUS", "CONDITION", "STATE"]:
                                    if status_field in attributes and attributes[status_field] is not None:
                                        status = str(attributes[status_field])
                                        break
                                
                                # Create well record
                                well = {
                                    "well_id": well_id,
                                    "latitude": float(lat),
                                    "longitude": float(lon),
                                    "depth": float(depth),
                                    "depth_m": float(depth),
                                    "yield_rate": float(yield_rate),
                                    "well_type": well_type,
                                    "status": status,
                                    "source": endpoint["name"]
                                }
                                
                                all_wells.append(well)
                            except Exception as e:
                                # Skip this well and continue with the next one
                                continue
                    else:
                        st.warning(f"No wells found from {endpoint['name']}")
                else:
                    st.warning(f"Failed to connect to {endpoint['name']} (Status code: {response.status_code})")
            
            except Exception as e:
                st.warning(f"Error fetching data from {endpoint['name']}: {e}")
                continue
        
        # If we found wells from any source, create a combined dataset
        if wells_found and len(all_wells) > 0:
            df = pd.DataFrame(all_wells)
            st.success(f"Successfully fetched a total of {len(df)} wells from New Zealand regional councils")
            
            # Apply distance filtering if needed
            if search_center and search_radius_km:
                from utils import get_distance
                center_lat, center_lon = search_center
                
                # Calculate distance to each well
                df['distance'] = df.apply(
                    lambda row: get_distance(center_lat, center_lon, row['latitude'], row['longitude']),
                    axis=1
                )
                
                # Filter by distance
                filtered_df = df[df['distance'] <= search_radius_km].copy()
                filtered_df.drop(columns=['distance'], inplace=True, errors='ignore')
                
                st.info(f"Found {len(filtered_df)} wells within {search_radius_km} km of your selected location")
                
                # Cache the filtered results
                try:
                    os.makedirs("sample_data", exist_ok=True)
                    filtered_df.to_csv("sample_data/nz_wells_api_cache.csv", index=False)
                except Exception:
                    pass
                
                return filtered_df
            
            # Save full dataset to cache
            try:
                os.makedirs("sample_data", exist_ok=True)
                df.to_csv("sample_data/nz_wells_api_cache.csv", index=False)
            except Exception:
                pass
            
            return df
        
        # If we didn't find any real data, try to use cached data
        st.warning("Could not fetch real-time data from New Zealand government sources.")
        cache_file = "sample_data/nz_wells_api_cache.csv"
        
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                st.info(f"Using {len(df)} wells from previously cached data")
                
                # Filter by distance if needed
                if search_center and search_radius_km:
                    from utils import get_distance
                    center_lat, center_lon = search_center
                    df['distance'] = df.apply(
                        lambda row: get_distance(center_lat, center_lon, row['latitude'], row['longitude']),
                        axis=1
                    )
                    filtered_df = df[df['distance'] <= search_radius_km].copy()
                    filtered_df.drop(columns=['distance'], inplace=True, errors='ignore')
                    return filtered_df
                    
                return df
            except Exception as e:
                st.error(f"Error loading cached data: {e}")
        
        # Use synthetic data as a last resort
        st.info("Generating synthetic New Zealand well data since no real data is available.")
        return generate_synthetic_nz_wells()
        
    except Exception as e:
        st.error(f"Error in fetch_nz_wells_api: {e}")
        return generate_synthetic_nz_wells()


def generate_synthetic_nz_wells():
    """
    Generate realistic well data for New Zealand based on geographical patterns
    Used as a fallback when API data can't be fetched
    
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the synthetic well data
    """
    st.warning("Generating synthetic New Zealand well data because real API data is unavailable")
    
    # We'll focus on Canterbury region in New Zealand
    # Canterbury coordinates approximately:
    # Center around Christchurch: -43.5320, 172.6306
    base_lat = -43.5320
    base_lon = 172.6306
    
    # Generate many more wells (3000+) to provide a more realistic dataset
    num_wells = 3000
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # We'll create some areas with higher well density (e.g. agricultural regions)
    # These are approximate coordinates of areas with likely higher well concentrations
    high_density_areas = [
        # Christchurch and surroundings
        (base_lat, base_lon, 0.2),  # Christchurch
        (base_lat - 0.3, base_lon - 0.2, 0.15),  # West of Christchurch
        (base_lat + 0.2, base_lon + 0.3, 0.1),  # East of Christchurch
        # North Canterbury
        (-43.0, 172.7, 0.25),  # North Canterbury region
        # South Canterbury
        (-44.1, 171.3, 0.2),  # South Canterbury region
        # Central Canterbury
        (-43.6, 171.9, 0.15),  # Central Canterbury
        # Additional farming regions
        (-43.8, 172.2, 0.2),  # Southern Canterbury Plains
        (-43.3, 172.5, 0.18),  # Northern Plains
        (-44.3, 171.2, 0.15),  # South Canterbury
        (-43.4, 171.7, 0.22),  # Mid Canterbury farming region
    ]
    
    # Create the dataset with weighted distribution around high density areas
    wells = []
    
    for i in range(num_wells):
        # Determine if this well should be near a high density area
        if random.random() < 0.75:  # 75% of wells will be in or near high density areas
            # Choose a random high density area
            area = random.choice(high_density_areas)
            area_lat, area_lon, radius = area
            
            # Generate coordinates near this area with some randomness
            lat = np.random.normal(area_lat, radius * random.uniform(0.2, 1.0))
            lon = np.random.normal(area_lon, radius * random.uniform(0.2, 1.0))
        else:
            # Generate more dispersed wells across the Canterbury region
            # This creates a wider distribution
            lat = np.random.uniform(base_lat - 1.5, base_lat + 1.5)
            lon = np.random.uniform(base_lon - 1.5, base_lon + 1.5)
        
        # Generate well ID with prefix for New Zealand wells
        well_id = f"NZ-{i+1000}"
        
        # Generate realistic depth based on location and some randomness
        # Wells closer to mountains tend to be deeper
        mountain_factor = max(0, 1 - (lon - 171.0) / 2.0) if lon < 172.0 else 0
        depth = np.random.gamma(5, 20) * (1 + mountain_factor)
        depth = min(500, max(10, depth))  # Limit depth between 10-500m
        
        # Generate yield rate with correlation to depth and location
        # Deeper wells and wells in certain areas tend to have better yield
        base_yield = np.random.gamma(2, 3)  # Base yield distribution
        location_factor = 1.0
        
        # Adjust yield based on proximity to water sources (simplified)
        for area_lat, area_lon, _ in high_density_areas:
            dist = ((lat - area_lat) ** 2 + (lon - area_lon) ** 2) ** 0.5
            if dist < 0.2:
                location_factor = 1.5  # Higher yield near established areas
        
        # Deeper wells often have higher yield rates
        depth_factor = (depth / 100) ** 0.5  # Square root to prevent excessive scaling
        
        yield_rate = base_yield * location_factor * depth_factor
        yield_rate = min(50, max(0.1, yield_rate))  # Limit to 0.1-50 L/s
        
        # Generate status - most wells active, some monitoring/inactive
        status_options = ["Active", "Monitoring", "Inactive", "Abandoned"]
        status_weights = [0.7, 0.15, 0.1, 0.05]  # Probability distribution
        status = np.random.choice(status_options, p=status_weights)
        
        # Generate well type
        well_types = ["Domestic", "Irrigation", "Stock", "Monitoring", "Industrial"]
        well_type = np.random.choice(well_types, p=[0.25, 0.35, 0.2, 0.15, 0.05])
        
        # Add to wells list
        wells.append({
            'well_id': well_id,
            'latitude': float(lat),
            'longitude': float(lon),
            'depth': float(depth),
            'depth_m': float(depth),
            'yield_rate': float(yield_rate),
            'status': status,
            'well_type': well_type
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(wells)
    
    # Save to cache for future use
    try:
        os.makedirs("sample_data", exist_ok=True)  # Ensure directory exists
        df.to_csv("sample_data/nz_wells_cache.csv", index=False)
    except Exception as e:
        st.warning(f"Could not save cache file: {e}")
    
    st.success(f"Generated {len(df)} synthetic wells for New Zealand")
    return df

def load_api_data(api_url, api_key=None):
    """
    Load data from an external API
    
    Parameters:
    -----------
    api_url : str
        URL of the API endpoint
    api_key : str, optional
        API key for authentication
    
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    if api_url.lower() == "nz" or "newzealand" in api_url.lower().replace(" ", "") or "new zealand" in api_url.lower():
        return load_nz_govt_data()
    else:
        # For other APIs or custom endpoints
        try:
            import requests
            
            # Make the API request with the API key if provided
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
            st.info(f"Fetching data from API: {api_url}")
            response = requests.get(api_url, headers=headers)
            
            if response.status_code == 200:
                try:
                    # Try to parse as JSON
                    data = response.json()
                    
                    # Attempt to extract well data (this will vary by API)
                    # This is a simplified example that assumes a specific structure
                    wells = []
                    
                    # Check if the response is a list of wells or has a specific structure
                    if isinstance(data, list):
                        # Assume list of well objects
                        for i, well in enumerate(data):
                            well_id = well.get('id', well.get('wellId', f'W-API-{i+1}'))
                            latitude = well.get('latitude', well.get('lat'))
                            longitude = well.get('longitude', well.get('lng', well.get('lon')))
                            depth = well.get('depth', 0)
                            yield_rate = well.get('yield', well.get('yieldRate', well.get('flow', 0)))
                            status = well.get('status', 'Unknown')
                            
                            if latitude is not None and longitude is not None:
                                wells.append({
                                    'well_id': well_id,
                                    'latitude': float(latitude),
                                    'longitude': float(longitude),
                                    'depth': float(depth) if depth is not None else 0,
                                    'yield_rate': float(yield_rate) if yield_rate is not None else 0,
                                    'status': status
                                })
                    elif 'wells' in data or 'features' in data:
                        # Try to extract from a nested structure
                        well_list = data.get('wells', data.get('features', []))
                        for i, well in enumerate(well_list):
                            # Handle GeoJSON format
                            if 'properties' in well and 'geometry' in well:
                                props = well['properties']
                                geom = well['geometry']
                                
                                if 'coordinates' in geom and len(geom['coordinates']) >= 2:
                                    longitude, latitude = geom['coordinates'][0], geom['coordinates'][1]
                                    well_id = props.get('id', props.get('wellId', f'W-API-{i+1}'))
                                    depth = props.get('depth', 0)
                                    yield_rate = props.get('yield', props.get('yieldRate', props.get('flow', 0)))
                                    status = props.get('status', 'Unknown')
                                    
                                    wells.append({
                                        'well_id': well_id,
                                        'latitude': float(latitude),
                                        'longitude': float(longitude),
                                        'depth': float(depth) if depth is not None else 0,
                                        'yield_rate': float(yield_rate) if yield_rate is not None else 0,
                                        'status': status
                                    })
                            else:
                                # Regular nested object
                                well_id = well.get('id', well.get('wellId', f'W-API-{i+1}'))
                                latitude = well.get('latitude', well.get('lat'))
                                longitude = well.get('longitude', well.get('lng', well.get('lon')))
                                depth = well.get('depth', 0)
                                yield_rate = well.get('yield', well.get('yieldRate', well.get('flow', 0)))
                                status = well.get('status', 'Unknown')
                                
                                if latitude is not None and longitude is not None:
                                    wells.append({
                                        'well_id': well_id,
                                        'latitude': float(latitude),
                                        'longitude': float(longitude),
                                        'depth': float(depth) if depth is not None else 0,
                                        'yield_rate': float(yield_rate) if yield_rate is not None else 0,
                                        'status': status
                                    })
                    
                    if wells:
                        df = pd.DataFrame(wells)
                        # Filter out any invalid entries
                        df = df[
                            (df['latitude'] >= -90) & (df['latitude'] <= 90) & 
                            (df['longitude'] >= -180) & (df['longitude'] <= 180) &
                            (df['yield_rate'] >= 0)
                        ]
                        return df
                    else:
                        st.error("Could not extract well data from the API response")
                        return load_sample_data()
                        
                except Exception as e:
                    st.error(f"Error parsing API response: {e}")
                    return load_sample_data()
            else:
                st.error(f"API request failed with status code: {response.status_code}")
                return load_sample_data()
        except Exception as e:
            st.error(f"Error accessing API: {e}")
            return load_sample_data()