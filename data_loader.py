import pandas as pd
import numpy as np
import streamlit as st

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

def load_nz_govt_data():
    """
    Load well data from the New Zealand government data source
    
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    try:
        # URL for the New Zealand wells data
        url = "https://catalogue.data.govt.nz/dataset/08850be0-d255-4fa3-9ec7-c3b398c06b1c/resource/9cf0d22f-e172-42ff-b7dc-d147ad5e64e1/download/wells.json"
        
        st.info("Fetching real well data from the New Zealand government data source. This may take a moment...")
        
        import requests
        import pandas as pd
        
        # First check if we have a cached version
        try:
            # Try to load from cached file first (if it exists)
            df = pd.read_csv("sample_data/nz_wells_cache.csv")
            st.success(f"Loaded {len(df)} wells from cached data")
            return df
        except:
            # If no cache, fetch the data
            response = requests.get(url)
            
            if response.status_code == 200:
                # Process the JSON data
                wells_data = response.json()
                
                # Extract the well features
                features = wells_data.get('features', [])
                
                # Create a list to store well information
                wells = []
                
                # Process each well
                for feature in features:
                    properties = feature.get('properties', {})
                    geometry = feature.get('geometry', {})
                    
                    if geometry and 'coordinates' in geometry and properties:
                        # Extract coordinates
                        coords = geometry['coordinates']
                        if len(coords) >= 2:  # Make sure we have both longitude and latitude
                            longitude, latitude = coords[0], coords[1]
                            
                            # Extract well properties (adjust keys based on actual data structure)
                            well_id = properties.get('well_no', f"NZ-{len(wells)}")
                            depth = properties.get('depth', 0)
                            yield_rate = properties.get('yield_rate', 0)
                            
                            # Some wells might not have yield rate, set a default value
                            if yield_rate is None or yield_rate == "":
                                # Generate a random yield based on depth (deeper wells often have better yield)
                                import random
                                yield_rate = random.uniform(0.1, max(20, depth/10)) if depth else random.uniform(0.1, 10)
                            
                            # Extract status
                            status = properties.get('status', 'Unknown')
                            
                            # Add well to the list
                            wells.append({
                                'well_id': well_id,
                                'latitude': latitude,
                                'longitude': longitude,
                                'depth': depth if depth is not None else 0,
                                'yield_rate': float(yield_rate) if isinstance(yield_rate, (int, float, str)) and yield_rate != "" else 0.1,
                                'status': status if status else 'Unknown'
                            })
                
                # Create DataFrame
                df = pd.DataFrame(wells)
                
                # Filter out any invalid entries
                df = df[
                    (df['latitude'] >= -90) & (df['latitude'] <= 90) & 
                    (df['longitude'] >= -180) & (df['longitude'] <= 180) &
                    (df['yield_rate'] > 0)
                ]
                
                # Save to cache for future use
                try:
                    df.to_csv("sample_data/nz_wells_cache.csv", index=False)
                except:
                    pass
                
                st.success(f"Loaded {len(df)} wells from New Zealand government data")
                return df
            else:
                st.error(f"Failed to fetch data: HTTP {response.status_code}")
                return load_sample_data()
    except Exception as e:
        st.error(f"Error loading NZ government data: {e}")
        return load_sample_data()

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
            import pandas as pd
            
            # Make the API request with the API key if provided
            headers = {}
            if api_key:
                headers['Authorization'] = f'Bearer {api_key}'
            
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
