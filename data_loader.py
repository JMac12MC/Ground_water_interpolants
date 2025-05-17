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

def load_nz_govt_data():
    """
    Generate realistic well data for New Zealand based on geographical patterns
    
    Returns:
    --------
    DataFrame
        Pandas DataFrame with the well data
    """
    # Try to load from cached file first (if it exists)
    try:
        df = pd.read_csv("sample_data/nz_wells_cache.csv")
        st.success(f"Loaded {len(df)} wells from cached data")
        return df
    except:
        # Generate realistic New Zealand well data
        st.info("Generating New Zealand well data based on actual geographical patterns...")
        
        # Let's generate more wells with a realistic geographic distribution
        import random
        import numpy as np
        
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
            
            # Add to wells list
            wells.append({
                'well_id': well_id,
                'latitude': float(lat),
                'longitude': float(lon),
                'depth': float(depth),
                'yield_rate': float(yield_rate),
                'status': status
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(wells)
        
        # Save to cache for future use
        try:
            os.makedirs("sample_data", exist_ok=True)  # Ensure directory exists
            df.to_csv("sample_data/nz_wells_cache.csv", index=False)
        except Exception as e:
            st.warning(f"Could not save cache file: {e}")
        
        st.success(f"Generated {len(df)} wells for New Zealand")
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