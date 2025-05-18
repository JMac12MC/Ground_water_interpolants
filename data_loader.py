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
    import requests
    from io import StringIO
    
    # Local cache file paths
    official_cache_file = "sample_data/nz_wells_official.csv"
    processed_cache_file = "sample_data/nz_wells_processed.csv"
    
    # Try to use the processed cache first if we're not using full dataset
    if not use_full_dataset and os.path.exists(processed_cache_file):
        try:
            df = pd.read_csv(processed_cache_file)
            
            # If we have a specific search area, filter the cached data
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
                    st.success(f"Loaded {len(filtered_df)} wells from cached data within {search_radius_km} km radius")
                    return filtered_df
                else:
                    st.info("No wells found in the cached data for your search area.")
            else:
                st.success(f"Loaded {len(df)} wells from processed cached data")
                return df
                
        except Exception as e:
            st.warning(f"Error loading processed cached data: {e}")
    
    # Check if we need to download the official dataset
    if use_full_dataset or not os.path.exists(official_cache_file):
        st.info("Downloading official well data from New Zealand government catalog...")
        
        try:
            # Direct download link for the CSV file from data.govt.nz
            # This is for the "Wells and Bores - All" dataset
            csv_url = "https://data.mfe.govt.nz/tabledata/89712/csv?_ga=2.267982213.1190558128.1620150524-536494945.1620150524"
            
            # Download the CSV file
            response = requests.get(csv_url, timeout=30)
            
            if response.status_code == 200:
                # Save the raw CSV to cache
                os.makedirs("sample_data", exist_ok=True)
                with open(official_cache_file, "wb") as f:
                    f.write(response.content)
                
                st.success("Successfully downloaded official New Zealand well data")
            else:
                st.error(f"Failed to download well data: Status code {response.status_code}")
                # If we have processed data, use that instead
                if os.path.exists(processed_cache_file):
                    return pd.read_csv(processed_cache_file)
                else:
                    return generate_synthetic_nz_wells()
        except Exception as e:
            st.error(f"Error downloading official well data: {e}")
            # If we have processed data, use that instead
            if os.path.exists(processed_cache_file):
                return pd.read_csv(processed_cache_file)
            else:
                return generate_synthetic_nz_wells()
    
    # Process the official data (whether newly downloaded or already cached)
    if os.path.exists(official_cache_file):
        try:
            # Load the official data
            st.info("Processing the New Zealand government well data...")
            raw_df = pd.read_csv(official_cache_file)
            
            # Process and clean the data
            # Map the columns to our standard format
            column_mapping = {
                # These column names are based on the expected CSV format from data.govt.nz
                # Adjust as needed based on the actual CSV file
                'well_id': 'well_id',
                'Well No': 'well_id',
                'Northing': 'northing',
                'Easting': 'easting',
                'Latitude': 'latitude',
                'NZTMY': 'northing',
                'NZTMX': 'easting',
                'Lat': 'latitude',
                'Long': 'longitude',
                'Longitude': 'longitude',
                'Depth': 'depth',
                'DEPTH': 'depth',
                'Depth_m': 'depth_m',
                'GROUND_WATER_LEVEL': 'ground_water_level',
                'Yield': 'yield_rate',
                'YIELD': 'yield_rate',
                'Yield_Rate': 'yield_rate',
                'Yield_L_s': 'yield_rate',
                'Yield_m3_d': 'yield_m3_d',
                'Well_Type': 'well_type',
                'WELL_TYPE': 'well_type',
                'Purpose': 'well_type',
                'Status': 'status',
                'STATE': 'status'
            }
            
            # Rename columns that exist in the DataFrame
            columns_to_rename = {old: new for old, new in column_mapping.items() if old in raw_df.columns}
            processed_df = raw_df.rename(columns=columns_to_rename)
            
            # Required columns for our application
            required_columns = ['well_id', 'latitude', 'longitude', 'depth', 'yield_rate']
            
            # Check if we need to convert NZTM (New Zealand Transverse Mercator) to lat/long
            if ('northing' in processed_df.columns and 'easting' in processed_df.columns and 
                ('latitude' not in processed_df.columns or 'longitude' not in processed_df.columns)):
                st.info("Converting NZTM coordinates to latitude/longitude...")
                
                # Simple approximation for NZTM to WGS84 (latitude/longitude)
                # This is a very rough conversion and should be replaced with proper projection methods
                # For a production app, use pyproj or a similar library for accurate conversion
                processed_df['latitude'] = (processed_df['northing'] - 5400000) / 111000 * -1
                processed_df['longitude'] = (processed_df['easting'] - 1600000) / 85000 + 172
            
            # Ensure we have all required columns
            for col in required_columns:
                if col not in processed_df.columns:
                    if col == 'well_id':
                        processed_df['well_id'] = [f"NZ-{i}" for i in range(len(processed_df))]
                    elif col in ['latitude', 'longitude', 'depth', 'yield_rate']:
                        processed_df[col] = 0.0
            
            # Ensure depth_m is present for UI consistency
            if 'depth' in processed_df.columns and 'depth_m' not in processed_df.columns:
                processed_df['depth_m'] = processed_df['depth']
            
            # Ensure well_type and status are present
            if 'well_type' not in processed_df.columns:
                processed_df['well_type'] = "Unknown"
            if 'status' not in processed_df.columns:
                processed_df['status'] = "Unknown"
            
            # Convert numeric columns to float
            for col in ['latitude', 'longitude', 'depth', 'depth_m', 'yield_rate']:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0)
            
            # Filter out rows with invalid coordinates
            processed_df = processed_df[
                (processed_df['latitude'] != 0) & 
                (processed_df['longitude'] != 0) &
                (processed_df['latitude'] >= -50) & (processed_df['latitude'] <= -30) &  # Valid range for NZ
                (processed_df['longitude'] >= 165) & (processed_df['longitude'] <= 180)   # Valid range for NZ
            ]
            
            # Save the processed data to cache
            processed_df.to_csv(processed_cache_file, index=False)
            
            st.success(f"Successfully processed {len(processed_df)} wells from New Zealand government data")
            
            # Apply spatial filtering if needed
            if search_center and search_radius_km:
                from utils import get_distance
                
                center_lat, center_lon = search_center
                processed_df['distance'] = processed_df.apply(
                    lambda row: get_distance(center_lat, center_lon, row['latitude'], row['longitude']),
                    axis=1
                )
                filtered_df = processed_df[processed_df['distance'] <= search_radius_km].copy()
                filtered_df.drop(columns=['distance'], inplace=True, errors='ignore')
                
                st.info(f"Found {len(filtered_df)} wells within {search_radius_km} km of your selected location")
                return filtered_df
            
            return processed_df
            
        except Exception as e:
            st.error(f"Error processing official well data: {e}")
            return generate_synthetic_nz_wells()
    
    # If all else fails, generate synthetic data
    st.warning("Unable to access real New Zealand well data. Using synthetic data as a fallback.")
    return generate_synthetic_nz_wells()
    
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