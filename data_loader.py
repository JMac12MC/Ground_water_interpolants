import pandas as pd
import numpy as np
import streamlit as st
import os
import random
import pyproj

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

def load_nz_govt_data(search_center=None, search_radius_km=None):
    """
    Load well data from the Wells_30k dataset

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
    # Create the sample_data directory if it doesn't exist
    os.makedirs("sample_data", exist_ok=True)

    # Path for the new Wells_30k dataset
    wells_30k_file = "attached_assets/Wells_30k_1751105715343.csv"

    if os.path.exists(wells_30k_file):
        with st.spinner("Loading well database..."):
            pass

        try:
            # Read the CSV file directly
            raw_df = pd.read_csv(wells_30k_file)

            # Create DataFrame with well IDs
            wells_df = pd.DataFrame()
            wells_df['well_id'] = raw_df['WELL_NO']

            # Process NZTM coordinates (New Zealand Transverse Mercator)
            # The Wells_30k dataset has NZTMX and NZTMY columns
            if 'NZTMX' in raw_df.columns and 'NZTMY' in raw_df.columns:
                # Convert to numeric, handling any non-numeric values
                nztmx = pd.to_numeric(raw_df['NZTMX'], errors='coerce')
                nztmy = pd.to_numeric(raw_df['NZTMY'], errors='coerce')

                # Use pyproj to perform proper coordinate transformation
                # NZTM2000 (EPSG:2193) to WGS84 (EPSG:4326)
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:2193",  # NZTM2000 
                    "EPSG:4326",  # WGS84 (standard lat/long)
                    always_xy=True
                )

                # Create dataframe with valid coordinates only
                valid_coords = ~nztmx.isna() & ~nztmy.isna()

                # Initialize with placeholder values
                lat = np.full(len(nztmy), np.nan)
                lon = np.full(len(nztmx), np.nan)

                # Only transform coordinates that are valid
                if valid_coords.any():
                    # Get valid coordinates
                    valid_x = nztmx[valid_coords].values
                    valid_y = nztmy[valid_coords].values

                    # Transform coordinates using pyproj
                    transformed_lon, transformed_lat = transformer.transform(valid_x, valid_y)

                    # Assign transformed coordinates back to arrays
                    lon[valid_coords] = transformed_lon
                    lat[valid_coords] = transformed_lat

                # Set latitude and longitude in wells_df
                wells_df['latitude'] = lat
                wells_df['longitude'] = lon

                # Also keep original NZTM coordinates
                wells_df['nztm_x'] = nztmx
                wells_df['nztm_y'] = nztmy
            
            # Fallback to NZMG coordinates if needed
            elif 'NZMGX' in raw_df.columns and 'NZMGY' in raw_df.columns:
                # Convert to numeric, handling any non-numeric values  
                nzmgx = pd.to_numeric(raw_df['NZMGX'], errors='coerce')
                nzmgy = pd.to_numeric(raw_df['NZMGY'], errors='coerce')

                # Use pyproj to perform proper coordinate transformation
                # NZMG (EPSG:27200) to WGS84 (EPSG:4326)
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:27200",  # NZMG (New Zealand Map Grid) 
                    "EPSG:4326",   # WGS84 (standard lat/long)
                    always_xy=True
                )

                # Create dataframe with valid coordinates only
                valid_coords = ~nzmgx.isna() & ~nzmgy.isna()

                # Initialize with placeholder values
                lat = np.full(len(nzmgy), np.nan)
                lon = np.full(len(nzmgx), np.nan)

                # Only transform coordinates that are valid
                if valid_coords.any():
                    # Get valid coordinates
                    valid_x = nzmgx[valid_coords].values
                    valid_y = nzmgy[valid_coords].values

                    # Transform coordinates using pyproj
                    transformed_lon, transformed_lat = transformer.transform(valid_x, valid_y)

                    # Assign transformed coordinates back to arrays
                    lon[valid_coords] = transformed_lon
                    lat[valid_coords] = transformed_lat

                # Set latitude and longitude in wells_df
                wells_df['latitude'] = lat
                wells_df['longitude'] = lon

                # Also keep original NZMG coordinates
                wells_df['nzmg_x'] = nzmgx
                wells_df['nzmgy'] = nzmgy

            # Calculate depth to groundwater using screen depths if available
            screen_depths = []
            for screen_col in ['TOP_SCREEN_1', 'TOP_SCREEN_2', 'TOP_SCREEN_3', 'BOTTOM_SCREEN_1', 'BOTTOM_SCREEN_2', 'BOTTOM_SCREEN_3']:
                if screen_col in raw_df.columns:
                    screen_depths.append(pd.to_numeric(raw_df[screen_col], errors='coerce'))

            if screen_depths:
                # Combine all screen depths and find the minimum (shallowest groundwater)
                all_screens = pd.concat(screen_depths, axis=1)
                wells_df['depth_to_groundwater'] = all_screens.min(axis=1, skipna=True)
                
                # Mark wells with no screen data as dry wells
                wells_df['is_dry_well'] = wells_df['depth_to_groundwater'].isna()
                
                # For display purposes, use screen depth or fall back to drill hole depth
                wells_df['display_depth'] = wells_df['depth_to_groundwater'].fillna(pd.to_numeric(raw_df['DEPTH'], errors='coerce'))
            else:
                # Use drill hole depth as fallback
                wells_df['depth_to_groundwater'] = pd.to_numeric(raw_df['DEPTH'], errors='coerce')
                wells_df['is_dry_well'] = wells_df['depth_to_groundwater'].isna()
                wells_df['display_depth'] = wells_df['depth_to_groundwater']

            # Keep original depth for reference
            wells_df['drill_hole_depth'] = pd.to_numeric(raw_df['DEPTH'], errors='coerce').fillna(0)
            # For general display, use display_depth
            wells_df['depth'] = wells_df['display_depth']
            wells_df['depth_m'] = wells_df['display_depth']

            # Add yield information - use MAX_YIELD from the Wells_30k dataset
            if 'MAX_YIELD' in raw_df.columns:
                # Convert to numeric and fill missing values with 0
                wells_df['yield_rate'] = pd.to_numeric(raw_df['MAX_YIELD'], errors='coerce').fillna(0)
            else:
                # Fallback - generate yields based on depth if MAX_YIELD not available
                wells_df['yield_rate'] = (wells_df['depth'] / 10).clip(lower=0.1, upper=40)

            # Only mark wells as dry if they have depth data but no yield
            # Wells with missing depth data will be excluded from interpolation entirely
            has_depth_data = wells_df['depth_to_groundwater'].notna() | wells_df['drill_hole_depth'].notna()
            wells_df['is_dry_well'] = wells_df['is_dry_well'] | (has_depth_data & (wells_df['yield_rate'] == 0.0))

            # Remove wells with no depth information from the dataset
            # These wells cannot contribute to either yield or depth interpolation
            wells_df = wells_df[has_depth_data].copy()

            # Add well type and status information using Wells_30k columns
            wells_df['well_type'] = raw_df['WELL_TYPE_DESC'].fillna('Unknown')
            wells_df['status'] = raw_df['WELL_STATUS_DESC'].fillna('Unknown')

            # Add well use information from USE_CODE_1_DESC
            wells_df['well_use'] = raw_df['USE_CODE_1_DESC'].fillna('Unknown')

            # Filter out geotechnical/geological investigation wells using well_use
            geotechnical_mask = wells_df['well_use'].str.contains(
                'Geotechnical.*Investigation|Geological.*Investigation', 
                case=False, 
                na=False, 
                regex=True
            )
            wells_df = wells_df[~geotechnical_mask].copy()

            # Add locality if available
            if 'LOCALITY' in raw_df.columns:
                wells_df['locality'] = raw_df['LOCALITY'].fillna('Unknown')
            else:
                wells_df['locality'] = 'Canterbury Region'

            # Filter to keep only valid wells
            # Keep wells with valid coordinates in New Zealand bounds
            valid_wells = wells_df[
                (wells_df['latitude'] >= -47) & 
                (wells_df['latitude'] <= -41) &
                (wells_df['longitude'] >= 168) & 
                (wells_df['longitude'] <= 174)
            ].copy()

            if len(valid_wells) == 0:
                st.warning("No valid wells found in the dataset. The coordinate conversion may need adjustment.")
                return generate_wells_for_area((-43.5, 172.5), 100)

            # Simple message showing well count without technical jargon
            if len(valid_wells) > 0:
                st.info(f"Using {len(valid_wells):,} wells from Wells_30k database")

            # If we have a specific search area, filter by distance
            if search_center and search_radius_km:
                from utils import get_distance

                # Calculate distance from search center
                center_lat, center_lon = search_center
                valid_wells['distance'] = valid_wells.apply(
                    lambda row: get_distance(center_lat, center_lon, row['latitude'], row['longitude']),
                    axis=1
                )

                # Keep only wells within the radius
                nearby_wells = valid_wells[valid_wells['distance'] <= search_radius_km].copy()

                if len(nearby_wells) > 0:
                    # ALWAYS show ALL wells within the search radius without ANY filtering whatsoever
                    # No limits, no sampling - show every single well even if there are thousands
                    return nearby_wells
                else:
                    st.info(f"No wells found within {search_radius_km} km of your location. Generating sample wells for the area.")
                    return generate_wells_for_area(search_center, search_radius_km)

            return valid_wells

        except Exception as e:
            st.error(f"Error processing Wells_30k data: {e}")
            # Generate fallback wells
            return generate_wells_for_area((-43.5, 172.5), 100)
    else:
        st.error(f"Wells_30k data file not found at {wells_30k_file}")
        # Generate fallback wells
        return generate_wells_for_area((-43.5, 172.5), 100)

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