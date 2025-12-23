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



def categorize_wells(wells_df):
    """
    Categorize wells into clear groups for different interpolation purposes.
    This is the single source of truth for well categorization.

    Returns:
    --------
    dict: Contains categorized DataFrames for different interpolation purposes
    """
    # Start with basic data validation
    wells_df = wells_df.copy()

    # Convert yield and depth columns to numeric
    wells_df['yield_rate'] = pd.to_numeric(wells_df['yield_rate'], errors='coerce')
    if 'depth_to_groundwater' in wells_df.columns:
        wells_df['depth_to_groundwater'] = pd.to_numeric(wells_df['depth_to_groundwater'], errors='coerce')
    if 'depth' in wells_df.columns:
        wells_df['depth'] = pd.to_numeric(wells_df['depth'], errors='coerce')

    # Check for screen data (indicates groundwater access)
    screen_columns = ['TOP_SCREEN_1', 'TOP_SCREEN_2', 'TOP_SCREEN_3', 'BOTTOM_SCREEN_1', 'BOTTOM_SCREEN_2', 'BOTTOM_SCREEN_3']
    has_screen_data = pd.Series(False, index=wells_df.index)

    for screen_col in screen_columns:
        if screen_col in wells_df.columns:
            screen_values = pd.to_numeric(wells_df[screen_col], errors='coerce')
            has_screen_data |= screen_values.notna()

    # Determine depth information
    has_depth_to_groundwater = False
    if 'depth_to_groundwater' in wells_df.columns:
        has_depth_to_groundwater = wells_df['depth_to_groundwater'].notna().any()

    has_depth = False
    if 'depth' in wells_df.columns:
        has_depth = wells_df['depth'].notna().any()

    # Create clear categories
    categories = {}

    # Category 1: Wells with actual yield data (for yield interpolation)
    # SIMPLIFIED: Only use wells with valid MAX_YIELD values
    yield_wells = wells_df[
        wells_df['yield_rate'].notna() & 
        (wells_df['yield_rate'] > 0)
    ].copy()
    categories['yield_wells'] = yield_wells

    # Category 2: Wells with depth data but NO yield data (for depth interpolation only)
    if has_depth_to_groundwater:
        depth_column = 'depth_to_groundwater'
        depth_wells = wells_df[
            wells_df[depth_column].notna() & 
            (wells_df[depth_column] > 0) &
            wells_df['yield_rate'].isna()
        ].copy()
    elif has_depth:
        depth_column = 'depth'
        depth_wells = wells_df[
            wells_df[depth_column].notna() & 
            (wells_df[depth_column] > 0) &
            wells_df['yield_rate'].isna()
        ].copy()
    else:
        depth_wells = pd.DataFrame()

    categories['depth_only_wells'] = depth_wells

    # Category 3: Wells with BOTH yield and depth data (can be used for both)
    if has_depth_to_groundwater:
        depth_column = 'depth_to_groundwater'
        both_wells = wells_df[
            wells_df['yield_rate'].notna() & 
            (wells_df['yield_rate'] > 0) &
            wells_df[depth_column].notna() & 
            (wells_df[depth_column] > 0)
        ].copy()
    elif has_depth:
        depth_column = 'depth'
        both_wells = wells_df[
            wells_df['yield_rate'].notna() & 
            (wells_df['yield_rate'] > 0) &
            wells_df[depth_column].notna() & 
            (wells_df[depth_column] > 0)
        ].copy()
    else:
        both_wells = pd.DataFrame()

    categories['both_wells'] = both_wells

    # Category 4: Wells with depth data but no yield data (used for depth interpolation)
    # These wells are excluded from yield interpolation but can be used for depth
    wells_without_yield = wells_df[
        wells_df['yield_rate'].isna() &
        has_screen_data
    ].copy()
    categories['wells_without_yield'] = wells_without_yield

    return categories

def get_wells_for_interpolation(wells_df, interpolation_type):
    """
    Get the appropriate wells for a specific interpolation type.

    Parameters:
    -----------
    wells_df : DataFrame
        Full wells dataset
    interpolation_type : str
        'yield' or 'depth'

    Returns:
    --------
    DataFrame
        Wells appropriate for the specified interpolation
    """
    categories = categorize_wells(wells_df)

    if interpolation_type == 'yield':
        # For yield interpolation: ONLY use wells with actual MAX_YIELD values
        # No dry well logic - just use the numeric values from MAX_YIELD
        yield_wells = categories['yield_wells']
        both_wells = categories['both_wells']

        # Combine wells that have actual yield measurements
        all_yield_wells = []

        if not yield_wells.empty:
            all_yield_wells.append(yield_wells)

        if not both_wells.empty:
            all_yield_wells.append(both_wells)

        if all_yield_wells:
            return pd.concat(all_yield_wells, ignore_index=True)
        else:
            return pd.DataFrame()

    elif interpolation_type == 'specific_capacity':
        # For specific capacity interpolation: ONLY use wells with actual specific capacity data
        # No fallback logic - if no specific capacity data exists, exclude the well entirely
        wells_with_specific_capacity = wells_df[
            wells_df['specific_capacity'].notna() & 
            (wells_df['specific_capacity'] > 0)
        ].copy()

        return wells_with_specific_capacity

    elif interpolation_type == 'ground_water_level':
        # For ground water level interpolation: ONLY use wells with actual ground water level data
        # Allow all numeric values including 0 and negative values (artesian conditions)
        # Check for the column that contains ground water level values
        if 'ground water level' in wells_df.columns:
            wells_with_gwl = wells_df[
                wells_df['ground water level'].notna()
                # Removed != 0 filter - zeros are valid (surface level or converted artesian)
            ].copy()
        else:
            # Fallback to empty DataFrame if column doesn't exist
            wells_with_gwl = pd.DataFrame()

        return wells_with_gwl



    elif interpolation_type == 'depth':
        # For depth interpolation: use wells with depth data (including wells without yield)
        depth_wells = categories['depth_only_wells']
        both_wells = categories['both_wells']
        wells_without_yield = categories.get('wells_without_yield', pd.DataFrame())

        # Combine wells that have depth information
        all_depth_wells = []

        if not depth_wells.empty:
            all_depth_wells.append(depth_wells)

        if not both_wells.empty:
            all_depth_wells.append(both_wells)

        # Include wells without yield if they have depth data
        if not wells_without_yield.empty:
            # Filter wells without yield to only those with depth data
            if 'depth_to_groundwater' in wells_without_yield.columns:
                wells_with_depth = wells_without_yield[
                    wells_without_yield['depth_to_groundwater'].notna() & 
                    (wells_without_yield['depth_to_groundwater'] > 0)
                ].copy()
            elif 'depth' in wells_without_yield.columns:
                wells_with_depth = wells_without_yield[
                    wells_without_yield['depth'].notna() & 
                    (wells_without_yield['depth'] > 0)
                ].copy()
            else:
                wells_with_depth = pd.DataFrame()

            if not wells_with_depth.empty:
                all_depth_wells.append(wells_with_depth)

        if all_depth_wells:
            return pd.concat(all_depth_wells, ignore_index=True)
        else:
            return pd.DataFrame()

    else:
        raise ValueError(f"Unknown interpolation type: {interpolation_type}")

def load_nz_govt_data(search_center=None, search_radius_km=None):
    """
    Load well data from the attached Wells and Bores dataset

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

    # Path for the new attached Wells and Bores dataset
    wells_30k_file = "attached_assets/19_07_2025 - Wells_and_Bores_1752906389250.csv"

    if os.path.exists(wells_30k_file):
        with st.spinner("Loading well database..."):
            pass

        try:
            # Read the CSV file directly
            raw_df = pd.read_csv(wells_30k_file)

            # Create DataFrame with well IDs
            wells_df = pd.DataFrame()
            wells_df['well_id'] = raw_df['WELL_NO']

            # Process coordinates - this dataset has X,Y coordinates and NZTM coordinates
            # First try NZTM coordinates if available
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

            # Try X,Y coordinates (these appear to be in a projected coordinate system)
            elif 'X' in raw_df.columns and 'Y' in raw_df.columns:
                # Convert to numeric, handling any non-numeric values
                x_coords = pd.to_numeric(raw_df['X'], errors='coerce')
                y_coords = pd.to_numeric(raw_df['Y'], errors='coerce')

                # Determine coordinate system based on coordinate ranges
                # If coordinates are in millions, likely NZTM; if small, likely lat/lon
                if x_coords.max() > 1000000:  # NZTM coordinates
                    # Use pyproj to transform from NZTM to WGS84
                    transformer = pyproj.Transformer.from_crs(
                        "EPSG:2193",  # NZTM2000 
                        "EPSG:4326",  # WGS84 (standard lat/long)
                        always_xy=True
                    )

                    # Create dataframe with valid coordinates only
                    valid_coords = ~x_coords.isna() & ~y_coords.isna()

                    # Initialize with placeholder values
                    lat = np.full(len(y_coords), np.nan)
                    lon = np.full(len(x_coords), np.nan)

                    # Only transform coordinates that are valid
                    if valid_coords.any():
                        # Get valid coordinates
                        valid_x = x_coords[valid_coords].values
                        valid_y = y_coords[valid_coords].values

                        # Transform coordinates using pyproj
                        transformed_lon, transformed_lat = transformer.transform(valid_x, valid_y)

                        # Assign transformed coordinates back to arrays
                        lon[valid_coords] = transformed_lon
                        lat[valid_coords] = transformed_lat

                    # Set latitude and longitude in wells_df
                    wells_df['latitude'] = lat
                    wells_df['longitude'] = lon

                else:  # Already in lat/lon format
                    wells_df['latitude'] = y_coords
                    wells_df['longitude'] = x_coords

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
                    # Keep the original screen columns for categorization
                    wells_df[screen_col] = pd.to_numeric(raw_df[screen_col], errors='coerce')

            # Set depth to groundwater
            if screen_depths:
                all_screens = pd.concat(screen_depths, axis=1)
                wells_df['depth_to_groundwater'] = all_screens.min(axis=1, skipna=True)
                wells_df['display_depth'] = wells_df['depth_to_groundwater'].fillna(pd.to_numeric(raw_df['DEPTH'], errors='coerce'))
            else:
                # Use drill hole depth as fallback
                wells_df['depth_to_groundwater'] = pd.to_numeric(raw_df['DEPTH'], errors='coerce')
                wells_df['display_depth'] = wells_df['depth_to_groundwater']

            # Keep original depth for reference
            wells_df['drill_hole_depth'] = pd.to_numeric(raw_df['DEPTH'], errors='coerce').fillna(0)
            # For general display, use display_depth
            wells_df['depth'] = wells_df['display_depth']
            wells_df['depth_m'] = wells_df['display_depth']

            # Add yield information - ONLY use MAX_YIELD from the new dataset
            if 'MAX_YIELD' in raw_df.columns:
                # Replace only empty strings with NaN, keep '0' as legitimate zero values
                max_yield_series = raw_df['MAX_YIELD'].astype(str).replace(['', 'nan'], np.nan)
                wells_df['yield_rate'] = pd.to_numeric(max_yield_series, errors='coerce')
            else:
                wells_df['yield_rate'] = pd.Series([np.nan] * len(wells_df))

            # Add specific capacity information if available
            if 'SPECIFIC_CAPACITY' in raw_df.columns:
                wells_df['specific_capacity'] = pd.to_numeric(raw_df['SPECIFIC_CAPACITY'], errors='coerce')
            else:
                wells_df['specific_capacity'] = pd.Series([np.nan] * len(wells_df))

            # Add initial SWL (Standing Water Level) information if available
            if 'INITIAL_SWL' in raw_df.columns:
                wells_df['initial_swl'] = pd.to_numeric(raw_df['INITIAL_SWL'], errors='coerce')
            else:
                wells_df['initial_swl'] = pd.Series([np.nan] * len(wells_df))

            # Add ground water level information from the 'ground water level' column
            if 'ground water level' in raw_df.columns:
                wells_df['ground water level'] = pd.to_numeric(raw_df['ground water level'], errors='coerce')
            else:
                wells_df['ground water level'] = pd.Series([np.nan] * len(wells_df))

            # Add well type and status information using new dataset columns
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
                st.info(f"Using {len(valid_wells):,} wells from Wells and Bores database")

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
            st.error(f"Error processing Wells and Bores data: {e}")
            # Generate fallback wells
            return generate_wells_for_area((-43.5, 172.5), 100)
    else:
        st.error(f"Wells and Bores data file not found at {wells_30k_file}")
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