import numpy as np
import pandas as pd
import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestRegressor
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import io
import base64
from PIL import Image
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from utils import get_distance

def create_indicator_polygon_geometry(indicator_mask, threshold=0.7):
    """
    Convert indicator kriging mask into polygon geometry for clipping
    
    Parameters:
    -----------
    indicator_mask : tuple
        Tuple containing indicator kriging mask data
    threshold : float
        Threshold for high-probability zones
        
    Returns:
    --------
    shapely.geometry or None
        Merged polygon geometry of high-probability zones
    """
    try:
        if indicator_mask is None:
            return None
            
        mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
        if mask_values is None:
            return None
            
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        
        # Create polygons for grid cells where probability >= threshold
        polygons = []
        for i in range(len(mask_lat_vals)-1):
            for j in range(len(mask_lon_vals)-1):
                if mask_values[i, j] >= threshold:
                    # Create polygon for this grid cell
                    min_lat, max_lat = mask_lat_vals[i], mask_lat_vals[i+1]
                    min_lon, max_lon = mask_lon_vals[j], mask_lon_vals[j+1]
                    
                    polygon = Polygon([
                        (min_lon, min_lat),
                        (max_lon, min_lat), 
                        (max_lon, max_lat),
                        (min_lon, max_lat),
                        (min_lon, min_lat)
                    ])
                    polygons.append(polygon)
        
        if not polygons:
            return None
            
        # Merge all polygons into a single geometry
        merged_geometry = unary_union(polygons)
        
        print(f"Created indicator clipping geometry from {len(polygons)} high-probability cells")
        return merged_geometry
        
    except Exception as e:
        print(f"Error creating indicator polygon geometry: {e}")
        return None

def generate_indicator_kriging_mask(wells_df, center_point, radius_km, resolution=50, soil_polygons=None, threshold=0.7):
    """
    Generate an indicator kriging mask for high-probability zones (≥ threshold)
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    center_point : tuple
        (latitude, longitude) of center point
    radius_km : float
        Radius in km to include in the grid
    resolution : int
        Grid resolution (number of cells per dimension)
    soil_polygons : GeoDataFrame, optional
        Soil polygon data for initial clipping
    threshold : float
        Threshold for high-probability zones (default 0.7)
        
    Returns:
    --------
    tuple
        (lat_grid, lon_grid, mask_values, grid_lats, grid_lons)
    """
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return None, None, None, None, None
        
    # Get indicator kriging interpolation for the area
    try:
        # Use indicator kriging to get probability surface
        geojson_data = generate_geo_json_grid(
            wells_df, center_point, radius_km, 
            resolution=resolution, method='indicator_kriging',
            soil_polygons=soil_polygons
        )
        
        if not geojson_data or not geojson_data.get('features'):
            return None, None, None, None, None
            
        # Extract center coordinates and setup grid with HIGH-PRECISION conversion
        center_lat, center_lon = center_point
        
        # Use same high-precision conversion factors as main interpolation
        from utils import get_distance
        TOLERANCE_KM = 0.0001  # 10cm tolerance - SAME as sequential_heatmap.py
        MAX_ITERATIONS = 200   # SAME as sequential_heatmap.py
        ADAPTIVE_STEP_SIZE = 0.000001  # SAME dynamic precision as sequential_heatmap.py
        
        def get_precise_conversion_factors(reference_lat, reference_lon):
            """Calculate ultra-precise km-to-degree conversion factors using iterative refinement"""
            import numpy as np
            test_distance = 1.0  # 1km test distance
            
            # Ultra-precise latitude conversion
            lat_offset_initial = test_distance / 111.0
            best_lat_factor = 111.0
            best_lat_error = float('inf')
            
            for i in range(MAX_ITERATIONS):
                test_lat = reference_lat + lat_offset_initial
                actual_distance = get_distance(reference_lat, reference_lon, test_lat, reference_lon)
                error = abs(actual_distance - test_distance)
                
                current_factor = test_distance / lat_offset_initial
                if error < best_lat_error:
                    best_lat_factor = current_factor
                    best_lat_error = error
                
                if error < TOLERANCE_KM:
                    break
                    
                # ENHANCED ADAPTIVE REFINEMENT - Same algorithm as sequential_heatmap.py
                if error > 0.001:  # > 1 meter error - proportional adjustment
                    adjustment_factor = test_distance / actual_distance  
                    lat_offset_initial *= adjustment_factor
                else:  # Precision phase - adaptive micro-adjustments
                    step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)  # Dynamic step based on error
                    if actual_distance > test_distance:
                        lat_offset_initial -= step_size
                    else:
                        lat_offset_initial += step_size
            
            # Ultra-precise longitude conversion
            lon_offset_initial = test_distance / (111.0 * abs(np.cos(np.radians(reference_lat))))
            best_lon_factor = 111.0 * abs(np.cos(np.radians(reference_lat)))
            best_lon_error = float('inf')
            
            for i in range(MAX_ITERATIONS):
                test_lon = reference_lon + lon_offset_initial
                actual_distance = get_distance(reference_lat, reference_lon, reference_lat, test_lon)
                error = abs(actual_distance - test_distance)
                
                current_factor = test_distance / lon_offset_initial
                if error < best_lon_error:
                    best_lon_factor = current_factor
                    best_lon_error = error
                
                if error < TOLERANCE_KM:
                    break
                    
                # ENHANCED ADAPTIVE REFINEMENT - Same algorithm as sequential_heatmap.py  
                if error > 0.001:  # > 1 meter error - proportional adjustment
                    adjustment_factor = test_distance / actual_distance
                    lon_offset_initial *= adjustment_factor
                else:  # Precision phase - adaptive micro-adjustments
                    step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)  # Dynamic step based on error
                    if actual_distance > test_distance:
                        lon_offset_initial -= step_size
                    else:
                        lon_offset_initial += step_size
            
            return best_lat_factor, best_lon_factor
        
        km_per_degree_lat, km_per_degree_lon = get_precise_conversion_factors(center_lat, center_lon)
        
        min_lat = center_lat - (radius_km / km_per_degree_lat)
        max_lat = center_lat + (radius_km / km_per_degree_lat)
        min_lon = center_lon - (radius_km / km_per_degree_lon)
        max_lon = center_lon + (radius_km / km_per_degree_lon)
        
        # Create grid
        grid_size = min(150, max(50, resolution))
        lat_vals = np.linspace(min_lat, max_lat, grid_size)
        lon_vals = np.linspace(min_lon, max_lon, grid_size)
        grid_lons, grid_lats = np.meshgrid(lon_vals, lat_vals)
        
        # Initialize mask with zeros
        mask_values = np.zeros_like(grid_lats)
        
        # Extract probability values from GeoJSON features
        for feature in geojson_data['features']:
            if feature['geometry']['type'] == 'Polygon':
                # Get polygon coordinates and yield value
                coords = feature['geometry']['coordinates'][0]
                yield_value = feature['properties']['yield']
                
                # Find centroid of polygon
                if len(coords) >= 4:  # Valid polygon
                    centroid_lon = np.mean([c[0] for c in coords[:-1]])
                    centroid_lat = np.mean([c[1] for c in coords[:-1]])
                    
                    # Find nearest grid point
                    lat_idx = np.argmin(np.abs(lat_vals - centroid_lat))
                    lon_idx = np.argmin(np.abs(lon_vals - centroid_lon))
                    
                    # Set mask value to the actual indicator kriging probability
                    mask_values[lat_idx, lon_idx] = yield_value
                        
        high_prob_count = np.sum(mask_values >= threshold)
        print(f"Indicator mask generated: {high_prob_count} high-probability points (≥{threshold}) out of {mask_values.size} total grid points")
        return grid_lats, grid_lons, mask_values, lat_vals, lon_vals
        
    except Exception as e:
        print(f"Error generating indicator mask: {e}")
        return None, None, None, None, None

def generate_geo_json_grid(wells_df, center_point, radius_km, resolution=50, method='kriging', show_variance=False, auto_fit_variogram=False, variogram_model='spherical', soil_polygons=None, indicator_mask=None, banks_peninsula_coords=None, adjacent_boundaries=None, boundary_vertices=None):
    """
    Generate GeoJSON grid with interpolated yield values for accurate visualization

    This function creates a smooth, continuous interpolation surface with GeoJSON format
    for optimal visualization of groundwater yield patterns.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    center_point : tuple
        (latitude, longitude) of center point
    radius_km : float
        Radius in km to include in the grid
    resolution : int
        Grid resolution (number of cells per dimension)
    method : str
        Interpolation method to use
    indicator_mask : tuple or None
        Tuple containing indicator kriging mask data for clipping

    Returns:
    --------
    dict
        GeoJSON data structure with interpolated yield values
    """
    
    # Debug indicator mask status and create polygon geometry for clipping
    print(f"GeoJSON {method}: indicator_mask is {'provided' if indicator_mask is not None else 'None'}")
    indicator_geometry = None
    
    # Create Banks Peninsula exclusion polygon if coordinates provided
    banks_peninsula_polygon = None
    if banks_peninsula_coords and len(banks_peninsula_coords) > 3:
        try:
            from shapely.geometry import Polygon
            # Ensure the polygon is closed (first and last points are the same)
            coords = list(banks_peninsula_coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            
            # Create polygon with (longitude, latitude) order for shapely
            # Note: banks_peninsula_coords should be in (latitude, longitude) format
            shapely_coords = [(coord[1], coord[0]) for coord in coords]  # Convert to (lon, lat)
            banks_peninsula_polygon = Polygon(shapely_coords)
            print(f"Banks Peninsula exclusion polygon created with {len(coords)} points")
        except Exception as e:
            print(f"Error creating Banks Peninsula exclusion polygon: {e}")
            banks_peninsula_polygon = None
    
    # Don't apply indicator clipping to indicator kriging methods themselves
    # Indicator methods should not be clipped by their own output
    indicator_methods = ['indicator_kriging', 'indicator_variance']
    clippable_methods = ['kriging', 'yield_kriging', 'specific_capacity_kriging', 'depth_kriging', 'swl_kriging', 'ground_water_level_kriging', 'idw', 'rf_kriging']
    
    if indicator_mask is not None and method in clippable_methods:
        mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
        print(f"Indicator mask grid shape: {mask_values.shape if mask_values is not None else 'None'}")
        if mask_values is not None:
            high_prob_count = np.sum(mask_values >= 0.7)
            print(f"Mask has {high_prob_count} high-probability points (≥0.7) out of {mask_values.size} total")
            # Create polygon geometry from indicator mask for clipping
            indicator_geometry = create_indicator_polygon_geometry(indicator_mask, threshold=0.7)
            if indicator_geometry is not None:
                print(f"Created indicator clipping geometry from {high_prob_count} high-probability cells")

    # Handle empty datasets
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return {"type": "FeatureCollection", "features": []}

    # Filter out geotechnical/geological investigation wells using well_use
    if 'well_use' in wells_df.columns:
        geotechnical_mask = wells_df['well_use'].str.contains(
            'Geotechnical.*Investigation|Geological.*Investigation', 
            case=False, 
            na=False, 
            regex=True
        )
        wells_df = wells_df[~geotechnical_mask].copy()

        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

    # Extract the original grid information with HIGH-PRECISION coordinate conversion
    center_lat, center_lon = center_point
    
    # HIGH-PRECISION COORDINATE CONVERSION SYSTEM
    # Use same ultra-precise geodetic calculations as sequential_heatmap.py
    # This ensures perfect alignment between heatmap centroids and clipping boundaries
    from utils import get_distance
    
    # Ultra-precise conversion factors using iterative refinement
    # Target: Match sequential_heatmap.py precision (10cm accuracy)
    TOLERANCE_KM = 0.0001  # 10cm tolerance - SAME as sequential_heatmap.py
    MAX_ITERATIONS = 200   # SAME as sequential_heatmap.py
    ADAPTIVE_STEP_SIZE = 0.000001  # SAME dynamic precision as sequential_heatmap.py
    
    def get_precise_conversion_factors(reference_lat, reference_lon):
        """Calculate ultra-precise km-to-degree conversion factors using iterative refinement"""
        import numpy as np
        
        # Step 1: Ultra-precise latitude conversion
        test_distance = 1.0  # 1km test distance
        lat_offset_initial = test_distance / 111.0  # Initial estimate
        
        best_lat_factor = 111.0
        best_lat_error = float('inf')
        
        for i in range(MAX_ITERATIONS):
            test_lat = reference_lat + lat_offset_initial
            actual_distance = get_distance(reference_lat, reference_lon, test_lat, reference_lon)
            error = abs(actual_distance - test_distance)
            
            current_factor = test_distance / lat_offset_initial
            if error < best_lat_error:
                best_lat_factor = current_factor
                best_lat_error = error
            
            if error < TOLERANCE_KM:
                break
                
            # ENHANCED ADAPTIVE REFINEMENT - Same algorithm as sequential_heatmap.py
            if error > 0.001:  # > 1 meter error - proportional adjustment
                adjustment_factor = test_distance / actual_distance  
                lat_offset_initial *= adjustment_factor
            else:  # Precision phase - adaptive micro-adjustments
                step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)  # Dynamic step based on error
                if actual_distance > test_distance:
                    lat_offset_initial -= step_size
                else:
                    lat_offset_initial += step_size
        
        # Step 2: Ultra-precise longitude conversion (latitude-dependent)
        lon_offset_initial = test_distance / (111.0 * abs(np.cos(np.radians(reference_lat))))
        
        best_lon_factor = 111.0 * abs(np.cos(np.radians(reference_lat)))
        best_lon_error = float('inf')
        
        for i in range(MAX_ITERATIONS):
            test_lon = reference_lon + lon_offset_initial
            actual_distance = get_distance(reference_lat, reference_lon, reference_lat, test_lon)
            error = abs(actual_distance - test_distance)
            
            current_factor = test_distance / lon_offset_initial
            if error < best_lon_error:
                best_lon_factor = current_factor
                best_lon_error = error
            
            if error < TOLERANCE_KM:
                break
                
            # ENHANCED ADAPTIVE REFINEMENT - Same algorithm as sequential_heatmap.py
            if error > 0.001:  # > 1 meter error - proportional adjustment
                adjustment_factor = test_distance / actual_distance
                lon_offset_initial *= adjustment_factor
            else:  # Precision phase - adaptive micro-adjustments
                step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)  # Dynamic step based on error
                if actual_distance > test_distance:
                    lon_offset_initial -= step_size
                else:
                    lon_offset_initial += step_size
        
        print(f"HIGH-PRECISION CONVERSION: lat_factor={best_lat_factor:.8f} km/deg (error: {best_lat_error:.8f}km)")
        print(f"HIGH-PRECISION CONVERSION: lon_factor={best_lon_factor:.8f} km/deg (error: {best_lon_error:.8f}km)")
        
        return best_lat_factor, best_lon_factor
    
    # Calculate ultra-precise conversion factors
    km_per_degree_lat, km_per_degree_lon = get_precise_conversion_factors(center_lat, center_lon)

    # Create grid in lat/lon space with EXACT EDGE ALIGNMENT for adjacent heatmaps
    if adjacent_boundaries is not None:
        # Use adjacent boundaries to ensure perfect edge alignment
        print(f"EDGE-ALIGNED CLIPPING: Using adjacent boundaries for seamless joining")
        
        # Extract adjacent boundary information
        west_boundary = adjacent_boundaries.get('west')
        east_boundary = adjacent_boundaries.get('east') 
        north_boundary = adjacent_boundaries.get('north')
        south_boundary = adjacent_boundaries.get('south')
        
        # Calculate default boundaries first
        default_min_lat = center_lat - (radius_km / km_per_degree_lat)
        default_max_lat = center_lat + (radius_km / km_per_degree_lat)
        default_min_lon = center_lon - (radius_km / km_per_degree_lon)
        default_max_lon = center_lon + (radius_km / km_per_degree_lon)
        
        # Use adjacent boundaries where available, defaults otherwise
        min_lat = south_boundary if south_boundary is not None else default_min_lat
        max_lat = north_boundary if north_boundary is not None else default_max_lat
        min_lon = west_boundary if west_boundary is not None else default_min_lon
        max_lon = east_boundary if east_boundary is not None else default_max_lon
        
        print(f"  West edge: {'ALIGNED' if west_boundary is not None else 'DEFAULT'} ({min_lon:.8f})")
        print(f"  East edge: {'ALIGNED' if east_boundary is not None else 'DEFAULT'} ({max_lon:.8f})")
        print(f"  North edge: {'ALIGNED' if north_boundary is not None else 'DEFAULT'} ({max_lat:.8f})")
        print(f"  South edge: {'ALIGNED' if south_boundary is not None else 'DEFAULT'} ({min_lat:.8f})")
        
    else:
        # Standard centroid-based boundaries
        min_lat = center_lat - (radius_km / km_per_degree_lat)
        max_lat = center_lat + (radius_km / km_per_degree_lat)
        min_lon = center_lon - (radius_km / km_per_degree_lon)
        max_lon = center_lon + (radius_km / km_per_degree_lon)
        print(f"STANDARD CLIPPING: Using centroid-based boundaries")

    # High resolution grid for smooth professional visualization
    # Increase resolution significantly for smoother appearance like kriging software
    wells_count = len(wells_df)
    if wells_count > 5000:
        grid_size = 80   # Higher resolution for very large datasets
    elif wells_count > 1000:
        grid_size = 120  # High resolution for large datasets
    else:
        grid_size = 150  # Very fine resolution for smaller datasets

    # Create the grid for our GeoJSON polygons
    lat_vals = np.linspace(min_lat, max_lat, grid_size)
    lon_vals = np.linspace(min_lon, max_lon, grid_size)

    # Extract coordinates and values from the wells dataframe
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)

    # Use the new categorization system
    from data_loader import get_wells_for_interpolation

    if method == 'depth_kriging':
        # Get wells appropriate for depth interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'depth')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)

        # Use depth_to_groundwater if available, otherwise fall back to depth
        if 'depth_to_groundwater' in wells_df.columns and wells_df['depth_to_groundwater'].notna().any():
            yields = wells_df['depth_to_groundwater'].values.astype(float)
        else:
            yields = wells_df['depth'].values.astype(float)
    elif method == 'specific_capacity_kriging':
        # Get wells appropriate for specific capacity interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'specific_capacity')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        # Double-check that all wells have valid specific capacity data
        wells_df = wells_df[
            wells_df['specific_capacity'].notna() & 
            (wells_df['specific_capacity'] > 0)
        ].copy()

        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        yields = wells_df['specific_capacity'].values.astype(float)
    elif method == 'ground_water_level_kriging':
        # Get wells appropriate for ground water level interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'ground_water_level')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        # Double-check that all wells have valid ground water level data
        # Only filter for non-null values, allow all numeric values (including 0 and negative)
        valid_gwl_mask = wells_df['ground water level'].notna()

        wells_df = wells_df[valid_gwl_mask].copy()

        if wells_df.empty:
            print("No valid ground water level data found for interpolation")
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        
        # Convert negative ground water levels to 0 for interpolation
        # Negative values indicate artesian conditions (water above surface)
        # For depth mapping, treat these as surface level (depth = 0)
        raw_gwl_values = wells_df['ground water level'].values.astype(float)
        yields = np.maximum(raw_gwl_values, 0)  # Convert negative values to 0
        
        negative_count = np.sum(raw_gwl_values < 0)
        if negative_count > 0:
            print(f"Ground water level interpolation: converted {negative_count} negative values (artesian) to 0 (surface level)")

        print(f"Ground water level interpolation: using {len(yields)} wells with values ranging from {yields.min():.2f} to {yields.max():.2f}")
    elif method == 'indicator_kriging':
        # For indicator kriging, we need wells with ACTUAL yield data (including 0.0)
        # But exclude wells that have missing yield data entirely
        wells_df_original = wells_df.copy()

        # Only filter for valid coordinates and yield_rate column existence
        if 'yield_rate' not in wells_df_original.columns:
            return {"type": "FeatureCollection", "features": []}

        # For indicator kriging, be more selective about 0.0 values
        # Only include wells that are explicitly water wells with yield measurements
        # Exclude wells that appear to be monitoring, geotechnical, or no-yield wells

        # Filter for valid coordinates and wells with actual yield measurements
        valid_coord_mask = (
            wells_df_original['latitude'].notna() & 
            wells_df_original['longitude'].notna() &
            wells_df_original['yield_rate'].notna()  # Must have yield data
        )

        # Additional filtering to exclude wells that shouldn't be in indicator kriging
        if 'well_use' in wells_df_original.columns:
            # Exclude geotechnical/geological investigation wells
            investigation_mask = ~wells_df_original['well_use'].str.contains(
                'Geotechnical.*Investigation|Geological.*Investigation|Monitoring', 
                case=False, na=False, regex=True
            )
            valid_coord_mask = valid_coord_mask & investigation_mask

        wells_df = wells_df_original[valid_coord_mask].copy()

        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)

        # Convert to BINARY indicator values for kriging input using COMBINED criteria
        # A well is viable (indicator = 1) if EITHER:
        # - yield_rate ≥ 0.1 L/s, OR  
        # - ground water level data exists (any valid depth means water was found)
        # Note: If ground water level is recorded, it means water was found at that depth = viable
        # Otherwise indicator = 0 (not viable)
        
        yield_threshold = 0.1
        
        raw_yields = wells_df['yield_rate'].values.astype(float)
        
        # Check if ground water level data is available
        has_gwl_data = 'ground water level' in wells_df.columns
        if has_gwl_data:
            gwl_values = wells_df['ground water level'].values.astype(float)
            # Handle NaN values in ground water level
            gwl_valid = ~np.isnan(gwl_values)
            gwl_viable = np.zeros_like(gwl_values, dtype=bool)
            # Wells are viable if they have ANY valid ground water level data (water was found)
            gwl_viable[gwl_valid] = True  # Any recorded depth means water was found = viable
        else:
            gwl_values = np.full_like(raw_yields, np.nan)
            gwl_viable = np.zeros_like(raw_yields, dtype=bool)
        
        # Combined viability logic: viable if EITHER condition is met
        yield_viable = raw_yields >= yield_threshold
        combined_viable = yield_viable | gwl_viable
        yields = combined_viable.astype(float)  # Binary: 1 or 0

        # Count wells in each category for detailed logging
        viable_count = np.sum(yields == 1)
        non_viable_count = np.sum(yields == 0)
        yield_only_viable = np.sum(yield_viable & ~gwl_viable)
        gwl_only_viable = np.sum(gwl_viable & ~yield_viable) if has_gwl_data else 0
        both_viable = np.sum(yield_viable & gwl_viable) if has_gwl_data else 0

        print(f"GeoJSON indicator kriging: using {len(yields)} wells with COMBINED binary classification")
        print(f"COMBINED RESULTS:")
        print(f"  Total viable: {viable_count} wells ({100*viable_count/len(yields):.1f}%)")
        print(f"  Total non-viable: {non_viable_count} wells ({100*non_viable_count/len(yields):.1f}%)")
        print(f"BREAKDOWN BY CRITERIA:")
        print(f"  Viable by yield only (≥{yield_threshold} L/s): {yield_only_viable} wells")
        if has_gwl_data:
            print(f"  Viable by ground water level only (has valid depth data): {gwl_only_viable} wells") 
            print(f"  Viable by both criteria: {both_viable} wells")
            valid_gwl_count = np.sum(~np.isnan(gwl_values))
            print(f"  Wells with ground water level data: {valid_gwl_count}/{len(wells_df)}")
        else:
            print(f"  No ground water level data available - using yield criteria only")
        print(f"RAW DATA RANGES:")
        print(f"  Yield range: {raw_yields.min():.3f} to {raw_yields.max():.3f} L/s")
        if has_gwl_data and valid_gwl_count > 0:
            valid_gwl = gwl_values[~np.isnan(gwl_values)]
            print(f"  Ground water level range: {valid_gwl.min():.3f} to {valid_gwl.max():.3f}")
            
            # Debug specific well M35/4191 if present
            if 'well_id' in wells_df.columns:
                test_well_mask = wells_df['well_id'].str.contains('M35/4191', na=False)
                if test_well_mask.any():
                    test_well = wells_df[test_well_mask].iloc[0]
                    test_yield = test_well['yield_rate']
                    test_gwl = test_well['ground water level'] if not pd.isna(test_well['ground water level']) else 'NaN'
                    test_yield_viable = test_yield >= yield_threshold
                    test_gwl_viable = not pd.isna(test_well['ground water level'])
                    test_combined = test_yield_viable or test_gwl_viable
                    print(f"DEBUG WELL M35/4191:")
                    print(f"    Yield: {test_yield} L/s (viable: {test_yield_viable})")
                    print(f"    Ground water level: {test_gwl} (viable: {test_gwl_viable})")
                    print(f"    Combined viable: {test_combined}")
                    print(f"    Criteria: yield>={yield_threshold}, gwl=has_valid_data")
                    
            # Sample of ground water level values showing range
            gwl_sample = valid_gwl[:10] if len(valid_gwl) > 0 else []
            if len(gwl_sample) > 0:
                print(f"  Sample GWL values (all viable as water was found): {gwl_sample}")
                print(f"  Note: Any recorded depth means water was found = viable well")
                
        print(f"  Wells with exactly 0.0 yield: {np.sum(raw_yields == 0.0)}")
        print(f"  Wells with NaN yield (excluded): {wells_df_original['yield_rate'].isna().sum()}")
        print(f"  Total wells excluded for quality: {len(wells_df_original) - len(wells_df)}")
    else:
        # Get wells appropriate for yield interpolation
        wells_df = get_wells_for_interpolation(wells_df, 'yield')
        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        yields = wells_df['yield_rate'].values.astype(float)

    # Convert to km-based coordinates for proper interpolation
    x_coords = (lons - center_lon) * km_per_degree_lon
    y_coords = (lats - center_lat) * km_per_degree_lat

    # Create grid in km space (square bounds)
    grid_x = np.linspace(-radius_km, radius_km, grid_size)
    grid_y = np.linspace(-radius_km, radius_km, grid_size)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    # Flatten for interpolation
    points = np.vstack([x_coords, y_coords]).T  # Well points in km
    xi = np.vstack([grid_X.flatten(), grid_Y.flatten()]).T  # Grid points in km

    # Use square bounds instead of circular radius
    mask = (np.abs(xi[:,0]) <= radius_km) & (np.abs(xi[:,1]) <= radius_km)

    # Define grid_points early to avoid UnboundLocalError
    grid_points = xi[mask]

    # Perform interpolation
    points = np.vstack([x_coords, y_coords]).T

    try:
        # Initialize variance array for kriging uncertainty
        kriging_variance = None

        # Choose interpolation method based on parameter and dataset size
        if show_variance and (method == 'kriging' or method == 'rf_kriging') and len(wells_df) >= 5:
            # Use actual kriging with variance calculation when variance is requested
            print("Calculating kriging with variance estimation")

            # Convert coordinates back to lat/lon for kriging (pykrige expects lon/lat)
            lon_values = x_coords / km_per_degree_lon + center_lon
            lat_values = y_coords / km_per_degree_lat + center_lat

            # Use already defined grid_points
            xi_lon = grid_points[:, 0] / km_per_degree_lon + center_lon
            xi_lat = grid_points[:, 1] / km_per_degree_lat + center_lat

            # Set up kriging with variance calculation
            if auto_fit_variogram:
                # Use auto-fitted variogram for more accurate uncertainty estimation
                print(f"Auto-fitting {variogram_model} variogram model...")
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model=variogram_model,
                    verbose=False,
                    enable_plotting=False,
                    variogram_parameters=None  # Let PyKrige auto-fit parameters
                )
            else:
                # Use fixed variogram model for speed
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model='linear',  # Fast and stable
                    verbose=False,
                    enable_plotting=False
                )

            # Execute kriging to get both predictions and variance
            interpolated_z, kriging_variance = OK.execute('points', xi_lon, xi_lat)

        elif method == 'indicator_kriging' and len(wells_df) >= 5:
            # Perform indicator kriging for binary yield suitability
            print("Performing indicator kriging for yield suitability mapping...")

            # Convert coordinates back to lat/lon for kriging (pykrige expects lon/lat)
            lon_values = x_coords / km_per_degree_lon + center_lon
            lat_values = y_coords / km_per_degree_lat + center_lat

            # Use already defined grid_points
            xi_lon = grid_points[:, 0] / km_per_degree_lon + center_lon
            xi_lat = grid_points[:, 1] / km_per_degree_lat + center_lat

            # Set up kriging for binary indicator data with constrained parameters
            # Use spherical model with limited range to prevent high values far from viable wells
            max_range_km = min(radius_km * 0.5, 5.0)  # Limit influence to half radius or 5km max
            range_degrees = max_range_km / 111.0  # Convert km to degrees (rough approximation)

            OK = OrdinaryKriging(
                lon_values, lat_values, yields,
                variogram_model='spherical',  # Better spatial control than linear
                verbose=False,
                enable_plotting=False,
                weight=True,  # Enable nugget effect for indicator data
                variogram_parameters=[0.2, 0.6, range_degrees]  # [nugget, sill, range] - constrained range
            )

            # Execute kriging to get probability predictions
            interpolated_z, _ = OK.execute('points', xi_lon, xi_lat)

            # Ensure values are in [0,1] range (probabilities)
            interpolated_z = np.clip(interpolated_z, 0.0, 1.0)

            # Apply distance-based decay to prevent high values far from good wells
            # Calculate distances from each grid point to nearest good well (yield >= 0.75)
            good_wells_mask = yields >= 0.75  # Wells with good yield indicators
            if np.any(good_wells_mask):
                good_coords = np.column_stack([x_coords[good_wells_mask], y_coords[good_wells_mask]])
                grid_coords = grid_points

                # Calculate distance to nearest good well for each grid point
                from scipy.spatial.distance import cdist
                distances = cdist(grid_coords, good_coords)
                min_distances_km = np.min(distances, axis=1) / 1000.0  # Convert to km

                # Apply exponential decay for distances > 3km
                decay_threshold_km = 3.0
                decay_factor = 0.3  # Gentle decay to preserve three-tier structure

                distance_mask = min_distances_km > decay_threshold_km
                excess_distance = min_distances_km[distance_mask] - decay_threshold_km
                decay_multiplier = np.exp(-decay_factor * excess_distance)
                interpolated_z[distance_mask] *= decay_multiplier

            # Count points in each tier based on OUTPUT ranges (continuous 0-1 function)
            # Keep the continuous kriging output values intact - colors applied during visualization
            red_count = np.sum((interpolated_z >= 0.0) & (interpolated_z < 0.4))     # Red: 0.0-0.4
            orange_count = np.sum((interpolated_z >= 0.4) & (interpolated_z < 0.7))  # Orange: 0.4-0.7  
            green_count = np.sum(interpolated_z >= 0.7)                              # Green: 0.7-1.0

            print(f"Indicator kriging results (output function ranges):")
            print(f"Red (0.0-0.4): {red_count} points, Orange (0.4-0.7): {orange_count} points, Green (0.7-1.0): {green_count} points")
            print(f"Output value range: {interpolated_z.min():.3f} to {interpolated_z.max():.3f}")

        elif (method == 'kriging' or method == 'depth_kriging') and auto_fit_variogram and len(wells_df) >= 5:
            # Perform kriging with auto-fitted variogram for yield/depth visualization (without variance output)
            if method == 'depth_kriging':
                print(f"Auto-fitting {variogram_model} variogram model for depth estimation...")
            else:
                print(f"Auto-fitting {variogram_model} variogram model for yield estimation...")

            # Convert coordinates back to lat/lon for kriging (pykrige expects lon/lat)
            lon_values = x_coords / km_per_degree_lon + center_lon
            lat_values = y_coords / km_per_degree_lat + center_lat

            # Use already defined grid_points
            xi_lon = grid_points[:, 0] / km_per_degree_lon + center_lon
            xi_lat = grid_points[:, 1] / km_per_degree_lat + center_lat

            # Set up kriging with auto-fitted variogram - ensure proper parameters for depth data
            if method == 'depth_kriging':
                # For depth data, use more appropriate variogram parameters
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model=variogram_model,
                    verbose=True,  # Enable verbose for debugging depth issues
                    enable_plotting=False,
                    variogram_parameters=None,  # Let PyKrige auto-fit parameters
                    weight=True,  # Enable nugget effect for depth data
                    anisotropy_scaling=1.0,  # No anisotropy scaling
                    anisotropy_angle=0.0
                )
            else:
                # Standard yield kriging
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model=variogram_model,
                    verbose=False,
                    enable_plotting=False,
                    variogram_parameters=None  # Let PyKrige auto-fit parameters
                )

            # Execute kriging to get predictions (ignore variance)
            interpolated_z, _ = OK.execute('points', xi_lon, xi_lat)

            # Additional validation for depth interpolation
            if method == 'depth_kriging':
                print(f"Depth interpolation stats: min={np.min(interpolated_z):.2f}, max={np.max(interpolated_z):.2f}, mean={np.mean(interpolated_z):.2f}")
                # Ensure reasonable depth values (depths should be positive and reasonable)
                interpolated_z = np.maximum(0.1, interpolated_z)  # Minimum depth of 0.1m
                interpolated_z = np.minimum(200.0, interpolated_z)  # Maximum reasonable depth of 200m

        else:
            # Use standard griddata interpolation for other cases
            # This is much faster than kriging for large datasets
            interpolated_z = griddata(
                points, yields, grid_points,
                method='linear', fill_value=0.0
            )

            # Fill any NaN values with nearest neighbor interpolation
            nan_mask = np.isnan(interpolated_z)
            if np.any(nan_mask):
                interpolated_z[nan_mask] = griddata(
                    points, yields, grid_points[nan_mask],
                    method='nearest', fill_value=0.0
                )

            # Apply advanced smoothing for professional kriging-like appearance
            from scipy.ndimage import gaussian_filter

            # Reshape to 2D grid for smoothing
            try:
                # Create full 2D grid for smoothing
                z_grid = np.zeros_like(grid_X)
                z_grid_flat = z_grid.flatten()
                z_grid_flat[mask] = interpolated_z
                z_grid = z_grid_flat.reshape(grid_X.shape)

                # Apply multiple smoothing passes for ultra-smooth appearance
                # First pass: moderate smoothing
                z_smooth = gaussian_filter(z_grid, sigma=1.5)
                # Second pass: fine smoothing for professional appearance
                z_smooth = gaussian_filter(z_smooth, sigma=0.8)

                # Extract smoothed values for our mask
                z_smooth_flat = z_smooth.flatten()
                interpolated_z = z_smooth_flat[mask]

                # Ensure values stay within reasonable bounds
                interpolated_z = np.maximum(0, interpolated_z)

            except Exception as e:
                # If smoothing fails, apply basic smoothing
                print(f"Advanced smoothing error: {e}, using basic smoothing")
                try:
                    z_grid = np.zeros_like(grid_X)
                    z_grid_flat = z_grid.flatten()
                    z_grid_flat[mask] = interpolated_z
                    z_grid = z_grid_flat.reshape(grid_X.shape)
                    z_smooth = gaussian_filter(z_grid, sigma=1.0)
                    z_smooth_flat = z_smooth.flatten()
                    interpolated_z = z_smooth_flat[mask]
                except:
                    print("Basic smoothing also failed, using raw interpolation")
    except Exception as e:
        # Fallback to simple IDW interpolation if the above methods fail
        print(f"Interpolation error: {e}, using fallback method")
        interpolated_z = np.zeros(grid_points.shape[0])
        for i, point in enumerate(grid_points):
            weights = 1.0 / (np.sqrt(np.sum((points - point)**2, axis=1)) + 1e-5)
            interpolated_z[i] = np.sum(weights * yields) / np.sum(weights)

    # Convert grid coordinates back to lat/lon
    grid_lats = (grid_points[:, 1] / km_per_degree_lat) + center_lat
    grid_lons = (grid_points[:, 0] / km_per_degree_lon) + center_lon

    # Prepare soil polygon geometry for later filtering (do not apply to interpolation)
    merged_soil_geometry = None

    if soil_polygons is not None and len(soil_polygons) > 0:
        try:
            # Create a unified geometry from all soil polygons for later filtering
            valid_geometries = []
            for idx, row in soil_polygons.iterrows():
                if row.geometry and row.geometry.is_valid:
                    valid_geometries.append(row.geometry)

            if valid_geometries:
                # Merge all polygons into a single multipolygon
                merged_soil_geometry = unary_union(valid_geometries)
                print(f"Prepared soil drainage geometry for display filtering")
            else:
                print("No valid soil polygon geometries found")
        except Exception as e:
            print(f"Error preparing soil polygon geometry: {e}")
            merged_soil_geometry = None

    # Build the GeoJSON structure
    features = []

    # Create final square clipping geometry (smaller than original search area)
    # Original search area is radius_km x radius_km square
    # Final clipping area is 50% of original (10km for 20km original)
    final_clip_factor = 0.5
    final_radius_km = radius_km * final_clip_factor
    
    # Create final square clipping polygon - RESPECT BOUNDARY SNAPPING
    final_clip_lat_radius = final_radius_km / km_per_degree_lat
    final_clip_lon_radius = final_radius_km / km_per_degree_lon
    
    # Use snapped boundaries if available, otherwise use center-based calculation
    if adjacent_boundaries is not None:
        # Use the same boundaries that were used for grid creation
        west_boundary = adjacent_boundaries.get('west')
        east_boundary = adjacent_boundaries.get('east') 
        north_boundary = adjacent_boundaries.get('north')
        south_boundary = adjacent_boundaries.get('south')
        
        # Calculate default final clip boundaries
        default_final_min_lat = center_lat - final_clip_lat_radius
        default_final_max_lat = center_lat + final_clip_lat_radius
        default_final_min_lon = center_lon - final_clip_lon_radius
        default_final_max_lon = center_lon + final_clip_lon_radius
        
        # CRITICAL FIX: Only use snapped boundaries for grid edges, maintain full heatmap size
        # The final clipping should maintain the standard size, but align one edge when snapping
        final_min_lat = south_boundary if south_boundary is not None else default_final_min_lat
        final_max_lat = north_boundary if north_boundary is not None else default_final_max_lat
        
        # For longitude: if we have a snapped west boundary, use it and calculate east from it
        if west_boundary is not None:
            final_min_lon = west_boundary  
            final_max_lon = west_boundary + (2 * final_clip_lon_radius)  # Full width eastward
        elif east_boundary is not None:
            final_max_lon = east_boundary
            final_min_lon = east_boundary - (2 * final_clip_lon_radius)  # Full width westward  
        else:
            final_min_lon = default_final_min_lon
            final_max_lon = default_final_max_lon
        
        final_clip_polygon_coords = [
            [final_min_lon, final_min_lat],  # SW
            [final_max_lon, final_min_lat],  # SE
            [final_max_lon, final_max_lat],  # NE
            [final_min_lon, final_max_lat],  # NW
            [final_min_lon, final_min_lat]   # Close
        ]
        print(f"Final clipping using SNAPPED BOUNDARIES: W={final_min_lon:.8f}, E={final_max_lon:.8f}")
    else:
        # Standard center-based clipping when no boundaries are snapped
        final_clip_polygon_coords = [
            [center_lon - final_clip_lon_radius, center_lat - final_clip_lat_radius],  # SW
            [center_lon + final_clip_lon_radius, center_lat - final_clip_lat_radius],  # SE
            [center_lon + final_clip_lon_radius, center_lat + final_clip_lat_radius],  # NE
            [center_lon - final_clip_lon_radius, center_lat + final_clip_lat_radius],  # NW
            [center_lon - final_clip_lon_radius, center_lat - final_clip_lat_radius]   # Close
        ]
    
    from shapely.geometry import Polygon as ShapelyPolygon
    final_clip_geometry = ShapelyPolygon(final_clip_polygon_coords)
    
    print(f"Final clipping: {radius_km}km -> {final_radius_km:.1f}km square ({final_clip_factor*100:.0f}% of original)")
    print(f"Final clipping geometry bounds: {final_clip_geometry.bounds}")

    # Create polygons only where needed - use a Delaunay triangulation approach
    # for a more organic-looking interpolation surface
    from scipy.spatial import Delaunay

    try:
        # Create a Delaunay triangulation of the interpolation points
        points_2d = np.vstack([grid_lons, grid_lats]).T

        # Initialize boundary snapping counters
        total_snapped_vertices = 0
        total_triangles_processed = 0

        # Only create triangulation if we have enough points
        if len(points_2d) > 3:
            tri = Delaunay(points_2d)

            # Process each triangle to create a polygon
            for simplex in tri.simplices:
                total_triangles_processed += 1
                # Get the three points of this triangle
                vertices = points_2d[simplex]

                # Get the values for these points (yield or variance)
                if show_variance and kriging_variance is not None:
                    vertex_values = kriging_variance[simplex]
                    avg_value = float(np.mean(vertex_values))
                    # For variance, show all values (including very small ones)
                    value_threshold = 0.0001  # Show almost all variance values
                else:
                    vertex_values = interpolated_z[simplex]
                    avg_value = float(np.mean(vertex_values))
                    value_threshold = 0.01  # Only show meaningful yield values

                avg_yield = avg_value  # Keep for backwards compatibility

                # Adjust value threshold based on interpolation method
                effective_threshold = value_threshold

                # Only add triangles with meaningful values and within our radius
                if avg_yield > effective_threshold:
                    # STEP 1: Apply boundary snapping BEFORE soil polygon clipping
                    snapped_vertices = 0
                    if adjacent_boundaries is not None:
                        # Apply boundary snapping to triangle vertices
                        snap_threshold_km = 3.0  # 3km threshold for snapping (covers typical boundary gaps)
                        
                        for vertex_idx in range(3):  # 3 vertices per triangle
                            vertex_lon, vertex_lat = vertices[vertex_idx]
                            original_lon, original_lat = vertex_lon, vertex_lat
                            
                            # Check each boundary for snapping
                            for boundary_type, boundary_value in adjacent_boundaries.items():
                                if boundary_value is None:
                                    continue
                                    
                                if boundary_type == 'west':
                                    distance_km = get_distance(vertex_lat, vertex_lon, vertex_lat, boundary_value)
                                    if distance_km <= snap_threshold_km:
                                        vertices[vertex_idx, 0] = boundary_value  # Snap longitude to west boundary
                                        snapped_vertices += 1
                                        total_snapped_vertices += 1
                                        if total_snapped_vertices <= 5:  # Log first few snaps for debugging
                                            print(f"🎯 WEST BOUNDARY SNAP: vertex ({original_lon:.6f}, {original_lat:.6f}) → ({boundary_value:.6f}, {original_lat:.6f}), distance: {distance_km:.3f}km")
                                        break  # Only snap to one boundary per vertex
                                        
                                elif boundary_type == 'east':
                                    distance_km = get_distance(vertex_lat, vertex_lon, vertex_lat, boundary_value)
                                    if distance_km <= snap_threshold_km:
                                        vertices[vertex_idx, 0] = boundary_value  # Snap longitude to east boundary
                                        snapped_vertices += 1
                                        total_snapped_vertices += 1
                                        if total_snapped_vertices <= 5:  # Log first few snaps for debugging
                                            print(f"🎯 EAST BOUNDARY SNAP: vertex ({original_lon:.6f}, {original_lat:.6f}) → ({boundary_value:.6f}, {original_lat:.6f}), distance: {distance_km:.3f}km")
                                        break  # Only snap to one boundary per vertex
                                        
                                elif boundary_type == 'north':
                                    distance_km = get_distance(vertex_lat, vertex_lon, boundary_value, vertex_lon)
                                    if distance_km <= snap_threshold_km:
                                        vertices[vertex_idx, 1] = boundary_value  # Snap latitude to north boundary
                                        snapped_vertices += 1
                                        total_snapped_vertices += 1
                                        if total_snapped_vertices <= 5:  # Log first few snaps for debugging
                                            print(f"🎯 NORTH BOUNDARY SNAP: vertex ({original_lon:.6f}, {original_lat:.6f}) → ({original_lon:.6f}, {boundary_value:.6f}), distance: {distance_km:.3f}km")
                                        break  # Only snap to one boundary per vertex
                                        
                                elif boundary_type == 'south':
                                    distance_km = get_distance(vertex_lat, vertex_lon, boundary_value, vertex_lon)
                                    if distance_km <= snap_threshold_km:
                                        vertices[vertex_idx, 1] = boundary_value  # Snap latitude to south boundary
                                        snapped_vertices += 1
                                        total_snapped_vertices += 1
                                        if total_snapped_vertices <= 5:  # Log first few snaps for debugging
                                            print(f"🎯 SOUTH BOUNDARY SNAP: vertex ({original_lon:.6f}, {original_lat:.6f}) → ({original_lon:.6f}, {boundary_value:.6f}), distance: {distance_km:.3f}km")
                                        break  # Only snap to one boundary per vertex

                    # STEP 2: Check if triangle should be included based on soil polygons (AFTER snapping)
                    include_triangle = True

                    if merged_soil_geometry is not None:
                        # Apply proper geometric intersection clipping
                        triangle_coords = [(float(v[0]), float(v[1])) for v in vertices]
                        triangle_coords.append(triangle_coords[0])  # Close the polygon
                        triangle_polygon = ShapelyPolygon(triangle_coords)
                        
                        # Only include if triangle is completely within soil drainage areas
                        include_triangle = merged_soil_geometry.contains(triangle_polygon)

                    # Additional clipping by indicator kriging geometry (high-probability zones)
                    # CRITICAL FIX: Skip indicator clipping when boundary snapping is active
                    # Boundary snapping takes precedence over indicator geometry restrictions
                    if include_triangle and indicator_geometry is not None and indicator_mask is not None and adjacent_boundaries is None:
                        centroid_lon = float(np.mean(vertices[:, 0]))
                        centroid_lat = float(np.mean(vertices[:, 1]))
                        was_included = include_triangle
                        
                        try:
                            # Use distance-based approach: check if centroid is near any high-probability indicator points
                            # Extract indicator mask data for distance checking
                            mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
                            
                            # Find high-probability points (≥0.7)
                            high_prob_mask = mask_values >= 0.7
                            if np.any(high_prob_mask):
                                high_prob_lats = mask_lat_grid[high_prob_mask]
                                high_prob_lons = mask_lon_grid[high_prob_mask]
                                
                                # Calculate distances to all high-probability points
                                distances = np.sqrt((high_prob_lats - centroid_lat)**2 + (high_prob_lons - centroid_lon)**2)
                                min_distance = np.min(distances)
                                
                                # Include if within reasonable distance (roughly grid spacing)
                                grid_spacing = abs(mask_lat_vals[1] - mask_lat_vals[0]) if len(mask_lat_vals) > 1 else 0.01
                                include_triangle = min_distance <= (grid_spacing * 1.5)  # 1.5x grid spacing tolerance
                            else:
                                include_triangle = False
                        except Exception as e:
                            # Fallback: use geometry-based clipping if distance approach fails
                            centroid_point = Point(centroid_lon, centroid_lat)
                            include_triangle = indicator_geometry.contains(centroid_point) or indicator_geometry.intersects(centroid_point)
                            
                        if was_included and not include_triangle:
                            print(f"Triangulation indicator clipping: excluded triangle at ({centroid_lat:.3f}, {centroid_lon:.3f}) with value {avg_yield:.2f}")

                    if include_triangle:
                        # Create polygon for this triangle
                        poly = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [float(vertices[0,0]), float(vertices[0,1])],
                                    [float(vertices[1,0]), float(vertices[1,1])],
                                    [float(vertices[2,0]), float(vertices[2,1])],
                                    [float(vertices[0,0]), float(vertices[0,1])]
                                ]]
                            },
                            "properties": {
                                "value": avg_yield,
                                "yield": avg_yield
                            }
                        }
                        
                        # Check if triangle should be excluded by Banks Peninsula
                        should_exclude = False
                        if banks_peninsula_polygon is not None:
                            try:
                                # Use the triangle centroid for exclusion check
                                centroid_lon = np.mean([vertices[0,0], vertices[1,0], vertices[2,0]])
                                centroid_lat = np.mean([vertices[0,1], vertices[1,1], vertices[2,1]])
                                center_point = Point(centroid_lon, centroid_lat)
                                
                                # Exclude if the center point is inside Banks Peninsula
                                if banks_peninsula_polygon.contains(center_point):
                                    should_exclude = True
                            except Exception as e:
                                print(f"Error checking Banks Peninsula exclusion for triangle: {e}")
                        
                        # Only add feature if it's not excluded
                        if not should_exclude:
                            features.append(poly)
        
        # Summary of boundary snapping results
        if adjacent_boundaries is not None:
            print(f"🔧 BOUNDARY SNAPPING SUMMARY:")
            print(f"  📊 Total triangles processed: {total_triangles_processed}")
            print(f"  📊 Total vertices snapped: {total_snapped_vertices}")
            if total_snapped_vertices > 0:
                snap_percentage = (total_snapped_vertices / (total_triangles_processed * 3)) * 100 if total_triangles_processed > 0 else 0
                print(f"  📊 Snap percentage: {snap_percentage:.1f}% of vertices")
                print(f"  🎯 Adjacent boundaries used: {list(adjacent_boundaries.keys())}")
            else:
                print(f"  ⚠️  No vertices were close enough to boundaries for snapping")
                
    except Exception as e:
        # If triangulation fails, fall back to the simpler grid method
        print(f"Triangulation error: {e}, using grid method")
        for i in range(len(grid_lats)):
            # Only process points with meaningful values
            if interpolated_z[i] > 0.01:
                # Check if point should be included based on soil polygons
                include_point = True

                if merged_soil_geometry is not None:
                    point = Point(grid_lons[i], grid_lats[i])
                    include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                # Additional clipping by indicator kriging geometry (high-probability zones)
                if include_point and indicator_geometry is not None and indicator_mask is not None:
                    was_included = include_point
                    
                    try:
                        # Use distance-based approach: check if point is near any high-probability indicator points
                        # Extract indicator mask data for distance checking
                        mask_lat_grid, mask_lon_grid, mask_values, mask_lat_vals, mask_lon_vals = indicator_mask
                        
                        # Find high-probability points (≥0.7)
                        high_prob_mask = mask_values >= 0.7
                        if np.any(high_prob_mask):
                            high_prob_lats = mask_lat_grid[high_prob_mask]
                            high_prob_lons = mask_lon_grid[high_prob_mask]
                            
                            # Calculate distances to all high-probability points
                            distances = np.sqrt((high_prob_lats - grid_lats[i])**2 + (high_prob_lons - grid_lons[i])**2)
                            min_distance = np.min(distances)
                            
                            # Include if within reasonable distance (roughly grid spacing)
                            grid_spacing = abs(mask_lat_vals[1] - mask_lat_vals[0]) if len(mask_lat_vals) > 1 else 0.01
                            include_point = min_distance <= (grid_spacing * 1.5)  # 1.5x grid spacing tolerance
                        else:
                            include_point = False
                    except Exception as e:
                        # Fallback: use geometry-based clipping if distance approach fails
                        point = Point(grid_lons[i], grid_lats[i])
                        include_point = indicator_geometry.contains(point) or indicator_geometry.intersects(point)
                        
                    if was_included and not include_point:
                        print(f"Indicator clipping: excluded point at ({grid_lats[i]:.3f}, {grid_lons[i]:.3f}) with value {interpolated_z[i]:.2f}")

                if include_point:
                    # Create a small circle as a polygon (approximated with 8 points)
                    radius_deg_lat = 0.5 * (lat_vals[1] - lat_vals[0])
                    radius_deg_lon = 0.5 * (lon_vals[1] - lon_vals[0])

                    # Create the polygon coordinates
                    coords = []
                    for angle in np.linspace(0, 2*np.pi, 9):
                        x = grid_lons[i] + radius_deg_lon * np.cos(angle)
                        y = grid_lats[i] + radius_deg_lat * np.sin(angle)
                        coords.append([float(x), float(y)])

                    # Create the polygon feature
                    poly = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        },
                        "properties": {
                            "value": float(interpolated_z[i]),
                            "yield": float(interpolated_z[i])
                        }
                    }
                    
                    # Check if polygon should be excluded by Banks Peninsula
                    should_exclude = False
                    if banks_peninsula_polygon is not None:
                        try:
                            # Use the grid point as the center for exclusion check
                            center_point = Point(grid_lons[i], grid_lats[i])
                            
                            # Exclude if the center point is inside Banks Peninsula
                            if banks_peninsula_polygon.contains(center_point):
                                should_exclude = True
                        except Exception as e:
                            print(f"Error checking Banks Peninsula exclusion: {e}")
                    
                    # Only add feature if it's not excluded
                    if not should_exclude:
                        features.append(poly)

    # Log filtering results
    if merged_soil_geometry is not None:
        print(f"GeoJSON features filtered by soil drainage areas: {len(features)} polygons displayed")
    
    # Apply final square clipping to existing features (triangle removal, not re-interpolation)
    features_before_final_clip = len(features)
    final_clipped_features = []
    
    for feature in features:
        try:
            # Get polygon centroid
            coords = feature['geometry']['coordinates'][0]
            if len(coords) >= 3:
                centroid_lon = sum(coord[0] for coord in coords) / len(coords)
                centroid_lat = sum(coord[1] for coord in coords) / len(coords)
                centroid_point = Point(centroid_lon, centroid_lat)
                
                # Create triangle polygon for proper intersection clipping
                triangle_coords = [(coord[0], coord[1]) for coord in coords[:-1]]  # Remove duplicate closing point
                if len(triangle_coords) >= 3:
                    triangle_polygon = ShapelyPolygon(triangle_coords)
                    
                    # CRITICAL FIX: When using boundary snapping, allow triangles that intersect boundary
                    if adjacent_boundaries is not None:
                        # With boundary snapping: keep triangles that intersect or are contained
                        if final_clip_geometry.intersects(triangle_polygon):
                            final_clipped_features.append(feature)
                    else:
                        # Without boundary snapping: strict containment only
                        if final_clip_geometry.contains(triangle_polygon):
                            final_clipped_features.append(feature)
                else:
                    # If we can't create a proper triangle, keep the feature
                    final_clipped_features.append(feature)
        except Exception as e:
            # If centroid calculation fails, keep the feature
            final_clipped_features.append(feature)
    
    features = final_clipped_features
    print(f"Final square clipping: {features_before_final_clip} -> {len(features)} features ({final_radius_km:.1f}km sides)")

    # Create the full GeoJSON object
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return geojson

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50, method='kriging', soil_polygons=None):
    """
    Generate heat map data using various interpolation techniques

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude and yield_rate columns
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point
    radius_km : float
        Radius in kilometers to generate heat map data for
    resolution : int
        Number of points to generate in each dimension
    method : str
        Interpolation method to use ('kriging', 'idw', 'rf_kriging')

    Returns:
    --------
    list
        List of [lat, lng, intensity] points for the heat map

    Notes:
    ------
    This function is called each time a new location is selected to create an
    entirely fresh interpolation. The results should always be specific to the
    current center_point and never reuse old interpolation data.
    """
    # Handle empty datasets
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return []

    # Filter out geotechnical/geological investigation wells using well_use
    if 'well_use' in wells_df.columns:
        geotechnical_mask = wells_df['well_use'].str.contains(
            'Geotechnical.*Investigation|Geological.*Investigation', 
            case=False, 
            na=False, 
            regex=True
        )
        wells_df = wells_df[~geotechnical_mask].copy()

        if wells_df.empty:
            return []

    # Use the new categorization system
    from data_loader import get_wells_for_interpolation

    if method == 'depth_kriging':
        # Get wells appropriate for depth interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'depth')
        if wells_df_filtered.empty:
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)

        # Use depth_to_groundwater if available, otherwise fall back to depth
        if 'depth_to_groundwater' in wells_df_filtered.columns and wells_df_filtered['depth_to_groundwater'].notna().any():
            yields = wells_df_filtered['depth_to_groundwater'].values.astype(float)
        else:
            yields = wells_df_filtered['depth'].values.astype(float)
    elif method == 'specific_capacity_kriging':
        # Get wells appropriate for specific capacity interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'specific_capacity')
        if wells_df_filtered.empty:
            return []

        # Ensure all wells have valid specific capacity data
        wells_df_filtered = wells_df_filtered[
            wells_df_filtered['specific_capacity'].notna() & 
            (wells_df_filtered['specific_capacity'] > 0)
        ].copy()

        if wells_df_filtered.empty:
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)
        yields = wells_df_filtered['specific_capacity'].values.astype(float)
    elif method == 'ground_water_level_kriging':
        # Get wells appropriate for ground water level interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'ground_water_level')
        if wells_df_filtered.empty:
            print("No wells found for ground water level interpolation")
            return []

        # Ensure all wells have valid ground water level data
        valid_gwl_mask = (
            wells_df_filtered['ground water level'].notna() & 
            (wells_df_filtered['ground water level'] != 0) &
            (wells_df_filtered['ground water level'].abs() > 0.1)  # Exclude very small values
        )

        wells_df_filtered = wells_df_filtered[valid_gwl_mask].copy()

        if wells_df_filtered.empty:
            print("No valid ground water level data after filtering")
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)
        yields = wells_df_filtered['ground water level'].values.astype(float)

        print(f"Heat map ground water level: using {len(yields)} wells with GWL values from {yields.min():.2f} to {yields.max():.2f}")
    elif method == 'indicator_kriging':
        # Get wells appropriate for indicator interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'yield')
        if wells_df_filtered.empty:
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)

        # Convert yields to BINARY indicator values for kriging input
        # 0 for wells with yield < 0.1 L/s (not viable)
        # 1 for wells with yield ≥ 0.1 L/s (viable)
        yield_threshold = 0.1
        raw_yields = wells_df_filtered['yield_rate'].values.astype(float)
        yields = (raw_yields >= yield_threshold).astype(float)  # Binary: 1 or 0

        # Count wells in each category for logging
        viable_count = np.sum(yields == 1)
        non_viable_count = np.sum(yields == 0)

        print(f"Heat map indicator kriging: using {len(yields)} wells with binary classification")
        print(f"Non-viable (<{yield_threshold} L/s): {non_viable_count} wells ({100*non_viable_count/len(yields):.1f}%)")
        print(f"Viable (≥{yield_threshold} L/s): {viable_count} wells ({100*viable_count/len(yields):.1f}%)")
        print(f"Raw yield range: {raw_yields.min():.3f} to {raw_yields.max():.3f} L/s")
    else:
        # Get wells appropriate for yield interpolation
        wells_df_filtered = get_wells_for_interpolation(wells_df, 'yield')
        if wells_df_filtered.empty:
            return []

        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)
        yields = wells_df_filtered['yield_rate'].values.astype(float)

    # Handle case with too few data points
    if len(wells_df) < 3:
        heat_data = []
        for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
            heat_data.append([float(lat), float(lon), float(yield_val)])
        return heat_data

    # Create simplified grid for interpolation
    center_lat, center_lon = center_point
    grid_size = min(50, max(30, resolution))

    try:
        # Convert to km-based coordinates (flat Earth approximation for small areas)
        # This is essential for proper interpolation
        km_per_degree_lat = 111.0  # km per degree of latitude
        km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))  # km per degree of longitude

        # Convert all coordinates to km from center
        x_coords = (lons - center_lon) * km_per_degree_lon
        y_coords = (lats - center_lat) * km_per_degree_lat

        # Create grid in km space (square bounds)
        grid_x = np.linspace(-radius_km, radius_km, grid_size)
        grid_y = np.linspace(-radius_km, radius_km, grid_size)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

        # Flatten for interpolation
        points = np.vstack([x_coords, y_coords]).T  # Well points in km
        xi = np.vstack([grid_X.flatten(), grid_Y.flatten()]).T  # Grid points in km

        # Filter points outside the square bounds
        mask = (np.abs(xi[:,0]) <= radius_km) & (np.abs(xi[:,1]) <= radius_km)
        xi_inside = xi[mask]

        # Define grid_points for compatibility
        grid_points = xi_inside

        # Choose interpolation method based on parameter and dataset size
        if (method == 'yield_kriging' or method == 'specific_capacity_kriging' or method == 'ground_water_level_kriging' or method == 'indicator_kriging') and len(wells_df) >= 5:
            try:
                if method == 'specific_capacity_kriging':
                    interpolation_name = "specific capacity kriging"
                elif method == 'ground_water_level_kriging':
                    interpolation_name = "ground water level kriging"
                elif method == 'indicator_kriging':
                    interpolation_name = "indicator kriging (yield suitability)"
                else:
                    interpolation_name = "yield kriging"
                print(f"Using {interpolation_name} interpolation for heat map")

                # Filter to meaningful data for better kriging
                if method == 'indicator_kriging':
                    # For indicator kriging, use all data (including 0s and 1s)
                    meaningful_data_mask = np.ones(len(yields), dtype=bool)  # Use all data
                else:
                    # For other methods, filter to meaningful yield data
                    meaningful_data_mask = yields > 0.1

                if meaningful_data_mask.any() and np.sum(meaningful_data_mask) >= 5:
                    # Use filtered data
                    filtered_x_coords = x_coords[meaningful_data_mask]
                    filtered_y_coords = y_coords[meaningful_data_mask] 
                    filtered_yields = yields[meaningful_data_mask]

                    # Convert to lat/lon for kriging
                    filtered_lons = filtered_x_coords / km_per_degree_lon + center_lon
                    filtered_lats = filtered_y_coords / km_per_degree_lat + center_lat

                    # Set up kriging with appropriate variogram model
                    if method == 'indicator_kriging':
                        # Use linear variogram for binary indicator data
                        variogram_model_to_use = 'linear'
                    else:
                        variogram_model_to_use = 'spherical'

                    OK = OrdinaryKriging(
                        filtered_lons, filtered_lats, filtered_yields,
                        variogram_model=variogram_model_to_use,
                        verbose=False,
                        enable_plotting=False,
                        variogram_parameters=None
                    )

                    # Execute kriging
                    xi_lon = xi_inside[:, 0] / km_per_degree_lon + center_lon
                    xi_lat = xi_inside[:, 1] / km_per_degree_lat + center_lat
                    interpolated_z, _ = OK.execute('points', xi_lon, xi_lat)

                    # Process results based on interpolation method
                    if method == 'indicator_kriging':
                        # Ensure probability values are in [0,1] range
                        interpolated_z = np.clip(interpolated_z, 0.0, 1.0)

                        # Apply binary threshold for clear visualization (0.5 = 50% probability)
                        binary_threshold = 0.5
                        interpolated_z = (interpolated_z >= binary_threshold).astype(float)

                        print(f"Heat map indicator kriging: binary classification with {np.sum(interpolated_z)}/{len(interpolated_z)} areas classified as 'likely' for groundwater")
                    else:
                        # Ensure non-negative yields for other methods
                        interpolated_z = np.maximum(0, interpolated_z)
                else:
                    # Fallback to griddata if insufficient data
                    interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)
                    nan_mask = np.isnan(interpolated_z)
                    if np.any(nan_mask):
                        interpolated_z[nan_mask] = griddata(
                            points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                        )

            except Exception as e:
                print(f"Yield kriging error: {e}, fallingback to standard interpolation")
                interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)
                nan_mask = np.isnan(interpolated_z)
                if np.any(nan_mask):
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                    )

        elif method == 'rf_kriging' and len(wells_df) >= 10:
            try:
                print("Using Random Forest + Kriging interpolation")
                # OPTIMIZATION: Reduce number of trees for faster performance
                # For large datasets, we need to prioritize speed over slight accuracy improvements

                # Prepare data for Random Forest
                features = np.vstack([x_coords, y_coords]).T  # Features are [x, y] coordinates in km
                target = yields  # Target is the yield values

                # OPTIMIZATION: Use fewer trees (50 instead of 100) and limit max_depth
                # This significantly speeds up training and prediction with minimal accuracy loss
                rf = RandomForestRegressor(
                    n_estimators=50,       # Reduced from 100 for faster performance
                    max_depth=15,          # Limit tree depth for faster training
                    min_samples_split=5,   # Require more samples per split (reduces overfitting)
                    n_jobs=-1,             # Use all available cores
                    random_state=42
                )
                rf.fit(features, target)

                # Get RF predictions for all grid points
                rf_predictions = rf.predict(xi_inside)

                # OPTIMIZATION: Skip kriging for very large datasets (>1000 points)
                # as it becomes the main performance bottleneck
                if len(features) < 1000:
                    # Calculate residuals on training data
                    rf_train_preds = rf.predict(features)
                    residuals = target - rf_train_preds

                    # If enough points, apply Kriging to the residuals
                    if len(features) >= 5 and len(features) < 1000:
                        # Convert back to lon/lat for kriging (pykrige expects lon/lat)
                        lon_values = x_coords / km_per_degree_lon + center_lon
                        lat_values = y_coords / km_per_degree_lat + center_lat
                        xi_lon = xi_inside[:, 0] / km_per_degree_lon + center_lon
                        xi_lat = xi_inside[:, 1] / km_per_degree_lat + center_lat

                        # OPTIMIZATION: Use a simpler variogram model and limit kriging calculations
                        OK = OrdinaryKriging(
                            lon_values, lat_values, residuals,
                            variogram_model='linear',  # Simpler model than spherical - much faster
                            verbose=False,
                            enable_plotting=False
                        )
                        # Execute kriging on grid points
                        kriged_residuals, _ = OK.execute('points', xi_lon, xi_lat)

                        # Combine RF predictions with kriged residuals
                        interpolated_z = rf_predictions + kriged_residuals
                    else:
                        # Not enough points for kriging, use RF predictions only
                        interpolated_z = rf_predictions
                else:
                    # Too many points for efficient kriging, use RF predictions only
                    print("Using RF predictions only (skipping kriging for large dataset)")
                    interpolated_z = rf_predictions
            except Exception as e:
                print(f"RF+Kriging error: {e}, falling back to standard interpolation")
                # Fall back to standard interpolation
                # Basic 2D interpolation (linear)
                from scipy.interpolate import griddata
                interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)

                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                    )
        else:
            # Choose interpolation method
            # Basic 2D interpolation using scipy.interpolate.griddata

            # OPTIMIZATION: For large datasets, use fewer points and faster method
            if len(points) > 2000:
                print(f"Large dataset optimization: Sampling from {len(points)} points")
                # For very large datasets, we'll use a random sample to improve performance
                sample_size = min(2000, len(points))
                # Use stratified sampling for better representation (divide area into regions)
                # Create grid cells (10x10)
                lat_bins = np.linspace(np.min(lats), np.max(lats), 10)
                lon_bins = np.linspace(np.min(lons), np.max(lons), 10)

                # Create a mask for sampling points from each cell
                sample_indices = []
                for i in range(9):
                    for j in range(9):
                        # Get points in this cell
                        cell_mask = (
                            (lats >= lat_bins[i]) & (lats < lat_bins[i+1]) & 
                            (lons >= lon_bins[j]) & (lons < lon_bins[j+1])
                        )
                        cell_indices = np.where(cell_mask)[0]

                        # If there are points in this cell, sample a proportional amount
                        if len(cell_indices) > 0:
                            # Calculate how many points to sample from this cell
                            # (proportional to the cell's share of total points)
                            cell_sample_size = max(1, int(sample_size * len(cell_indices) / len(points)))
                            # Sample randomly from this cell
                            if len(cell_indices) > cell_sample_size:
                                cell_sample = np.random.choice(cell_indices, cell_sample_size, replace=False)
                                sample_indices.extend(cell_sample)
                            else:
                                # If cell has fewer points than needed, take all of them
                                sample_indices.extend(cell_indices)

                # Ensure we have enough sample points
                if len(sample_indices) < sample_size:
                    # Add more random points if needed
                    remaining = sample_size - len(sample_indices)
                    all_indices = np.arange(len(points))
                    remaining_indices = np.setdiff1d(all_indices, sample_indices)
                    if len(remaining_indices) > 0:
                        additional_samples = np.random.choice(
                            remaining_indices, 
                            min(remaining, len(remaining_indices)), 
                            replace=False
                        )
                        sample_indices.extend(additional_samples)

                # Use the sampled points for interpolation
                sampled_points = points[sample_indices]
                sampled_yields = yields[sample_indices]

                # Use linear interpolation for speed with large datasets
                interpolated_z = griddata(
                    sampled_points, sampled_yields, xi_inside, 
                    method='linear', fill_value=0.0
                )

                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        sampled_points, sampled_yields, xi_inside[nan_mask],
                        method='nearest', fill_value=0.0
                    )
            else:
                # Standard approach for smaller datasets
                # First try linear interpolation
                interpolated_z = griddata(
                    points, yields, xi_inside, 
                    method='linear', fill_value=0.0
                )

                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask],
                        method='nearest', fill_value=0.0
                    )

        # Make sure we don't have negative values
        interpolated_z = np.maximum(0, interpolated_z)

        # Convert back to lat/lon coordinates
        lat_points = (xi_inside[:, 1] / km_per_degree_lat) + center_lat
        lon_points = (xi_inside[:, 0] / km_per_degree_lon) + center_lon

        # Prepare soil polygon geometry for filtering heat map display
        merged_soil_geometry = None
        if soil_polygons is not None and len(soil_polygons) > 0:
            try:
                # Create a unified geometry from all soil polygons
                valid_geometries = []
                for idx, row in soil_polygons.iterrows():
                    if row.geometry and row.geometry.is_valid:
                        valid_geometries.append(row.geometry)

                if valid_geometries:
                    # Merge all polygons into a single multipolygon
                    merged_soil_geometry = unary_union(valid_geometries)
                    print(f"Prepared soil drainage geometry for heat map filtering")
                else:
                    print("No valid soil polygon geometries found for heat map")
            except Exception as e:
                print(f"Error preparing soil polygon geometry for heat map: {e}")
                merged_soil_geometry = None

        # Create heat map data
        heat_data = []

        # OPTIMIZATION: For large datasets, reduce the number of heat map points for better performance
        # First, detect if we need to sample points due to a large dataset
        max_heat_points = 2500  # Maximum points for smooth performance

        if len(lat_points) > max_heat_points:
            # Use a grid-based sampling approach to maintain visual accuracy with fewer points
            print(f"Optimizing heatmap visualization: sampling {max_heat_points} points from {len(lat_points)} total")

            # Create a grid with target number of cells
            grid_size = int(np.sqrt(max_heat_points))

            # Find min and max lat/lon
            min_lat, max_lat = np.min(lat_points), np.max(lat_points)
            min_lon, max_lon = np.min(lon_points), np.max(lon_points)

            # Create grid
            lat_grid = np.linspace(min_lat, max_lat, grid_size)
            lon_grid = np.linspace(min_lon, max_lon, grid_size)

            # Sample points by selecting representative points in each grid cell
            heat_data = []
            for i in range(len(lat_grid)-1):
                for j in range(len(lon_grid)-1):
                    # Find points in this grid cell
                    cell_mask = (
                        (lat_points >= lat_grid[i]) & (lat_points < lat_grid[i+1]) &
                        (lon_points >= lon_grid[j]) & (lon_points < lon_grid[j+1])
                    )

                    if np.any(cell_mask):
                        # Select a point with the maximum value from this cell (important for yield visualization)
                        cell_values = interpolated_z[cell_mask]
                        cell_lat = lat_points[cell_mask]
                        cell_lon = lon_points[cell_mask]

                        if np.max(cell_values) > 0.01:  # Only add significant values
                            max_idx = np.argmax(cell_values)
                            point_lat = float(cell_lat[max_idx])
                            point_lon = float(cell_lon[max_idx])

                            # Check if point should be included based on soil polygons
                            include_point = True
                            if merged_soil_geometry is not None:
                                point = Point(point_lon, point_lat)
                                include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                            if include_point:
                                # Double-check: ensure point is actually within soil polygons
                                if merged_soil_geometry is not None:
                                    point = Point(point_lon, point_lat)
                                    # Use strict containment for heat map points
                                    strictly_contained = merged_soil_geometry.contains(point)

                                    # If not strictly contained, check if very close to boundary
                                    if not strictly_contained:
                                        buffer_distance = 0.0001  # roughly 10 meters
                                        buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                                        strictly_contained = buffered_geometry.contains(point)

                                    if strictly_contained:
                                        heat_data.append([
                                            point_lat,  # Latitude
                                            point_lon,  # Longitude
                                            float(cell_values[max_idx])  # Yield value (actual value)
                                        ])
                                else:
                                    heat_data.append([
                                        point_lat,  # Latitude
                                        point_lon,  # Longitude
                                        float(cell_values[max_idx])  # Yield value (actual value)
                                    ])
        else:
            # Standard approach for smaller datasets
            heat_data = []
            # Add interpolated points
            for i in range(len(lat_points)):
                # Adjust threshold based on interpolation method
                meaningful_threshold = 0.01

                # Only add points with meaningful values
                if interpolated_z[i] > meaningful_threshold:
                    # Check if point should be included based on soil polygons
                    include_point = True
                    if merged_soil_geometry is not None:
                        # For non-SWL methods, apply normal soil polygon filtering
                        point = Point(lon_points[i], lat_points[i])
                        include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                    if include_point:
                        heat_data.append([
                            float(lat_points[i]),  # Latitude
                            float(lon_points[i]),  # Longitude
                            float(interpolated_z[i])  # SWL/yield value (actual value, not normalized)
                        ])

        # Always make sure well points themselves are included for accuracy
        # These are the actual data points we have, so they should be shown
        well_points_added = 0
        for j in range(len(lats)):
            # Check if well is within search radius
            well_dist_km = np.sqrt(
                ((lats[j] - center_lat) * km_per_degree_lat)**2 +
                ((lons[j] - center_lon) * km_per_degree_lon)**2
            )

            if well_dist_km <= radius_km:
                # Check if well should be included based on soil polygons
                include_well = True
                if merged_soil_geometry is not None:
                    well_point = Point(lons[j], lats[j])
                    include_well = merged_soil_geometry.contains(well_point) or merged_soil_geometry.intersects(well_point)

                if include_well:
                    # Check soil polygon containment based on method
                    well_point = Point(lons[j], lats[j])
                    well_contained = merged_soil_geometry.contains(well_point) or merged_soil_geometry.intersects(well_point)

                    if well_contained:
                        heat_data.append([
                            float(lats[j]),
                            float(lons[j]),
                            float(yields[j])
                        ])
                        well_points_added += 1
                    else:
                        # No soil polygon filtering
                        heat_data.append([
                            float(lats[j]),
                            float(lons[j]),
                            float(yields[j])
                        ])
                        well_points_added += 1

        # Log filtering results
        if merged_soil_geometry is not None:
            print(f"Heat map filtered by soil drainage areas: {len(heat_data)} points displayed")

        return heat_data

    except Exception as e:
        print(f"Interpolation error: {e}")
        return fallback_interpolation(wells_df, center_point, radius_km)

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """
    Simplified IDW (Inverse Distance Weighting) interpolation as fallback method
    Creates a continuous interpolated surface based on actual well yield values
    """
    # Filter to only use wells with actual yield data (exclude unknown yields)
    if 'has_unknown_yield' in wells_df.columns:
        valid_wells = wells_df[(~wells_df['has_unknown_yield']) & (wells_df['yield_rate'].notna())].copy()
    else:
        valid_wells = wells_df[wells_df['yield_rate'].notna()].copy()

    if valid_wells.empty:
        return []

    # Extract coordinates and yields
    lats = valid_wells['latitude'].values.astype(float)
    lons = valid_wells['longitude'].values.astype(float)
    yields = valid_wells['yield_rate'].values.astype(float)

    # Handle empty dataset
    if len(yields) == 0:
        return []

    center_lat,center_lon = center_point

    # Create a grid for interpolation
    grid_resolution = min(50, resolution)

    # Calculate area to cover
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))

    # Grid boundaries
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius

    # Create grid
    grid_lats = np.linspace(min_lat, max_lat, grid_resolution)
    grid_lons = np.linspace(min_lon, max_lon, grid_resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)

    # Flatten grid for calculations
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()

    # Heat map data collection
    heat_data = []

    # Convert to a simple projected space (km from center)
    x_points = (lons - center_lon) * 111.0 * np.cos(np.radians(center_lat))
    y_points = (lats - center_lat) * 111.0

    # For each grid point, calculate IDW interpolation
    for i in range(len(grid_lat_flat)):
        grid_point_lat = grid_lat_flat[i]
        grid_point_lon = grid_lon_flat[i]

        # Skip if outside search radius
        dist_from_center_km = np.sqrt(
            ((grid_point_lat - center_lat) * 111.0)**2 + 
            ((grid_point_lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )

        if dist_from_center_km > radius_km:
            continue

        # Convert grid point to projected space
        grid_x = (grid_point_lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))
        grid_y = (grid_point_lat - center_lat) * 111.0

        # Calculate distances to all wells
        distances = np.sqrt((grid_x - x_points)**2 + (grid_y - y_points)**2)

        # IDW Parameters
        power = 2.0  # Standard IDW power parameter

        # Standard IDW formula
        if np.min(distances) < 0.05:  # If very close to a well (within 50m)
            # Find the closest well
            closest_idx = np.argmin(distances)
            interpolated_value = yields[closest_idx]
        else:
            # Apply IDW with distance power weighting
            weights = 1.0 / (distances**power)

            # Handle potential division by zero
            if np.any(np.isinf(weights)):
                mask = np.isinf(weights)
                weights[mask] = 1.0
                weights[~mask] = 0.0

            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                # Calculate weighted average
                interpolated_value = np.sum(weights * yields)
            else:
                # This shouldn't happen, but just in case
                interpolated_value = 0.0

        # Add to heat map with interpolated value (actual yield value)
        if interpolated_value > 0.01:  # Only add significant points
            heat_data.append([
                float(grid_point_lat), 
                float(grid_point_lon),
                float(interpolated_value)  # Use actual yield value
            ])

    # Always add the actual well points with their values
    for j in range(len(lats)):
        # Check if within search radius
        dist_from_center_km = np.sqrt(
            ((lats[j] - center_lat) * 111.0)**2 + 
            ((lons[j] - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )

        if dist_from_center_km <= radius_km:
            heat_data.append([
                float(lats[j]),
                float(lons[j]),
                float(yields[j])  # Use actual yield value
            ])

    return heat_data

def get_prediction_at_point(wells_df, point_lat, point_lon):
    """
    Get a predicted yield at a specific point based on nearby wells using modified IDW

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    point_lat : float
        Latitude of the point to predict
    point_lon : float
        Longitude of the point to predict

    Returns:
    --------
    float
        Predicted yield rate at the specified point
    """
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        return 0

    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)

    # Convert to kilometer distances for better accuracy
    # First, convert to flat projection (rough approximation using kilometers)
    origin_lat, origin_lon = np.mean(lats), np.mean(lons)
    x_coords = (lons - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    y_coords = (lats - origin_lat) * 111.0

    # Convert prediction point to same projection
    point_x = (point_lon - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    point_y = (point_lat - origin_lat) * 111.0

    try:
        # Calculate distances to all wells (in kilometers)
        distances = np.sqrt((point_x - x_coords)**2 + (point_y - y_coords)**2)

        # Parameters for modified IDW
        # Higher power value creates sharper transitions between high and low yield areas
        power = 2.5
        smoothing = 0.1  # Small value to prevent division by zero

        # Maximum distance influence (in km)
        # Points beyond this have minimal impact on prediction
        max_influence_distance = 5.0  # 5km influence radius

        # Apply exponential distance decay to sharply reduce influence of distant wells
        exp_weights = np.exp(-distances / (max_influence_distance / 3))

        # Apply traditional inverse distance power weighting with higher power
        idw_weights = 1.0 / (distances + smoothing)**power

        # Combine both weighting strategies
        weights = exp_weights * idw_weights

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)

            # Calculate weighted average of yields
            predicted_yield = np.sum(weights * yields)

            # If point is very far from any well, reduce confidence
            nearest_dist = np.min(distances)

            if nearest_dist > max_influence_distance:
                # Adjust prediction closer to area average for distant points
                confidence = max(0.2, np.exp(-(nearest_dist - max_influence_distance) / max_influence_distance))
                avg_yield = np.mean(yields)
                predicted_yield = predicted_yield * confidence + avg_yield * (1 - confidence)

            return float(max(0, predicted_yield))  # Ensure non-negative yield
        else:
            return 0.0

    except Exception as e:
        print(f"Prediction error: {e}")
        # Fall back to basic IDW
        return basic_idw_prediction(wells_df, point_lat, point_lon)

def basic_idw_prediction(wells_df, point_lat, point_lon):
    """
    Calculate yield using basic Inverse Distance Weighting (IDW)
    Used as a fallback method - only uses wells with actual yield data
    """
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        return 0

    # Filter to only use wells with actual yield data
    if 'has_unknown_yield' in wells_df.columns:
        valid_wells = wells_df[(~wells_df['has_unknown_yield']) & (wells_df['yield_rate'].notna())].copy()
    else:
        valid_wells = wells_df[wells_df['yield_rate'].notna()].copy()

    if valid_wells.empty:
        return 0

    # Calculate distance from each well to the point (in km)
    distances = []
    for idx, row in valid_wells.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        # Convert to kilometers using approximate conversion
        lat_dist = (lat - point_lat) * 111.0  # 1 degree latitude ≈ 111 km
        lon_dist = (lon - point_lon) * 111.0 * np.cos(np.radians((lat + point_lat) / 2))
        distance = np.sqrt(lat_dist**2 + lon_dist**2)
        distances.append(max(0.1, distance))  # Prevent division by zero

    # Calculate inverse distance weights with power of 2
    weights = [1 / (d**2) for d in distances]
    total_weight = sum(weights)

    if total_weight == 0:
        # If all weights are zero (should be impossible with our minimum distance)
        return 0

    # Calculate weighted average of yields
    weighted_yield = sum(w * float(row['yield_rate']) for w, (idx, row) in zip(weights, valid_wells.iterrows())) / total_weight
    return float(max(0, weighted_yield))  # Ensure non-negative yieldld

def calculate_kriging_variance(wells_df, center_point, radius_km, resolution=50, variogram_model='spherical', use_depth=False, soil_polygons=None):
    """
    Calculate kriging variance for yield or depth to groundwater interpolations.

    This function performs ordinary kriging to estimate the variance (uncertainty)
    of the interpolated values. It's designed to work with both yield and
    depth-to-groundwater data, and applies soil polygon filtering like other interpolants.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data, including latitude, longitude, and
        either yield_rate (for yield interpolation) or depth_to_groundwater/depth
        (for depth interpolation).
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point.
    radius_km : float
        Radius in kilometers to generate the variance data for.
    resolution : int
        Number of points to generate in each dimension.
    variogram_model : str, optional
        Variogram model to use for kriging (e.g., 'linear', 'spherical', 'gaussian').
        Defaults to 'spherical'.
    use_depth : bool
        Whether to use depth data instead of yield data.
    soil_polygons : GeoDataFrame, optional
        Soil polygon data for clipping the variance output.

    Returns:
    --------
    list
        List of [lat, lng, variance] points for the variance heat map.
        Returns an empty list if there is not enough data or if an error occurs.

    Notes:
    ------
    - The function automatically detects whether to interpolate yield or depth
      based on the columns available in `wells_df`.
    - It handles cases where depth_to_groundwater is not available and falls
      back to using the 'depth' column if it exists.
    - It returns an empty list if there is insufficient data (less than 5 wells)
      to perform kriging.
    - Applies the same soil polygon clipping as other interpolants.
    """
    try:
        # Filter wells data similar to yield kriging for consistency
        wells_df_filtered = wells_df.copy()

        # Filter out geotechnical/geological investigation wells using well_use
        if 'well_use' in wells_df_filtered.columns:
            geotechnical_mask = wells_df_filtered['well_use'].str.contains(
                'Geotechnical.*Investigation|Geological.*Investigation', 
                case=False, 
                na=False, 
                regex=True
            )
            wells_df_filtered = wells_df_filtered[~geotechnical_mask].copy()

            if wells_df_filtered.empty:
                return []

        # Determine whether to use yield or depth data based on use_depth parameter
        if use_depth:
            # For depth variance, filter wells similar to depth kriging
            if 'is_dry_well' in wells_df.columns:
                wells_df_filtered = wells_df_filtered[~wells_df_filtered['is_dry_well']]

            if 'depth_to_groundwater' in wells_df_filtered.columns and not wells_df_filtered['depth_to_groundwater'].isna().all():
                wells_df_filtered = wells_df_filtered[wells_df_filtered['depth_to_groundwater'].notna() & (wells_df_filtered['depth_to_groundwater'] > 0)]
                values = wells_df_filtered['depth_to_groundwater'].values.astype(float)
                interpolation_type = 'depth_to_groundwater'
            elif 'depth' in wells_df_filtered.columns:
                wells_df_filtered = wells_df_filtered[wells_df_filtered['depth'].notna() & (wells_df_filtered['depth'] > 0)]
                values = wells_df_filtered['depth'].values.astype(float)
                interpolation_type = 'depth'
            else:
                print("Error: No depth data found in wells_df.")
                return []
        else:
            # For yield variance, filter to meaningful yield data like yield kriging
            meaningful_yield_mask = wells_df_filtered['yield_rate'].fillna(0) > 0.1

            if meaningful_yield_mask.any() and np.sum(meaningful_yield_mask) >= 5:
                wells_df_filtered = wells_df_filtered[meaningful_yield_mask]
                values = wells_df_filtered['yield_rate'].values.astype(float)
                interpolation_type = 'yield'
            else:
                # If insufficient meaningful yield data, use all available yield data
                values = wells_df_filtered['yield_rate'].fillna(0).values.astype(float)
                interpolation_type = 'yield'

        # Prepare data: Extract coordinates and values from filtered data
        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)

        # Ensure there's enough data for kriging
        if len(wells_df_filtered) < 5:
            print("Warning: Insufficient data points for kriging variance calculation after filtering.")
            return []

        # Create grid for interpolation
        center_lat, center_lon = center_point
        grid_size = min(50, max(30, resolution))  # Adjust grid size if necessary

        # Convert to km-based coordinates
        km_per_degree_lat = 111.0
        km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))

        # Convert coordinates to km from center
        x_coords = (lons - center_lon) * km_per_degree_lon
        y_coords = (lats - center_lat) * km_per_degree_lat

        # Create grid in km space (square bounds)
        grid_x = np.linspace(-radius_km, radius_km, grid_size)
        grid_y = np.linspace(-radius_km, radius_km, grid_size)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

        # Flatten grid for interpolation
        xi = np.vstack([grid_X.flatten(), grid_Y.flatten()]).T

        # Filter points outside the square bounds (instead of circular)
        mask = (np.abs(xi[:,0]) <= radius_km) & (np.abs(xi[:,1]) <= radius_km)
        xi_inside = xi[mask]
        # Convert back to lat/lon for kriging
        lon_values = x_coords / km_per_degree_lon + center_lon
        lat_values = y_coords / km_per_degree_lat + center_lat
        xi_lon = xi_inside[:, 0] / km_per_degree_lon + center_lon
        xi_lat = xi_inside[:, 1] / km_per_degree_lat + center_lat

        # Perform Ordinary Kriging with enhanced variance calculation
        try:
            # First, try standard kriging with optimized parameters for variance
            OK = OrdinaryKriging(
                lon_values, lat_values, values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
                weight=False,  # Disable additional weighting for cleaner variance
                exact_values=False,  # Allow interpolation for better variance estimation
                pseudo_inv=False  # Use standard inversion
            )

            # Execute kriging to get both predictions and variance
            predictions, kriging_variance = OK.execute('points', xi_lon, xi_lat)

            print(f"Initial kriging variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}, Mean: {np.mean(kriging_variance):.6f}")

            # Ensure variance values are reasonable (variance should always be positive)
            kriging_variance = np.maximum(kriging_variance, 1e-8)

            # Check if variance is too uniform (indicates calculation issue)
            variance_std = np.std(kriging_variance)
            variance_mean = np.mean(kriging_variance)
            variance_cv = variance_std / variance_mean if variance_mean > 0 else 0

            print(f"Variance coefficient of variation: {variance_cv:.4f}")

            if variance_cv < 0.1:  # If variance is too uniform (less than 10% variation)
                print("Warning: Variance appears too uniform, enhancing with multiple variogram models...")

                # Try multiple variogram models and combine results
                variance_results = []
                variogram_models = ['linear', 'power', 'gaussian', 'exponential']

                for model in variogram_models:
                    try:
                        OK_test = OrdinaryKriging(
                            lon_values, lat_values, values,
                            variogram_model=model,
                            verbose=False,
                            enable_plotting=False
                        )
                        _, test_variance = OK_test.execute('points', xi_lon, xi_lat)
                        test_variance = np.maximum(test_variance, 1e-8)

                        # Check if this model produces better variation
                        test_cv = np.std(test_variance) / np.mean(test_variance) if np.mean(test_variance) > 0 else 0
                        variance_results.append((model, test_variance, test_cv))
                        print(f"Model {model}: CV = {test_cv:.4f}")

                    except Exception as model_error:
                        print(f"Model {model} failed: {model_error}")
                        continue

                # Use the model with the highest coefficient of variation (most spatial variation)
                if variance_results:
                    best_model, best_variance, best_cv = max(variance_results, key=lambda x: x[2])
                    print(f"Using {best_model} model with CV = {best_cv:.4f}")
                    kriging_variance = best_variance
                    variance_cv = best_cv

            # If still too uniform, enhance with distance-based variation
            if variance_cv < 0.05:
                print("Enhancing variance with distance-based component...")

                # Calculate distance from each grid point to nearest data point
                from scipy.spatial.distance import cdist

                # Create coordinate arrays for distance calculation
                data_coords = np.column_stack([lat_values, lon_values])
                grid_coords = np.column_stack([xi_lat, xi_lon])

                # Calculate distances (in decimal degrees)
                distances = cdist(grid_coords, data_coords)
                min_distances = np.min(distances, axis=1)

                # Create distance-based variance enhancement
                max_distance = np.max(min_distances) if len(min_distances) > 0 else 1.0
                if max_distance > 0:
                    # Normalize distances and create variance enhancement factor
                    normalized_distances = min_distances / max_distance

                    # Create exponential distance decay for variance
                    distance_factor = 1.0 + 2.0 * np.exp(normalized_distances * 3.0)  # Exponential growth with distance

                    # Enhance the original kriging variance with distance component
                    base_variance = np.mean(kriging_variance)
                    enhanced_variance = kriging_variance * distance_factor

                    # Smooth the transition between original and enhanced variance
                    alpha = 0.6  # Blend factor
                    kriging_variance = alpha * enhanced_variance + (1 - alpha) * kriging_variance

                    print(f"Enhanced variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}")

            # Final variance validation and scaling
            kriging_variance = np.maximum(kriging_variance, 1e-6)

            # Scale variance to reasonable range for visualization
            min_var = np.min(kriging_variance)
            max_var = np.max(kriging_variance)
            var_range = max_var - min_var

            if var_range > 0:
                # Normalize to 0-1 range then scale to meaningful variance range
                normalized_var = (kriging_variance - min_var) / var_range
                # Scale to range that makes sense for the data type
                if use_depth:
                    kriging_variance = normalized_var * (np.var(values) * 2.0) + (np.var(values) * 0.1)
                else:
                    kriging_variance = normalized_var * (np.var(values) * 1.5) + (np.var(values) * 0.05)

            print(f"Final variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}, CV: {np.std(kriging_variance)/np.mean(kriging_variance):.4f}")

        except Exception as e:
            print(f"Error in kriging variance calculation: {e}")
            # Fallback: create synthetic variance based on distance to nearest data point
            print("Using enhanced distance-based variance estimation as fallback")

            # Calculate distance from each grid point to nearest data point
            from scipy.spatial.distance import cdist

            # Create coordinate arrays for distance calculation
            data_coords = np.column_stack([lat_values, lon_values])
            grid_coords = np.column_stack([xi_lat, xi_lon])

            # Calculate distances
            distances = cdist(grid_coords, data_coords)
            min_distances = np.min(distances, axis=1)

            # Create variance based on distance and data density
            max_distance = np.max(min_distances)
            if max_distance > 0:
                normalized_distances = min_distances / max_distance

                # Create more realistic variance based on distance and local data density
                # Areas far from wells should have higher uncertainty
                base_variance = np.var(values) if len(values) > 1 else 1.0

                # Exponential increase in variance with distance
                distance_variance = base_variance * (0.2 + 1.8 * np.exp(normalized_distances * 2.0))

                # Add some local variation based on coordinate position
                lat_variation = np.sin(xi_lat * 100) * 0.1 * base_variance
                lon_variation = np.cos(xi_lon * 100) * 0.1 * base_variance

                kriging_variance = distance_variance + lat_variation + lon_variation
                kriging_variance = np.maximum(kriging_variance, base_variance * 0.1)
            else:
                kriging_variance = np.full(len(xi_lat), np.var(values) * 0.5)

        # Prepare variance data for heat map
        lat_points = (xi_inside[:, 1] / km_per_degree_lat) + center_lat
        lon_points = (xi_inside[:, 0] / km_per_degree_lon) + center_lon

        # Prepare soil polygon geometry for filtering variance display (same as other interpolants)
        merged_soil_geometry = None
        if soil_polygons is not None and len(soil_polygons) > 0:
            try:
                # Create a unified geometry from all soil polygons
                valid_geometries = []
                for idx, row in soil_polygons.iterrows():
                    if row.geometry and row.geometry.is_valid:
                        valid_geometries.append(row.geometry)

                if valid_geometries:
                    # Merge all polygons into a single multipolygon
                    merged_soil_geometry = unary_union(valid_geometries)
                    print(f"Prepared soil drainage geometry for kriging variance filtering")
                else:
                    print("No valid soil polygon geometries found for kriging variance")
            except Exception as e:
                print(f"Error preparing soil polygon geometry for kriging variance: {e}")
                merged_soil_geometry = None

        # Apply stricter soil polygon filtering for variance data
        variance_data = []
        points_inside_count = 0
        points_total_count = len(lat_points)

        for i in range(len(lat_points)):
            # Only add points with meaningful variance values (lower threshold for variance)
            if kriging_variance[i] > 1e-6:
                # Check if point should be included based on soil polygons
                include_point = True
                if merged_soil_geometry is not None:
                    point = Point(lon_points[i], lat_points[i])

                    # Use ONLY strict containment - no intersection tolerance
                    strictly_contained = merged_soil_geometry.contains(point)

                    # For variance visualization, we want very precise clipping
                    # Only allow points that are clearly inside the soil polygons
                    if not strictly_contained:
                        # Check if point is very close to boundary (smaller buffer than before)
                        buffer_distance = 0.00005  # roughly 5 meters - much smaller buffer
                        buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                        strictly_contained = buffered_geometry.contains(point)

                    include_point = strictly_contained
                    if include_point:
                        points_inside_count += 1

                if include_point:
                    # Add data point with latitude, longitude, and variance
                    variance_data.append([
                        float(lat_points[i]),
                        float(lon_points[i]),
                        float(kriging_variance[i])
                    ])
                else:
                    # Add small amount of debugging for clipping
                    if merged_soil_geometry is not None and i % 100 == 0:  # Sample every 100th point for debug
                        point = Point(lon_points[i], lat_points[i])
                        distance_to_boundary = merged_soil_geometry.distance(point)
                        if distance_to_boundary < 0.001:  # If very close to boundary
                            print(f"Debug: Point {i} excluded - distance to boundary: {distance_to_boundary:.6f}")

        if merged_soil_geometry is not None:
            print(f"Variance clipping: {points_inside_count}/{points_total_count} points inside soil polygons ({100*points_inside_count/points_total_count:.1f}%)")

        # Log filtering results
        if merged_soil_geometry is not None:
            print(f"Kriging variance filtered by soil drainage areas: {len(variance_data)} points displayed")

        return variance_data

    except Exception as e:
        print(f"Error calculating kriging variance: {e}")
        return []  # Return empty list in case of error

# The following function is for create_base_map,
# I am adding the code here for generating the map with three-tier color system with red (0-0.5), orange (0.5-0.75), and green (0.75-1.0).

def create_base_map(location, zoom_start=10, tiles='OpenStreetMap'):
    """
    Create a base map centered at the specified location.

    Parameters:
    -----------
    location : tuple
        Tuple containing (latitude, longitude) of the center point.
    zoom_start : int, optional
        Initial zoom level. Defaults to 10.
    tiles : str, optional
        Tile provider for the map. Defaults to 'OpenStreetMap'.

    Returns:
    --------
    folium.Map
        A folium Map object.
    """
    import folium

    m = folium.Map(location=location, zoom_start=zoom_start, tiles=tiles)
    return m

def add_heatmap(m, data, radius=25, blur=15, max_value=1.0):
    """
    Add a heatmap layer to the map.

    Parameters:
    -----------
    m : folium.Map
        The folium Map object to add the heatmap to.
    data : list
        List of [lat, lng, intensity] points for the heatmap.
    radius : int, optional
        Radius of each point. Defaults to 25.
    blur : int, optional
        Amount of blur. Defaults to 15.
    max_value : float, optional
        Maximum intensity value. Defaults to 1.0.
    """
    from folium.plugins import HeatMap

    # Ensure data is not empty
    if not data:
        print("Warning: Heatmap data is empty. No heatmap will be added.")
        return

    # Create HeatMap layer and add it to the map
    hm = HeatMap(
        data,
        radius=radius,
        blur=blur,
        max_val=max_value,
        gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}  # Custom gradient
    )
    hm.add_to(m)

def add_circle_markers(m, wells_df, color='blue', radius=5, popup_text=None):
    """
    Add circle markers to the map for each well.

    Parameters:
    -----------
    m : folium.Map
        The folium Map object to add the markers to.
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude, and optional
        columns for popup text.
    color : str, optional
        Color of the markers. Defaults to 'blue'.
    radius : int, optional
        Radius of the markers. Defaults to 5.
    popup_text : list, optional
        List of column names to include in the popup text. Defaults to None.
    """
    import folium

    # Ensure wells_df is not empty
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        print("Warning: wells_df is empty. No circle markers will be added.")
        return

    # Iterate through each well and add a circle marker
    for idx, row in wells_df.iterrows():
        lat = float(row['latitude'])
        lon = float(row['longitude'])

        # Create popup text if popup_text is specified
        if popup_text:
            popup_content = ""
            for col in popup_text:
                if col in row:
                    popup_content += f"<b>{col}:</b> {row[col]}<br>"
            popup = folium.Popup(popup_content, max_width=300)
        else:
            popup = None

        # Create circle marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            popup=popup
        ).add_to(m)

def create_map_with_interpolated_data(wells_df, center_point, radius_km, resolution=50, interpolation_method='kriging', show_variance=False, variogram_model='spherical', soil_polygons=None):
    """
    Create a Folium map with interpolated groundwater yield data as a heat map.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude, and yield_rate columns.
    center_point : tuple
        Tuple containing (latitude, longitude) of the map's center point.
    radius_km : float
        Radius in kilometers to generate the heat map data for.
    resolution : int
        Number of points to generate in each dimension for interpolation.
    interpolation_method : str, optional
        Interpolation method to use ('kriging', 'idw', 'rf_kriging'). Defaults to 'kriging'.
    show_variance : bool, optional
        Whether to show kriging variance instead of interpolated values.
    variogram_model : str, optional
        Variogram model to use for kriging (e.g., 'linear', 'spherical', 'gaussian').
        Defaults to 'spherical'.
    soil_polygons : GeoDataFrame, optional
        Soil polygon data for clipping the interpolation output.

    Returns:
    --------
    folium.Map
        A Folium Map object with a heat map layer of interpolated yield data.
    """
    import folium
    import streamlit as st

    # Create the base map
    m = create_base_map(center_point, zoom_start=10)

    # Try to get the heat map data
    try:
        if show_variance:
            # Calculate kriging variance for uncertainty visualization
            heat_data = calculate_kriging_variance(wells_df, center_point, radius_km, resolution, variogram_model, soil_polygons=soil_polygons)
            # Set a maximum variance value for the heatmap
            max_value = 0.1  # Adjust this value as needed based on your data
        else:
            # Generate heat map data using the specified interpolation method
            heat_data = generate_heat_map_data(wells_df, center_point, radius_km, resolution, interpolation_method, soil_polygons=soil_polygons)

            # Calculate max_value
            if heat_data:
                max_value = max(point[2] for point in heat_data)
            else:
                max_value = 1.0

        if heat_data:
            if show_variance:
                add_heatmap(m, heat_data, radius=20, blur=15, max_value=max_value)  # Show variance as heatmap
            else:
                add_heatmap(m, heat_data, radius=25, blur=15, max_value=max_value)  # Show yield/depth as heatmap
        else:
            print("No heat data generated")

    except Exception as e:
        print(f"Error generating or adding heatmap: {e}")

    # Add circle markers for the well locations
    add_circle_markers(m, wells_df, color='black', radius=4, popup_text=['name', 'yield_rate', 'depth_to_groundwater', 'depth'])

    # Add a colormap legend to the map
    try:
        if show_variance:
            # Variance colormap legend
            colormap = folium.LinearColormap(
                colors=['blue', 'lime', 'yellow', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Kriging Variance (Uncertainty)'
            )
            m.add_child(colormap)
        elif st.session_state.interpolation_method == 'indicator_kriging':
            # Three-tier indicator kriging legend
            colormap = folium.StepColormap(
                colors=['#FF0000', '#FFA500', '#00FF00'],  # Red, Orange, Green
                vmin=0,
                vmax=1.0,
                index=[0, 0.5, 0.75, 1.0],  # Three-tier thresholds
                caption='Groundwater Yield Probability: Red (Poor 0-0.5), Orange (Moderate 0.5-0.75), Green (Good 0.75-1.0)'
            )
            m.add_child(colormap)
        elif st.session_state.interpolation_method == 'depth_kriging':
            # Depth colormap legend
            colormap = folium.LinearColormap(
                colors=['green', 'yellow', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Depth to Groundwater (m)'
            )
            m.add_child(colormap)
        elif st.session_state.interpolation_method == 'ground_water_level_kriging':
            # Ground water level colormap legend
            colormap = folium.LinearColormap(
                colors=['blue', 'cyan', 'yellow', 'orange', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Ground Water Level'
            )
            m.add_child(colormap)
        else:
            # Yield colormap legend
            colormap = folium.LinearColormap(
                colors=['blue', 'lime', 'yellow', 'red'],
                vmin=0,
                vmax=max_value,
                caption='Groundwater Yield (L/s)'
            )
            m.add_child(colormap)
    except Exception as e:
        print(f"Error adding colormap legend: {e}")

    return m

def generate_smooth_raster_overlay(geojson_data, bounds, raster_size=(512, 512), global_colormap_func=None, opacity=0.7):
    """
    Convert GeoJSON triangular mesh to smooth raster overlay for Windy.com-style visualization
    
    Parameters:
    -----------
    geojson_data : dict
        GeoJSON FeatureCollection containing triangular mesh data
    bounds : dict
        Dictionary with 'north', 'south', 'east', 'west' bounds
    raster_size : tuple
        (width, height) of output raster in pixels
    global_colormap_func : function
        Function to map values to colors consistently across all heatmaps
    opacity : float
        Transparency level (0.0 to 1.0) matching triangle mesh fillOpacity
        
    Returns:
    --------
    dict
        Dictionary containing base64 encoded image and bounds for Folium overlay
    """
    try:
        if not geojson_data or not geojson_data.get('features'):
            return None
            
        # Extract values and coordinates from triangular mesh
        values = []
        coords = []
        
        for feature in geojson_data['features']:
            if feature.get('properties', {}).get('value') is not None:
                value = feature['properties']['value']
                
                # Get triangle centroid
                if feature['geometry']['type'] == 'Polygon':
                    triangle_coords = feature['geometry']['coordinates'][0]
                    if len(triangle_coords) >= 3:
                        centroid_lon = sum(coord[0] for coord in triangle_coords) / len(triangle_coords)
                        centroid_lat = sum(coord[1] for coord in triangle_coords) / len(triangle_coords)
                        
                        values.append(value)
                        coords.append([centroid_lon, centroid_lat])
        
        if not values or not coords:
            return None
            
        values = np.array(values)
        coords = np.array(coords)
        
        # Create high-resolution grid for smooth interpolation
        width, height = raster_size
        west, east = bounds['west'], bounds['east']
        south, north = bounds['south'], bounds['north']
        
        # Create coordinate grids
        x = np.linspace(west, east, width)
        y = np.linspace(south, north, height)
        xi, yi = np.meshgrid(x, y)
        
        # Interpolate values onto high-resolution grid using cubic interpolation
        try:
            # Use cubic interpolation for smoothest results
            zi = griddata(coords, values, (xi, yi), method='cubic', fill_value=np.nan)
            
            # Fill any remaining NaN values with linear interpolation
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_linear = griddata(coords, values, (xi, yi), method='linear', fill_value=np.nan)
                zi[nan_mask] = zi_linear[nan_mask]
                
            # Final pass with nearest neighbor for any remaining NaN
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_nearest = griddata(coords, values, (xi, yi), method='nearest')
                zi[nan_mask] = zi_nearest[nan_mask]
                
        except Exception as e:
            print(f"Cubic interpolation failed, using linear: {e}")
            # Fallback to linear interpolation
            zi = griddata(coords, values, (xi, yi), method='linear', fill_value=np.nan)
            
            # Fill NaN with nearest neighbor
            nan_mask = np.isnan(zi)
            if np.any(nan_mask):
                zi_nearest = griddata(coords, values, (xi, yi), method='nearest')
                zi[nan_mask] = zi_nearest[nan_mask]
        
        # Apply natural boundary clipping - keep the natural interpolation boundaries 
        # No artificial rectangular clipping, let interpolation naturally fade to NaN at edges
        print(f"Smooth raster using natural interpolation boundaries (no artificial clipping)")
        
        # Apply Gaussian smoothing for even smoother appearance
        from scipy.ndimage import gaussian_filter
        zi_smooth = gaussian_filter(zi, sigma=1.0, mode='nearest')
        
        # Convert values to colors using global colormap function
        if global_colormap_func:
            # Create RGBA image
            rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
            
            for i in range(height):
                for j in range(width):
                    if not np.isnan(zi_smooth[i, j]):
                        color_hex = global_colormap_func(zi_smooth[i, j])
                        # Convert hex to RGB
                        color_hex = color_hex.lstrip('#')
                        if len(color_hex) == 6:
                            r = int(color_hex[0:2], 16)
                            g = int(color_hex[2:4], 16)
                            b = int(color_hex[4:6], 16)
                            rgba_image[i, j] = [r, g, b, int(opacity * 255)]  # Match triangle mesh opacity
                        else:
                            rgba_image[i, j] = [0, 0, 0, 0]  # Transparent for invalid colors
                    else:
                        rgba_image[i, j] = [0, 0, 0, 0]  # Transparent for NaN values
        else:
            # Fallback: use matplotlib colormap
            from matplotlib.cm import viridis
            from matplotlib.colors import Normalize
            
            # Normalize values
            valid_mask = ~np.isnan(zi_smooth)
            if np.any(valid_mask):
                vmin, vmax = np.nanmin(zi_smooth), np.nanmax(zi_smooth)
                norm = Normalize(vmin=vmin, vmax=vmax)
                
                # Apply colormap
                colored = viridis(norm(zi_smooth))
                rgba_image = (colored * 255).astype(np.uint8)
                
                # Set transparency for NaN values
                rgba_image[~valid_mask, 3] = 0
                rgba_image[valid_mask, 3] = int(opacity * 255)  # Match triangle mesh opacity
            else:
                rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        
        # Convert to PIL Image and then to base64
        # Flip vertically because raster coordinates are different from image coordinates
        rgba_image_flipped = np.flipud(rgba_image)
        pil_image = Image.fromarray(rgba_image_flipped, 'RGBA')
        
        # Save to bytes buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG', optimize=True)
        img_buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return {
            'image_base64': img_base64,
            'bounds': [[south, west], [north, east]],
            'opacity': opacity
        }
        
    except Exception as e:
        print(f"Error generating smooth raster overlay: {e}")
        return None