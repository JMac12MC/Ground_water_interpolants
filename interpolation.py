import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50):
    """
    Generate heat map data based on well yield rates using fixed radius approach
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude and yield_rate columns
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point
    radius_km : float
        Radius in kilometers to generate heat map data for
    resolution : int
        Number of points to generate in each dimension (higher = more detailed but slower)
        
    Returns:
    --------
    list
        List of [lat, lng, intensity] points for the heat map
    """
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return []
    
    # Extract coordinates and yields for wells within search radius
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    # Find global min and max yields for normalization reference
    min_yield = np.min(yields) if len(yields) > 0 else 0
    max_yield = np.max(yields) if len(yields) > 0 else 1
    yield_range = max(0.1, max_yield - min_yield)  # Prevent division by zero
    
    # Normalize yields to 0-1 range for the heat map
    # Important: This normalization is used ONLY for the heat map relative intensity
    normalized_yields = (yields - min_yield) / yield_range
    
    # Handle single or few wells case
    if len(wells_df) < 2:
        heat_data = []
        for i, (lat, lon, yield_val, norm_yield) in enumerate(zip(lats, lons, yields, normalized_yields)):
            # Use normalized value for heat map intensity (0-1 range)
            heat_data.append([float(lat), float(lon), float(norm_yield)])
        return heat_data
    
    # Create bounding box based on center and radius
    center_lat, center_lon = center_point
    
    # Calculate degree radii
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))
    
    # Set boundaries for grid
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius
    
    # Create grid points
    adjusted_resolution = min(80, max(40, resolution))  # Higher resolution for better detail
    grid_lats = np.linspace(min_lat, max_lat, adjusted_resolution)
    grid_lons = np.linspace(min_lon, max_lon, adjusted_resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Flatten grid
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()
    
    try:
        # Create empty heat map data collection
        heat_data = []
        
        # FIXED RADIUS APPROACH:
        # Each well has a consistent 2km influence radius
        # Color represents the well's yield (normalized to 0-1 range)
        FIXED_RADIUS_KM = 2.0  # Exact 2km radius around each well
        
        # First pass - add the exact well locations with normalized yield values
        for j in range(len(lats)):
            lat, lon = lats[j], lons[j]
            norm_yield = normalized_yields[j]  # Normalized yield value (0-1)
            
            # Skip wells outside search radius
            dist_from_center_km = np.sqrt(
                ((lat - center_lat) * 111.0)**2 + 
                ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
            )
            
            if dist_from_center_km <= radius_km:
                # Add well point with normalized yield value for intensity
                heat_data.append([
                    float(lat),
                    float(lon),
                    float(norm_yield)  # Normalized yield (0-1 range)
                ])
        
        # Second pass - add grid points around each well
        for i in range(len(grid_lat_flat)):
            grid_point_lat = grid_lat_flat[i]
            grid_point_lon = grid_lon_flat[i]
            
            # Skip points outside search radius
            dist_from_center_km = np.sqrt(
                ((grid_point_lat - center_lat) * 111.0)**2 + 
                ((grid_point_lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
            )
            
            if dist_from_center_km > radius_km:
                continue
            
            # Find closest well and its distance
            closest_dist = float('inf')
            closest_well_idx = None
            
            for j in range(len(lats)):
                dist_km = np.sqrt(
                    ((grid_point_lat - lats[j]) * 111.0)**2 + 
                    ((grid_point_lon - lons[j]) * 111.0 * np.cos(np.radians(grid_point_lat)))**2
                )
                
                if dist_km < closest_dist:
                    closest_dist = dist_km
                    closest_well_idx = j
            
            # Skip points not within influence radius of any well
            if closest_well_idx is None or closest_dist > FIXED_RADIUS_KM:
                continue
                
            # Skip points too close to an actual well point (to avoid double counting)
            if closest_dist < 0.05:  # Within 50m of a well
                continue
            
            # Get the normalized yield value of the closest well (0-1 scale)
            well_norm_yield = normalized_yields[closest_well_idx]
            
            # Linear falloff with distance (1.0 at well, 0.0 at edge of radius)
            distance_factor = max(0.0, 1.0 - closest_dist / FIXED_RADIUS_KM)
            
            # Adjust intensity based on distance, but keep the well's yield color
            point_intensity = well_norm_yield * distance_factor
            
            # Only add significant points to reduce clutter
            if point_intensity > 0.01:
                heat_data.append([
                    float(grid_point_lat),
                    float(grid_point_lon),
                    float(point_intensity)
                ])
        
        return heat_data
        
    except Exception as e:
        print(f"Interpolation error: {e}")
        
        # Fall back to simple visualization method
        return fallback_interpolation(wells_df, center_point, radius_km)

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """
    Simple fallback interpolation method that uses fixed radius around wells
    with color directly based on normalized yield value
    """
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    # Normalize yields to 0-1 range for consistent colors
    min_yield = np.min(yields) if len(yields) > 0 else 0
    max_yield = np.max(yields) if len(yields) > 0 else 1
    yield_range = max(0.1, max_yield - min_yield)
    normalized_yields = (yields - min_yield) / yield_range
    
    center_lat, center_lon = center_point
    
    # Heat map data collection
    heat_data = []
    
    # Fixed parameters for all wells
    fixed_radius_km = 2.0  # Each well has same 2km radius of influence
    
    # Create a simpler grid
    grid_resolution = min(40, resolution)
    
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
    
    # Flatten grid
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()
    
    # First add all well points with normalized yield values
    for j in range(len(lats)):
        lat = lats[j]
        lon = lons[j]
        norm_yield = normalized_yields[j]
        
        # Check if within search radius
        dist_from_center_km = np.sqrt(
            ((lat - center_lat) * 111.0)**2 + 
            ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )
        
        if dist_from_center_km <= radius_km:
            # Add exact well location with normalized yield value (0-1 scale)
            heat_data.append([
                float(lat),
                float(lon),
                float(norm_yield)
            ])
    
    # For each grid point
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
        
        # Find closest well
        closest_well_idx = None
        closest_dist = float('inf')
        
        for j in range(len(lats)):
            dist_km = np.sqrt(
                ((grid_point_lat - lats[j]) * 111.0)**2 + 
                ((grid_point_lon - lons[j]) * 111.0 * np.cos(np.radians(grid_point_lat)))**2
            )
            
            if dist_km < closest_dist:
                closest_dist = dist_km
                closest_well_idx = j
        
        # Skip if no wells in range or outside fixed radius
        if closest_well_idx is None or closest_dist > fixed_radius_km:
            continue
            
        # Skip points very close to a well (already represented)
        if closest_dist < 0.05:  # 50 meters
            continue
        
        # Use normalized yield value (0-1 scale) for color
        norm_yield_value = normalized_yields[closest_well_idx]
        
        # Distance factor (1.0 at well, 0.0 at edge of radius)
        distance_factor = max(0.0, 1.0 - (closest_dist / fixed_radius_km))
        
        # Final intensity combines normalized yield with distance falloff
        final_intensity = norm_yield_value * distance_factor
            
        # Add to heat map if significant
        if final_intensity > 0.01:
            heat_data.append([
                float(grid_point_lat),
                float(grid_point_lon),
                float(final_intensity)
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
    Used as a fallback method
    """
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        return 0
        
    # Calculate distance from each well to the point (in km)
    distances = []
    for idx, row in wells_df.iterrows():
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
    weighted_yield = sum(w * float(row['yield_rate']) for w, (idx, row) in zip(weights, wells_df.iterrows())) / total_weight
    return float(max(0, weighted_yield))  # Ensure non-negative yield
