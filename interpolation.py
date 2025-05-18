import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50):
    """
    Generate heat map data based on well yield rates
    
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
    
    if len(wells_df) < 2:
        # For very few wells, just show the exact well locations with their colors
        max_yield = np.max(yields) if yields.size > 0 and np.max(yields) > 0 else 1
        normalized_yields = yields / max_yield
        simple_heat_data = [[float(lat), float(lon), float(y)] for lat, lon, y in zip(lats, lons, normalized_yields)]
        return simple_heat_data
    
    # Important: Normalize yields for color intensity (not distance)
    max_yield = np.max(yields) if yields.size > 0 and np.max(yields) > 0 else 1
    min_yield = np.min(yields)
    
    # Create normalized yield values (for colors only)
    if max_yield > min_yield:
        # Scale from 0.2 to 1.0 to keep colors visible but maintain differences
        normalized_yields = 0.2 + 0.8 * (yields - min_yield) / (max_yield - min_yield)
    else:
        # If all yields are the same
        normalized_yields = np.ones_like(yields) * 0.5
    
    # Calculate bounding box based on center and radius
    center_lat, center_lon = center_point
    
    # Approximate conversion from km to degrees (varies by latitude)
    # At the equator, 1 degree is approximately 111 km
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))
    
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius
    
    # Create a grid of points for the heat map
    adjusted_resolution = min(70, max(40, resolution))  # Higher resolution for better detail
    grid_lats = np.linspace(min_lat, max_lat, adjusted_resolution)
    grid_lons = np.linspace(min_lon, max_lon, adjusted_resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Flatten grid for calculations
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()
    
    try:
        # FIXED APPROACH: Use fixed radius circles around each well
        # with color intensity based on yield but same distance coverage
        
        # Parameters for heat map construction
        fixed_influence_radius_km = 2.0  # Fixed 2km radius of influence around each well
        
        # Heat map data points collection
        heat_data = []
        
        # For each grid point, calculate its color based on nearby wells
        for i in range(len(grid_lat_flat)):
            grid_lat_point = grid_lat_flat[i]
            grid_lon_point = grid_lon_flat[i]
            
            # Calculate distance from center of search radius
            # Skip points outside the search radius
            dist_from_center_km = np.sqrt(
                ((grid_lat_point - center_lat) * 111.0)**2 + 
                ((grid_lon_point - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
            )
            
            if dist_from_center_km > radius_km:
                continue
                
            # Calculate distances to all wells (in km)
            well_distances = []
            for j in range(len(lats)):
                dist_km = np.sqrt(
                    ((grid_lat_point - lats[j]) * 111.0)**2 + 
                    ((grid_lon_point - lons[j]) * 111.0 * np.cos(np.radians(grid_lat_point)))**2
                )
                well_distances.append(dist_km)
            
            # Find the nearest well
            if not well_distances:
                continue
                
            min_dist = min(well_distances)
            nearest_well_idx = well_distances.index(min_dist)
            
            # Only include points within the fixed radius of any well
            if min_dist > fixed_influence_radius_km:
                continue
            
            # Calculate the distance factor (0-1) based on distance from well
            # Use an exponential decay for smoother effect
            # This makes intensity drop off sharply as you move away from a well
            # but the distance is FIXED regardless of yield
            distance_factor = np.exp(-3.0 * min_dist / fixed_influence_radius_km)
            
            # Important: Final intensity combines:
            # 1. Color from the well's yield (normalized_yields)
            # 2. Intensity dropoff from distance (completely separate from yield)
            color_intensity = normalized_yields[nearest_well_idx]
            
            # Don't fade out color entirely, maintain at least 30% of the original color
            final_intensity = color_intensity * (0.3 + 0.7 * distance_factor)
            
            # Skip very low intensity points
            if final_intensity < 0.05:
                continue
                
            # Add to heat map
            heat_data.append([
                float(grid_lat_point),
                float(grid_lon_point),
                float(final_intensity)
            ])
        
        # Always add points directly at well locations to ensure they show correctly
        for j in range(len(lats)):
            lat = lats[j]
            lon = lons[j]
            
            # Check if in search radius
            dist_from_center_km = np.sqrt(
                ((lat - center_lat) * 111.0)**2 + 
                ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
            )
            
            if dist_from_center_km <= radius_km:
                heat_data.append([
                    float(lat),
                    float(lon),
                    float(normalized_yields[j])  # Full intensity at exact well location
                ])
        
        return heat_data
        
    except Exception as e:
        print(f"Interpolation error: {e}")
        
        # Fall back to simple visualization method
        return fallback_interpolation(wells_df, center_point, radius_km)

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """
    Simple fallback interpolation method that uses fixed radius around wells
    with color based on yield
    """
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    center_lat, center_lon = center_point
    
    # Normalize yields for color only (not distance)
    max_yield = np.max(yields) if yields.size > 0 and np.max(yields) > 0 else 1
    min_yield = np.min(yields)
    
    # Scale colors from 0.2 to 1.0 for better visibility
    if max_yield > min_yield:
        normalized_yields = 0.2 + 0.8 * (yields - min_yield) / (max_yield - min_yield)
    else:
        normalized_yields = np.ones_like(yields) * 0.5
    
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
        
        # Color is from well yield
        color_intensity = normalized_yields[closest_well_idx]
        
        # Distance factor (0=far, 1=at well location)
        distance_factor = max(0, 1.0 - (closest_dist / fixed_radius_km))
        
        # Final intensity fades with distance but retains color from yield
        final_intensity = color_intensity * distance_factor
        
        # Skip very low values
        if final_intensity < 0.05:
            continue
            
        # Add to heat map
        heat_data.append([
            float(grid_point_lat),
            float(grid_point_lon),
            float(final_intensity)
        ])
    
    # Always add the actual well points
    for j in range(len(lats)):
        # Check if in search radius
        dist_from_center_km = np.sqrt(
            ((lats[j] - center_lat) * 111.0)**2 + 
            ((lons[j] - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )
        
        if dist_from_center_km <= radius_km:
            # Well points should have full color intensity
            heat_data.append([
                float(lats[j]),
                float(lons[j]),
                float(normalized_yields[j])
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
