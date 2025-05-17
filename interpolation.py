import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50):
    """
    Generate interpolated heat map data based on well yield rates
    
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
    
    if len(wells_df) < 3:
        # For very few wells, just show the exact well locations with their yields
        max_yield = np.max(yields) if yields.size > 0 and np.max(yields) > 0 else 1
        normalized_yields = yields / max_yield
        simple_heat_data = [[float(lat), float(lon), float(y)] for lat, lon, y in zip(lats, lons, normalized_yields)]
        return simple_heat_data
    
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
    adjusted_resolution = min(60, max(30, resolution))  # Balancing detail vs performance
    grid_lats = np.linspace(min_lat, max_lat, adjusted_resolution)
    grid_lons = np.linspace(min_lon, max_lon, adjusted_resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Flatten grid for calculations
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()
    
    # Convert to kilometer distances for better accuracy
    # First, convert to flat projection (rough approximation using kilometers)
    origin_lat, origin_lon = center_lat, center_lon
    x_coords = (lons - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    y_coords = (lats - origin_lat) * 111.0
    
    # Convert grid points to same projection
    grid_x = (grid_lon_flat - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    grid_y = (grid_lat_flat - origin_lat) * 111.0
    
    try:
        # Use a modified Inverse Distance Weighting with distance-based falloff
        # This gives much sharper contrast between high and low yield areas
        
        # Parameters for the modified IDW
        power = 2.5  # Higher power = sharper transitions (2 is standard IDW)
        smoothing = 0.1  # Small value to prevent division by zero
        
        # Maximum distance influence (in km) - points beyond this have minimal impact
        # This is key to preventing high interpolation between distant points
        max_influence_distance = min(radius_km / 3, 5.0)  # Smaller of 1/3 radius or 5km
        
        # Create array to store interpolated values
        grid_values = np.zeros(len(grid_x))
        
        # For each point in the grid, calculate weighted yield
        for i in range(len(grid_x)):
            # Calculate distance to each well (in kilometers)
            distances = np.sqrt((grid_x[i] - x_coords)**2 + (grid_y[i] - y_coords)**2)
            
            # Apply inverse distance weighting with sharp distance cutoff
            # Use exponential decay for smoother transition
            weights = np.exp(-distances / (max_influence_distance / 3))
            
            # Apply additional power-law decay (traditional IDW)
            weights = weights / (distances + smoothing)**power
            
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                
                # Calculate weighted average
                grid_values[i] = np.sum(weights * yields)
            else:
                # If no wells have influence, set to very low value
                grid_values[i] = 0.0001
        
        # Normalize values for heat map (0-1 scale)
        min_val = np.min(grid_values)
        max_val = np.max(grid_values)
        
        if max_val > min_val:
            # Scale values, keeping a true zero and maintaining proportions
            # Generate a distinct range from 0.05-1.0 for better visualization
            if min_val > 0:
                # If all values are positive, maintain proportions but start from a small value
                normalized_values = 0.05 + 0.95 * (grid_values - min_val) / (max_val - min_val)
            else:
                # If we have zeroes, keep them close to zero
                normalized_values = np.zeros_like(grid_values)
                positive_mask = grid_values > 0
                if np.any(positive_mask):
                    normalized_values[positive_mask] = 0.05 + 0.95 * (grid_values[positive_mask] - min_val) / (max_val - min_val)
        else:
            # If all values are the same
            normalized_values = np.ones_like(grid_values) * 0.5
        
        # Filter points to create a cleaner display
        heat_data = []
        
        # Extra filtering to create more distinct high/low areas
        for i in range(len(grid_x)):
            # Skip very low values completely
            if normalized_values[i] < 0.05:
                continue
                
            # Calculate distance from center of search area (in km)
            dist_from_center = np.sqrt((grid_x[i])**2 + (grid_y[i])**2)
            
            # Skip points outside search radius
            if dist_from_center > radius_km:
                continue
            
            # Apply distance-based attenuation from center
            dist_factor = max(0.1, 1 - (dist_from_center / radius_km)**2)
            
            # Find distance to nearest actual well
            nearest_well_dist = np.min(np.sqrt((grid_x[i] - x_coords)**2 + (grid_y[i] - y_coords)**2))
            
            # Adjust intensity based on distance to nearest well
            # Very important: Sharply reduce intensity for points far from any well
            proximity_factor = np.exp(-nearest_well_dist / (max_influence_distance / 3))
            
            # Combine all factors
            intensity = normalized_values[i] * dist_factor * proximity_factor
            
            # Skip very low intensity points
            if intensity < 0.05:
                continue
                
            # Add to heat map
            heat_data.append([
                float(grid_lat_flat[i]), 
                float(grid_lon_flat[i]), 
                float(intensity)
            ])
        
        # Ensure actual well locations have correct intensity in the heatmap
        # This is important to show exact well yields at their locations
        for lat, lon, yield_val in zip(lats, lons, yields):
            # Convert yield to normalized value
            intensity = yield_val / max_val if max_val > 0 else 0.5
            
            # Check if within radius
            dist_km = np.sqrt(
                ((lat - center_lat) * 111.0)**2 + 
                ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
            )
            
            if dist_km <= radius_km:
                # Add well point to heat map with adjusted intensity
                heat_data.append([float(lat), float(lon), float(intensity)])
        
        return heat_data
        
    except Exception as e:
        print(f"Interpolation error: {e}")
        
        # Fall back to simple IDW interpolation
        return fallback_interpolation(wells_df, center_point, radius_km, resolution)

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """
    Simple fallback interpolation method using inverse distance weighting
    with sharper distance decay
    """
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    center_lat, center_lon = center_point
    
    # Calculate bounding box
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))
    
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius
    
    # Create grid for interpolation
    grid_lats = np.linspace(min_lat, max_lat, resolution)
    grid_lons = np.linspace(min_lon, max_lon, resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Flatten grid for calculations
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()
    
    # Calculate a simple IDW interpolation with very sharp dropoff
    heat_data = []
    
    # Parameters for a very sharp dropoff
    power = 3.0  # Higher than standard IDW (2.0)
    max_influence_km = 3.0  # Very limited influence radius
    
    for i in range(len(grid_lat_flat)):
        grid_point_lat = grid_lat_flat[i]
        grid_point_lon = grid_lon_flat[i]
        
        # Skip points outside search radius
        dist_from_center = np.sqrt(
            ((grid_point_lat - center_lat) * 111.0)**2 + 
            ((grid_point_lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )
        
        if dist_from_center > radius_km:
            continue
        
        # Calculate distances to all wells (in km)
        weights = []
        weight_sum = 0
        value_sum = 0
        
        for j in range(len(lats)):
            # Calculate distance in km
            dist = np.sqrt(
                ((grid_point_lat - lats[j]) * 111.0)**2 + 
                ((grid_point_lon - lons[j]) * 111.0 * np.cos(np.radians(grid_point_lat)))**2
            )
            
            # Apply very sharp distance cutoff
            if dist > max_influence_km:
                weight = 0
            else:
                # For points within influence, use inverse power law with exponential decay
                weight = np.exp(-dist / (max_influence_km / 3)) / (dist + 0.1)**power
            
            weights.append(weight)
            weight_sum += weight
            value_sum += weight * yields[j]
        
        # Calculate interpolated value
        if weight_sum > 0:
            interpolated_value = value_sum / weight_sum
            
            # Normalize to a 0-1 range using the max yield in the dataset
            max_yield = np.max(yields) if np.max(yields) > 0 else 1
            normalized_value = interpolated_value / max_yield
            
            # Apply distance-based attenuation from center
            dist_factor = max(0.1, 1 - (dist_from_center / radius_km)**2)
            
            # Add point to heatmap if it has any significant value
            intensity = normalized_value * dist_factor
            if intensity > 0.05:
                heat_data.append([float(grid_point_lat), float(grid_point_lon), float(intensity)])
    
    # If we couldn't generate any heat map points, just use the actual well points
    if not heat_data:
        max_yield = np.max(yields) if np.max(yields) > 0 else 1
        for i in range(len(lats)):
            # Check if within search radius
            dist_from_center = np.sqrt(
                ((lats[i] - center_lat) * 111.0)**2 + 
                ((lons[i] - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
            )
            
            if dist_from_center <= radius_km:
                normalized_yield = yields[i] / max_yield
                heat_data.append([float(lats[i]), float(lons[i]), float(normalized_yield)])
    
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
