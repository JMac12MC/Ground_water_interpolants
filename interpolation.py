import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import base64
from io import BytesIO
import geojson
from utils import get_distance

def calculate_interpolated_grid(wells_df, center_point, radius_km, min_yield=None, max_yield=None, num_points=60):
    """
    Create an interpolated grid of yield values using IDW (Inverse Distance Weighting)
    for visualizing as an isopach map.
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude and yield_rate columns
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point
    radius_km : float
        Radius in kilometers to generate isopach map for
    min_yield : float, optional
        Minimum yield value for normalization
    max_yield : float, optional
        Maximum yield value for normalization
    num_points : int
        Grid resolution (higher = more detailed but slower)
        
    Returns:
    --------
    DataFrame
        DataFrame with columns: latitude, longitude, yield_value, yield_class
        Where yield_class is an integer 0-4 representing which of 5 categories
        the yield value falls into.
    """
    # Handle empty dataframe
    if wells_df is None or wells_df.empty:
        return None
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    # Set default min/max yield if not provided
    if min_yield is None:
        min_yield = np.min(yields) if len(yields) > 0 else 0
    if max_yield is None:
        max_yield = np.max(yields) if len(yields) > 0 else 1
    
    # For very few wells, just return a simple grid around them
    if len(wells_df) < 3:
        grid_data = []
        for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
            # Normalize yield for consistent coloring
            norm_yield = (yield_val - min_yield) / (max_yield - min_yield) if max_yield > min_yield else 0.5
            # Determine yield class (0-4)
            yield_class = min(4, max(0, int(norm_yield * 5)))
            grid_data.append({
                'latitude': float(lat), 
                'longitude': float(lon),
                'yield_value': float(yield_val),
                'yield_class': yield_class
            })
        return pd.DataFrame(grid_data)
    
    center_lat, center_lon = center_point
    
    # Calculate area to cover
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))
    
    # Grid boundaries
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius
    
    # Create grid
    grid_lats = np.linspace(min_lat, max_lat, num_points)
    grid_lons = np.linspace(min_lon, max_lon, num_points)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Flatten grid for calculations
    grid_lat_flat = grid_lat.flatten()
    grid_lon_flat = grid_lon.flatten()
    
    # Data collection for the interpolated grid
    grid_data = []
    
    # Convert to a simple projected space (km from center) for distance calculations
    x_points = (lons - center_lon) * 111.0 * np.cos(np.radians(center_lat))
    y_points = (lats - center_lat) * 111.0
    
    # Calculate yield range
    yield_range = max_yield - min_yield
    
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
        
        # Calculate interpolated value
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
                interpolated_value = 0.0
        
        # Determine yield class (0-4)
        norm_yield = (interpolated_value - min_yield) / yield_range if yield_range > 0 else 0.5
        norm_yield = min(1.0, max(0.0, norm_yield))  # Clamp to 0-1 range
        yield_class = min(4, max(0, int(norm_yield * 5)))
        
        # Add to grid data
        grid_data.append({
            'latitude': float(grid_point_lat), 
            'longitude': float(grid_point_lon),
            'yield_value': float(interpolated_value),
            'yield_class': yield_class
        })
    
    # Also add the actual well points with their yield_class
    for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
        # Check if within search radius
        dist_from_center_km = np.sqrt(
            ((lat - center_lat) * 111.0)**2 + 
            ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )
        
        if dist_from_center_km <= radius_km:
            # Determine yield class
            norm_yield = (yield_val - min_yield) / yield_range if yield_range > 0 else 0.5
            norm_yield = min(1.0, max(0.0, norm_yield))  # Clamp to 0-1 range
            yield_class = min(4, max(0, int(norm_yield * 5)))
            
            grid_data.append({
                'latitude': float(lat), 
                'longitude': float(lon),
                'yield_value': float(yield_val),
                'yield_class': yield_class
            })
    
    return pd.DataFrame(grid_data)

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """
    Simplified IDW (Inverse Distance Weighting) interpolation as fallback method
    Creates a continuous interpolated surface based on actual well yield values
    """
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    # Handle empty dataset
    if len(yields) == 0:
        return []
    
    center_lat, center_lon = center_point
    
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
    
    # Find global min/max yield across all wells for consistent color mapping
    min_yield = np.min(yields) if len(yields) > 0 else 0
    max_yield = np.max(yields) if len(yields) > 0 else 1
    yield_range = max(0.1, max_yield - min_yield)
    
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
        
        # Normalize using the SAME GLOBAL scale for consistent colors
        normalized_value = (interpolated_value - min_yield) / yield_range
        normalized_value = max(0.0, min(1.0, normalized_value))  # Clamp to 0-1 range
        
        # Add to heat map with normalized value for consistent colors
        if normalized_value > 0.01:  # Only add significant points
            heat_data.append([
                float(grid_point_lat), 
                float(grid_point_lon),
                float(normalized_value)  # Use normalized value for consistent colors
            ])
    
    # Always add the actual well points to ensure they're visible
    for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
        # Check if within search radius
        dist_from_center_km = np.sqrt(
            ((lat - center_lat) * 111.0)**2 + 
            ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
        )
        
        if dist_from_center_km <= radius_km:
            # Normalize the well's yield
            normalized_yield = (yield_val - min_yield) / yield_range
            normalized_yield = max(0.0, min(1.0, normalized_yield))  # Clamp to 0-1
            
            # Add the exact well with its normalized yield value
            heat_data.append([float(lat), float(lon), float(normalized_yield)])
    
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
    if wells_df.empty:
        return 0.0
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    # Calculate distances (in km) from point to each well
    distances = np.array([
        get_distance(point_lat, point_lon, lat, lon) 
        for lat, lon in zip(lats, lons)
    ])
    
    # If we have a very close well, just use its value
    if np.min(distances) < 0.1:  # Within 100m
        return yields[np.argmin(distances)]
    
    # Use IDW with a power parameter of 2
    power = 2.0
    weights = 1.0 / (distances**power)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate the weighted average
    predicted_yield = np.sum(weights * yields)
    
    return predicted_yield