import numpy as np
import pandas as pd
from scipy.interpolate import griddata

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
    if wells_df.empty:
        return []
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values
    lons = wells_df['longitude'].values
    yields = wells_df['yield_rate'].values
    
    # Normalize yields for better visualization (0 to 1 scale)
    max_yield = np.max(yields) if np.max(yields) > 0 else 1
    normalized_yields = yields / max_yield
    
    # Calculate bounding box based on center and radius
    center_lat, center_lon = center_point
    
    # Approximate conversion from km to degrees (varies by latitude)
    # At the equator, 1 degree is approximately 111 km
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius
    
    # Create a grid of points
    grid_lats = np.linspace(min_lat, max_lat, resolution)
    grid_lons = np.linspace(min_lon, max_lon, resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Reshape for interpolation
    points = np.column_stack((lats, lons))
    grid_points = np.column_stack((grid_lat.flatten(), grid_lon.flatten()))
    
    # Perform interpolation
    try:
        # Use 'cubic' for smoother results, 'linear' for speed
        grid_yields = griddata(points, normalized_yields, grid_points, method='linear', fill_value=0)
        
        # Filter out points outside the radius
        heat_data = []
        for i, (lat, lon) in enumerate(grid_points):
            # Only include points with some yield value
            if grid_yields[i] > 0.01:  # Filter very low values for cleaner visualization
                heat_data.append([lat, lon, float(grid_yields[i])])
        
        return heat_data
    except Exception as e:
        # If interpolation fails (e.g., not enough points), return simple heat map
        print(f"Interpolation error: {e}")
        simple_heat_data = [[float(lat), float(lon), float(y)] for lat, lon, y in zip(lats, lons, normalized_yields)]
        return simple_heat_data

def get_prediction_at_point(wells_df, point_lat, point_lon):
    """
    Get a predicted yield at a specific point based on nearby wells
    
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
        return 0
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values
    lons = wells_df['longitude'].values
    yields = wells_df['yield_rate'].values
    
    # Create the point
    points = np.column_stack((lats, lons))
    point = np.array([point_lat, point_lon])
    
    # Perform interpolation at the single point
    try:
        predicted_yield = griddata(points, yields, point, method='linear')
        
        # If the point is outside the convex hull of the data points, use nearest neighbor
        if np.isnan(predicted_yield):
            predicted_yield = griddata(points, yields, point, method='nearest')
            
        return float(predicted_yield)
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0
