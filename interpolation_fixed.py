import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
from scipy.spatial import Voronoi, voronoi_plot_2d
import base64
from io import BytesIO
import json
from utils import get_distance

def generate_contour_geojson(wells_df, center_point, radius_km, min_yield=None, max_yield=None, num_points=80):
    """
    Generate GeoJSON contours for an isopach map showing zones of equal yield values
    
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
    dict
        GeoJSON object containing contour polygon features with yield value properties
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
    
    # For very few wells, can't generate contours properly
    if len(wells_df) < 3:
        return None
    
    center_lat, center_lon = center_point
    
    # Calculate area to cover
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))
    
    # Grid boundaries (square grid that encompasses the circular search area)
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius
    
    # Create high-resolution grid for smooth contours
    grid_lats = np.linspace(min_lat, max_lat, num_points)
    grid_lons = np.linspace(min_lon, max_lon, num_points)
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)  # Note: meshgrid swaps order for matplotlib
    
    # Stack well coordinates for griddata input
    points = np.column_stack((lons, lats))
    
    # Use griddata for interpolation (Natural Neighbor interpolation)
    grid_z = griddata(points, yields, (grid_lon, grid_lat), method='cubic', fill_value=np.nan)
    
    # Apply distance mask to limit to circular area
    # Calculate distance grid from center
    x_grid = (grid_lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))
    y_grid = (grid_lat - center_lat) * 111.0
    dist_grid = np.sqrt(x_grid**2 + y_grid**2)
    grid_mask = dist_grid > radius_km
    grid_z[grid_mask] = np.nan  # Set points outside radius to NaN
    
    # If cubic interpolation fails (needs at least 3 points), try a simpler method
    if np.all(np.isnan(grid_z)):
        grid_z = griddata(points, yields, (grid_lon, grid_lat), method='linear', fill_value=np.nan)
        
    # Fill remaining NaN values with nearest neighbor interpolation
    if np.any(np.isnan(grid_z)):
        grid_z_nearest = griddata(points, yields, (grid_lon, grid_lat), method='nearest', fill_value=min_yield)
        # Replace NaN values with nearest values
        grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
    
    # Make sure all values are within min/max range
    grid_z = np.clip(grid_z, min_yield, max_yield)
    
    try:
        # Generate isopach contours using matplotlib
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create contour levels - more levels for smoother gradients
        level_count = 12
        # Nonlinear levels to emphasize areas of interest
        levels = np.linspace(min_yield, max_yield, level_count)
        
        # Generate filled contours
        contourf = ax.contourf(grid_lon, grid_lat, grid_z, levels=levels, cmap='jet')
        
        # Create contour lines
        contour = ax.contour(grid_lon, grid_lat, grid_z, levels=levels, colors='black', linewidths=0.5)
        
        # Extract contour polygons to GeoJSON format
        contour_json = {"type": "FeatureCollection", "features": []}
        
        # Extract contour coordinates
        for level_idx, collection in enumerate(contourf.collections):
            # Get the yield value for this level
            level_value = float(levels[min(level_idx, len(levels) - 1)])
            
            # Get paths from contourf collection
            for path in collection.get_paths():
                # Skip very small contours
                if len(path.vertices) < 5:  # Need enough points for a meaningful polygon
                    continue
                
                # Extract path coordinates
                polygon_coords = []
                for point in path.vertices:
                    # Store as [lon, lat] for GeoJSON (not [lat, lon])
                    polygon_coords.append([float(point[0]), float(point[1])])
                
                # Ensure polygon is closed
                if len(polygon_coords) > 0 and polygon_coords[0] != polygon_coords[-1]:
                    polygon_coords.append(polygon_coords[0])
                
                # Create GeoJSON feature if we have enough points
                if len(polygon_coords) >= 4:  # Minimum for a valid polygon (3 points + closing point)
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "yield_value": level_value
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [polygon_coords]
                        }
                    }
                    contour_json["features"].append(feature)
        
        # Clean up
        plt.close(fig)
        
        return contour_json
    
    except Exception as e:
        print(f"Error generating contours: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_prediction_at_point(wells_df, point_lat, point_lon):
    """
    Get a predicted yield at a specific point based on nearby wells using IDW
    
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