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

def create_contour_json(wells_df, center_point=None, radius_km=None, min_yield=None, max_yield=None, num_points=100):
    """
    Generate contour data for an isopach map showing zones of equal yield values
    using scipy's griddata for interpolation.
    
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
        GeoJSON object containing contour polygon features
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
    
    # If center point not provided, use center of all well points
    if center_point is None:
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
    else:
        center_lat, center_lon = center_point
    
    # If radius not provided, calculate appropriate radius to cover all or most wells
    if radius_km is None:
        # Calculate distances from center to all points
        distances = []
        for lat, lon in zip(lats, lons):
            distance = np.sqrt(((lat - center_lat) * 111.0)**2 + 
                             ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2)
            distances.append(distance)
        
        # Use a radius that includes most wells (95th percentile)
        if distances:
            radius_km = np.percentile(distances, 95)
            # Make sure radius is reasonable (between 5km and 50km)
            radius_km = max(5.0, min(50.0, radius_km))
        else:
            radius_km = 10.0  # Default if no wells
    
    # Calculate area to cover
    lat_radius = radius_km / 111.0
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat))))
    
    # Grid boundaries (square grid that encompasses the region)
    min_lat = center_lat - lat_radius
    max_lat = center_lat + lat_radius
    min_lon = center_lon - lon_radius
    max_lon = center_lon + lon_radius
    
    # Create high-resolution grid for smooth contours
    grid_lons = np.linspace(min_lon, max_lon, num_points)
    grid_lats = np.linspace(min_lat, max_lat, num_points)
    
    # Create meshgrid for interpolation
    grid_lon, grid_lat = np.meshgrid(grid_lons, grid_lats)
    
    # Stack coordinates for scipy's griddata
    points = np.column_stack((lons, lats))
    
    # Try cubic interpolation first (for smooth contours)
    try:
        # Interpolate using cubic method
        grid_z = griddata(points, yields, (grid_lon, grid_lat), method='cubic')
        
        # Check if interpolation worked - if there are too many NaN values, use linear instead
        if np.isnan(grid_z).sum() > 0.5 * grid_z.size:
            grid_z = griddata(points, yields, (grid_lon, grid_lat), method='linear')
    except:
        # Fall back to linear interpolation if cubic fails
        grid_z = griddata(points, yields, (grid_lon, grid_lat), method='linear')
    
    # As a final fallback, use nearest neighbor to fill remaining NaN values
    if np.isnan(grid_z).any():
        grid_z_nearest = griddata(points, yields, (grid_lon, grid_lat), method='nearest')
        grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
    
    # Create circular mask for the grid (only show within radius)
    y_dist = grid_lat - center_lat
    x_dist = grid_lon - center_lon
    dist = np.sqrt((y_dist * 111.0)**2 + (x_dist * 111.0 * np.cos(np.radians(center_lat)))**2)
    mask = dist > radius_km
    
    # Apply mask - set values outside radius to NaN
    masked_grid_z = grid_z.copy()
    masked_grid_z[mask] = np.nan
    
    # Create contour levels
    level_count = 10  # Number of contour levels
    levels = np.linspace(min_yield, max_yield, level_count)
    
    # Manual construction of GeoJSON for contours
    contour_geojson = {
        "type": "FeatureCollection",
        "features": []
    }
    
    # For each level, create a polygon feature in the GeoJSON
    # We'll manually trace boundaries between adjacent levels
    for i in range(level_count - 1):
        lower_level = levels[i]
        upper_level = levels[i+1]
        level_mask = (masked_grid_z >= lower_level) & (masked_grid_z < upper_level)
        
        # If no cells match this level, skip
        if not np.any(level_mask):
            continue
        
        # Extract contiguous regions for this level
        from skimage import measure
        
        # Use scikit-image to find contours (polygon boundaries)
        try:
            # Find contours at the level
            contours = measure.find_contours(level_mask.astype(float), 0.5)
            
            # Process each contour
            for contour in contours:
                # Skip contours that are too small
                if len(contour) < 4:
                    continue
                
                # Convert from grid indices to lat/lon coordinates
                contour_lon = grid_lons[np.minimum(np.maximum(0, contour[:, 1].astype(int)), len(grid_lons)-1)]
                contour_lat = grid_lats[np.minimum(np.maximum(0, contour[:, 0].astype(int)), len(grid_lats)-1)]
                
                # Create coordinates for GeoJSON (lon, lat order)
                coords = [[float(lon), float(lat)] for lon, lat in zip(contour_lon, contour_lat)]
                
                # Ensure polygon is closed
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                
                # Create feature
                feature = {
                    "type": "Feature",
                    "properties": {
                        "yield_value": float(lower_level)
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords]
                    }
                }
                
                contour_geojson["features"].append(feature)
        except Exception as e:
            print(f"Error processing contour level {lower_level}-{upper_level}: {e}")
            continue
    
    # If no features were created, attempt a different approach
    if len(contour_geojson["features"]) == 0:
        # Create simplified representation with circles
        for i in range(level_count - 1):
            lower_level = levels[i]
            upper_level = levels[i+1]
            
            # Find center of this yield level
            mid_yield = (lower_level + upper_level) / 2
            points_in_range = []
            
            for lat, lon, yield_val in zip(lats, lons, yields):
                if lower_level <= yield_val < upper_level:
                    points_in_range.append((lat, lon))
            
            # If no points in this range, skip
            if not points_in_range:
                continue
                
            # Create a representative point for this yield level
            avg_lat = sum(p[0] for p in points_in_range) / len(points_in_range)
            avg_lon = sum(p[1] for p in points_in_range) / len(points_in_range)
            
            # Add a circle feature
            feature = {
                "type": "Feature",
                "properties": {
                    "yield_value": float(mid_yield)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(avg_lon), float(avg_lat)]
                }
            }
            contour_geojson["features"].append(feature)
    
    return contour_geojson

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