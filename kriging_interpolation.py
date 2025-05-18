import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Rbf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import json
from utils import get_distance

def create_kriging_contours(wells_df, center_point=None, radius_km=None, min_yield=None, max_yield=None, num_points=100):
    """
    Generate contour data for an isopach map showing zones of equal yield values
    using Radial Basis Function (similar to kriging) for interpolation.
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude and yield_rate columns
    center_point : tuple, optional
        Tuple containing (latitude, longitude) of the center point
        If None, will use center of wells
    radius_km : float, optional
        Radius in kilometers to limit visualization
        If None, will show the entire interpolated area
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
    
    # If center point not provided, use center of all well points
    if center_point is None:
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
    else:
        center_lat, center_lon = center_point
    
    # If radius not provided, calculate appropriate radius
    if radius_km is None:
        # Calculate distances from center to all points
        distances = []
        for lat, lon in zip(lats, lons):
            distance = np.sqrt(((lat - center_lat) * 111.0)**2 + 
                             ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2)
            distances.append(distance)
        
        # Use a radius that includes most wells (95% percentile)
        if distances:
            radius_km = np.percentile(distances, 95)
            # Make sure radius is reasonable (between 5km and 50km)
            radius_km = max(5.0, min(50.0, radius_km))
        else:
            radius_km = 20.0  # Default if no wells
    
    # Calculate area to cover
    lat_radius = radius_km / 111.0 * 1.5  # Expand area by 50% for better visualization
    lon_radius = radius_km / (111.0 * np.cos(np.radians(float(center_lat)))) * 1.5
    
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
    
    try:
        # Use Radial Basis Function for kriging-like interpolation
        # This smoothly interpolates between points, similar to the example from the link
        rbf = Rbf(lons, lats, yields, function='multiquadric', epsilon=2)
        
        # Apply RBF interpolation to the grid
        grid_z = rbf(grid_lon, grid_lat)
        
        # Ensure values are within the yield range for consistency
        grid_z = np.clip(grid_z, min_yield, max_yield)
        
    except Exception as e:
        print(f"RBF interpolation failed, falling back to griddata: {e}")
        
        # Fallback to griddata
        points = np.column_stack((lons, lats))
        grid_z = griddata(points, yields, (grid_lon, grid_lat), method='cubic', fill_value=min_yield)
        
        # Fill any NaN values with linear interpolation
        if np.isnan(grid_z).any():
            grid_z_linear = griddata(points, yields, (grid_lon, grid_lat), method='linear', fill_value=min_yield)
            grid_z = np.where(np.isnan(grid_z), grid_z_linear, grid_z)
            
            # As a last resort, use nearest neighbor interpolation for any remaining NaNs
            if np.isnan(grid_z).any():
                grid_z_nearest = griddata(points, yields, (grid_lon, grid_lat), method='nearest', fill_value=min_yield)
                grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
    
    # Generate contours using matplotlib
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Define contour levels for smooth transitions
        level_count = 12
        levels = np.linspace(min_yield, max_yield, level_count)
        
        # Generate filled contours
        contourf = ax.contourf(grid_lon, grid_lat, grid_z, levels=levels, cmap='jet')
        
        # Extract contour polygons to GeoJSON format
        contour_json = {"type": "FeatureCollection", "features": []}
        
        # Iterate through contour levels to create GeoJSON polygons
        for i in range(level_count-1):
            level_min = levels[i]
            level_max = levels[i+1]
            level_value = (level_min + level_max) / 2.0  # Middle value for each level
            
            # Create boolean mask for this level range
            level_mask = (grid_z >= level_min) & (grid_z < level_max)
            
            # Use scikit-image to find contour polygons
            from skimage import measure
            contours = measure.find_contours(level_mask.astype(float), 0.5)
            
            # Process each contour
            for contour in contours:
                # Skip tiny contours
                if len(contour) < 5:
                    continue
                    
                # Convert from array indices to lat/lon coordinates
                # Note: contour indices are (row, col) but we need (lat, lon)
                contour_coords = []
                for point in contour:
                    row, col = point
                    # Convert indices to actual coordinates
                    if 0 <= row < len(grid_lats) and 0 <= col < len(grid_lons):
                        lat = grid_lats[int(min(row, len(grid_lats)-1))]
                        lon = grid_lons[int(min(col, len(grid_lons)-1))]
                        # Store as [lon, lat] for GeoJSON
                        contour_coords.append([float(lon), float(lat)])
                
                # Ensure polygon is closed
                if len(contour_coords) > 3 and contour_coords[0] != contour_coords[-1]:
                    contour_coords.append(contour_coords[0])
                
                # Create GeoJSON feature if we have enough points
                if len(contour_coords) > 3:
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "yield_value": float(level_value)
                        },
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [contour_coords]
                        }
                    }
                    contour_json["features"].append(feature)
        
        # Clean up the figure
        plt.close(fig)
        
        return contour_json
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error generating contours: {e}")
        return None

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

def get_well_stats(wells_df, center_lat, center_lon, radius_km):
    """
    Calculate statistics for wells within the given radius
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    center_lat : float
        Center latitude
    center_lon : float
        Center longitude
    radius_km : float
        Search radius in kilometers
        
    Returns:
    --------
    dict
        Dictionary containing statistics
    """
    if wells_df.empty:
        return {
            "well_count": 0,
            "avg_yield": 0,
            "max_yield": 0,
            "avg_depth": 0,
            "predicted_yield": 0
        }
    
    # Filter wells within radius
    wells_in_radius = []
    for _, row in wells_df.iterrows():
        dist = get_distance(center_lat, center_lon, row['latitude'], row['longitude'])
        if dist <= radius_km:
            wells_in_radius.append(row)
    
    # Create dataframe of nearby wells
    nearby_df = pd.DataFrame(wells_in_radius) if wells_in_radius else pd.DataFrame()
    
    # Calculate statistics
    stats = {
        "well_count": len(nearby_df),
        "avg_yield": 0,
        "max_yield": 0,
        "avg_depth": 0,
        "predicted_yield": 0
    }
    
    if not nearby_df.empty:
        stats["avg_yield"] = round(nearby_df['yield_rate'].mean(), 2)
        stats["max_yield"] = round(nearby_df['yield_rate'].max(), 2)
        stats["avg_depth"] = round(nearby_df['depth'].mean(), 2) if 'depth' in nearby_df.columns else 0
        stats["predicted_yield"] = round(get_prediction_at_point(wells_df, center_lat, center_lon), 2)
    
    return stats