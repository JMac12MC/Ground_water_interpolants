import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Rbf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import folium
import branca.colormap as cm
from utils import get_distance

def create_smooth_contours(wells_df, center_point=None, radius_km=None, min_yield=None, max_yield=None, num_points=200):
    """
    Generate smooth continuous contours for ground water yield visualization
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude and yield_rate columns
    center_point : tuple, optional
        Tuple containing (latitude, longitude) of the center point
    radius_km : float, optional
        Radius in kilometers to limit visualization
    min_yield : float, optional
        Minimum yield value for normalization
    max_yield : float, optional
        Maximum yield value for normalization
    num_points : int
        Grid resolution (higher = more detailed but slower)
        
    Returns:
    --------
    dict
        Dict containing contours and color information for map overlay
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
        min_yield = float(np.min(yields)) if len(yields) > 0 else 0
    if max_yield is None:
        max_yield = float(np.max(yields)) if len(yields) > 0 else 1
    
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
    
    # Calculate grid boundaries to cover the entire area
    min_lat = np.min(lats) - 0.1
    max_lat = np.max(lats) + 0.1
    min_lon = np.min(lons) - 0.1
    max_lon = np.max(lons) + 0.1
    
    # Create a regular grid for interpolation
    grid_x, grid_y = np.mgrid[min_lon:max_lon:complex(0, num_points), 
                            min_lat:max_lat:complex(0, num_points)]
    
    # Create points array from well locations
    points = np.column_stack((lons, lats))
    
    try:
        # Use Radial Basis Function for smoother interpolation
        rbf = Rbf(lons, lats, yields, function='multiquadric', epsilon=2)
        grid_z = rbf(grid_x, grid_y)
        
        # Ensure values are within range for consistency
        grid_z = np.clip(grid_z, min_yield, max_yield)
    except Exception as e:
        print(f"RBF interpolation failed, falling back to griddata: {e}")
        
        # Fallback to griddata interpolation
        grid_z = griddata(points, yields, (grid_x, grid_y), method='cubic')
        
        # Fill NaN values
        if np.isnan(grid_z).any():
            grid_z_linear = griddata(points, yields, (grid_x, grid_y), method='linear')
            grid_z = np.where(np.isnan(grid_z), grid_z_linear, grid_z)
        
        # Final fallback to nearest neighbor for any remaining NaNs
        if np.isnan(grid_z).any():
            grid_z_nearest = griddata(points, yields, (grid_x, grid_y), method='nearest')
            grid_z = np.where(np.isnan(grid_z), grid_z_nearest, grid_z)
    
    # Create a colormap for the yields
    colormap = cm.LinearColormap(
        ['blue', 'cyan', 'green', 'yellow', 'orange', 'red'],
        vmin=min_yield,
        vmax=max_yield
    )
    
    # Add the contour fill layer to the map
    contour_layer = {
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_z': grid_z,
        'min_yield': min_yield,
        'max_yield': max_yield,
        'colormap': colormap
    }
    
    return contour_layer

def add_contour_to_map(m, contour_layer, name="Isopach Map"):
    """
    Add a smooth contour overlay to a Folium map
    
    Parameters:
    -----------
    m : folium.Map
        The map to add the contours to
    contour_layer : dict
        Contour data from create_smooth_contours
    name : str
        Name for the layer in the layer control
    
    Returns:
    --------
    folium.Map
        Updated map with contour overlay
    """
    if contour_layer is None:
        return m
        
    grid_x = contour_layer['grid_x']
    grid_y = contour_layer['grid_y']
    grid_z = contour_layer['grid_z']
    colormap = contour_layer['colormap']
    
    # Add contour overlay
    contour_group = folium.FeatureGroup(name=name)
    
    # Create grid points with interpolated values
    for i in range(grid_z.shape[0]-1):
        for j in range(grid_z.shape[1]-1):
            # Create small polygon for each grid cell
            grid_cell = [
                [grid_y[i, j], grid_x[i, j]],
                [grid_y[i, j+1], grid_x[i, j+1]],
                [grid_y[i+1, j+1], grid_x[i+1, j+1]],
                [grid_y[i+1, j], grid_x[i+1, j]]
            ]
            
            # Use average value for the grid cell
            cell_value = np.mean([
                grid_z[i, j], 
                grid_z[i, j+1], 
                grid_z[i+1, j+1], 
                grid_z[i+1, j]
            ])
            
            # Add a polygon for this grid cell with appropriate color
            folium.Polygon(
                locations=grid_cell,
                color=None,
                fill=True,
                fill_color=colormap(cell_value),
                fill_opacity=0.5,
                tooltip=f"Yield: {cell_value:.2f}"
            ).add_to(contour_group)
    
    # Add the contour layer to the map
    contour_group.add_to(m)
    
    # Add colormap legend
    colormap.caption = 'Groundwater Yield (L/s)'
    colormap.add_to(m)
    
    return m

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
    if wells_df is None or wells_df.empty:
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
        return float(yields[np.argmin(distances)])
    
    # Use IDW with a power parameter of 2
    power = 2.0
    weights = 1.0 / (distances**power)
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Calculate the weighted average
    predicted_yield = float(np.sum(weights * yields))
    
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
    if wells_df is None or wells_df.empty:
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
        stats["avg_yield"] = float(np.round(nearby_df['yield_rate'].mean(), 2))
        stats["max_yield"] = float(np.round(nearby_df['yield_rate'].max(), 2))
        stats["avg_depth"] = float(np.round(nearby_df['depth'].mean(), 2)) if 'depth' in nearby_df.columns else 0
        stats["predicted_yield"] = float(np.round(get_prediction_at_point(wells_df, center_lat, center_lon), 2))
    
    return stats