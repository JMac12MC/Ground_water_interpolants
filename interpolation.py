import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50):
    """
    Generate heat map data using 2D Kriging interpolation for scientific groundwater mapping
    
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
    
    # For special case with too few wells, just show the wells as they are
    if len(wells_df) < 3:
        heat_data = []
        for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
            heat_data.append([float(lat), float(lon), float(yield_val)])
        return heat_data
    
    # Create bounding box based on center and radius
    center_lat, center_lon = center_point
    
    # Convert geographic area to a "flat" projection for better interpolation
    # This is a simple approximation for small areas
    # 111 km per degree of latitude, longitude varies with cosine of latitude
    x_points = (lons - center_lon) * 111.0 * np.cos(np.radians(center_lat))  # km
    y_points = (lats - center_lat) * 111.0  # km
    z_points = yields  # The yield values we want to interpolate
    
    # Define the grid
    grid_density = min(100, max(50, resolution))  # Higher resolution grid
    
    # Convert radius to a square in the local projection
    x_grid = np.linspace(-radius_km, radius_km, grid_density)
    y_grid = np.linspace(-radius_km, radius_km, grid_density)
    
    # Create a grid of coordinates where we'll predict the yield
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    
    try:
        # Use scikit-gstat's kriging implementation
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        
        # Prepare the input points for Kriging
        xy_points = np.column_stack((x_points, y_points))
        
        # Define the Gaussian Process model with RBF kernel - this is the core of Kriging
        # RBF length scale ~5km - regional groundwater correlation typical scale
        # WhiteKernel handles noise in the measurements
        kernel = 1.0 * RBF(length_scale=5.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        
        # Fit the model to our data points (wells)
        gp.fit(xy_points, z_points)
        
        # Prepare grid points for prediction
        # Flatten the grid for prediction and filter points outside search radius
        grid_xs = X_grid.flatten()
        grid_ys = Y_grid.flatten()
        
        # Create prediction points array
        grid_points = np.column_stack((grid_xs, grid_ys))
        
        # Only keep points within the radius
        mask = np.sqrt(grid_xs**2 + grid_ys**2) <= radius_km
        grid_points = grid_points[mask]
        
        if len(grid_points) == 0:
            return []
        
        # Predict the yield at each grid point using the GP model (Kriging)
        predictions = gp.predict(grid_points)
        
        # Make sure we don't predict negative yields
        predictions = np.maximum(0, predictions)
        
        # Convert back to geographic coordinates
        geo_lons = (grid_points[:, 0] / (111.0 * np.cos(np.radians(center_lat)))) + center_lon
        geo_lats = (grid_points[:, 1] / 111.0) + center_lat
        
        # Create heat map data
        heat_data = []
        
        # We need to normalize the predictions for the heatmap (0-1 scale)
        # Find min/max of the predicted values
        min_pred = np.min(predictions) if len(predictions) > 0 else 0
        max_pred = np.max(predictions) if len(predictions) > 0 else 1
        pred_range = max(0.1, max_pred - min_pred)  # Avoid division by zero
        
        # Add the interpolated grid points to the heat map
        for i in range(len(geo_lats)):
            # Normalize the predicted value to 0-1 range for the heat map
            normalized_pred = (predictions[i] - min_pred) / pred_range
            
            # Add to heat map if significant
            if normalized_pred > 0.01:
                heat_data.append([
                    float(geo_lats[i]),
                    float(geo_lons[i]),
                    float(predictions[i])  # Use actual yield value for color intensity
                ])
        
        # Add the actual well points to ensure they're visible
        for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
            # Check if within search radius
            dist_from_center_km = np.sqrt(
                ((lat - center_lat) * 111.0)**2 + 
                ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
            )
            
            if dist_from_center_km <= radius_km:
                # Add the exact well with its actual yield
                heat_data.append([float(lat), float(lon), float(yield_val)])
        
        return heat_data
        
    except Exception as e:
        print(f"Kriging interpolation error: {e}")
        # Fall back to basic interpolation method
        return fallback_interpolation(wells_df, center_point, radius_km)

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
