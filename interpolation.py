import numpy as np
import pandas as pd
import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestRegressor
from pykrige.ok import OrdinaryKriging

def generate_geo_json_grid(wells_df, center_point, radius_km, resolution=50, method='kriging'):
    """
    Generate GeoJSON grid with interpolated yield values for accurate visualization
    
    This function converts the interpolated data into a GeoJSON format with small polygon
    cells, each colored according to the exact yield value at that location.
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data
    center_point : tuple
        (latitude, longitude) of center point
    radius_km : float
        Radius in km to include in the grid
    resolution : int
        Grid resolution (number of cells per dimension)
    method : str
        Interpolation method to use
        
    Returns:
    --------
    dict
        GeoJSON data structure with interpolated yield values
    """
    # First get the interpolated data using our existing function
    interpolated_data = generate_heat_map_data(wells_df, center_point, radius_km, resolution, method)
    
    # Extract the original grid information
    center_lat, center_lon = center_point
    km_per_degree_lat = 111.0  # ~111km per degree of latitude
    km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))  # Longitude degrees vary with latitude
    
    # Create grid in lat/lon space
    min_lat = center_lat - (radius_km / km_per_degree_lat)
    max_lat = center_lat + (radius_km / km_per_degree_lat)
    min_lon = center_lon - (radius_km / km_per_degree_lon)
    max_lon = center_lon + (radius_km / km_per_degree_lon)
    
    # Create a more reasonable grid size for GeoJSON (too many cells makes it slow)
    grid_size = min(40, resolution)  # Limit to reasonable number for browser performance
    lat_vals = np.linspace(min_lat, max_lat, grid_size)
    lon_vals = np.linspace(min_lon, max_lon, grid_size)
    
    # Get a grid of values using our existing interpolated data
    grid_values = np.zeros((grid_size-1, grid_size-1))
    
    # Build the GeoJSON structure
    features = []
    for i in range(len(lat_vals)-1):
        for j in range(len(lon_vals)-1):
            # Calculate center point of this grid cell
            cell_lat = (lat_vals[i] + lat_vals[i+1]) / 2
            cell_lon = (lon_vals[j] + lon_vals[j+1]) / 2
            
            # Find yield value for this cell by looking for the closest point
            # in our interpolated data
            closest_value = 0
            min_dist = float('inf')
            
            for point in interpolated_data:
                point_lat, point_lon, value = point
                dist = ((point_lat - cell_lat)**2 + (point_lon - cell_lon)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_value = value
            
            # Only add cells with meaningful values
            if closest_value > 0.01:
                # Create polygon for this grid cell
                poly = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [float(lon_vals[j]), float(lat_vals[i])],
                            [float(lon_vals[j+1]), float(lat_vals[i])],
                            [float(lon_vals[j+1]), float(lat_vals[i+1])],
                            [float(lon_vals[j]), float(lat_vals[i+1])],
                            [float(lon_vals[j]), float(lat_vals[i])]
                        ]]
                    },
                    "properties": {
                        "yield": float(closest_value)
                    }
                }
                features.append(poly)
    
    # Create the full GeoJSON object
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50, method='kriging'):
    """
    Generate heat map data using various interpolation techniques
    
    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with latitude, longitude and yield_rate columns
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point
    radius_km : float
        Radius in kilometers to generate heat map data for
    resolution : int
        Number of points to generate in each dimension
    method : str
        Interpolation method to use ('kriging', 'idw', 'rf_kriging')
        
    Returns:
    --------
    list
        List of [lat, lng, intensity] points for the heat map
    
    Notes:
    ------
    This function is called each time a new location is selected to create an
    entirely fresh interpolation. The results should always be specific to the
    current center_point and never reuse old interpolation data.
    """
    # Handle empty datasets
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return []
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    # Handle case with too few data points
    if len(wells_df) < 3:
        heat_data = []
        for i, (lat, lon, yield_val) in enumerate(zip(lats, lons, yields)):
            heat_data.append([float(lat), float(lon), float(yield_val)])
        return heat_data
        
    # Create simplified grid for interpolation
    center_lat, center_lon = center_point
    grid_size = min(50, max(30, resolution))
    
    try:
        # Convert to km-based coordinates (flat Earth approximation for small areas)
        # This is essential for proper interpolation
        km_per_degree_lat = 111.0  # km per degree of latitude
        km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))  # km per degree of longitude
        
        # Convert all coordinates to km from center
        x_coords = (lons - center_lon) * km_per_degree_lon
        y_coords = (lats - center_lat) * km_per_degree_lat
        
        # Create grid in km space
        grid_x = np.linspace(-radius_km, radius_km, grid_size)
        grid_y = np.linspace(-radius_km, radius_km, grid_size)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)
        
        # Flatten for interpolation
        points = np.vstack([x_coords, y_coords]).T  # Well points in km
        xi = np.vstack([grid_X.flatten(), grid_Y.flatten()]).T  # Grid points in km
        
        # Filter points outside the radius
        distances = np.sqrt(xi[:,0]**2 + xi[:,1]**2)
        mask = distances <= radius_km
        xi_inside = xi[mask]
        
        # Choose interpolation method
        if method == 'rf_kriging' and len(wells_df) >= 10:
            try:
                print("Using Random Forest + Kriging interpolation")
                # OPTIMIZATION: Reduce number of trees for faster performance
                # For large datasets, we need to prioritize speed over slight accuracy improvements
                
                # Prepare data for Random Forest
                features = np.vstack([x_coords, y_coords]).T  # Features are [x, y] coordinates in km
                target = yields  # Target is the yield values
                
                # OPTIMIZATION: Use fewer trees (50 instead of 100) and limit max_depth
                # This significantly speeds up training and prediction with minimal accuracy loss
                rf = RandomForestRegressor(
                    n_estimators=50,       # Reduced from 100 for faster performance
                    max_depth=15,          # Limit tree depth for faster training
                    min_samples_split=5,   # Require more samples per split (reduces overfitting)
                    n_jobs=-1,             # Use all available cores
                    random_state=42
                )
                rf.fit(features, target)
                
                # Get RF predictions for all grid points
                rf_predictions = rf.predict(xi_inside)
                
                # OPTIMIZATION: Skip kriging for very large datasets (>1000 points)
                # as it becomes the main performance bottleneck
                if len(features) < 1000:
                    # Calculate residuals on training data
                    rf_train_preds = rf.predict(features)
                    residuals = target - rf_train_preds
                    
                    # If enough points, apply Kriging to the residuals
                    if len(features) >= 5 and len(features) < 1000:
                        # Convert back to lon/lat for kriging (pykrige expects lon/lat)
                        lon_values = x_coords / km_per_degree_lon + center_lon
                        lat_values = y_coords / km_per_degree_lat + center_lat
                        xi_lon = xi_inside[:, 0] / km_per_degree_lon + center_lon
                        xi_lat = xi_inside[:, 1] / km_per_degree_lat + center_lat
                        
                        # OPTIMIZATION: Use a simpler variogram model and limit kriging calculations
                        OK = OrdinaryKriging(
                            lon_values, lat_values, residuals,
                            variogram_model='linear',  # Simpler model than spherical - much faster
                            verbose=False,
                            enable_plotting=False
                        )
                        # Execute kriging on grid points
                        kriged_residuals, _ = OK.execute('points', xi_lon, xi_lat)
                        
                        # Combine RF predictions with kriged residuals
                        interpolated_z = rf_predictions + kriged_residuals
                    else:
                        # Not enough points for kriging, use RF predictions only
                        interpolated_z = rf_predictions
                else:
                    # Too many points for efficient kriging, use RF predictions only
                    print("Using RF predictions only (skipping kriging for large dataset)")
                    interpolated_z = rf_predictions
            except Exception as e:
                print(f"RF+Kriging error: {e}, falling back to standard interpolation")
                # Fall back to standard interpolation
                # Basic 2D interpolation (linear)
                from scipy.interpolate import griddata
                interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)
                
                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                    )
        else:
            # OPTIMIZATION: For standard kriging method
            # Basic 2D interpolation - import statement at top of file
            from scipy.interpolate import griddata
            
            # OPTIMIZATION: For large datasets, use fewer points and faster method
            if len(points) > 2000:
                print(f"Large dataset optimization: Sampling from {len(points)} points")
                # For very large datasets, we'll use a random sample to improve performance
                sample_size = min(2000, len(points))
                # Use stratified sampling for better representation (divide area into regions)
                # Create grid cells (10x10)
                lat_bins = np.linspace(np.min(lats), np.max(lats), 10)
                lon_bins = np.linspace(np.min(lons), np.max(lons), 10)
                
                # Create a mask for sampling points from each cell
                sample_indices = []
                for i in range(9):
                    for j in range(9):
                        # Get points in this cell
                        cell_mask = (
                            (lats >= lat_bins[i]) & (lats < lat_bins[i+1]) & 
                            (lons >= lon_bins[j]) & (lons < lon_bins[j+1])
                        )
                        cell_indices = np.where(cell_mask)[0]
                        
                        # If there are points in this cell, sample a proportional amount
                        if len(cell_indices) > 0:
                            # Calculate how many points to sample from this cell
                            # (proportional to the cell's share of total points)
                            cell_sample_size = max(1, int(sample_size * len(cell_indices) / len(points)))
                            # Sample randomly from this cell
                            if len(cell_indices) > cell_sample_size:
                                cell_sample = np.random.choice(cell_indices, cell_sample_size, replace=False)
                                sample_indices.extend(cell_sample)
                            else:
                                # If cell has fewer points than needed, take all of them
                                sample_indices.extend(cell_indices)
                
                # Ensure we have enough sample points
                if len(sample_indices) < sample_size:
                    # Add more random points if needed
                    remaining = sample_size - len(sample_indices)
                    all_indices = np.arange(len(points))
                    remaining_indices = np.setdiff1d(all_indices, sample_indices)
                    if len(remaining_indices) > 0:
                        additional_samples = np.random.choice(
                            remaining_indices, 
                            min(remaining, len(remaining_indices)), 
                            replace=False
                        )
                        sample_indices.extend(additional_samples)
                
                # Use the sampled points for interpolation
                sampled_points = points[sample_indices]
                sampled_yields = yields[sample_indices]
                
                # Use linear interpolation for speed with large datasets
                interpolated_z = griddata(
                    sampled_points, sampled_yields, xi_inside, 
                    method='linear', fill_value=0.0
                )
                
                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        sampled_points, sampled_yields, xi_inside[nan_mask],
                        method='nearest', fill_value=0.0
                    )
            else:
                # Standard approach for smaller datasets
                # First try linear interpolation
                interpolated_z = griddata(
                    points, yields, xi_inside, 
                    method='linear', fill_value=0.0
                )
                
                # For areas with NaNs, apply nearest neighbor to fill gaps
                if np.any(np.isnan(interpolated_z)):
                    nan_mask = np.isnan(interpolated_z)
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask],
                        method='nearest', fill_value=0.0
                    )
        
        # Make sure we don't have negative values
        interpolated_z = np.maximum(0, interpolated_z)
        
        # Convert back to lat/lon coordinates
        lat_points = (xi_inside[:, 1] / km_per_degree_lat) + center_lat
        lon_points = (xi_inside[:, 0] / km_per_degree_lon) + center_lon
        
        # Create heat map data
        heat_data = []
        
        # OPTIMIZATION: For large datasets, reduce the number of heat map points for better performance
        # First, detect if we need to sample points due to a large dataset
        max_heat_points = 2500  # Maximum points for smooth performance
        
        if len(lat_points) > max_heat_points:
            # Use a grid-based sampling approach to maintain visual accuracy with fewer points
            print(f"Optimizing heatmap visualization: sampling {max_heat_points} points from {len(lat_points)} total")
            
            # Create a grid with target number of cells
            grid_size = int(np.sqrt(max_heat_points))
            
            # Find min and max lat/lon
            min_lat, max_lat = np.min(lat_points), np.max(lat_points)
            min_lon, max_lon = np.min(lon_points), np.max(lon_points)
            
            # Create grid
            lat_grid = np.linspace(min_lat, max_lat, grid_size)
            lon_grid = np.linspace(min_lon, max_lon, grid_size)
            
            # Sample points by selecting representative points in each grid cell
            heat_data = []
            for i in range(len(lat_grid)-1):
                for j in range(len(lon_grid)-1):
                    # Find points in this grid cell
                    cell_mask = (
                        (lat_points >= lat_grid[i]) & (lat_points < lat_grid[i+1]) &
                        (lon_points >= lon_grid[j]) & (lon_points < lon_grid[j+1])
                    )
                    
                    if np.any(cell_mask):
                        # Select a point with the maximum value from this cell (important for yield visualization)
                        cell_values = interpolated_z[cell_mask]
                        cell_lat = lat_points[cell_mask]
                        cell_lon = lon_points[cell_mask]
                        
                        if np.max(cell_values) > 0.01:  # Only add significant values
                            max_idx = np.argmax(cell_values)
                            heat_data.append([
                                float(cell_lat[max_idx]),  # Latitude
                                float(cell_lon[max_idx]),  # Longitude
                                float(cell_values[max_idx])  # Yield value (actual value)
                            ])
        else:
            # Standard approach for smaller datasets
            heat_data = []
            # Add interpolated points
            for i in range(len(lat_points)):
                # Only add points with meaningful values
                if interpolated_z[i] > 0.01:
                    heat_data.append([
                        float(lat_points[i]),  # Latitude
                        float(lon_points[i]),  # Longitude
                        float(interpolated_z[i])  # Yield value (actual value, not normalized)
                    ])
        
        # Always make sure well points themselves are included for accuracy
        # These are the actual data points we have, so they should be shown
        well_points_added = 0
        for j in range(len(lats)):
            # Check if well is within search radius
            well_dist_km = np.sqrt(
                ((lats[j] - center_lat) * km_per_degree_lat)**2 +
                ((lons[j] - center_lon) * km_per_degree_lon)**2
            )
            
            if well_dist_km <= radius_km:
                heat_data.append([
                    float(lats[j]),
                    float(lons[j]),
                    float(yields[j])
                ])
                well_points_added += 1
        
        return heat_data
    
    except Exception as e:
        print(f"Interpolation error: {e}")
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
