import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from skgstat import Variogram
from skgstat.models import spherical, gaussian, exponential
import matplotlib.pyplot as plt

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50):
    """
    Generate interpolated heat map data based on well yield rates using Kriging
    
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
    
    # Simple handling for very few data points
    if len(wells_df) < 4:
        # Create a simple heat map directly from well points
        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        yields = wells_df['yield_rate'].values.astype(float)
        
        if len(yields) == 0 or np.max(yields) == 0:
            return []
            
        max_yield = np.max(yields)
        normalized_yields = yields / max_yield
        simple_heat_data = [[float(lat), float(lon), float(y)] for lat, lon, y in zip(lats, lons, normalized_yields)]
        return simple_heat_data
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
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
    
    # Create a grid of points for kriging prediction
    adjusted_resolution = min(50, max(25, resolution))  # Balancing detail vs performance
    grid_lats = np.linspace(min_lat, max_lat, adjusted_resolution)
    grid_lons = np.linspace(min_lon, max_lon, adjusted_resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Convert to projection coordinates (rough approximation using kilometers)
    # This makes distances more accurate for variogram calculation
    origin_lat, origin_lon = np.mean(lats), np.mean(lons)
    x_coords = (lons - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    y_coords = (lats - origin_lat) * 111.0
    
    # Create coordinates for grid points
    grid_x = (grid_lon.flatten() - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    grid_y = (grid_lat.flatten() - origin_lat) * 111.0
    
    try:
        # Combine coordinates for variogram calculation
        coords = np.column_stack((x_coords, y_coords))
        grid_coords = np.column_stack((grid_x, grid_y))
        
        # Apply Kriging for spatial interpolation
        # Create and fit a variogram model
        try:
            # Try different variogram models to find the best fit
            models = [gaussian, spherical, exponential]
            best_model = None
            best_score = float('inf')
            
            for model in models:
                try:
                    # Create variogram with appropriate range for the dataset
                    max_dist = np.sqrt(radius_km**2 + radius_km**2)  # Max possible distance in study area
                    V = Variogram(coords, yields, model=model, maxlag=max_dist)
                    
                    # Calculate goodness of fit for this model
                    score = V.rmse
                    
                    if score < best_score:
                        best_score = score
                        best_model = model
                except:
                    continue
                    
            # If no model worked, use spherical as default
            if best_model is None:
                best_model = spherical
                
            # Create final variogram with best model
            V = Variogram(coords, yields, model=best_model, maxlag=radius_km)
            
            # Ordinary Kriging
            grid_z = np.zeros(len(grid_coords))
            
            # For each point in the grid, apply Kriging with nearby wells
            # This is computationally intensive but gives better results
            for i, point in enumerate(grid_coords):
                # Calculate distances to all sample points
                dists = np.sqrt(np.sum((coords - point)**2, axis=1))
                
                # Use only points within a reasonable distance (performance optimization)
                # and ensure we use at least 3 points
                max_neighbors = min(10, len(coords))  # Use at most 10 neighbors
                idx = np.argsort(dists)[:max_neighbors]
                
                if len(idx) < 3:
                    continue
                
                # Get nearby coordinates and values
                local_coords = coords[idx]
                local_values = yields[idx]
                
                try:
                    # Create local variogram for this neighborhood
                    local_V = Variogram(local_coords, local_values, model=best_model)
                    
                    # Calculate kriging weights
                    n = len(local_coords)
                    distances = np.sqrt(np.sum((local_coords - point)**2, axis=1))
                    
                    # Build kriging matrix
                    K = np.zeros((n+1, n+1))
                    for i1 in range(n):
                        for i2 in range(i1, n):
                            # Distance between sample points
                            h = np.sqrt(np.sum((local_coords[i1] - local_coords[i2])**2))
                            gamma = local_V.transform(h)  # Semivariance
                            K[i1, i2] = gamma
                            K[i2, i1] = gamma
                    
                    # Add lagrange multiplier row/column
                    K[:n, n] = 1.0
                    K[n, :n] = 1.0
                    K[n, n] = 0.0
                    
                    # Right-hand side: variogram values at prediction location
                    k = np.array([local_V.transform(d) for d in distances] + [1.0])
                    
                    # Solve for weights
                    try:
                        weights = np.linalg.solve(K, k)[:n]  # Exclude lagrange multiplier
                        
                        # Calculate prediction
                        grid_z[i] = np.sum(weights * local_values)
                    except:
                        # Fallback if matrix is singular
                        # Use inverse distance weighting as fallback
                        weights = 1.0 / (distances**2 + 1e-8)
                        weights /= np.sum(weights)
                        grid_z[i] = np.sum(weights * local_values)
                except:
                    # Fallback if variogram fails
                    # Simple inverse distance weighting
                    weights = 1.0 / (distances**2 + 1e-8)
                    weights /= np.sum(weights)
                    grid_z[i] = np.sum(weights * local_values)
            
            # Normalize results for heat map display (0-1 scale)
            min_z = np.min(grid_z)
            max_z = np.max(grid_z)
            
            if max_z > min_z:
                # Scale to 0.1-1.0 range to make low values visible but distinct from zero
                norm_z = 0.1 + 0.9 * (grid_z - min_z) / (max_z - min_z)
            else:
                norm_z = np.ones_like(grid_z) * 0.5
            
            # Create heat map data
            heat_data = []
            for i, (lat, lon) in enumerate(zip(grid_lat.flatten(), grid_lon.flatten())):
                if norm_z[i] > 0.05:  # Filter out very low values
                    # Distance from center
                    dist_from_center = np.sqrt(
                        ((lat - center_lat) * 111.0)**2 + 
                        ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
                    )
                    
                    # Reduce intensity for points far from center
                    if dist_from_center <= radius_km:
                        dist_factor = max(0, 1 - (dist_from_center / radius_km)**2)
                        
                        # Create heat map point with intensity adjusted by distance from center
                        heat_data.append([float(lat), float(lon), float(norm_z[i] * dist_factor)])
            
            return heat_data
            
        except Exception as e:
            print(f"Kriging failed: {e}")
            # Fall back to simpler interpolation method
            return fallback_interpolation(wells_df, center_point, radius_km, resolution)
    
    except Exception as e:
        print(f"Interpolation error: {e}")
        # If all else fails, just show points at the well locations
        normalized_yields = yields / np.max(yields) if np.max(yields) > 0 else np.ones_like(yields) * 0.5
        simple_heat_data = [[float(lat), float(lon), float(y)] for lat, lon, y in zip(lats, lons, normalized_yields)]
        return simple_heat_data

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """Fallback method using simpler interpolation if Kriging fails"""
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    center_lat, center_lon = center_point
    
    # Normalize yields
    max_yield = np.max(yields) if np.max(yields) > 0 else 1
    min_yield = np.min(yields)
    yield_range = max_yield - min_yield
    
    if yield_range > 0:
        normalized_yields = 0.1 + 0.9 * (yields - min_yield) / yield_range
    else:
        normalized_yields = np.ones_like(yields) * 0.5
    
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
    
    # Reshape for interpolation
    points = np.column_stack((lats, lons))
    grid_points = np.column_stack((grid_lat.flatten(), grid_lon.flatten()))
    
    try:
        # Try simple linear interpolation
        grid_yields = griddata(points, normalized_yields, grid_points, method='linear', fill_value=0)
        
        # Create heat map data
        heat_data = []
        for i, (lat, lon) in enumerate(grid_points):
            if grid_yields[i] > 0.05:
                # Calculate distance from center
                dist_from_center = np.sqrt(
                    ((lat - center_lat) * 111.0)**2 + 
                    ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2
                )
                
                if dist_from_center <= radius_km:
                    dist_factor = max(0, 1 - (dist_from_center / radius_km)**2)
                    intensity = float(grid_yields[i]) * dist_factor
                    heat_data.append([float(lat), float(lon), intensity])
        
        return heat_data
    except:
        # Last resort: just show points at the well locations
        simple_heat_data = [[float(lat), float(lon), float(y)] for lat, lon, y in zip(lats, lons, normalized_yields)]
        return simple_heat_data

def get_prediction_at_point(wells_df, point_lat, point_lon):
    """
    Get a predicted yield at a specific point based on nearby wells using Kriging
    
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
    
    # If we have very few wells, use inverse distance weighting
    if len(wells_df) < 4:
        return idw_prediction(wells_df, point_lat, point_lon)
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].values.astype(float)
    
    # Convert to projection coordinates for better distance calculation
    # Use mean of coordinates as origin to minimize distortion
    origin_lat, origin_lon = np.mean(lats), np.mean(lons)
    x_coords = (lons - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    y_coords = (lats - origin_lat) * 111.0
    
    # Convert prediction point to same coordinate system
    point_x = (point_lon - origin_lon) * 111.0 * np.cos(np.radians(origin_lat))
    point_y = (point_lat - origin_lat) * 111.0
    
    try:
        # Create coordinates array for variogram calculation
        coords = np.column_stack((x_coords, y_coords))
        
        # Calculate max distance for variogram
        max_dist = np.sqrt(np.max(np.sum((coords - np.array([point_x, point_y]))**2, axis=1)))
        max_dist = max(max_dist, 20)  # At least 20km to get reliable variogram
        
        # Try different variogram models to find best fit
        models = [gaussian, spherical, exponential]
        best_model = None
        best_score = float('inf')
        
        for model in models:
            try:
                V = Variogram(coords, yields, model=model, maxlag=max_dist)
                score = V.rmse
                
                if score < best_score:
                    best_score = score
                    best_model = model
            except:
                continue
        
        # If no model worked, use spherical as default
        if best_model is None:
            best_model = spherical
        
        # Create variogram with best model
        V = Variogram(coords, yields, model=best_model, maxlag=max_dist)
        
        # Calculate distances to all points
        dists = np.sqrt(np.sum((coords - np.array([point_x, point_y]))**2, axis=1))
        
        # Sort points by distance and take the closest ones for kriging
        # This improves accuracy and performance
        max_neighbors = min(12, len(coords))  # Use up to 12 nearest neighbors
        idx = np.argsort(dists)[:max_neighbors]
        
        local_coords = coords[idx]
        local_values = yields[idx]
        local_dists = dists[idx]
        
        # Apply Ordinary Kriging with the local points
        try:
            # Calculate kriging matrix
            n = len(local_coords)
            K = np.zeros((n+1, n+1))
            
            # Fill the kriging matrix with semivariance values
            for i in range(n):
                for j in range(i, n):
                    # Distance between sample points
                    h = np.sqrt(np.sum((local_coords[i] - local_coords[j])**2))
                    gamma = V.transform(h)  # Semivariance
                    K[i, j] = gamma
                    K[j, i] = gamma
            
            # Add Lagrange multiplier constraints
            K[:n, n] = 1.0
            K[n, :n] = 1.0
            K[n, n] = 0.0
            
            # Right hand side: variogram values for distances to prediction point
            k = np.array([V.transform(d) for d in local_dists] + [1.0])
            
            # Solve for weights
            try:
                weights = np.linalg.solve(K, k)[:n]  # Exclude lagrange multiplier
                
                # Calculate prediction
                predicted_yield = np.sum(weights * local_values)
                
                # Apply confidence adjustment based on distance to nearest well
                # This reduces prediction certainty in areas far from any measured points
                nearest_dist = np.min(local_dists)
                confidence_factor = max(0.5, min(1.0, 2.0 / (1.0 + nearest_dist/5)))
                
                # Blend with average for stability
                avg_yield = np.mean(yields)
                final_yield = predicted_yield * confidence_factor + avg_yield * (1 - confidence_factor)
                
                return float(max(0, final_yield))  # Ensure non-negative yield
                
            except:
                # Matrix solution failed, use IDW as fallback
                return idw_prediction(wells_df, point_lat, point_lon)
                
        except Exception as e:
            print(f"Kriging prediction error: {e}")
            return idw_prediction(wells_df, point_lat, point_lon)
            
    except Exception as e:
        print(f"Variogram calculation error: {e}")
        # Fall back to simpler interpolation
        try:
            # Create points for griddata interpolation
            points = np.column_stack((lats, lons))
            point = np.array([point_lat, point_lon])
            
            # Try cubic first if we have enough points
            if len(wells_df) >= 5:
                predicted_yield = griddata(points, yields, point, method='cubic')
                
                if np.isnan(predicted_yield):
                    predicted_yield = griddata(points, yields, point, method='linear')
                    
                    if np.isnan(predicted_yield):
                        predicted_yield = griddata(points, yields, point, method='nearest')
            else:
                predicted_yield = griddata(points, yields, point, method='linear')
                
                if np.isnan(predicted_yield):
                    predicted_yield = griddata(points, yields, point, method='nearest')
            
            # If all methods failed, use IDW
            if np.isnan(predicted_yield):
                return idw_prediction(wells_df, point_lat, point_lon)
                
            return float(max(0, predicted_yield))  # Ensure non-negative yield
            
        except:
            # Last resort is IDW
            return idw_prediction(wells_df, point_lat, point_lon)
            
def idw_prediction(wells_df, point_lat, point_lon):
    """
    Calculate yield using Inverse Distance Weighting (IDW)
    Used as a fallback when Kriging fails
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
    
    # Calculate inverse distance weights
    # Use power of 2 for standard IDW
    weights = [1 / (d**2) for d in distances]
    total_weight = sum(weights)
    
    if total_weight == 0:
        # If all weights are zero (should be impossible with our minimum distance)
        return 0
    
    # Calculate weighted average of yields
    weighted_yield = sum(w * float(row['yield_rate']) for w, (idx, row) in zip(weights, wells_df.iterrows())) / total_weight
    return float(max(0, weighted_yield))  # Ensure non-negative yield
