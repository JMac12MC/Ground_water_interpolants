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
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return []
    
    # Make sure we have at least 4 points for proper interpolation
    if len(wells_df) < 4:
        # Create a simple heat map directly from well points
        lats = wells_df['latitude'].values
        lons = wells_df['longitude'].values
        yields = wells_df['yield_rate'].values
        max_yield = np.max(yields) if yields.size > 0 and np.max(yields) > 0 else 1
        normalized_yields = yields / max_yield
        simple_heat_data = [[float(lat), float(lon), float(y)] for lat, lon, y in zip(lats, lons, normalized_yields)]
        return simple_heat_data
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values
    lons = wells_df['longitude'].values
    yields = wells_df['yield_rate'].values
    
    # Normalize yields using a more representative scale
    min_yield = np.min(yields)
    max_yield = np.max(yields)
    yield_range = max_yield - min_yield
    
    # If all values are the same, avoid division by zero
    if yield_range == 0:
        normalized_yields = np.ones_like(yields) * 0.5  # Mid-range if all values are equal
    else:
        # Scale to 0.1-1.0 instead of 0-1 to make sure low values are still visible but distinct
        normalized_yields = 0.1 + 0.9 * (yields - min_yield) / yield_range
    
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
    
    # Create a grid of points - use higher resolution for more detailed maps
    adjusted_resolution = min(80, max(30, resolution))  # Ensure reasonable bounds
    grid_lats = np.linspace(min_lat, max_lat, adjusted_resolution)
    grid_lons = np.linspace(min_lon, max_lon, adjusted_resolution)
    grid_lat, grid_lon = np.meshgrid(grid_lats, grid_lons)
    
    # Reshape for interpolation
    points = np.column_stack((lats, lons))
    grid_points = np.column_stack((grid_lat.flatten(), grid_lon.flatten()))
    
    # Perform interpolation with fallbacks
    try:
        # Try 'cubic' method first for smoother results if we have enough points
        if len(wells_df) >= 6:
            try:
                grid_yields = griddata(points, normalized_yields, grid_points, method='cubic', fill_value=np.nan)
                # Fill NaN values using linear interpolation
                nan_mask = np.isnan(grid_yields)
                if np.any(nan_mask):
                    grid_yields[nan_mask] = griddata(points, normalized_yields, grid_points[nan_mask], 
                                                     method='linear', fill_value=0)
            except Exception:
                # Fall back to linear if cubic fails
                grid_yields = griddata(points, normalized_yields, grid_points, method='linear', fill_value=0)
        else:
            # Use linear for smaller datasets
            grid_yields = griddata(points, normalized_yields, grid_points, method='linear', fill_value=0)
        
        # Create heat map data points, using distance weighting from center
        heat_data = []
        for i, (lat, lon) in enumerate(grid_points):
            # Skip points with no interpolated value
            if grid_yields[i] <= 0:
                continue
                
            # Calculate distance from center point as fraction of radius
            dist_from_center = np.sqrt(((lat - center_lat) * 111.0)**2 + 
                                      ((lon - center_lon) * 111.0 * np.cos(np.radians(center_lat)))**2)
            dist_factor = max(0, 1 - (dist_from_center / radius_km)**2)
            
            # Adjust intensity based on both yield and distance from any actual well
            nearest_well_dist = np.min(np.sqrt(((lat - lats) * 111.0)**2 + 
                                             ((lon - lons) * 111.0 * np.cos(np.radians(lat)))**2))
            proximity_factor = max(0.2, min(1.0, 1.0 / (1.0 + nearest_well_dist/5)))
            
            # Final intensity combines yield value with proximity adjustments
            intensity = float(grid_yields[i]) * dist_factor * proximity_factor
            
            # Only add points with meaningful intensity
            if intensity > 0.05:
                heat_data.append([lat, lon, intensity])
        
        return heat_data
        
    except Exception as e:
        # If interpolation fails, fallback to direct visualization of well points
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
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        return 0
    
    # If we have less than 3 wells, use inverse distance weighting directly
    if len(wells_df) < 3:
        # Calculate distance from each well to the point (in km)
        distances = []
        for idx, row in wells_df.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            # Convert to kilometers using approximate conversion
            lat_dist = (lat - point_lat) * 111.0  # 1 degree latitude ≈ 111 km
            lon_dist = (lon - point_lon) * 111.0 * np.cos(np.radians((lat + point_lat) / 2))
            distance = np.sqrt(lat_dist**2 + lon_dist**2)
            distances.append(max(0.1, distance))  # Prevent division by zero
        
        # Calculate inverse distance weights
        weights = [1 / (d**2) for d in distances]
        total_weight = sum(weights)
        
        # Calculate weighted average of yields
        weighted_yield = sum(w * row['yield_rate'] for w, (idx, row) in zip(weights, wells_df.iterrows())) / total_weight
        return float(weighted_yield)
    
    # Extract coordinates and yields
    lats = wells_df['latitude'].values
    lons = wells_df['longitude'].values
    yields = wells_df['yield_rate'].values
    
    # Create the point
    points = np.column_stack((lats, lons))
    point = np.array([point_lat, point_lon])
    
    # Perform interpolation at the single point
    try:
        # Try cubic first if we have enough points
        if len(wells_df) >= 5:
            try:
                predicted_yield = griddata(points, yields, point, method='cubic')
                
                # If the point is outside the convex hull of the data points
                if np.isnan(predicted_yield):
                    # Try linear
                    predicted_yield = griddata(points, yields, point, method='linear')
                    
                    if np.isnan(predicted_yield):
                        # Fall back to nearest neighbor as a last resort
                        predicted_yield = griddata(points, yields, point, method='nearest')
            except Exception:
                # Fall back to linear if cubic fails
                predicted_yield = griddata(points, yields, point, method='linear')
                
                if np.isnan(predicted_yield):
                    predicted_yield = griddata(points, yields, point, method='nearest')
        else:
            # For fewer points, start with linear
            predicted_yield = griddata(points, yields, point, method='linear')
            
            if np.isnan(predicted_yield):
                predicted_yield = griddata(points, yields, point, method='nearest')
        
        # Apply distance-based adjustment
        # Find the distance to the nearest well (in km)
        distances = []
        for lat, lon in zip(lats, lons):
            lat_dist = (lat - point_lat) * 111.0
            lon_dist = (lon - point_lon) * 111.0 * np.cos(np.radians((lat + point_lat) / 2))
            distance = np.sqrt(lat_dist**2 + lon_dist**2)
            distances.append(distance)
        
        nearest_dist = min(distances)
        
        # Adjust prediction based on distance to nearest well
        # The further from a known well, the more we reduce confidence
        confidence_factor = max(0.5, min(1.0, 2.0 / (1.0 + nearest_dist/2)))
        
        # Calculate the average yield as a fallback/blending value
        avg_yield = np.mean(yields)
        
        # Blend the prediction with the average based on confidence
        final_yield = predicted_yield * confidence_factor + avg_yield * (1 - confidence_factor)
        
        return float(final_yield)
    except Exception as e:
        print(f"Prediction error: {e}")
        
        # Fall back to inverse distance weighting
        try:
            distances = []
            for lat, lon in zip(lats, lons):
                lat_dist = (lat - point_lat) * 111.0
                lon_dist = (lon - point_lon) * 111.0 * np.cos(np.radians((lat + point_lat) / 2))
                distance = np.sqrt(lat_dist**2 + lon_dist**2)
                distances.append(max(0.1, distance))  # Prevent division by zero
            
            weights = [1 / (d**2) for d in distances]
            total_weight = sum(weights)
            weighted_yield = sum(w * y for w, y in zip(weights, yields)) / total_weight
            return float(weighted_yield)
        except:
            # Last resort - return average
            return float(np.mean(yields)) if len(yields) > 0 else 0
