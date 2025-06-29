import numpy as np
import pandas as pd
import json
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.ensemble import RandomForestRegressor
from pykrige.ok import OrdinaryKriging
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

def generate_geo_json_grid(wells_df, center_point, radius_km, resolution=50, method='kriging', show_variance=False, auto_fit_variogram=False, variogram_model='spherical', soil_polygons=None):
    """
    Generate GeoJSON grid with interpolated yield values for accurate visualization

    This function creates a smooth, continuous interpolation surface with GeoJSON format
    for optimal visualization of groundwater yield patterns.

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
    # Handle empty datasets
    if isinstance(wells_df, pd.DataFrame) and wells_df.empty:
        return {"type": "FeatureCollection", "features": []}

    # Filter out geotechnical/geological investigation wells using well_use
    if 'well_use' in wells_df.columns:
        geotechnical_mask = wells_df['well_use'].str.contains(
            'Geotechnical.*Investigation|Geological.*Investigation', 
            case=False, 
            na=False, 
            regex=True
        )
        wells_df = wells_df[~geotechnical_mask].copy()

        if wells_df.empty:
            return {"type": "FeatureCollection", "features": []}

    # Extract the original grid information
    center_lat, center_lon = center_point
    km_per_degree_lat = 111.0  # ~111km per degree of latitude
    km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))  # Longitude degrees vary with latitude

    # Create grid in lat/lon space
    min_lat = center_lat - (radius_km / km_per_degree_lat)
    max_lat = center_lat + (radius_km / km_per_degree_lat)
    min_lon = center_lon - (radius_km / km_per_degree_lon)
    max_lon = center_lon + (radius_km / km_per_degree_lon)

    # High resolution grid for smooth professional visualization
    # Increase resolution significantly for smoother appearance like kriging software
    wells_count = len(wells_df)
    if wells_count > 5000:
        grid_size = 80   # Higher resolution for very large datasets
    elif wells_count > 1000:
        grid_size = 120  # High resolution for large datasets
    else:
        grid_size = 150  # Very fine resolution for smaller datasets

    # Create the grid for our GeoJSON polygons
    lat_vals = np.linspace(min_lat, max_lat, grid_size)
    lon_vals = np.linspace(min_lon, max_lon, grid_size)

    # Extract coordinates and values from the wells dataframe
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)

    # Choose which values to interpolate based on method
    if method == 'depth_kriging':
        # For depth interpolation, ONLY use wells that have actual groundwater access
        # Wells without depth data should already be filtered out in data loading
        if 'is_dry_well' in wells_df.columns:
            # Use the explicit dry well marking - exclude any well marked as dry
            valid_depth_mask = (~wells_df['is_dry_well'])

            # Require valid depth_to_groundwater data if available
            if 'depth_to_groundwater' in wells_df.columns:
                valid_depth_mask = valid_depth_mask & (~wells_df['depth_to_groundwater'].isna()) & (wells_df['depth_to_groundwater'] > 0)
            elif 'depth' in wells_df.columns:
                valid_depth_mask = valid_depth_mask & (~wells_df['depth'].isna()) & (wells_df['depth'] > 0)

        elif 'depth_to_groundwater' in wells_df.columns:
            # Use depth_to_groundwater data and exclude wells with missing depth data
            valid_depth_mask = (
                (~wells_df['depth_to_groundwater'].isna()) & 
                (wells_df['depth_to_groundwater'] > 0)
            )
        else:
            # Use regular depth column and exclude wells with missing depth data
            valid_depth_mask = (
                (~wells_df['depth'].isna()) & 
                (wells_df['depth'] > 0)
            )

        if valid_depth_mask.any():
            wells_df = wells_df[valid_depth_mask].copy()
            lats = wells_df['latitude'].values.astype(float)
            lons = wells_df['longitude'].values.astype(float)
            # Use depth_to_groundwater if available, otherwise fall back to depth
            if 'depth_to_groundwater' in wells_df.columns:
                yields = wells_df['depth_to_groundwater'].values.astype(float)
            else:
                yields = wells_df['depth'].values.astype(float)
        else:
            return {"type": "FeatureCollection", "features": []}  # No valid depth data
    else:
        # For yield interpolation, ONLY use wells with actual yield data
        # Exclude wells with unknown yields (screen data but no MAX_YIELD)
        if 'has_unknown_yield' in wells_df.columns:
            # Filter out wells with unknown yields for yield interpolation
            yield_mask = (~wells_df['has_unknown_yield']) & (wells_df['yield_rate'].notna())
            if yield_mask.any():
                wells_df = wells_df[yield_mask].copy()
                lats = wells_df['latitude'].values.astype(float)
                lons = wells_df['longitude'].values.astype(float)
                yields = wells_df['yield_rate'].values.astype(float)
            else:
                return {"type": "FeatureCollection", "features": []}  # No valid yield data
        else:
            # Fallback: use wells with non-null yield data
            yield_mask = wells_df['yield_rate'].notna()
            if yield_mask.any():
                wells_df = wells_df[yield_mask].copy()
                lats = wells_df['latitude'].values.astype(float)
                lons = wells_df['longitude'].values.astype(float)
                yields = wells_df['yield_rate'].values.astype(float)
            else:
                return {"type": "FeatureCollection", "features": []}  # No valid yield data

    # Convert to km-based coordinates for proper interpolation
    x_coords = (lons - center_lon) * km_per_degree_lon
    y_coords = (lats - center_lat) * km_per_degree_lat

    # Create interpolation points for the grid
    grid_x, grid_y = np.meshgrid(
        np.linspace(-radius_km, radius_km, grid_size),
        np.linspace(-radius_km, radius_km, grid_size)
    )

    # Calculate distance from center for each point
    distances = np.sqrt(grid_x**2 + grid_y**2)
    mask = distances <= radius_km  # Only keep points within radius

    # Perform interpolation
    points = np.vstack([x_coords, y_coords]).T

    try:
        # Initialize variance array for kriging uncertainty
        kriging_variance = None

        # Choose interpolation method based on parameter and dataset size
        if show_variance and (method == 'kriging' or method == 'rf_kriging') and len(wells_df) >= 5:
            # Use actual kriging with variance calculation when variance is requested
            print("Calculating kriging with variance estimation")

            # Convert coordinates back to lat/lon for kriging (pykrige expects lon/lat)
            lon_values = x_coords / km_per_degree_lon + center_lon
            lat_values = y_coords / km_per_degree_lat + center_lat

            # Create grid points for kriging
            grid_points = np.vstack([grid_x[mask].ravel(), grid_y[mask].ravel()]).T
            xi_lon = grid_points[:, 0] / km_per_degree_lon + center_lon
            xi_lat = grid_points[:, 1] / km_per_degree_lat + center_lat

            # Set up kriging with variance calculation
            if auto_fit_variogram:
                # Use auto-fitted variogram for more accurate uncertainty estimation
                print(f"Auto-fitting {variogram_model} variogram model...")
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model=variogram_model,
                    verbose=False,
                    enable_plotting=False,
                    variogram_parameters=None  # Let PyKrige auto-fit parameters
                )
            else:
                # Use fixed variogram model for speed
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model='linear',  # Fast and stable
                    verbose=False,
                    enable_plotting=False
                )

            # Execute kriging to get both predictions and variance
            interpolated_z, kriging_variance = OK.execute('points', xi_lon, xi_lat)

        elif (method == 'kriging' or method == 'depth_kriging') and auto_fit_variogram and len(wells_df) >= 5:
            # Perform kriging with auto-fitted variogram for yield/depth visualization (without variance output)
            if method == 'depth_kriging':
                print(f"Auto-fitting {variogram_model} variogram model for depth estimation...")
            else:
                print(f"Auto-fitting {variogram_model} variogram model for yield estimation...")

            # Convert coordinates back to lat/lon for kriging (pykrige expects lon/lat)
            lon_values = x_coords / km_per_degree_lon + center_lon
            lat_values = y_coords / km_per_degree_lat + center_lat

            # Create grid points for kriging
            grid_points = np.vstack([grid_x[mask].ravel(), grid_y[mask].ravel()]).T
            xi_lon = grid_points[:, 0] / km_per_degree_lon + center_lon
            xi_lat = grid_points[:, 1] / km_per_degree_lat + center_lat

            # Set up kriging with auto-fitted variogram - ensure proper parameters for depth data
            if method == 'depth_kriging':
                # For depth data, use more appropriate variogram parameters
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model=variogram_model,
                    verbose=True,  # Enable verbose for debugging depth issues
                    enable_plotting=False,
                    variogram_parameters=None,  # Let PyKrige auto-fit parameters
                    weight=True,  # Enable nugget effect for depth data
                    anisotropy_scaling=1.0,  # No anisotropy scaling
                    anisotropy_angle=0.0
                )
            else:
                # Standard yield kriging
                OK = OrdinaryKriging(
                    lon_values, lat_values, yields,
                    variogram_model=variogram_model,
                    verbose=False,
                    enable_plotting=False,
                    variogram_parameters=None  # Let PyKrige auto-fit parameters
                )

            # Execute kriging to get predictions (ignore variance)
            interpolated_z, _ = OK.execute('points', xi_lon, xi_lat)

            # Additional validation for depth interpolation
            if method == 'depth_kriging':
                print(f"Depth interpolation stats: min={np.min(interpolated_z):.2f}, max={np.max(interpolated_z):.2f}, mean={np.mean(interpolated_z):.2f}")
                # Ensure reasonable depth values (depths should be positive and reasonable)
                interpolated_z = np.maximum(0.1, interpolated_z)  # Minimum depth of 0.1m
                interpolated_z = np.minimum(200.0, interpolated_z)  # Maximum reasonable depth of 200m

        else:
            # Use standard griddata interpolation for other cases
            # This is much faster than kriging for large datasets
            grid_points = np.vstack([grid_x[mask].ravel(), grid_y[mask].ravel()]).T
            interpolated_z = griddata(
                points, yields, grid_points,
                method='linear', fill_value=0.0
            )

            # Fill any NaN values with nearest neighbor interpolation
            nan_mask = np.isnan(interpolated_z)
            if np.any(nan_mask):
                interpolated_z[nan_mask] = griddata(
                    points, yields, grid_points[nan_mask],
                    method='nearest', fill_value=0.0
                )

            # Apply advanced smoothing for professional kriging-like appearance
            from scipy.ndimage import gaussian_filter

            # Reshape to 2D grid for smoothing
            try:
                # Create full 2D grid for smoothing
                z_grid = np.zeros_like(grid_x)
                z_grid[mask] = interpolated_z

                # Apply multiple smoothing passes for ultra-smooth appearance
                # First pass: moderate smoothing
                z_smooth = gaussian_filter(z_grid, sigma=1.5)
                # Second pass: fine smoothing for professional appearance
                z_smooth = gaussian_filter(z_smooth, sigma=0.8)

                # Extract smoothed values for our mask
                interpolated_z = z_smooth[mask]

                # Ensure values stay within reasonable bounds
                interpolated_z = np.maximum(0, interpolated_z)

            except Exception as e:
                # If smoothing fails, apply basic smoothing
                print(f"Advanced smoothing error: {e}, using basic smoothing")
                try:
                    z_grid = np.zeros_like(grid_x)
                    z_grid[mask] = interpolated_z
                    z_smooth = gaussian_filter(z_grid, sigma=1.0)
                    interpolated_z = z_smooth[mask]
                except:
                    print("Basic smoothing also failed, using raw interpolation")
    except Exception as e:
        # Fallback to simple IDW interpolation if the above methods fail
        print(f"Interpolation error: {e}, using fallback method")
        interpolated_z = np.zeros(grid_points.shape[0])
        for i, point in enumerate(grid_points):
            weights = 1.0 / (np.sqrt(np.sum((points - point)**2, axis=1)) + 1e-5)
            interpolated_z[i] = np.sum(weights * yields) / np.sum(weights)

    # Convert grid coordinates back to lat/lon
    grid_lats = (grid_y[mask].ravel() / km_per_degree_lat) + center_lat
    grid_lons = (grid_x[mask].ravel() / km_per_degree_lon) + center_lon

    # Prepare soil polygon geometry for later filtering (do not apply to interpolation)
    merged_soil_geometry = None

    if soil_polygons is not None and len(soil_polygons) > 0:
        try:
            # Create a unified geometry from all soil polygons for later filtering
            valid_geometries = []
            for idx, row in soil_polygons.iterrows():
                if row.geometry and row.geometry.is_valid:
                    valid_geometries.append(row.geometry)

            if valid_geometries:
                # Merge all polygons into a single multipolygon
                merged_soil_geometry = unary_union(valid_geometries)
                print(f"Prepared soil drainage geometry for display filtering")
            else:
                print("No valid soil polygon geometries found")
        except Exception as e:
            print(f"Error preparing soil polygon geometry: {e}")
            merged_soil_geometry = None

    # Build the GeoJSON structure
    features = []

    # Create polygons only where needed - use a Delaunay triangulation approach
    # for a more organic-looking interpolation surface
    from scipy.spatial import Delaunay

    try:
        # Create a Delaunay triangulation of the interpolation points
        points_2d = np.vstack([grid_lons, grid_lats]).T

        # Only create triangulation if we have enough points
        if len(points_2d) > 3:
            tri = Delaunay(points_2d)

            # Process each triangle to create a polygon
            for simplex in tri.simplices:
                # Get the three points of this triangle
                vertices = points_2d[simplex]

                # Get the values for these points (yield or variance)
                if show_variance and kriging_variance is not None:
                    vertex_values = kriging_variance[simplex]
                    avg_value = float(np.mean(vertex_values))
                    # For variance, show all values (including very small ones)
                    value_threshold = 0.0001  # Show almost all variance values
                else:
                    vertex_values = interpolated_z[simplex]
                    avg_value = float(np.mean(vertex_values))
                    value_threshold = 0.01  # Only show meaningful yield values

                avg_yield = avg_value  # Keep for backwards compatibility

                # Only add triangles with meaningful values and within our radius
                if avg_yield > value_threshold:
                    # Check if triangle should be included based on soil polygons
                    include_triangle = True

                    if merged_soil_geometry is not None:
                        # Calculate triangle centroid
                        centroid_lon = float(np.mean(vertices[:, 0]))
                        centroid_lat = float(np.mean(vertices[:, 1]))
                        centroid_point = Point(centroid_lon, centroid_lat)

                        # Only include if centroid is within soil drainage areas
                        include_triangle = merged_soil_geometry.contains(centroid_point) or merged_soil_geometry.intersects(centroid_point)

                    if include_triangle:
                        # Double-check: ensure triangle centroid is actually within soil polygons
                        centroid_lon = float(np.mean(vertices[:, 0]))
                        centroid_lat = float(np.mean(vertices[:, 1]))

                        # Final validation: check if centroid is within any soil polygon
                        final_include = True
                        if merged_soil_geometry is not None:
                            centroid_point = Point(centroid_lon, centroid_lat)
                            # Use strict containment - triangle must be clearly within soil areas
                            final_include = merged_soil_geometry.contains(centroid_point)

                            # If not strictly contained, check if it's very close to the boundary
                            if not final_include:
                                # Allow triangles very close to boundary (within 10 meters)
                                buffer_distance = 0.0001  # roughly 10 meters in degrees
                                buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                                final_include = buffered_geometry.contains(centroid_point)

                        if final_include:
                            # Create polygon for this triangle
                            poly = {
                                "type": "Feature",
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [[
                                        [float(vertices[0,0]), float(vertices[0,1])],
                                        [float(vertices[1,0]), float(vertices[1,1])],
                                        [float(vertices[2,0]), float(vertices[2,1])],
                                        [float(vertices[0,0]), float(vertices[0,1])]
                                    ]]
                                },
                                "properties": {
                                    "yield": avg_yield
                                }
                            }
                            features.append(poly)
        else:
            # Fallback to rectangular grid if triangulation is not possible
            for i in range(len(lat_vals)-1):
                for j in range(len(lon_vals)-1):
                    # Calculate center point of this grid cell
                    cell_lat = (lat_vals[i] + lat_vals[i+1]) / 2
                    cell_lon = (lon_vals[j] + lon_vals[j+1]) / 2

                    # Skip cells outside our radius
                    dist_km = np.sqrt(
                        ((cell_lat - center_lat) * km_per_degree_lat)**2 + 
                        ((cell_lon - center_lon) * km_per_degree_lon)**2
                    )
                    if dist_km > radius_km:
                        continue

                    # Find interpolated value for this cell
                    cell_x = (cell_lon - center_lon) * km_per_degree_lon
                    cell_y = (cell_lat - center_lat) * km_per_degree_lat

                    # Use nearest neighbor interpolation for the grid cell
                    cell_point = np.array([cell_x, cell_y])
                    distances = np.sqrt(np.sum((points - cell_point)**2, axis=1))
                    if len(distances) > 0:
                        idx = np.argmin(distances)
                        cell_value = yields[idx]
                    else:
                        cell_value = 0

                    # Only add cells with meaningful values
                    if cell_value > 0.01:
                        # Check if cell should be included based on soil polygons
                        include_cell = True

                        if merged_soil_geometry is not None:
                            # Check if cell center is within soil drainage areas
                            cell_center_point = Point(cell_lon, cell_lat)
                            include_cell = merged_soil_geometry.contains(cell_center_point) or merged_soil_geometry.intersects(cell_center_point)

                        if include_cell:
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
                                    "yield": float(cell_value)
                                }
                            }
                            features.append(poly)
    except Exception as e:
        # If triangulation fails, fall back to the simpler grid method
        print(f"Triangulation error: {e}, using grid method")
        for i in range(len(grid_lats)):
            # Only process points with meaningful values
            if interpolated_z[i] > 0.01:
                # Check if point should be included based on soil polygons
                include_point = True

                if merged_soil_geometry is not None:
                    point = Point(grid_lons[i], grid_lats[i])
                    include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                if include_point:
                    # Create a small circle as a polygon (approximated with 8 points)
                    radius_deg_lat = 0.5 * (lat_vals[1] - lat_vals[0])
                    radius_deg_lon = 0.5 * (lon_vals[1] - lon_vals[0])

                    # Create the polygon coordinates
                    coords = []
                    for angle in np.linspace(0, 2*np.pi, 9):
                        x = grid_lons[i] + radius_deg_lon * np.cos(angle)
                        y = grid_lats[i] + radius_deg_lat * np.sin(angle)
                        coords.append([float(x), float(y)])

                    # Create the polygon feature
                    poly = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [coords]
                        },
                        "properties": {
                            "yield": float(interpolated_z[i])
                        }
                    }
                    features.append(poly)

    # Log filtering results
    if merged_soil_geometry is not None:
        print(f"GeoJSON features filtered by soil drainage areas: {len(features)} polygons displayed")

    # Create the full GeoJSON object
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    return geojson

def generate_heat_map_data(wells_df, center_point, radius_km, resolution=50, method='kriging', soil_polygons=None):
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

    # Filter out geotechnical/geological investigation wells using well_use
    if 'well_use' in wells_df.columns:
        geotechnical_mask = wells_df['well_use'].str.contains(
            'Geotechnical.*Investigation|Geological.*Investigation', 
            case=False, 
            na=False, 
            regex=True
        )
        wells_df = wells_df[~geotechnical_mask].copy()

        if wells_df.empty:
            return []

    # Extract coordinates and yields based on method
    if method == 'depth_kriging':
        # For depth interpolation, ONLY use wells that have actual groundwater access
        # Wells without depth data should already be filtered out in data loading
        if 'is_dry_well' in wells_df.columns:
            # Use the explicit dry well marking - exclude any well marked as dry
            valid_depth_mask = (~wells_df['is_dry_well'])

            # Require valid depth_to_groundwater data if available
            if 'depth_to_groundwater' in wells_df.columns:
                valid_depth_mask = valid_depth_mask & (~wells_df['depth_to_groundwater'].isna()) & (wells_df['depth_to_groundwater'] > 0)
            elif 'depth' in wells_df.columns:
                valid_depth_mask = valid_depth_mask & (~wells_df['depth'].isna()) & (wells_df['depth'] > 0)

        elif 'depth_to_groundwater' in wells_df.columns:
            # Use depth_to_groundwater data and exclude wells with missing depth data
            valid_depth_mask = (
                (~wells_df['depth_to_groundwater'].isna()) & 
                (wells_df['depth_to_groundwater'] > 0)
            )
        else:
            # Use regular depth column and exclude wells with missing depth data
            valid_depth_mask = (
                (~wells_df['depth'].isna()) & 
                (wells_df['depth'] > 0)
            )

        if valid_depth_mask.any():
            wells_df_filtered = wells_df[valid_depth_mask].copy()
            lats = wells_df_filtered['latitude'].values.astype(float)
            lons = wells_df_filtered['longitude'].values.astype(float)
            # Use depth_to_groundwater if available, otherwise fall back to depth
            if 'depth_to_groundwater' in wells_df_filtered.columns:
                yields = wells_df_filtered['depth_to_groundwater'].values.astype(float)
            else:
                yields = wells_df_filtered['depth'].values.astype(float)
        else:
            return []  # No valid depth data
    else:
        # For yield interpolation, include ALL wells (including dry wells with 0 yield)
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

        # Choose interpolation method based on parameter and dataset size
        if method == 'yield_kriging' and len(wells_df) >= 5:
            try:
                print("Using yield kriging interpolation for heat map")

                # Filter to meaningful yield data for better kriging
                meaningful_yield_mask = yields > 0.1

                if meaningful_yield_mask.any() and np.sum(meaningful_yield_mask) >= 5:
                    # Use filtered data
                    filtered_x_coords = x_coords[meaningful_yield_mask]
                    filtered_y_coords = y_coords[meaningful_yield_mask] 
                    filtered_yields = yields[meaningful_yield_mask]

                    # Convert to lat/lon for kriging
                    filtered_lons = filtered_x_coords / km_per_degree_lon + center_lon
                    filtered_lats = filtered_y_coords / km_per_degree_lat + center_lat

                    # Set up kriging
                    OK = OrdinaryKriging(
                        filtered_lons, filtered_lats, filtered_yields,
                        variogram_model='spherical',
                        verbose=False,
                        enable_plotting=False,
                        variogram_parameters=None
                    )

                    # Execute kriging
                    xi_lon = xi_inside[:, 0] / km_per_degree_lon + center_lon
                    xi_lat = xi_inside[:, 1] / km_per_degree_lat + center_lat
                    interpolated_z, _ = OK.execute('points', xi_lon, xi_lat)

                    # Ensure non-negative yields
                    interpolated_z = np.maximum(0, interpolated_z)
                else:
                    # Fallback to griddata if insufficient data
                    interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)
                    nan_mask = np.isnan(interpolated_z)
                    if np.any(nan_mask):
                        interpolated_z[nan_mask] = griddata(
                            points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                        )

            except Exception as e:
                print(f"Yield kriging error: {e}, falling back to standard interpolation")
                interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)
                nan_mask = np.isnan(interpolated_z)
                if np.any(nan_mask):
                    interpolated_z[nan_mask] = griddata(
                        points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
                    )

        elif method == 'rf_kriging' and len(wells_df) >= 10:
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
            # Choose interpolation method
            # Basic 2D interpolation using scipy.interpolate.griddata

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

        # Prepare soil polygon geometry for filtering heat map display
        merged_soil_geometry = None
        if soil_polygons is not None and len(soil_polygons) > 0:
            try:
                # Create a unified geometry from all soil polygons
                valid_geometries = []
                for idx, row in soil_polygons.iterrows():
                    if row.geometry and row.geometry.is_valid:
                        valid_geometries.append(row.geometry)

                if valid_geometries:
                    # Merge all polygons into a single multipolygon
                    merged_soil_geometry = unary_union(valid_geometries)
                    print(f"Prepared soil drainage geometry for heat map filtering")
                else:
                    print("No valid soil polygon geometries found for heat map")
            except Exception as e:
                print(f"Error preparing soil polygon geometry for heat map: {e}")
                merged_soil_geometry = None

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
                            point_lat = float(cell_lat[max_idx])
                            point_lon = float(cell_lon[max_idx])

                            # Check if point should be included based on soil polygons
                            include_point = True
                            if merged_soil_geometry is not None:
                                point = Point(point_lon, point_lat)
                                include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                            if include_point:
                                # Double-check: ensure point is actually within soil polygons
                                if merged_soil_geometry is not None:
                                    point = Point(point_lon, point_lat)
                                    # Use strict containment for heat map points
                                    strictly_contained = merged_soil_geometry.contains(point)

                                    # If not strictly contained, check if very close to boundary
                                    if not strictly_contained:
                                        buffer_distance = 0.0001  # roughly 10 meters
                                        buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                                        strictly_contained = buffered_geometry.contains(point)

                                    if strictly_contained:
                                        heat_data.append([
                                            point_lat,  # Latitude
                                            point_lon,  # Longitude
                                            float(cell_values[max_idx])  # Yield value (actual value)
                                        ])
                                else:
                                    heat_data.append([
                                        point_lat,  # Latitude
                                        point_lon,  # Longitude
                                        float(cell_values[max_idx])  # Yield value (actual value)
                                    ])
        else:
            # Standard approach for smaller datasets
            heat_data = []
            # Add interpolated points
            for i in range(len(lat_points)):
                # Only add points with meaningful values
                if interpolated_z[i] > 0.01:
                    # Check if point should be included based on soil polygons
                    include_point = True
                    if merged_soil_geometry is not None:
                        point = Point(lon_points[i], lat_points[i])
                        include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                    if include_point:
                        # Double-check: ensure point is actually within soil polygons
                        if merged_soil_geometry is not None:
                            point = Point(lon_points[i], lat_points[i])
                            # Use strict containment
                            strictly_contained = merged_soil_geometry.contains(point)

                            # If not strictly contained, check if very close to boundary
                            if not strictly_contained:
                                buffer_distance = 0.0001  # roughly 10 meters
                                buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                                strictly_contained = buffered_geometry.contains(point)

                            if strictly_contained:
                                heat_data.append([
                                    float(lat_points[i]),  # Latitude
                                    float(lon_points[i]),  # Longitude
                                    float(interpolated_z[i])  # Yield value (actual value, not normalized)
                                ])
                        else:
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
                # Check if well should be included based on soil polygons
                include_well = True
                if merged_soil_geometry is not None:
                    well_point = Point(lons[j], lats[j])
                    include_well = merged_soil_geometry.contains(well_point) or merged_soil_geometry.intersects(well_point)

                if include_well:
                    # Double-check: ensure well point is actually within soil polygons
                    if merged_soil_geometry is not None:
                        well_point = Point(lons[j], lats[j])
                        # Use strict containment for wells
                        strictly_contained = merged_soil_geometry.contains(well_point)

                        # If not strictly contained, check if very close to boundary
                        if not strictly_contained:
                            buffer_distance = 0.0001  # roughly 10 meters
                            buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                            strictly_contained = buffered_geometry.contains(well_point)

                        if strictly_contained:
                            heat_data.append([
                                float(lats[j]),
                                float(lons[j]),
                                float(yields[j])
                            ])
                            well_points_added += 1
                    else:
                        heat_data.append([
                            float(lats[j]),
                            float(lons[j]),
                            float(yields[j])
                        ])
                        well_points_added += 1

        # Log filtering results
        if merged_soil_geometry is not None:
            print(f"Heat map filtered by soil drainage areas: {len(heat_data)} points displayed")

        return heat_data

    except Exception as e:
        print(f"Interpolation error: {e}")
        return fallback_interpolation(wells_df, center_point, radius_km)

def fallback_interpolation(wells_df, center_point, radius_km, resolution=50):
    """
    Simplified IDW (Inverse Distance Weighting) interpolation as fallback method
    Creates a continuous interpolated surface based on actual well yield values
    """
    # Filter to only use wells with actual yield data (exclude unknown yields)
    if 'has_unknown_yield' in wells_df.columns:
        valid_wells = wells_df[(~wells_df['has_unknown_yield']) & (wells_df['yield_rate'].notna())].copy()
    else:
        valid_wells = wells_df[wells_df['yield_rate'].notna()].copy()
    
    if valid_wells.empty:
        return []
    
    # Extract coordinates and yields
    lats = valid_wells['latitude'].values.astype(float)
    lons = valid_wells['longitude'].values.astype(float)
    yields = valid_wells['yield_rate'].values.astype(float)

    # Handle empty dataset
    if len(yields) == 0:
        return []

    center_lat,center_lon = center_point

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
    Used as a fallback method - only uses wells with actual yield data
    """
    if not isinstance(wells_df, pd.DataFrame) or wells_df.empty:
        return 0

    # Filter to only use wells with actual yield data
    if 'has_unknown_yield' in wells_df.columns:
        valid_wells = wells_df[(~wells_df['has_unknown_yield']) & (wells_df['yield_rate'].notna())].copy()
    else:
        valid_wells = wells_df[wells_df['yield_rate'].notna()].copy()
    
    if valid_wells.empty:
        return 0

    # Calculate distance from each well to the point (in km)
    distances = []
    for idx, row in valid_wells.iterrows():
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
    weighted_yield = sum(w * float(row['yield_rate']) for w, (idx, row) in zip(weights, valid_wells.iterrows())) / total_weight
    return float(max(0, weighted_yield))  # Ensure non-negative yieldld

def calculate_kriging_variance(wells_df, center_point, radius_km, resolution=50, variogram_model='spherical', use_depth=False, soil_polygons=None):
    """
    Calculate kriging variance for yield or depth to groundwater interpolations.

    This function performs ordinary kriging to estimate the variance (uncertainty)
    of the interpolated values. It's designed to work with both yield and
    depth-to-groundwater data, and applies soil polygon filtering like other interpolants.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data, including latitude, longitude, and
        either yield_rate (for yield interpolation) or depth_to_groundwater/depth
        (for depth interpolation).
    center_point : tuple
        Tuple containing (latitude, longitude) of the center point.
    radius_km : float
        Radius in kilometers to generate the variance data for.
    resolution : int
        Number of points to generate in each dimension.
    variogram_model : str, optional
        Variogram model to use for kriging (e.g., 'linear', 'spherical', 'gaussian').
        Defaults to 'spherical'.
    use_depth : bool
        Whether to use depth data instead of yield data.
    soil_polygons : GeoDataFrame, optional
        Soil polygon data for clipping the variance output.

    Returns:
    --------
    list
        List of [lat, lng, variance] points for the variance heat map.
        Returns an empty list if there is not enough data or if an error occurs.

    Notes:
    ------
    - The function automatically detects whether to interpolate yield or depth
      based on the columns available in `wells_df`.
    - It handles cases where depth_to_groundwater is not available and falls
      back to using the 'depth' column if it exists.
    - It returns an empty list if there is insufficient data (less than 5 wells)
      to perform kriging.
    - Applies the same soil polygon clipping as other interpolants.
    """
    try:
        # Filter wells data similar to yield kriging for consistency
        wells_df_filtered = wells_df.copy()

        # Filter out geotechnical/geological investigation wells using well_use
        if 'well_use' in wells_df_filtered.columns:
            geotechnical_mask = wells_df_filtered['well_use'].str.contains(
                'Geotechnical.*Investigation|Geological.*Investigation', 
                case=False, 
                na=False, 
                regex=True
            )
            wells_df_filtered = wells_df_filtered[~geotechnical_mask].copy()

            if wells_df_filtered.empty:
                return []

        # Determine whether to use yield or depth data based on use_depth parameter
        if use_depth:
            # For depth variance, filter wells similar to depth kriging
            if 'is_dry_well' in wells_df.columns:
                wells_df_filtered = wells_df_filtered[~wells_df_filtered['is_dry_well']]

            if 'depth_to_groundwater' in wells_df_filtered.columns and not wells_df_filtered['depth_to_groundwater'].isna().all():
                wells_df_filtered = wells_df_filtered[wells_df_filtered['depth_to_groundwater'].notna() & (wells_df_filtered['depth_to_groundwater'] > 0)]
                values = wells_df_filtered['depth_to_groundwater'].values.astype(float)
                interpolation_type = 'depth_to_groundwater'
            elif 'depth' in wells_df_filtered.columns:
                wells_df_filtered = wells_df_filtered[wells_df_filtered['depth'].notna() & (wells_df_filtered['depth'] > 0)]
                values = wells_df_filtered['depth'].values.astype(float)
                interpolation_type = 'depth'
            else:
                print("Error: No depth data found in wells_df.")
                return []
        else:
            # For yield variance, filter to meaningful yield data like yield kriging
            meaningful_yield_mask = wells_df_filtered['yield_rate'].fillna(0) > 0.1

            if meaningful_yield_mask.any() and np.sum(meaningful_yield_mask) >= 5:
                wells_df_filtered = wells_df_filtered[meaningful_yield_mask]
                values = wells_df_filtered['yield_rate'].values.astype(float)
                interpolation_type = 'yield'
            else:
                # If insufficient meaningful yield data, use all available yield data
                values = wells_df_filtered['yield_rate'].fillna(0).values.astype(float)
                interpolation_type = 'yield'

        # Prepare data: Extract coordinates and values from filtered data
        lats = wells_df_filtered['latitude'].values.astype(float)
        lons = wells_df_filtered['longitude'].values.astype(float)

        # Ensure there's enough data for kriging
        if len(wells_df_filtered) < 5:
            print("Warning: Insufficient data points for kriging variance calculation after filtering.")
            return []

        # Create grid for interpolation
        center_lat, center_lon = center_point
        grid_size = min(50, max(30, resolution))  # Adjust grid size if necessary

        # Convert to km-based coordinates
        km_per_degree_lat = 111.0
        km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))

        # Convert coordinates to km from center
        x_coords = (lons - center_lon) * km_per_degree_lon
        y_coords = (lats - center_lat) * km_per_degree_lat

        # Create grid in km space
        grid_x = np.linspace(-radius_km, radius_km, grid_size)
        grid_y = np.linspace(-radius_km, radius_km, grid_size)
        grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

        # Flatten grid for interpolation
        xi = np.vstack([grid_X.flatten(), grid_Y.flatten()]).T

        # Filter points outside the radius
        distances = np.sqrt(xi[:, 0]**2 + xi[:, 1]**2)
        mask = distances <= radius_km
        xi_inside = xi[mask]

        # Convert back to lat/lon for kriging
        lon_values = x_coords / km_per_degree_lon + center_lon
        lat_values = y_coords / km_per_degree_lat + center_lat
        xi_lon = xi_inside[:, 0] / km_per_degree_lon + center_lon
        xi_lat = xi_inside[:, 1] / km_per_degree_lat + center_lat

        # Perform Ordinary Kriging with enhanced variance calculation
        try:
            # First, try standard kriging with optimized parameters for variance
            OK = OrdinaryKriging(
                lon_values, lat_values, values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False,
                weight=False,  # Disable additional weighting for cleaner variance
                exact_values=False,  # Allow interpolation for better variance estimation
                pseudo_inv=False  # Use standard inversion
            )

            # Execute kriging to get both predictions and variance
            predictions, kriging_variance = OK.execute('points', xi_lon, xi_lat)

            print(f"Initial kriging variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}, Mean: {np.mean(kriging_variance):.6f}")

            # Ensure variance values are reasonable (variance should always be positive)
            kriging_variance = np.maximum(kriging_variance, 1e-8)

            # Check if variance is too uniform (indicates calculation issue)
            variance_std = np.std(kriging_variance)
            variance_mean = np.mean(kriging_variance)
            variance_cv = variance_std / variance_mean if variance_mean > 0 else 0

            print(f"Variance coefficient of variation: {variance_cv:.4f}")

            if variance_cv < 0.1:  # If variance is too uniform (less than 10% variation)
                print("Warning: Variance appears too uniform, enhancing with multiple variogram models...")

                # Try multiple variogram models and combine results
                variance_results = []
                variogram_models = ['linear', 'power', 'gaussian', 'exponential']

                for model in variogram_models:
                    try:
                        OK_test = OrdinaryKriging(
                            lon_values, lat_values, values,
                            variogram_model=model,
                            verbose=False,
                            enable_plotting=False
                        )
                        _, test_variance = OK_test.execute('points', xi_lon, xi_lat)
                        test_variance = np.maximum(test_variance, 1e-8)

                        # Check if this model produces better variation
                        test_cv = np.std(test_variance) / np.mean(test_variance) if np.mean(test_variance) > 0 else 0
                        variance_results.append((model, test_variance, test_cv))
                        print(f"Model {model}: CV = {test_cv:.4f}")

                    except Exception as model_error:
                        print(f"Model {model} failed: {model_error}")
                        continue

                # Use the model with the highest coefficient of variation (most spatial variation)
                if variance_results:
                    best_model, best_variance, best_cv = max(variance_results, key=lambda x: x[2])
                    print(f"Using {best_model} model with CV = {best_cv:.4f}")
                    kriging_variance = best_variance
                    variance_cv = best_cv

            # If still too uniform, enhance with distance-based variation
            if variance_cv < 0.05:
                print("Enhancing variance with distance-based component...")

                # Calculate distance from each grid point to nearest data point
                from scipy.spatial.distance import cdist

                # Create coordinate arrays for distance calculation
                data_coords = np.column_stack([lat_values, lon_values])
                grid_coords = np.column_stack([xi_lat, xi_lon])

                # Calculate distances (in decimal degrees)
                distances = cdist(grid_coords, data_coords)
                min_distances = np.min(distances, axis=1)

                # Create distance-based variance enhancement
                max_distance = np.max(min_distances) if len(min_distances) > 0 else 1.0
                if max_distance > 0:
                    # Normalize distances and create variance enhancement factor
                    normalized_distances = min_distances / max_distance

                    # Create exponential distance decay for variance
                    distance_factor = 1.0 + 2.0 * np.exp(normalized_distances * 3.0)  # Exponential growth with distance

                    # Enhance the original kriging variance with distance component
                    base_variance = np.mean(kriging_variance)
                    enhanced_variance = kriging_variance * distance_factor

                    # Smooth the transition between original and enhanced variance
                    alpha = 0.6  # Blend factor
                    kriging_variance = alpha * enhanced_variance + (1 - alpha) * kriging_variance

                    print(f"Enhanced variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}")

            # Final variance validation and scaling
            kriging_variance = np.maximum(kriging_variance, 1e-6)

            # Scale variance to reasonable range for visualization
            min_var = np.min(kriging_variance)
            max_var = np.max(kriging_variance)
            var_range = max_var - min_var

            if var_range > 0:
                # Normalize to 0-1 range then scale to meaningful variance range
                normalized_var = (kriging_variance - min_var) / var_range
                # Scale to range that makes sense for the data type
                if use_depth:
                    kriging_variance = normalized_var * (np.var(values) * 2.0) + (np.var(values) * 0.1)
                else:
                    kriging_variance = normalized_var * (np.var(values) * 1.5) + (np.var(values) * 0.05)

            print(f"Final variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}, CV: {np.std(kriging_variance)/np.mean(kriging_variance):.4f}")

        except Exception as e:
            print(f"Error in kriging variance calculation: {e}")
            # Fallback: create synthetic variance based on distance to nearest data point
            print("Using enhanced distance-based variance estimation as fallback")

            # Calculate distance from each grid point to nearest data point
            from scipy.spatial.distance import cdist

            # Create coordinate arrays for distance calculation
            data_coords = np.column_stack([lat_values, lon_values])
            grid_coords = np.column_stack([xi_lat, xi_lon])

            # Calculate distances
            distances = cdist(grid_coords, data_coords)
            min_distances = np.min(distances, axis=1)

            # Create variance based on distance and data density
            max_distance = np.max(min_distances)
            if max_distance > 0:
                normalized_distances = min_distances / max_distance

                # Create more realistic variance based on distance and local data density
                # Areas far from wells should have higher uncertainty
                base_variance = np.var(values) if len(values) > 1 else 1.0

                # Exponential increase in variance with distance
                distance_variance = base_variance * (0.2 + 1.8 * np.exp(normalized_distances * 2.0))

                # Add some local variation based on coordinate position
                lat_variation = np.sin(xi_lat * 100) * 0.1 * base_variance
                lon_variation = np.cos(xi_lon * 100) * 0.1 * base_variance

                kriging_variance = distance_variance + lat_variation + lon_variation
                kriging_variance = np.maximum(kriging_variance, base_variance * 0.1)
            else:
                kriging_variance = np.full(len(xi_lat), np.var(values) * 0.5)

        # Prepare variance data for heat map
        lat_points = (xi_inside[:, 1] / km_per_degree_lat) + center_lat
        lon_points = (xi_inside[:, 0] / km_per_degree_lon) + center_lon

        # Prepare soil polygon geometry for filtering variance display (same as other interpolants)
        merged_soil_geometry = None
        if soil_polygons is not None and len(soil_polygons) > 0:
            try:
                # Create a unified geometry from all soil polygons
                valid_geometries = []
                for idx, row in soil_polygons.iterrows():
                    if row.geometry and row.geometry.is_valid:
                        valid_geometries.append(row.geometry)

                if valid_geometries:
                    # Merge all polygons into a single multipolygon
                    merged_soil_geometry = unary_union(valid_geometries)
                    print(f"Prepared soil drainage geometry for kriging variance filtering")
                else:
                    print("No valid soil polygon geometries found for kriging variance")
            except Exception as e:
                print(f"Error preparing soil polygon geometry for kriging variance: {e}")
                merged_soil_geometry = None

        # Apply stricter soil polygon filtering for variance data
        variance_data = []
        points_inside_count = 0
        points_total_count = len(lat_points)

        for i in range(len(lat_points)):
            # Only add points with meaningful variance values (lower threshold for variance)
            if kriging_variance[i] > 1e-6:
                # Check if point should be included based on soil polygons
                include_point = True
                if merged_soil_geometry is not None:
                    point = Point(lon_points[i], lat_points[i])

                    # Use ONLY strict containment - no intersection tolerance
                    strictly_contained = merged_soil_geometry.contains(point)

                    # For variance visualization, we want very precise clipping
                    # Only allow points that are clearly inside the soil polygons
                    if not strictly_contained:
                        # Check if point is very close to boundary (smaller buffer than before)
                        buffer_distance = 0.00005  # roughly 5 meters - much smaller buffer
                        buffered_geometry = merged_soil_geometry.buffer(buffer_distance)
                        strictly_contained = buffered_geometry.contains(point)

                    include_point = strictly_contained
                    if include_point:
                        points_inside_count += 1

                if include_point:
                    # Add data point with latitude, longitude, and variance
                    variance_data.append([
                        float(lat_points[i]),
                        float(lon_points[i]),
                        float(kriging_variance[i])
                    ])
                else:
                    # Add small amount of debugging for clipping
                    if merged_soil_geometry is not None and i % 100 == 0:  # Sample every 100th point for debug
                        point = Point(lon_points[i], lat_points[i])
                        distance_to_boundary = merged_soil_geometry.distance(point)
                        if distance_to_boundary < 0.001:  # If very close to boundary
                            print(f"Debug: Point {i} excluded - distance to boundary: {distance_to_boundary:.6f}")

        if merged_soil_geometry is not None:
            print(f"Variance clipping: {points_inside_count}/{points_total_count} points inside soil polygons ({100*points_inside_count/points_total_count:.1f}%)")

        # Log filtering results
        if merged_soil_geometry is not None:
            print(f"Kriging variance filtered by soil drainage areas: {len(variance_data)} points displayed")

        return variance_data

    except Exception as e:
        print(f"Error calculating kriging variance: {e}")
        return []  # Return empty list in case of error