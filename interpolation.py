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
        # For depth interpolation, use wells that have valid depth to groundwater measurements
        if 'depth_to_groundwater' in wells_df.columns:
            valid_depth_mask = wells_df['depth_to_groundwater'].notna() & (wells_df['depth_to_groundwater'] > 0)
            if valid_depth_mask.any():
                wells_df = wells_df[valid_depth_mask].copy()
                lats = wells_df['latitude'].values.astype(float)
                lons = wells_df['longitude'].values.astype(float)
                yields = wells_df['depth_to_groundwater'].values.astype(float)
            else:
                return {"type": "FeatureCollection", "features": []}
        else:
            # Fallback to regular depth column if depth_to_groundwater not available
            valid_depth_mask = wells_df['depth'].notna() & (wells_df['depth'] > 0)
            if valid_depth_mask.any():
                wells_df = wells_df[valid_depth_mask].copy()
                lats = wells_df['latitude'].values.astype(float)
                lons = wells_df['longitude'].values.astype(float)
                yields = wells_df['depth'].values.astype(float)
            else:
                return {"type": "FeatureCollection", "features": []}
    else:
        # For yield interpolation, use ALL wells (including dry wells with 0 yield)
        yields = wells_df['yield_rate'].fillna(0).values.astype(float)

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
        # Choose interpolation method based on parameter and dataset size
        if (method == 'kriging' or method == 'depth_kriging') and auto_fit_variogram and len(wells_df) >= 5:
            # Perform kriging with auto-fitted variogram
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

            # Set up kriging with auto-fitted variogram
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
                z_smooth = gaussian_filter(z_grid, sigma=1.5)
                z_smooth = gaussian_filter(z_smooth, sigma=0.8)

                # Extract smoothed values for our mask
                interpolated_z = z_smooth[mask]

                # Ensure values stay within reasonable bounds
                interpolated_z = np.maximum(0, interpolated_z)

            except Exception as e:
                print(f"Smoothing error: {e}, using basic smoothing")
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

    # Prepare soil polygon geometry for later filtering
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

    # Build the GeoJSON structure using Delaunay triangulation
    features = []
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
                vertex_values = interpolated_z[simplex]
                avg_value = float(np.mean(vertex_values))

                # Only add triangles with meaningful values
                if avg_value > 0.01:
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
                                "yield": avg_value
                            }
                        }
                        features.append(poly)

    except Exception as e:
        print(f"Triangulation error: {e}, using grid method")
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
        # For depth interpolation, use wells that have valid depth to groundwater measurements
        if 'depth_to_groundwater' in wells_df.columns:
            valid_depth_mask = wells_df['depth_to_groundwater'].notna() & (wells_df['depth_to_groundwater'] > 0)
            if valid_depth_mask.any():
                wells_df_filtered = wells_df[valid_depth_mask].copy()
                lats = wells_df_filtered['latitude'].values.astype(float)
                lons = wells_df_filtered['longitude'].values.astype(float)
                yields = wells_df_filtered['depth_to_groundwater'].values.astype(float)
            else:
                return []
        else:
            # Fallback to regular depth column if depth_to_groundwater not available
            valid_depth_mask = wells_df['depth'].notna() & (wells_df['depth'] > 0)
            if valid_depth_mask.any():
                wells_df_filtered = wells_df[valid_depth_mask].copy()
                lats = wells_df_filtered['latitude'].values.astype(float)
                lons = wells_df_filtered['longitude'].values.astype(float)
                yields = wells_df_filtered['depth'].values.astype(float)
            else:
                return []
    else:
        # For yield interpolation, include ALL wells (including dry wells with 0 yield)
        lats = wells_df['latitude'].values.astype(float)
        lons = wells_df['longitude'].values.astype(float)
        yields = wells_df['yield_rate'].fillna(0).values.astype(float)

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
        # Convert to km-based coordinates
        km_per_degree_lat = 111.0
        km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))

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

        # Use standard griddata interpolation
        interpolated_z = griddata(points, yields, xi_inside, method='linear', fill_value=0.0)

        # Fill any NaN values with nearest neighbor interpolation
        nan_mask = np.isnan(interpolated_z)
        if np.any(nan_mask):
            interpolated_z[nan_mask] = griddata(
                points, yields, xi_inside[nan_mask], method='nearest', fill_value=0.0
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
                    heat_data.append([
                        float(lat_points[i]),
                        float(lon_points[i]),
                        float(interpolated_z[i])
                    ])

        # Always make sure well points themselves are included for accuracy
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
    """
    # Extract coordinates and yields
    lats = wells_df['latitude'].values.astype(float)
    lons = wells_df['longitude'].values.astype(float)
    yields = wells_df['yield_rate'].fillna(0).values.astype(float)

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
                interpolated_value = 0.0

        # Add to heat map with interpolated value
        if interpolated_value > 0.01:
            heat_data.append([
                float(grid_point_lat), 
                float(grid_point_lon),
                float(interpolated_value)
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
                float(yields[j])
            ])

    return heat_data

def calculate_kriging_variance(wells_df, center_point, radius_km, resolution=50, variogram_model='spherical', use_depth=False, soil_polygons=None):
    """
    Calculate kriging variance for yield or depth to groundwater interpolations
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
            # For depth variance, use wells with valid depth to groundwater measurements
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
        grid_size = min(50, max(30, resolution))

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

        # Perform Ordinary Kriging with variance calculation
        try:
            OK = OrdinaryKriging(
                lon_values, lat_values, values,
                variogram_model=variogram_model,
                verbose=False,
                enable_plotting=False
            )

            # Execute kriging to get both predictions and variance
            predictions, kriging_variance = OK.execute('points', xi_lon, xi_lat)

            print(f"Kriging variance - Min: {np.min(kriging_variance):.6f}, Max: {np.max(kriging_variance):.6f}, Mean: {np.mean(kriging_variance):.6f}")

            # Ensure variance values are reasonable
            kriging_variance = np.maximum(kriging_variance, 1e-8)

        except Exception as e:
            print(f"Error in kriging variance calculation: {e}")
            # Fallback: create synthetic variance based on distance to nearest data point
            from scipy.spatial.distance import cdist

            # Create coordinate arrays for distance calculation
            data_coords = np.column_stack([lat_values, lon_values])
            grid_coords = np.column_stack([xi_lat, xi_lon])

            # Calculate distances
            distances = cdist(grid_coords, data_coords)
            min_distances = np.min(distances, axis=1)

            # Create variance based on distance
            max_distance = np.max(min_distances)
            if max_distance > 0:
                normalized_distances = min_distances / max_distance
                base_variance = np.var(values) if len(values) > 1 else 1.0
                kriging_variance = base_variance * (0.2 + 1.8 * np.exp(normalized_distances * 2.0))
                kriging_variance = np.maximum(kriging_variance, base_variance * 0.1)
            else:
                kriging_variance = np.full(len(xi_lat), np.var(values) * 0.5)

        # Prepare variance data for heat map
        lat_points = (xi_inside[:, 1] / km_per_degree_lat) + center_lat
        lon_points = (xi_inside[:, 0] / km_per_degree_lon) + center_lon

        # Apply soil polygon filtering if available
        variance_data = []
        for i in range(len(lat_points)):
            if kriging_variance[i] > 1e-6:
                variance_data.append([
                    float(lat_points[i]),
                    float(lon_points[i]),
                    float(kriging_variance[i])
                ])

        return variance_data

    except Exception as e:
        print(f"Error calculating kriging variance: {e}")
        return []

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