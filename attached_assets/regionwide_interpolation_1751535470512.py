import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd
from interpolation import generate_geo_json_grid  # Import from existing geology.txt
import warnings

def generate_regionwide_gwl_interpolation(wells_df, bounds, radius_km=15, grid_spacing_km=10, resolution=50, variogram_model='spherical', soil_polygons=None):
    """
    Generate a region-wide groundwater level interpolation by tiling the region with overlapping sub-regions.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with 'latitude', 'longitude', and 'ground water level' columns.
    bounds : dict
        Dictionary with 'north', 'south', 'east', 'west' keys defining the region bounds in decimal degrees.
    radius_km : float
        Radius for each local interpolation (default: 15 km).
    grid_spacing_km : float
        Spacing between grid points in kilometers (default: 10 km for overlap).
    resolution : int
        Grid resolution for each sub-region interpolation (default: 50).
    variogram_model : str
        Variogram model for kriging (default: 'spherical').
    soil_polygons : GeoDataFrame, optional
        Soil polygons for filtering interpolation results.

    Returns:
    --------
    dict
        GeoJSON FeatureCollection with interpolated groundwater level values across the region.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Initialize GeoJSON structure
    geojson = {"type": "FeatureCollection", "features": []}

    # Handle empty or invalid input
    if wells_df is None or wells_df.empty:
        print("Error: Empty or invalid wells DataFrame")
        return geojson

    # Validate bounds
    if not all(key in bounds for key in ['north', 'south', 'east', 'west']):
        print("Error: Invalid bounds. Must include 'north', 'south', 'east', 'west'.")
        return geojson

    # Calculate km per degree for the region's approximate center
    center_lat = (bounds['north'] + bounds['south']) / 2
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))

    # Convert bounds to km for grid creation
    min_x = (bounds['west'] - bounds['west']) * km_per_degree_lon  # Reference to west
    max_x = (bounds['east'] - bounds['west']) * km_per_degree_lon
    min_y = (bounds['south'] - bounds['south']) * km_per_degree_lat  # Reference to south
    max_y = (bounds['north'] - bounds['south']) * km_per_degree_lat

    # Create grid points for interpolation centers
    x_points = np.arange(min_x, max_x + grid_spacing_km, grid_spacing_km)
    y_points = np.arange(min_y, max_y + grid_spacing_km, grid_spacing_km)
    grid_x, grid_y = np.meshgrid(x_points, y_points)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Convert grid points back to lat/lon
    grid_lons = grid_points[:, 0] / km_per_degree_lon + bounds['west']
    grid_lats = grid_points[:, 1] / km_per_degree_lat + bounds['south']

    print(f"Generated {len(grid_lons)} grid points for interpolation")

    # Prepare soil polygons if provided
    merged_soil_geometry = None
    if soil_polygons is not None and not soil_polygons.empty:
        try:
            valid_geometries = [geom for geom in soil_polygons.geometry if geom and geom.is_valid]
            if valid_geometries:
                merged_soil_geometry = unary_union(valid_geometries)
                print("Prepared soil drainage geometry for filtering")
            else:
                print("No valid soil polygon geometries found")
        except Exception as e:
            print(f"Error preparing soil polygons: {e}")
            merged_soil_geometry = None

    # Initialize list to store all features
    all_features = []
    point_values = []  # Store points and values for final triangulation

    # Filter wells with valid ground water level data once
    valid_wells = wells_df[
        wells_df['ground water level'].notna() &
        (wells_df['ground water level'] != 0) &
        (wells_df['ground water level'].abs() > 0.1)
    ].copy()

    if valid_wells.empty:
        print("No valid ground water level data found")
        return geojson

    print(f"Using {len(valid_wells)} wells with valid ground water level data")

    # Process each grid point
    for idx, (center_lon, center_lat) in enumerate(zip(grid_lons, grid_lats)):
        if idx % 10 == 0:  # Progress update every 10 points
            print(f"Processing grid point {idx + 1}/{len(grid_lons)} at ({center_lat:.4f}, {center_lon:.4f})")

        # Filter wells within 15km radius
        wells_df['distance'] = wells_df.apply(
            lambda row: np.sqrt(
                ((row['latitude'] - center_lat) * km_per_degree_lat)**2 +
                ((row['longitude'] - center_lon) * km_per_degree_lon)**2
            ),
            axis=1
        )
        local_wells = valid_wells[valid_wells['distance'] <= radius_km].copy()

        if len(local_wells) < 5:
            print(f"Skipping grid point {idx + 1}: only {len(local_wells)} wells found")
            continue

        # Generate interpolation for this sub-region
        try:
            local_geojson = generate_geo_json_grid(
                local_wells,
                center_point=(center_lat, center_lon),
                radius_km=radius_km,
                resolution=resolution,
                method='ground_water_level_kriging',
                show_variance=False,
                auto_fit_variogram=True,
                variogram_model=variogram_model,
                soil_polygons=soil_polygons if merged_soil_geometry is not None else None
            )

            # Collect features and points for final triangulation
            for feature in local_geojson['features']:
                # Extract centroid of each polygon for final triangulation
                coords = feature['geometry']['coordinates'][0]
                centroid_x = np.mean([p[0] for p in coords[:-1]])  # Exclude closing point
                centroid_y = np.mean([p[1] for p in coords[:-1]])
                value = feature['properties']['yield']

                # Check if point is within bounds and soil polygons (if applicable)
                include_point = True
                if merged_soil_geometry is not None:
                    point = Point(centroid_x, centroid_y)
                    include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                if include_point and value > 0.1:  # Only include significant values
                    point_values.append([centroid_x, centroid_y, value])
                    all_features.append(feature)

        except Exception as e:
            print(f"Error processing grid point {idx + 1}: {e}")
            continue

    # Remove duplicate points based on proximity (within 100m)
    if point_values:
        point_values = np.array(point_values)
        unique_points = []
        unique_values = []

        for i, (x, y, val) in enumerate(point_values):
            keep = True
            for j, (ux, uy, _) in enumerate(unique_points):
                dist_km = np.sqrt(
                    ((x - ux) * km_per_degree_lon)**2 +
                    ((y - uy) * km_per_degree_lat)**2
                )
                if dist_km < 0.1:  # 100m threshold
                    # Average values for close points
                    unique_values[j] = (unique_values[j] + val) / 2
                    keep = False
                    break
            if keep:
                unique_points.append([x, y])
                unique_values.append(val)

        point_values = np.array(unique_points)
        unique_values = np.array(unique_values)

        # Perform Delaunay triangulation for final surface
        try:
            if len(point_values) > 3:
                tri = Delaunay(point_values)
                final_features = []

                for simplex in tri.simplices:
                    vertices = point_values[simplex]
                    vertex_values = unique_values[simplex]
                    avg_value = float(np.mean(vertex_values))

                    if avg_value > 0.1:  # Only include significant values
                        poly = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [float(vertices[0, 0]), float(vertices[0, 1])],
                                    [float(vertices[1, 0]), float(vertices[1, 1])],
                                    [float(vertices[2, 0]), float(vertices[2, 1])],
                                    [float(vertices[0, 0]), float(vertices[0, 1])]
                                ]]
                            },
                            "properties": {
                                "yield": avg_value
                            }
                        }
                        final_features.append(poly)

                geojson['features'] = final_features
                print(f"Generated {len(final_features)} final GeoJSON features")
            else:
                print("Insufficient points for final triangulation, using raw features")
                geojson['features'] = all_features
        except Exception as e:
            print(f"Final triangulation error: { radius_km=15, grid_spacing_km=10, resolution=50, variogram_model='spherical', soil_polygons=None)
    """
    Generate a region-wide groundwater level interpolation by tiling the region with overlapping sub-regions.

    Parameters:
    -----------
    wells_df : DataFrame
        DataFrame containing well data with 'latitude', 'longitude', and 'ground water level' columns.
    bounds : dict
        Dictionary with 'north', 'south', 'east', 'west' keys defining the region bounds in decimal degrees.
    radius_km : float
        Radius for each local interpolation (default: 15 km).
    grid_spacing_km : float
        Spacing between grid points in kilometers (default: 10 km for overlap).
    resolution : int
        Grid resolution for each sub-region interpolation (default: 50).
    variogram_model : str
        Variogram model for kriging (default: 'spherical').
    soil_polygons : GeoDataFrame, optional
        Soil polygons for filtering interpolation results.

    Returns:
    --------
    dict
        GeoJSON FeatureCollection with interpolated groundwater level values across the region.
    """
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

    # Initialize GeoJSON structure
    geojson = {"type": "FeatureCollection", "features": []}

    # Handle empty or invalid input
    if wells_df is None or wells_df.empty:
        print("Error: Empty or invalid wells DataFrame")
        return geojson

    # Validate bounds
    if not all(key in bounds for key in ['north', 'south', 'east', 'west']):
        print("Error: Invalid bounds. Must include 'north', 'south', 'east', 'west'.")
        return geojson

    # Calculate km per degree for the region's approximate center
    center_lat = (bounds['north'] + bounds['south']) / 2
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))

    # Convert bounds to km for grid creation
    min_x = (bounds['west'] - bounds['west']) * km_per_degree_lon  # Reference to west
    max_x = (bounds['east'] - bounds['west']) * km_per_degree_lon
    min_y = (bounds['south'] - bounds['south']) * km_per_degree_lat  # Reference to south
    max_y = (bounds['north'] - bounds['south']) * km_per_degree_lat

    # Create grid points for interpolation centers
    x_points = np.arange(min_x, max_x + grid_spacing_km, grid_spacing_km)
    y_points = np.arange(min_y, max_y + grid_spacing_km, grid_spacing_km)
    grid_x, grid_y = np.meshgrid(x_points, y_points)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Convert grid points back to lat/lon
    grid_lons = grid_points[:, 0] / km_per_degree_lon + bounds['west']
    grid_lats = grid_points[:, 1] / km_per_degree_lat + bounds['south']

    print(f"Generated {len(grid_lons)} grid points for interpolation")

    # Prepare soil polygons if provided
    merged_soil_geometry = None
    if soil_polygons is not None and not soil_polygons.empty:
        try:
            valid_geometries = [geom for geom in soil_polygons.geometry if geom and geom.is_valid]
            if valid_geometries:
                merged_soil_geometry = unary_union(valid_geometries)
                print("Prepared soil drainage geometry for filtering")
            else:
                print("No valid soil polygon geometries found")
        except Exception as e:
            print(f"Error preparing soil polygons: {e}")
            merged_soil_geometry = None

    # Initialize list to store all features
    all_features = []
    point_values = []  # Store points and values for final triangulation

    # Filter wells with valid ground water level data once
    valid_wells = wells_df[
        wells_df['ground water level'].notna() &
        (wells_df['ground water level'] != 0) &
        (wells_df['ground water level'].abs() > 0.1)
    ].copy()

    if valid_wells.empty:
        print("No valid ground water level data found")
        return geojson

    print(f"Using {len(valid_wells)} wells with valid ground water level data")

    # Process each grid point
    for idx, (center_lon, center_lat) in enumerate(zip(grid_lons, grid_lats)):
        if idx % 10 == 0:  # Progress update every 10 points
            print(f"Processing grid point {idx + 1}/{len(grid_lons)} at ({center_lat:.4f}, {center_lon:.4f})")

        # Filter wells within 15km radius
        wells_df['distance'] = wells_df.apply(
            lambda row: np.sqrt(
                ((row['latitude'] - center_lat) * km_per_degree_lat)**2 +
                ((row['longitude'] - center_lon) * km_per_degree_lon)**2
            ),
            axis=1
        )
        local_wells = valid_wells[valid_wells['distance'] <= radius_km].copy()

        if len(local_wells) < 5:
            print(f"Skipping grid point {idx + 1}: only {len(local_wells)} wells found")
            continue

        # Generate interpolation for this sub-region
        try:
            local_geojson = generate_geo_json_grid(
                local_wells,
                center_point=(center_lat, center_lon),
                radius_km=radius_km,
                resolution=resolution,
                method='ground_water_level_kriging',
                show_variance=False,
                auto_fit_variogram=True,
                variogram_model=variogram_model,
                soil_polygons=soil_polygons if merged_soil_geometry is not None else None
            )

            # Collect features and points for final triangulation
            for feature in local_geojson['features']:
                # Extract centroid of each polygon for final triangulation
                coords = feature['geometry']['coordinates'][0]
                centroid_x = np.mean([p[0] for p in coords[:-1]])  # Exclude closing point
                centroid_y = np.mean([p[1] for p in coords[:-1]])
                value = feature['properties']['yield']

                # Check if point is within bounds and soil polygons (if applicable)
                include_point = True
                if merged_soil_geometry is not None:
                    point = Point(centroid_x, centroid_y)
                    include_point = merged_soil_geometry.contains(point) or merged_soil_geometry.intersects(point)

                if include_point and value > 0.1:  # Only include significant values
                    point_values.append([centroid_x, centroid_y, value])
                    all_features.append(feature)

        except Exception as e:
            print(f"Error processing grid point {idx + 1}: {e}")
            continue

    # Remove duplicate points based on proximity (within 100m)
    if point_values:
        point_values = np.array(point_values)
        unique_points = []
        unique_values = []

        for i, (x, y, val) in enumerate(point_values):
            keep = True
            for j, (ux, uy, _) in enumerate(unique_points):
                dist_km = np.sqrt(
                    ((x - ux) * km_per_degree_lon)**2 +
                    ((y - uy) * km_per_degree_lat)**2
                )
                if dist_km < 0.1:  # 100m threshold
                    # Average values for close points
                    unique_values[j] = (unique_values[j] + val) / 2
                    keep = False
                    break
            if keep:
                unique_points.append([x, y])
                unique_values.append(val)

        point_values = np.array(unique_points)
        unique_values = np.array(unique_values)

        # Perform Delaunay triangulation for final surface
        try:
            if len(point_values) > 3:
                tri = Delaunay(point_values)
                final_features = []

                for simplex in tri.simplices:
                    vertices = point_values[simplex]
                    vertex_values = unique_values[simplex]
                    avg_value = float(np.mean(vertex_values))

                    if avg_value > 0.1:  # Only include significant values
                        poly = {
                            "type": "Feature",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [[
                                    [float(vertices[0, 0]), float(vertices[0,1])],
                                    [float(vertices[1, 0]), float(vertices[1, 1])],
                                    [float(vertices[2, 0]), float(vertices[2, 1])],
                                    [float(vertices[0, 0]), float(vertices[0, 1])]
                                ]]
                            },
                            "properties": {
                                "yield": avg_value
                            }
                        }
                        final_features.append(poly)

                geojson['features'] = final_features
                print(f"Generated {len(final_features)} final GeoJSON features")
            else:
                print("Insufficient points for final triangulation, using raw features")
                geojson['features'] = all_features
        except Exception as e:
            print(f"Final triangulation error: {e}")
            geojson['features'] = all_features
    else:
        print("No valid interpolation points generated")
        geojson['features'] = all_features

    print(f"Final GeoJSON contains {len(geojson['features'])} features")
    return geojson