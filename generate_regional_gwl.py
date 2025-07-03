
import numpy as np
import pandas as pd
import json
import os
from scipy.spatial import Delaunay
from shapely.geometry import Point
from shapely.ops import unary_union
import geopandas as gpd
from interpolation import generate_geo_json_grid
from data_loader import load_nz_govt_data
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def get_canterbury_bounds():
    """
    Define the Canterbury region bounds
    """
    return {
        'north': -42.5,    # Northern boundary
        'south': -45.0,    # Southern boundary  
        'east': 173.5,     # Eastern boundary
        'west': 170.5      # Western boundary
    }

def load_soil_polygons():
    """
    Load soil polygons for filtering (if available)
    """
    try:
        soil_files = [
            "attached_assets/s-map-soil-drainage-aug-2024_1749379069732.shp",
            "attached_assets/s-map-soil-drainage-aug-2024_1749427998471.shp"
        ]
        
        for soil_file in soil_files:
            if os.path.exists(soil_file):
                print(f"Loading soil polygons from {soil_file}")
                soil_polygons = gpd.read_file(soil_file)
                return soil_polygons
        
        print("No soil polygon files found, proceeding without soil filtering")
        return None
    except Exception as e:
        print(f"Error loading soil polygons: {e}")
        return None

def process_grid_point(args):
    """
    Process a single grid point for interpolation
    """
    idx, center_lat, center_lon, wells_df, radius_km, resolution, variogram_model, soil_polygons = args
    
    try:
        # Calculate km per degree for this location
        km_per_degree_lat = 111.0
        km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))
        
        # Filter wells within radius
        wells_df_copy = wells_df.copy()
        wells_df_copy['distance'] = wells_df_copy.apply(
            lambda row: np.sqrt(
                ((row['latitude'] - center_lat) * km_per_degree_lat)**2 +
                ((row['longitude'] - center_lon) * km_per_degree_lon)**2
            ),
            axis=1
        )
        
        local_wells = wells_df_copy[wells_df_copy['distance'] <= radius_km].copy()
        
        if len(local_wells) < 5:
            return idx, None, f"Only {len(local_wells)} wells found"
        
        # Generate interpolation for this sub-region
        local_geojson = generate_geo_json_grid(
            local_wells,
            center_point=(center_lat, center_lon),
            radius_km=radius_km,
            resolution=resolution,
            method='ground_water_level_kriging',
            show_variance=False,
            auto_fit_variogram=True,
            variogram_model=variogram_model,
            soil_polygons=soil_polygons
        )
        
        # Extract point values for merging
        point_values = []
        features = []
        
        for feature in local_geojson['features']:
            coords = feature['geometry']['coordinates'][0]
            centroid_x = np.mean([p[0] for p in coords[:-1]])
            centroid_y = np.mean([p[1] for p in coords[:-1]])
            value = feature['properties']['yield']
            
            if value > 0.1:  # Only include significant values
                point_values.append([centroid_x, centroid_y, value])
                features.append(feature)
        
        return idx, {'features': features, 'points': point_values}, f"Success: {len(features)} features"
        
    except Exception as e:
        return idx, None, f"Error: {str(e)}"

def generate_canterbury_gwl_interpolation(
    wells_df=None, 
    radius_km=15, 
    grid_spacing_km=10, 
    resolution=50, 
    variogram_model='spherical',
    max_workers=4
):
    """
    Generate a region-wide groundwater level interpolation for Canterbury
    
    Parameters:
    -----------
    wells_df : DataFrame, optional
        Wells data. If None, will load from NZ government data
    radius_km : float
        Radius for each local interpolation (default: 15 km)
    grid_spacing_km : float
        Spacing between grid points (default: 10 km for overlap)
    resolution : int
        Grid resolution for each sub-region (default: 50)
    variogram_model : str
        Variogram model for kriging (default: 'spherical')
    max_workers : int
        Maximum number of parallel workers (default: 4)
    
    Returns:
    --------
    dict
        GeoJSON FeatureCollection with interpolated groundwater level values
    """
    warnings.filterwarnings('ignore')
    
    print("Starting Canterbury region-wide groundwater level interpolation...")
    start_time = time.time()
    
    # Load wells data if not provided
    if wells_df is None:
        print("Loading wells data...")
        wells_df = load_nz_govt_data()
    
    # Get Canterbury bounds
    bounds = get_canterbury_bounds()
    print(f"Canterbury bounds: {bounds}")
    
    # Load soil polygons
    soil_polygons = load_soil_polygons()
    
    # Filter wells to Canterbury region and valid ground water level data
    canterbury_wells = wells_df[
        (wells_df['latitude'] >= bounds['south']) &
        (wells_df['latitude'] <= bounds['north']) &
        (wells_df['longitude'] >= bounds['west']) &
        (wells_df['longitude'] <= bounds['east']) &
        wells_df['ground water level'].notna() &
        (wells_df['ground water level'] != 0) &
        (wells_df['ground water level'].abs() > 0.1)
    ].copy()
    
    if canterbury_wells.empty:
        print("No valid ground water level data found in Canterbury region")
        return {"type": "FeatureCollection", "features": []}
    
    print(f"Using {len(canterbury_wells)} wells with valid ground water level data in Canterbury")
    
    # Calculate grid parameters
    center_lat = (bounds['north'] + bounds['south']) / 2
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(center_lat))
    
    # Create grid points
    lat_range = bounds['north'] - bounds['south']
    lon_range = bounds['east'] - bounds['west']
    
    lat_steps = int((lat_range * km_per_degree_lat) / grid_spacing_km) + 1
    lon_steps = int((lon_range * km_per_degree_lon) / grid_spacing_km) + 1
    
    grid_lats = np.linspace(bounds['south'], bounds['north'], lat_steps)
    grid_lons = np.linspace(bounds['west'], bounds['east'], lon_steps)
    
    # Create all grid point combinations
    grid_points = []
    for lat in grid_lats:
        for lon in grid_lons:
            grid_points.append((lat, lon))
    
    print(f"Generated {len(grid_points)} grid points for interpolation")
    print(f"Grid dimensions: {lat_steps} x {lon_steps}")
    
    # Prepare arguments for parallel processing
    args_list = []
    for idx, (center_lat, center_lon) in enumerate(grid_points):
        args_list.append((
            idx, center_lat, center_lon, canterbury_wells, 
            radius_km, resolution, variogram_model, soil_polygons
        ))
    
    # Process grid points in parallel
    all_point_values = []
    all_features = []
    successful_interpolations = 0
    
    print(f"Starting parallel processing with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_grid_point, args): args[0] for args in args_list}
        
        # Process completed tasks
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            
            try:
                result_idx, result_data, message = future.result()
                
                if result_data is not None:
                    all_features.extend(result_data['features'])
                    all_point_values.extend(result_data['points'])
                    successful_interpolations += 1
                
                # Progress update every 10 completed tasks
                if (result_idx + 1) % 10 == 0:
                    print(f"Completed {result_idx + 1}/{len(grid_points)}: {message}")
                    
            except Exception as e:
                print(f"Error processing grid point {idx}: {e}")
    
    print(f"Completed parallel processing. {successful_interpolations}/{len(grid_points)} successful interpolations")
    
    # Remove duplicate points and create final surface
    if all_point_values:
        print("Removing duplicate points and creating final surface...")
        
        point_values = np.array(all_point_values)
        unique_points = []
        unique_values = []
        
        # Remove duplicates within 100m
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
        
        print(f"Reduced {len(point_values)} points to {len(unique_points)} unique points")
        
        # Create final triangulated surface
        if len(unique_points) > 3:
            try:
                point_values = np.array(unique_points)
                unique_values = np.array(unique_values)
                
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
                
                geojson = {
                    "type": "FeatureCollection",
                    "features": final_features
                }
                
            except Exception as e:
                print(f"Triangulation error: {e}, using raw features")
                geojson = {
                    "type": "FeatureCollection", 
                    "features": all_features
                }
        else:
            geojson = {
                "type": "FeatureCollection",
                "features": all_features
            }
    else:
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
    
    total_time = time.time() - start_time
    print(f"Regional interpolation completed in {total_time:.1f} seconds")
    print(f"Final GeoJSON contains {len(geojson['features'])} features")
    
    return geojson

def save_regional_interpolation(output_file="canterbury_gwl_interpolation.json"):
    """
    Generate and save the Canterbury groundwater level interpolation
    """
    print("Generating Canterbury region-wide groundwater level interpolation...")
    
    # Generate the interpolation
    geojson = generate_canterbury_gwl_interpolation(
        radius_km=15,
        grid_spacing_km=10,
        resolution=50,
        variogram_model='spherical',
        max_workers=4
    )
    
    # Save to file
    os.makedirs("sample_data", exist_ok=True)
    output_path = os.path.join("sample_data", output_file)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"Regional interpolation saved to {output_path}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
        return output_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None

def load_regional_interpolation(input_file="canterbury_gwl_interpolation.json"):
    """
    Load the pre-computed Canterbury groundwater level interpolation
    """
    input_path = os.path.join("sample_data", input_file)
    
    if os.path.exists(input_path):
        try:
            with open(input_path, 'r') as f:
                geojson = json.load(f)
            print(f"Loaded regional interpolation from {input_path}")
            print(f"Contains {len(geojson['features'])} features")
            return geojson
        except Exception as e:
            print(f"Error loading regional interpolation: {e}")
            return None
    else:
        print(f"Regional interpolation file not found: {input_path}")
        return None

if __name__ == "__main__":
    # Generate and save the regional interpolation
    save_regional_interpolation()
