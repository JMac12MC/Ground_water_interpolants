
#!/usr/bin/env python3
"""
Resumable regional groundwater level interpolation system for Canterbury
Uses database-backed sub-interpolations that can be resumed if interrupted
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pykrige.ok import OrdinaryKriging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from database import PolygonDatabase
from data_loader import load_data
from utils import filter_by_soil_polygons

def create_grid_points(bounds, spacing_km=10):
    """
    Create a grid of points covering the specified bounds
    
    Parameters:
    -----------
    bounds : dict
        Dictionary with 'south', 'north', 'west', 'east' keys
    spacing_km : float
        Spacing between grid points in kilometers
    
    Returns:
    --------
    list
        List of (lat, lon) tuples
    """
    # Convert km to degrees (approximate)
    lat_spacing = spacing_km / 111.0  # 1 degree ≈ 111 km
    lon_spacing = spacing_km / (111.0 * np.cos(np.radians((bounds['south'] + bounds['north']) / 2)))
    
    lats = np.arange(bounds['south'], bounds['north'] + lat_spacing, lat_spacing)
    lons = np.arange(bounds['west'], bounds['east'] + lon_spacing, lon_spacing)
    
    grid_points = []
    for lat in lats:
        for lon in lons:
            grid_points.append((lat, lon))
    
    return grid_points

def get_wells_in_radius(wells_df, center_lat, center_lon, radius_km=15):
    """
    Get wells within specified radius of a center point
    
    Parameters:
    -----------
    wells_df : DataFrame
        Wells data with 'latitude', 'longitude', 'ground water level' columns
    center_lat : float
        Center latitude
    center_lon : float
        Center longitude
    radius_km : float
        Radius in kilometers
    
    Returns:
    --------
    DataFrame
        Filtered wells within radius
    """
    # Calculate distances using approximate conversion
    lat_diff = wells_df['latitude'] - center_lat
    lon_diff = wells_df['longitude'] - center_lon
    
    # Convert to km (approximate)
    lat_km = lat_diff * 111.0
    lon_km = lon_diff * 111.0 * np.cos(np.radians(center_lat))
    
    distance_km = np.sqrt(lat_km**2 + lon_km**2)
    
    return wells_df[distance_km <= radius_km].copy()

def generate_sub_interpolation(grid_idx, center_lat, center_lon, wells_df, radius_km=15, resolution=50):
    """
    Generate interpolation for a single grid point
    
    Parameters:
    -----------
    grid_idx : int
        Grid point index
    center_lat : float
        Center latitude
    center_lon : float
        Center longitude
    wells_df : DataFrame
        Full wells dataset
    radius_km : float
        Interpolation radius in km
    resolution : int
        Number of interpolation points per dimension
    
    Returns:
    --------
    dict or None
        GeoJSON data for the sub-interpolation
    """
    try:
        # Get wells within radius
        local_wells = get_wells_in_radius(wells_df, center_lat, center_lon, radius_km)
        
        if len(local_wells) < 3:
            print(f"Grid {grid_idx}: Insufficient wells ({len(local_wells)}) - skipping")
            return None
        
        # Filter out invalid values
        local_wells = local_wells[
            (local_wells['ground water level'].notna()) & 
            (local_wells['ground water level'] > 0) &
            (local_wells['ground water level'] < 1000)  # Reasonable max depth
        ].copy()
        
        if len(local_wells) < 3:
            print(f"Grid {grid_idx}: Insufficient valid wells after filtering - skipping")
            return None
        
        print(f"Grid {grid_idx}: Processing {len(local_wells)} wells at ({center_lat:.3f}, {center_lon:.3f})")
        
        # Create interpolation grid around center point
        lat_range = radius_km / 111.0  # Convert km to degrees
        lon_range = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        grid_lat = np.linspace(center_lat - lat_range, center_lat + lat_range, resolution)
        grid_lon = np.linspace(center_lon - lon_range, center_lon + lon_range, resolution)
        
        # Perform kriging
        try:
            krig = OrdinaryKriging(
                local_wells['longitude'].values,
                local_wells['latitude'].values,
                local_wells['ground water level'].values,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False
            )
            
            z, ss = krig.execute('grid', grid_lon, grid_lat)
            
        except Exception as e:
            print(f"Grid {grid_idx}: Kriging failed - {e}")
            return None
        
        # Convert to GeoJSON
        features = []
        for i in range(len(grid_lat)-1):
            for j in range(len(grid_lon)-1):
                if not np.isnan(z[i, j]):
                    # Create polygon for this grid cell
                    polygon = Polygon([
                        (grid_lon[j], grid_lat[i]),
                        (grid_lon[j+1], grid_lat[i]),
                        (grid_lon[j+1], grid_lat[i+1]),
                        (grid_lon[j], grid_lat[i+1]),
                        (grid_lon[j], grid_lat[i])
                    ])
                    
                    feature = {
                        "type": "Feature",
                        "geometry": polygon.__geo_interface__,
                        "properties": {
                            "yield": float(z[i, j]),
                            "variance": float(ss[i, j]) if not np.isnan(ss[i, j]) else 0.0,
                            "grid_idx": grid_idx
                        }
                    }
                    features.append(feature)
        
        if not features:
            print(f"Grid {grid_idx}: No valid features generated")
            return None
        
        geojson = {
            "type": "FeatureCollection",
            "features": features
        }
        
        print(f"Grid {grid_idx}: Generated {len(features)} features")
        return geojson
        
    except Exception as e:
        print(f"Grid {grid_idx}: Error - {e}")
        return None

def process_grid_point(args):
    """
    Process a single grid point (for parallel execution)
    """
    grid_idx, center_lat, center_lon, wells_df, db, radius_km, resolution = args
    
    # Check if already completed
    existing = db.get_sub_interpolation(grid_idx, 'canterbury', 'ground_water_level')
    if existing:
        print(f"Grid {grid_idx}: Already completed - skipping")
        return True
    
    # Generate the interpolation
    geojson = generate_sub_interpolation(
        grid_idx, center_lat, center_lon, wells_df, radius_km, resolution
    )
    
    if geojson:
        # Store in database
        feature_count = len(geojson['features'])
        success = db.store_sub_interpolation(
            grid_idx, 'canterbury', 'ground_water_level', 
            geojson, center_lat, center_lon, feature_count
        )
        
        if success:
            print(f"Grid {grid_idx}: Stored {feature_count} features in database")
            return True
        else:
            print(f"Grid {grid_idx}: Failed to store in database")
            return False
    
    return False

def generate_canterbury_gwl_interpolation(
    wells_df=None, 
    radius_km=15, 
    grid_spacing_km=10,  # 15km radius - 5km overlap = 10km spacing
    resolution=50, 
    variogram_model='spherical',
    max_workers=4
):
    """
    Generate resumable Canterbury region-wide groundwater level interpolation
    
    Parameters:
    -----------
    wells_df : DataFrame, optional
        Wells data. If None, will load from data_loader
    radius_km : float
        Interpolation radius in kilometers
    grid_spacing_km : float
        Grid spacing in kilometers
    resolution : int
        Number of interpolation points per sub-region
    variogram_model : str
        Kriging variogram model
    max_workers : int
        Number of parallel workers
    
    Returns:
    --------
    dict or None
        GeoJSON FeatureCollection with interpolated data
    """
    print("=" * 60)
    print("Canterbury Regional Groundwater Level Interpolation")
    print("=" * 60)
    print(f"Search radius: {radius_km}km")
    print(f"Grid spacing: {grid_spacing_km}km")
    print(f"Overlap: {radius_km - grid_spacing_km}km")
    print(f"Resolution: {resolution} points per sub-region")
    print(f"Workers: {max_workers}")
    print()
    
    # Load wells data if not provided
    if wells_df is None:
        print("Loading wells data...")
        wells_df = load_data()
        if wells_df is None or wells_df.empty:
            print("❌ Failed to load wells data")
            return None
    
    # Filter for Canterbury region and valid data
    canterbury_bounds = {
        'south': -45.0,
        'north': -42.5,
        'west': 170.5,
        'east': 173.5
    }
    
    wells_df = wells_df[
        (wells_df['latitude'] >= canterbury_bounds['south']) &
        (wells_df['latitude'] <= canterbury_bounds['north']) &
        (wells_df['longitude'] >= canterbury_bounds['west']) &
        (wells_df['longitude'] <= canterbury_bounds['east']) &
        (wells_df['ground water level'].notna()) &
        (wells_df['ground water level'] > 0)
    ].copy()
    
    print(f"📊 Using {len(wells_df)} wells in Canterbury region")
    
    # Initialize database
    try:
        db = PolygonDatabase()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None
    
    # Create grid points
    grid_points = create_grid_points(canterbury_bounds, grid_spacing_km)
    total_points = len(grid_points)
    
    print(f"🗺️  Created {total_points} grid points")
    
    # Check progress
    completed_count = db.count_sub_interpolations('canterbury', 'ground_water_level')
    print(f"📈 Progress: {completed_count}/{total_points} sub-interpolations completed ({completed_count/total_points*100:.1f}%)")
    
    if completed_count < total_points:
        print("\n🔄 Processing remaining grid points...")
        
        # Prepare arguments for parallel processing
        process_args = []
        for idx, (lat, lon) in enumerate(grid_points):
            process_args.append((idx, lat, lon, wells_df, db, radius_km, resolution))
        
        # Process in parallel
        start_time = time.time()
        successful = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_grid_point, args): args[0] 
                for args in process_args
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    success = future.result()
                    if success:
                        successful += 1
                    
                    # Progress update
                    current_completed = db.count_sub_interpolations('canterbury', 'ground_water_level')
                    elapsed = time.time() - start_time
                    progress = current_completed / total_points * 100
                    
                    if current_completed > 0:
                        eta = (elapsed / current_completed) * (total_points - current_completed)
                        print(f"Progress: {current_completed}/{total_points} ({progress:.1f}%) - ETA: {eta/60:.1f}min")
                    
                except Exception as e:
                    print(f"Grid {idx}: Processing failed - {e}")
        
        print(f"\n✅ Completed {successful} new sub-interpolations in {(time.time()-start_time)/60:.1f} minutes")
    
    # Merge all sub-interpolations
    print("\n🔗 Merging sub-interpolations into final surface...")
    
    try:
        sub_interpolations = db.list_sub_interpolations('canterbury', 'ground_water_level')
        
        if not sub_interpolations:
            print("❌ No sub-interpolations found to merge")
            return None
        
        print(f"📋 Merging {len(sub_interpolations)} sub-interpolations...")
        
        # Combine all features
        all_features = []
        for sub in sub_interpolations:
            if 'features' in sub['geojson_data']:
                all_features.extend(sub['geojson_data']['features'])
        
        if not all_features:
            print("❌ No features found in sub-interpolations")
            return None
        
        # Create final GeoJSON
        final_geojson = {
            "type": "FeatureCollection",
            "features": all_features
        }
        
        print(f"✅ Merged {len(all_features)} features into final surface")
        
        # Optional: Filter by soil drainage areas
        try:
            print("🌱 Applying soil drainage filtering...")
            final_geojson = filter_by_soil_polygons(final_geojson)
            print(f"✅ Soil filtering complete: {len(final_geojson['features'])} features remain")
        except Exception as e:
            print(f"⚠️  Soil filtering failed: {e} - continuing without filtering")
        
        return final_geojson
        
    except Exception as e:
        print(f"❌ Error merging sub-interpolations: {e}")
        return None

def save_regional_interpolation(
    filename="sample_data/canterbury_gwl_interpolation.json", 
    store_in_database=True
):
    """
    Generate and save the regional interpolation
    
    Parameters:
    -----------
    filename : str
        Output filename for GeoJSON
    store_in_database : bool
        Whether to store in database
    
    Returns:
    --------
    str or None
        Output filename if successful
    """
    # Generate the interpolation with 15km radius and 5km overlap
    overlap_km = 5
    radius_km = 15
    grid_spacing_km = radius_km - overlap_km  # 10km spacing for 5km overlap
    
    geojson = generate_canterbury_gwl_interpolation(
        radius_km=radius_km,
        grid_spacing_km=grid_spacing_km,
        resolution=50,
        variogram_model='spherical',
        max_workers=4
    )
    
    if geojson is None:
        print("❌ Failed to generate regional interpolation")
        return None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to file
    try:
        with open(filename, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"💾 Saved regional interpolation to: {filename}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        return None
    
    # Store in database
    if store_in_database:
        try:
            db = PolygonDatabase()
            success = db.store_regional_interpolation(
                'canterbury', geojson, 'ground_water_level'
            )
            if success:
                print("💾 Stored regional interpolation in database")
            else:
                print("⚠️  Failed to store in database")
        except Exception as e:
            print(f"⚠️  Database storage failed: {e}")
    
    return filename

def load_regional_interpolation(from_database=True):
    """
    Load regional interpolation from database or file
    
    Parameters:
    -----------
    from_database : bool
        Whether to load from database first
    
    Returns:
    --------
    dict or None
        GeoJSON data if found
    """
    # Try database first
    if from_database:
        try:
            db = PolygonDatabase()
            geojson = db.get_regional_interpolation('canterbury', 'ground_water_level')
            if geojson:
                print("📊 Loaded regional interpolation from database")
                return geojson
        except Exception as e:
            print(f"⚠️  Database load failed: {e}")
    
    # Fallback to file
    filename = "sample_data/canterbury_gwl_interpolation.json"
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                geojson = json.load(f)
            print(f"📊 Loaded regional interpolation from: {filename}")
            return geojson
        except Exception as e:
            print(f"❌ Failed to load file: {e}")
    
    return None

if __name__ == "__main__":
    # Generate and save the regional interpolation
    save_regional_interpolation(store_in_database=True)
