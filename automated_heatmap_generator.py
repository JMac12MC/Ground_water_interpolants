"""
Automated Heatmap Generator
Generates heatmaps covering all available well data without manual clicking
Uses NZTM2000 projection and grid-based tile processing
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from interpolation import generate_geo_json_grid
import pyproj
from pyproj import Transformer

def convert_degrees_to_nztm(lat, lon):
    """Convert WGS84 degrees to NZTM2000 coordinates"""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    x, y = transformer.transform(lon, lat)
    return x, y

def convert_nztm_to_degrees(x, y):
    """Convert NZTM2000 coordinates to WGS84 degrees"""
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon

def calculate_tile_size_meters():
    """
    Calculate tile size in meters based on existing 19.82km spacing
    This maintains compatibility with existing heatmap system
    """
    return 19820  # 19.82km in meters

def get_wells_bounds_nztm(wells_data):
    """
    Get the bounds of all wells in NZTM2000 coordinates
    """
    print("🌐 Converting well coordinates to NZTM2000...")
    
    # Convert all well coordinates to NZTM2000
    nztm_coords = []
    valid_wells = []
    
    # Check what columns are available
    print(f"📋 Available columns: {list(wells_data.columns)}")
    
    # Try different possible column names for coordinates
    lat_col = None
    lon_col = None
    
    for col in wells_data.columns:
        col_upper = col.upper()
        if col_upper == 'Y':  # Y is latitude in this dataset
            lat_col = col
        elif col_upper == 'X':  # X is longitude in this dataset
            lon_col = col
        elif 'LAT' in col_upper and lat_col is None:
            lat_col = col
        elif 'LON' in col_upper and lon_col is None:
            lon_col = col
    
    print(f"📍 Using latitude column: {lat_col}")
    print(f"📍 Using longitude column: {lon_col}")
    
    if not lat_col or not lon_col:
        raise ValueError(f"Could not find latitude/longitude columns. Available: {list(wells_data.columns)}")
    
    for idx, well in wells_data.iterrows():
        try:
            if pd.notna(well[lat_col]) and pd.notna(well[lon_col]):
                x_val, y_val = float(well[lon_col]), float(well[lat_col])
                
                # Check if coordinates are already in NZTM format (large numbers) or WGS84 (decimal degrees)
                if x_val > 1000000:  # NZTM coordinates are > 1,000,000
                    x, y = x_val, y_val  # Already NZTM
                else:
                    # Convert from WGS84 degrees to NZTM
                    x, y = convert_degrees_to_nztm(y_val, x_val)  # Note: Y is lat, X is lon
                
                nztm_coords.append((x, y))
                valid_wells.append(well)
        except Exception as e:
            print(f"Warning: Could not convert coordinates for well {idx}: {e}")
            continue
    
    if not nztm_coords:
        raise ValueError("No valid well coordinates found")
    
    # Get bounds
    x_coords = [coord[0] for coord in nztm_coords]
    y_coords = [coord[1] for coord in nztm_coords]
    
    minx, maxx = min(x_coords), max(x_coords)
    miny, maxy = min(y_coords), max(y_coords)
    
    print(f"📍 Wells extent in NZTM2000:")
    print(f"   X: {minx:.0f} to {maxx:.0f} m ({(maxx-minx)/1000:.1f} km wide)")
    print(f"   Y: {miny:.0f} to {maxy:.0f} m ({(maxy-miny)/1000:.1f} km tall)")
    print(f"   Total wells processed: {len(valid_wells)}")
    
    return minx, miny, maxx, maxy, valid_wells, nztm_coords

def snap_bounds_to_grid(minx, miny, maxx, maxy, tile_size):
    """
    Snap bounds to grid aligned with tile size
    """
    # Snap to grid
    grid_minx = np.floor(minx / tile_size) * tile_size
    grid_miny = np.floor(miny / tile_size) * tile_size
    grid_maxx = np.ceil(maxx / tile_size) * tile_size
    grid_maxy = np.ceil(maxy / tile_size) * tile_size
    
    print(f"🔲 Grid-snapped bounds:")
    print(f"   X: {grid_minx:.0f} to {grid_maxx:.0f} m")
    print(f"   Y: {grid_miny:.0f} to {grid_maxy:.0f} m")
    
    return grid_minx, grid_miny, grid_maxx, grid_maxy

def generate_tile_centroids(grid_minx, grid_miny, grid_maxx, grid_maxy, tile_size):
    """
    Generate all tile centroids within the grid bounds
    """
    tile_centroids = []
    
    for x in np.arange(grid_minx + tile_size/2, grid_maxx, tile_size):
        for y in np.arange(grid_miny + tile_size/2, grid_maxy, tile_size):
            tile_centroids.append((x, y))
    
    print(f"🗂️ Generated {len(tile_centroids)} tile centroids")
    
    return tile_centroids

def filter_wells_for_tile(wells_data, tile_center_x, tile_center_y, tile_size, search_radius_km=20):
    """
    Filter wells that are within the tile bounds and search radius
    """
    # Convert tile center back to degrees for compatibility with existing functions
    center_lat, center_lon = convert_nztm_to_degrees(tile_center_x, tile_center_y)
    center_point = [center_lat, center_lon]
    
    # Use existing well filtering logic with search radius
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate great circle distance between two points on Earth (specified in decimal degrees)"""
        import math
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
    
    filtered_wells = []
    # Find coordinate columns dynamically
    lat_col = None
    lon_col = None
    
    for col in wells_data.columns:
        col_upper = col.upper()
        if col_upper == 'Y':  # Y is latitude in this dataset
            lat_col = col
        elif col_upper == 'X':  # X is longitude in this dataset
            lon_col = col
        elif 'LAT' in col_upper and lat_col is None:
            lat_col = col
        elif 'LON' in col_upper and lon_col is None:
            lon_col = col
    
    if not lat_col or not lon_col:
        return pd.DataFrame()  # Return empty if no coordinates found
    
    for idx, well in wells_data.iterrows():
        try:
            if pd.notna(well[lat_col]) and pd.notna(well[lon_col]):
                x_val, y_val = float(well[lon_col]), float(well[lat_col])
                
                # Convert to WGS84 if needed for distance calculation
                if x_val > 1000000:  # NZTM coordinates
                    well_lat, well_lon = convert_nztm_to_degrees(x_val, y_val)
                else:
                    well_lat, well_lon = y_val, x_val  # Already WGS84
                
                distance = haversine_distance(
                    center_point[0], center_point[1],
                    well_lat, well_lon
                )
                if distance <= search_radius_km:
                    filtered_wells.append(well)
        except Exception as e:
            continue
    
    return pd.DataFrame(filtered_wells) if filtered_wells else pd.DataFrame()

def generate_automated_heatmaps(wells_data, interpolation_method, polygon_db, 
                               soil_polygons=None, new_clipping_polygon=None, 
                               search_radius_km=20, max_tiles=None):
    """
    Generate heatmaps automatically covering all available well data
    
    Args:
        wells_data: DataFrame with well data
        interpolation_method: 'kriging', 'ground_water_level_kriging', etc.
        polygon_db: Database connection for storing heatmaps
        soil_polygons: Optional soil drainage polygons
        new_clipping_polygon: Optional comprehensive clipping polygon
        search_radius_km: Search radius for wells around each tile center
        max_tiles: Maximum number of tiles to process (for testing)
    """
    
    print("🚀 AUTOMATED HEATMAP GENERATION STARTING...")
    print(f"📊 Processing {len(wells_data)} wells")
    
    # Step 1: Calculate tile size in meters
    tile_size = calculate_tile_size_meters()
    print(f"📏 Tile size: {tile_size/1000:.2f} km")
    
    # Step 2: Get wells bounds in NZTM2000
    try:
        minx, miny, maxx, maxy, valid_wells, nztm_coords = get_wells_bounds_nztm(wells_data)
        valid_wells_df = pd.DataFrame(valid_wells)
    except Exception as e:
        print(f"❌ Error processing well coordinates: {e}")
        return 0, [], [str(e)]
    
    # Step 3: Snap bounds to grid
    grid_minx, grid_miny, grid_maxx, grid_maxy = snap_bounds_to_grid(minx, miny, maxx, maxy, tile_size)
    
    # Step 4: Generate tile centroids
    tile_centroids = generate_tile_centroids(grid_minx, grid_miny, grid_maxx, grid_maxy, tile_size)
    
    # Apply max_tiles limit if specified
    if max_tiles and len(tile_centroids) > max_tiles:
        tile_centroids = tile_centroids[:max_tiles]
        print(f"⚠️ Limited to first {max_tiles} tiles for testing")
    
    # Step 5: Process each tile
    success_count = 0
    stored_heatmap_ids = []
    error_messages = []
    
    for i, (center_x, center_y) in enumerate(tile_centroids):
        print(f"\n🔄 Processing tile {i+1}/{len(tile_centroids)}")
        
        # Convert center back to degrees
        center_lat, center_lon = convert_nztm_to_degrees(center_x, center_y)
        center_point = [center_lat, center_lon]
        
        print(f"   📍 Tile center: {center_lat:.6f}, {center_lon:.6f}")
        print(f"   📍 NZTM center: {center_x:.0f}, {center_y:.0f}")
        
        # Filter wells for this tile
        tile_wells = filter_wells_for_tile(valid_wells_df, center_x, center_y, tile_size, search_radius_km)
        
        if tile_wells.empty:
            print(f"   ⏭️ No wells in tile - skipping")
            continue
        
        print(f"   🎯 Found {len(tile_wells)} wells in tile")
        
        try:
            # Generate indicator mask if needed (use None for now, let interpolation handle it)
            indicator_mask = None
            print(f"   📊 Using interpolation method: {interpolation_method}")
            if interpolation_method in ['indicator_kriging', 'ground_water_level_kriging']:
                print(f"   📊 Indicator kriging will be handled by interpolation function")
            
            # Generate heatmap using existing interpolation system
            print(f"   🎨 Generating {interpolation_method} heatmap...")
            
            geojson_data = generate_geo_json_grid(
                tile_wells.copy(),
                center_point,
                search_radius_km,
                resolution=100,  # Maintain existing resolution
                method=interpolation_method,
                show_variance=False,
                auto_fit_variogram=True,
                variogram_model='spherical',
                soil_polygons=soil_polygons,
                indicator_mask=indicator_mask,
                new_clipping_polygon=new_clipping_polygon
            )
            
            if geojson_data and len(geojson_data.get('features', [])) > 0:
                # Store heatmap in database
                heatmap_name = f"{interpolation_method}_auto_{center_lat:.3f}_{center_lon:.3f}"
                
                print(f"   💾 Storing heatmap: {heatmap_name}")
                print(f"   📐 Features generated: {len(geojson_data['features'])}")
                
                # Store in database using existing system
                heatmap_id = polygon_db.store_heatmap(
                    name=heatmap_name,
                    geojson_data=geojson_data,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    search_radius=search_radius_km,
                    interpolation_method=interpolation_method,
                    wells_count=len(tile_wells),
                    features_count=len(geojson_data['features'])
                )
                
                if heatmap_id:
                    stored_heatmap_ids.append(heatmap_id)
                    success_count += 1
                    print(f"   ✅ Stored as ID: {heatmap_id}")
                else:
                    print(f"   ❌ Failed to store heatmap")
                    error_messages.append(f"Storage failed for tile {center_lat:.3f}, {center_lon:.3f}")
            else:
                print(f"   ❌ No features generated")
                error_messages.append(f"No features generated for tile {center_lat:.3f}, {center_lon:.3f}")
                
        except Exception as e:
            print(f"   ❌ Error processing tile: {e}")
            error_messages.append(f"Tile {center_lat:.3f}, {center_lon:.3f}: {str(e)}")
    
    print(f"\n🏁 AUTOMATED GENERATION COMPLETE")
    print(f"✅ Successfully generated: {success_count} heatmaps")
    print(f"❌ Errors: {len(error_messages)}")
    print(f"💾 Stored heatmap IDs: {stored_heatmap_ids}")
    
    return success_count, stored_heatmap_ids, error_messages

def test_automated_generation(wells_data, polygon_db, max_tiles=5):
    """
    Test automated generation with a limited number of tiles
    """
    print("🧪 TESTING AUTOMATED HEATMAP GENERATION")
    
    success_count, stored_ids, errors = generate_automated_heatmaps(
        wells_data=wells_data,
        interpolation_method='ground_water_level_kriging',  # Default method
        polygon_db=polygon_db,
        search_radius_km=20,
        max_tiles=max_tiles
    )
    
    print(f"\n📋 TEST RESULTS:")
    print(f"   Tiles processed: {max_tiles}")
    print(f"   Successful: {success_count}")
    print(f"   Errors: {len(errors)}")
    
    if errors:
        print("   Error details:")
        for error in errors[:3]:  # Show first 3 errors
            print(f"     • {error}")
    
    return success_count > 0