"""
Automated Heatmap Generator
Leverages the proven sequential heatmap system to generate comprehensive coverage
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_automated_generation(wells_data, interpolation_method, polygon_db, soil_polygons=None, new_clipping_polygon=None, num_tiles=5):
    """
    Generate heatmaps automatically using the exact sequential system logic but extended to cover all well data.
    Uses the proven 19.82km spacing and coordinate conversion from sequential_heatmap.py.
    """
    
    from sequential_heatmap import generate_quad_heatmaps_sequential
    
    print(f"🚀 AUTOMATED GENERATION: Using proven sequential system to cover all well data")
    print(f"📋 Available columns: {list(wells_data.columns)}")
    
    # Find the southwest corner of wells data to start our grid from
    # Check for different possible coordinate column names
    if 'latitude' in wells_data.columns and 'longitude' in wells_data.columns:
        # Data already has lat/lon columns
        valid_wells = wells_data.dropna(subset=['latitude', 'longitude'])
        
        if len(valid_wells) == 0:
            return {"success": False, "error": "No valid well coordinates found"}
        
        lat_coords = valid_wells['latitude'].astype(float)
        lon_coords = valid_wells['longitude'].astype(float)
        
        # Get bounds directly from lat/lon
        sw_lat, ne_lat = lat_coords.min(), lat_coords.max()
        sw_lon, ne_lon = lon_coords.min(), lon_coords.max()
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    elif 'nztm_x' in wells_data.columns and 'nztm_y' in wells_data.columns:
        # Data has NZTM coordinates, convert to lat/lon
        valid_wells = wells_data.dropna(subset=['nztm_x', 'nztm_y'])
        
        if len(valid_wells) == 0:
            return {"success": False, "error": "No valid well coordinates found"}
        
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
        
        x_coords = valid_wells['nztm_x'].astype(float)
        y_coords = valid_wells['nztm_y'].astype(float)
        
        # Get bounds and convert to lat/lon
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        # Convert corners to lat/lon
        sw_lon, sw_lat = transformer.transform(min_x, min_y)
        ne_lon, ne_lat = transformer.transform(max_x, max_y)
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    elif 'X' in wells_data.columns and 'Y' in wells_data.columns:
        # Legacy data format
        valid_wells = wells_data.dropna(subset=['X', 'Y'])
        
        if len(valid_wells) == 0:
            return {"success": False, "error": "No valid well coordinates found"}
        
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
        
        x_coords = valid_wells['X'].astype(float)
        y_coords = valid_wells['Y'].astype(float)
        
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        sw_lon, sw_lat = transformer.transform(min_x, min_y)
        ne_lon, ne_lat = transformer.transform(max_x, max_y)
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    else:
        return {"success": False, "error": f"Could not find coordinate columns. Available: {list(wells_data.columns)}"}
    
    # Calculate how many grid positions we need using 19.82km spacing
    # Each heatmap covers 40km diameter, centers are 19.82km apart
    from utils import get_distance
    
    # Calculate grid dimensions needed
    lat_span = ne_lat - sw_lat
    lon_span = ne_lon - sw_lon
    
    # Convert to approximate km using center point
    center_lat = (sw_lat + ne_lat) / 2
    center_lon = (sw_lon + ne_lon) / 2
    
    lat_km = lat_span * 111.0  # degrees to km
    lon_km = lon_span * 111.0 * np.cos(np.radians(center_lat))
    
    # Calculate grid size needed (with 19.82km spacing)
    grid_spacing_km = 19.82
    rows_needed = max(1, int(np.ceil(lat_km / grid_spacing_km)) + 1)
    cols_needed = max(1, int(np.ceil(lon_km / grid_spacing_km)) + 1)
    
    print(f"📐 Data extent: {lat_km:.1f}km × {lon_km:.1f}km")
    print(f"📐 Grid needed: {rows_needed} × {cols_needed} = {rows_needed * cols_needed} heatmaps")
    print(f"📐 Using {min(num_tiles, rows_needed * cols_needed)} tiles for this test")
    
    # Use the existing sequential system but start from a position that will cover the well data
    # Position the starting point so that the grid will encompass the well data area
    center_lat = (sw_lat + ne_lat) / 2
    center_lon = (sw_lon + ne_lon) / 2
    
    # Start from center of data area for better coverage
    start_point = [center_lat, center_lon]
    
    # Call the proven sequential generation function with appropriate grid size
    # For test, use a smaller grid that will fit within the data area
    if num_tiles <= 4:
        actual_grid = (2, 2)  # 2x2 grid for small tests
    elif num_tiles <= 6:
        actual_grid = (2, 3)  # 2x3 grid (original default)
    else:
        test_grid_size = int(np.ceil(np.sqrt(num_tiles)))
        actual_grid = (min(test_grid_size, 3), min(test_grid_size, 3))  # Cap at 3x3 for testing
    
    print(f"📐 Test grid size: {actual_grid[0]} × {actual_grid[1]} (center-positioned for data coverage)")
    
    try:
        result = generate_quad_heatmaps_sequential(
            wells_data, 
            start_point, 
            20,  # search_radius in km
            interpolation_method, 
            polygon_db, 
            soil_polygons, 
            new_clipping_polygon, 
            actual_grid
        )
        
        # Extract results
        if isinstance(result, tuple) and len(result) >= 2:
            success_count, stored_heatmap_ids = result[0], result[1]
            
            print(f"📋 GENERATION RESULTS:")
            print(f"   Grid processed: {actual_grid[0]} × {actual_grid[1]}")
            print(f"   Successful heatmaps: {success_count}")
            print(f"   Stored heatmap IDs: {len(stored_heatmap_ids)}")
            
            return {
                "success": success_count > 0,
                "success_count": success_count,
                "total_heatmaps": len(stored_heatmap_ids),
                "heatmap_ids": stored_heatmap_ids,
                "grid_size": actual_grid,
                "errors": []
            }
        else:
            error_msg = "Unexpected result format from sequential generation"
            print(f"❌ {error_msg}")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        error_msg = f"Error in sequential generation: {str(e)}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}


def generate_automated_heatmaps(wells_data, interpolation_method, polygon_db, soil_polygons=None, new_clipping_polygon=None, search_radius_km=20, max_tiles=1000):
    """
    Generate comprehensive heatmap coverage using the proven sequential system.
    This is the main function called by app.py for full automated generation.
    """
    
    print(f"🚀 FULL AUTOMATED GENERATION: Covering all well data with up to {max_tiles} heatmaps")
    print(f"📋 Available columns: {list(wells_data.columns)}")
    
    from sequential_heatmap import generate_quad_heatmaps_sequential
    
    # Find the bounds of wells data - handle both lat/lon and NZTM coordinates
    valid_wells = None
    
    if 'latitude' in wells_data.columns and 'longitude' in wells_data.columns:
        # Already has lat/lon coordinates
        valid_wells = wells_data.dropna(subset=['latitude', 'longitude'])
        if len(valid_wells) > 0:
            lat_coords = valid_wells['latitude'].astype(float)
            lon_coords = valid_wells['longitude'].astype(float)
            print(f"📍 Using existing lat/lon coordinates")
        
    elif 'NZTMX' in wells_data.columns and 'NZTMY' in wells_data.columns:
        # Convert NZTM to lat/lon
        from pyproj import Transformer
        transformer = Transformer.from_crs('EPSG:2193', 'EPSG:4326', always_xy=True)
        
        valid_wells = wells_data.dropna(subset=['NZTMX', 'NZTMY'])
        if len(valid_wells) > 0:
            nztm_x = valid_wells['NZTMX'].astype(float).values
            nztm_y = valid_wells['NZTMY'].astype(float).values
            lon_coords, lat_coords = transformer.transform(nztm_x, nztm_y)
            
            # Add converted coordinates to dataframe
            valid_wells = valid_wells.copy()
            valid_wells['latitude'] = lat_coords
            valid_wells['longitude'] = lon_coords
            print(f"📍 Converted NZTM coordinates to lat/lon")
    
    if valid_wells is None or len(valid_wells) == 0:
        return 0, [], [f"No valid well coordinates found. Available columns: {list(wells_data.columns)}"]
    
    lat_coords = valid_wells['latitude'].astype(float)
    lon_coords = valid_wells['longitude'].astype(float)
    
    sw_lat, ne_lat = lat_coords.min(), lat_coords.max()
    sw_lon, ne_lon = lon_coords.min(), lon_coords.max()
    
    print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
    print(f"📊 Processing {len(valid_wells)} wells with span: {ne_lat-sw_lat:.6f}° × {ne_lon-sw_lon:.6f}°")
    
    # Calculate optimal grid size using CONVEX HULL for efficient coverage
    from utils import get_distance
    from scipy.spatial import ConvexHull
    from pyproj import Transformer
    
    # Create convex hull around well data for smarter boundary calculation
    print(f"🔷 Calculating convex hull boundary around {len(valid_wells)} wells...")
    
    # Convert lat/lon to NZTM for accurate area calculation - use ALL wells, no sampling
    transformer_to_nztm = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    transformer_to_latlon = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    
    # Convert ALL wells to NZTM for accurate convex hull
    all_lons = lon_coords
    all_lats = lat_coords
    nztm_coords = [transformer_to_nztm.transform(lon, lat) for lat, lon in zip(all_lats, all_lons)]
    nztm_x = [coord[0] for coord in nztm_coords]
    nztm_y = [coord[1] for coord in nztm_coords]
    
    # Calculate convex hull
    points = np.column_stack([nztm_x, nztm_y])
    hull = ConvexHull(points)
    hull_area_km2 = hull.volume / 1e6  # ConvexHull.volume is area in 2D, convert to km²
    
    # Get hull bounds for grid calculation
    hull_vertices = points[hull.vertices]
    hull_min_x, hull_max_x = hull_vertices[:, 0].min(), hull_vertices[:, 0].max()
    hull_min_y, hull_max_y = hull_vertices[:, 1].min(), hull_vertices[:, 1].max()
    
    # Convert hull bounds back to lat/lon for display
    hull_sw_lon, hull_sw_lat = transformer_to_latlon.transform(hull_min_x, hull_min_y)
    hull_ne_lon, hull_ne_lat = transformer_to_latlon.transform(hull_max_x, hull_max_y)
    
    print(f"📐 Convex hull boundary: SW({hull_sw_lat:.6f}, {hull_sw_lon:.6f}) to NE({hull_ne_lat:.6f}, {hull_ne_lon:.6f})")
    print(f"📐 Hull area: {hull_area_km2:.0f} km² vs rectangular area: {((ne_lat-sw_lat)*111)*((ne_lon-sw_lon)*111*np.cos(np.radians((sw_lat+ne_lat)/2))):.0f} km²")
    
    # Calculate grid based on hull bounds (in NZTM for accuracy)
    hull_x_km = (hull_max_x - hull_min_x) / 1000.0
    hull_y_km = (hull_max_y - hull_min_y) / 1000.0
    
    # Calculate optimal grid size (19.82km spacing) with small buffer
    grid_spacing_km = 19.82
    
    # Add smaller buffer since convex hull already bounds the data efficiently
    rows_needed = max(1, int(np.ceil(hull_y_km / grid_spacing_km)) + 1)  # +1 for buffer coverage
    cols_needed = max(1, int(np.ceil(hull_x_km / grid_spacing_km)) + 1)  # +1 for buffer coverage
    
    # Use hull center for positioning
    center_lat = (hull_sw_lat + hull_ne_lat) / 2
    center_lon = (hull_sw_lon + hull_ne_lon) / 2
    
    total_needed = rows_needed * cols_needed
    
    print(f"📐 Hull extent: {hull_x_km:.1f}km × {hull_y_km:.1f}km")
    print(f"📐 Optimal grid with buffer: {rows_needed} × {cols_needed} = {total_needed} heatmaps needed")
    print(f"📍 Convex hull provides efficient coverage following data distribution")
    
    # Limit to max_tiles
    if total_needed > max_tiles:
        # Scale down proportionally
        scale_factor = np.sqrt(max_tiles / total_needed)
        rows_limited = max(1, int(rows_needed * scale_factor))
        cols_limited = max(1, int(cols_needed * scale_factor))
        actual_grid = (rows_limited, cols_limited)
        print(f"📐 Limited to: {rows_limited} × {cols_limited} = {rows_limited * cols_limited} heatmaps (max {max_tiles})")
    else:
        actual_grid = (rows_needed, cols_needed)
        print(f"📐 Using full grid: {rows_needed} × {cols_needed} = {total_needed} heatmaps")
    
    # For large grids, use convex hull bounds instead of center for better coverage
    # Calculate starting point to cover the full convex hull area
    if actual_grid[0] * actual_grid[1] > 100:  # Large grid - use strategic positioning
        # Start from SW corner of convex hull with slight buffer
        lat_buffer = (hull_ne_lat - hull_sw_lat) * 0.05  # 5% buffer
        lon_buffer = (hull_ne_lon - hull_sw_lon) * 0.05  # 5% buffer
        start_point = [hull_sw_lat - lat_buffer, hull_sw_lon - lon_buffer]
        print(f"📍 Large grid: Starting from SW corner with buffer: {start_point[0]:.6f}, {start_point[1]:.6f}")
    else:
        # Small grid - use center
        start_point = [center_lat, center_lon]
        print(f"📍 Small grid: Starting from convex hull center: {start_point[0]:.6f}, {start_point[1]:.6f}")
    
    try:
        # For large automated generation, use adaptive approach that skips sparse areas
        if actual_grid[0] * actual_grid[1] > 50:  # Large grid - use adaptive generation
            print(f"🔍 Using adaptive generation for large grid - will skip areas with insufficient well density")
            result = generate_adaptive_heatmaps(
                wells_data,
                hull_sw_lat, hull_sw_lon, hull_ne_lat, hull_ne_lon,
                search_radius_km,
                interpolation_method,
                polygon_db,
                soil_polygons,
                new_clipping_polygon,
                actual_grid,
                min_wells_required=10  # Minimum wells needed for interpolation
            )
        else:
            # Small grid - use original sequential approach
            result = generate_quad_heatmaps_sequential(
                wells_data, 
                start_point, 
                search_radius_km,  # use the parameter value
                interpolation_method, 
                polygon_db, 
                soil_polygons, 
                new_clipping_polygon, 
                actual_grid
            )
        
        if isinstance(result, tuple) and len(result) >= 3:
            success_count, stored_heatmap_ids, error_messages = result[0], result[1], result[2]
            
            print(f"📋 FULL GENERATION RESULTS:")
            print(f"   Grid processed: {actual_grid[0]} × {actual_grid[1]}")
            print(f"   Successful heatmaps: {success_count}")
            print(f"   Stored heatmap IDs: {len(stored_heatmap_ids)}")
            print(f"   Errors: {len(error_messages)}")
            
            return success_count, stored_heatmap_ids, error_messages
        else:
            error_msg = "Unexpected result format from sequential generation"
            print(f"❌ {error_msg}")
            return 0, [], [error_msg]
            
    except Exception as e:
        error_msg = f"Error in full automated generation: {str(e)}"
        print(f"❌ {error_msg}")
        return 0, [], [error_msg]


def generate_adaptive_heatmaps(wells_data, sw_lat, sw_lon, ne_lat, ne_lon, search_radius_km, 
                             interpolation_method, polygon_db, soil_polygons, new_clipping_polygon, 
                             grid_size, min_wells_required=10):
    """
    Generate heatmaps adaptively across the convex hull area, only creating heatmaps 
    where there's sufficient well density.
    """
    from utils import is_within_square, get_distance
    from interpolation import generate_geo_json_grid
    import numpy as np
    import pandas as pd
    
    rows, cols = grid_size
    print(f"🔍 ADAPTIVE GENERATION: Scanning {rows}×{cols} grid for viable heatmap locations")
    
    # Calculate step sizes across the convex hull bounds
    lat_step = (ne_lat - sw_lat) / (rows - 1) if rows > 1 else 0
    lon_step = (ne_lon - sw_lon) / (cols - 1) if cols > 1 else 0
    
    viable_locations = []
    skipped_locations = []
    
    # Scan the entire grid area for viable locations
    for row in range(rows):
        for col in range(cols):
            # Calculate position in the convex hull area
            lat = sw_lat + (row * lat_step)
            lon = sw_lon + (col * lon_step)
            
            # Check well density at this location
            try:
                wells_df_temp = wells_data.copy()
                
                # Ensure we have valid coordinate columns
                if 'latitude' not in wells_df_temp.columns or 'longitude' not in wells_df_temp.columns:
                    print(f"❌ Missing lat/lon columns in wells_data. Available columns: {list(wells_df_temp.columns)}")
                    well_count = 0
                else:
                    # Filter out wells with NaN coordinates
                    valid_coords = wells_df_temp['latitude'].notna() & wells_df_temp['longitude'].notna()
                    wells_valid = wells_df_temp[valid_coords]
                    
                    if len(wells_valid) == 0:
                        well_count = 0
                    else:
                        wells_valid['within_square'] = wells_valid.apply(
                            lambda row_data: is_within_square(
                                row_data['latitude'], 
                                row_data['longitude'],
                                lat, lon, search_radius_km
                            ), 
                            axis=1
                        )
                        
                        local_wells = wells_valid[wells_valid['within_square']]
                        well_count = len(local_wells)
            except Exception as e:
                print(f"❌ Error checking well density at {lat:.3f}, {lon:.3f}: {str(e)}")
                well_count = 0
            
            location_name = f"r{row}c{col}_{lat:.3f}_{lon:.3f}"
            
            if well_count >= min_wells_required:
                viable_locations.append((location_name, [lat, lon], well_count))
                print(f"  ✓ Viable: {location_name} ({well_count} wells)")
            else:
                skipped_locations.append((location_name, [lat, lon], well_count))
                if well_count > 0:
                    print(f"  ⚠ Sparse: {location_name} ({well_count} wells, need {min_wells_required})")
    
    print(f"🎯 ADAPTIVE RESULTS: {len(viable_locations)} viable locations, {len(skipped_locations)} skipped")
    print(f"   Efficiency: {len(viable_locations)}/{rows*cols} = {(len(viable_locations)/(rows*cols))*100:.1f}% coverage")
    
    # Now generate heatmaps only for viable locations
    generated_heatmaps = []
    stored_heatmap_ids = []
    error_messages = []
    
    for i, (location_name, center_point, well_count) in enumerate(viable_locations):
        print(f"\n🔄 GENERATING {i+1}/{len(viable_locations)}: {location_name} with {well_count} wells")
        
        try:
            # Generate heatmap for this viable location
            center_lat, center_lon = center_point
            
            # Filter wells for this specific location
            wells_df_filtered = wells_data.copy()
            
            # Ensure we have valid coordinate columns and data
            if 'latitude' not in wells_df_filtered.columns or 'longitude' not in wells_df_filtered.columns:
                print(f"❌ Missing lat/lon columns for {location_name}")
                filtered_wells = pd.DataFrame()
            else:
                # Filter out wells with NaN coordinates
                valid_coords = wells_df_filtered['latitude'].notna() & wells_df_filtered['longitude'].notna()
                wells_valid = wells_df_filtered[valid_coords]
                
                if len(wells_valid) == 0:
                    filtered_wells = pd.DataFrame()
                else:
                    wells_valid['within_square'] = wells_valid.apply(
                        lambda row: is_within_square(
                            row['latitude'], 
                            row['longitude'],
                            center_lat, center_lon,
                            search_radius_km
                        ), 
                        axis=1
                    )
                    
                    filtered_wells = wells_valid[wells_valid['within_square']]
            
            if len(filtered_wells) >= min_wells_required:
                # Generate the heatmap
                geojson_result = generate_geo_json_grid(
                    filtered_wells,
                    center_lat, center_lon,
                    search_radius_km,
                    interpolation_method,
                    soil_polygons=soil_polygons,
                    new_clipping_polygon=new_clipping_polygon
                )
                
                if geojson_result and 'features' in geojson_result:
                    # Store in database
                    heatmap_id = f"{interpolation_method}_{location_name}"
                    
                    success = polygon_db.store_heatmap(
                        heatmap_id=heatmap_id,
                        geojson_data=geojson_result,
                        method=interpolation_method,
                        center_lat=center_lat,
                        center_lon=center_lon,
                        well_count=len(filtered_wells)
                    )
                    
                    if success:
                        stored_heatmap_ids.append(heatmap_id)
                        generated_heatmaps.append((location_name, geojson_result))
                        print(f"✅ SUCCESS: {location_name} stored as {heatmap_id}")
                    else:
                        error_msg = f"Failed to store {location_name} in database"
                        error_messages.append(error_msg)
                        print(f"❌ {error_msg}")
                else:
                    error_msg = f"Failed to generate valid GeoJSON for {location_name}"
                    error_messages.append(error_msg)
                    print(f"❌ {error_msg}")
            else:
                error_msg = f"Insufficient wells for {location_name}: {len(filtered_wells)} < {min_wells_required}"
                error_messages.append(error_msg)
                print(f"❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"Error generating {location_name}: {str(e)}"
            error_messages.append(error_msg)
            print(f"❌ {error_msg}")
    
    success_count = len(stored_heatmap_ids)
    print(f"\n📊 ADAPTIVE GENERATION COMPLETE:")
    print(f"   Successful heatmaps: {success_count}")
    print(f"   Failed attempts: {len(error_messages)}")
    print(f"   Coverage efficiency: {success_count}/{rows*cols} grid positions")
    
    return success_count, stored_heatmap_ids, error_messages


def full_automated_generation(wells_data, interpolation_method, polygon_db, soil_polygons=None, new_clipping_polygon=None):
    """
    Legacy function name - redirect to generate_automated_heatmaps
    """
    return generate_automated_heatmaps(
        wells_data=wells_data,
        interpolation_method=interpolation_method, 
        polygon_db=polygon_db,
        soil_polygons=soil_polygons,
        new_clipping_polygon=new_clipping_polygon,
        max_tiles=100
    )