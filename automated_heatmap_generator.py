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
    Generate comprehensive heatmap coverage using well-density-based positioning.
    Only generates heatmaps where wells actually exist within the search radius.
    """
    
    print(f"🚀 SMART AUTOMATED GENERATION: Covering areas with actual well data (up to {max_tiles} heatmaps)")
    print(f"📋 Available columns: {list(wells_data.columns)}")
    
    from utils import get_distance
    import numpy as np
    
    # Find the bounds of wells data 
    if 'latitude' in wells_data.columns and 'longitude' in wells_data.columns:
        valid_wells = wells_data.dropna(subset=['latitude', 'longitude'])
        
        if len(valid_wells) == 0:
            return 0, [], ["No valid well coordinates found"]
        
        lat_coords = valid_wells['latitude'].astype(float)
        lon_coords = valid_wells['longitude'].astype(float)
        
        sw_lat, ne_lat = lat_coords.min(), lat_coords.max()
        sw_lon, ne_lon = lon_coords.min(), lon_coords.max()
        
        print(f"📍 Well data bounds: SW({sw_lat:.6f}, {sw_lon:.6f}) to NE({ne_lat:.6f}, {ne_lon:.6f})")
        print(f"📊 Processing {len(valid_wells)} wells")
        
    else:
        return 0, [], [f"Could not find coordinate columns. Available: {list(wells_data.columns)}"]
    
    # NEW APPROACH: Generate heatmaps based on well density, not empty grid
    # Create a grid of potential heatmap positions with 19.82km spacing
    grid_spacing_km = 19.82
    
    # Calculate grid dimensions
    lat_span = ne_lat - sw_lat
    lon_span = ne_lon - sw_lon
    lat_km = lat_span * 111.0
    lon_km = lon_span * 111.0 * np.cos(np.radians((sw_lat + ne_lat) / 2))
    
    # Calculate grid spacing in degrees
    lat_spacing = grid_spacing_km / 111.0  # degrees
    lon_spacing = grid_spacing_km / (111.0 * np.cos(np.radians((sw_lat + ne_lat) / 2)))  # degrees
    
    # Generate potential grid positions
    potential_positions = []
    
    # Start from southwest corner and work across the entire area
    current_lat = sw_lat
    while current_lat <= ne_lat + lat_spacing:  # Extra buffer
        current_lon = sw_lon
        while current_lon <= ne_lon + lon_spacing:  # Extra buffer
            potential_positions.append([current_lat, current_lon])
            current_lon += lon_spacing
        current_lat += lat_spacing
    
    print(f"📐 Data extent: {lat_km:.1f}km × {lon_km:.1f}km")
    print(f"🎯 Generated {len(potential_positions)} potential heatmap positions")
    
    # Filter positions to only those with wells within search radius
    valid_positions = []
    for pos in potential_positions:
        pos_lat, pos_lon = pos
        
        # Check if there are wells within search_radius_km of this position
        wells_in_range = 0
        for _, well in valid_wells.iterrows():
            well_lat, well_lon = well['latitude'], well['longitude']
            distance = get_distance(pos_lat, pos_lon, well_lat, well_lon)
            
            if distance <= search_radius_km:
                wells_in_range += 1
                if wells_in_range >= 5:  # Need at least 5 wells for good interpolation
                    valid_positions.append(pos)
                    break
    
    print(f"🎯 Found {len(valid_positions)} positions with sufficient well data (≥5 wells within {search_radius_km}km)")
    
    # Limit to max_tiles
    if len(valid_positions) > max_tiles:
        print(f"📐 Limiting to {max_tiles} positions (from {len(valid_positions)} valid positions)")
        valid_positions = valid_positions[:max_tiles]
    
    # Generate individual heatmaps at each valid position
    success_count = 0
    stored_heatmap_ids = []
    error_messages = []
    
    print(f"🚀 Starting generation of {len(valid_positions)} heatmaps...")
    
    for i, position in enumerate(valid_positions):
        try:
            print(f"📍 Generating heatmap {i+1}/{len(valid_positions)} at {position[0]:.6f}, {position[1]:.6f}")
            
            # Use the interpolation system to generate individual heatmap
            from interpolation import generate_geo_json_grid
            
            # Generate the heatmap at this position
            center_point = [position[0], position[1]]
            
            geojson_data = generate_geo_json_grid(
                wells_df=wells_data,
                center_point=center_point,
                radius_km=search_radius_km,
                resolution=100,  # Use consistent resolution
                method=interpolation_method,
                soil_polygons=soil_polygons,
                new_clipping_polygon=new_clipping_polygon
            )
            
            if geojson_data and len(geojson_data.get('features', [])) > 0:
                # Store the heatmap in database
                heatmap_name = f"{interpolation_method}_{position[0]:.3f}_{position[1]:.3f}"
                
                try:
                    heatmap_id = polygon_db.store_heatmap(
                        name=heatmap_name,
                        geojson_data=geojson_data,
                        center_lat=position[0],
                        center_lon=position[1],
                        interpolation_method=interpolation_method
                    )
                    
                    result = {
                        'success': True,
                        'heatmap_id': heatmap_id,
                        'triangles': len(geojson_data.get('features', []))
                    }
                except Exception as store_error:
                    result = {
                        'success': False,
                        'error': f"Failed to store heatmap: {str(store_error)}"
                    }
            else:
                result = {
                    'success': False,
                    'error': "No triangular features generated"
                }
            
            if result and result.get('success', False):
                success_count += 1
                if 'heatmap_id' in result:
                    stored_heatmap_ids.append(result['heatmap_id'])
                print(f"✅ Success: {result.get('triangles', 0)} triangles generated")
            else:
                error_msg = f"Failed at position {position[0]:.6f}, {position[1]:.6f}: {result.get('error', 'Unknown error')}"
                error_messages.append(error_msg)
                print(f"❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"Exception at position {position[0]:.6f}, {position[1]:.6f}: {str(e)}"
            error_messages.append(error_msg)
            print(f"❌ {error_msg}")
    
    print(f"📋 SMART GENERATION RESULTS:")
    print(f"   Positions evaluated: {len(valid_positions)}")
    print(f"   Successful heatmaps: {success_count}")
    print(f"   Stored heatmap IDs: {len(stored_heatmap_ids)}")
    print(f"   Errors: {len(error_messages)}")
    
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