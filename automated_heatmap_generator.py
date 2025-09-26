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
    Uses dynamic spacing (search_radius - 0.180km) and coordinate conversion from sequential_heatmap.py.
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
    
    # Calculate grid size needed (with dynamic spacing based on search area)
    # Use search radius from function call minus 0.180km for optimal spacing
    search_radius_km = 20  # This function uses fixed 20km for testing
    grid_spacing_km = search_radius_km - 0.180
    rows_needed = max(1, int(np.ceil(lat_km / grid_spacing_km)) + 1)
    cols_needed = max(1, int(np.ceil(lon_km / grid_spacing_km)) + 1)
    
    print(f"📐 Data extent: {lat_km:.1f}km × {lon_km:.1f}km")
    print(f"📐 Grid needed: {rows_needed} × {cols_needed} = {rows_needed * cols_needed} heatmaps ({grid_spacing_km:.2f}km spacing)")
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
    all_lons = valid_wells['longitude'].astype(float)
    all_lats = valid_wells['latitude'].astype(float)
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
    
    # Generate dynamic grid points based on search area size
    from shapely.geometry import Point, Polygon
    
    # Create Shapely polygon from convex hull
    hull_polygon = Polygon(hull_vertices)
    
    # Get bounds for grid generation
    min_x, min_y, max_x, max_y = hull_polygon.bounds
    
    # Generate grid points at dynamic spacing based on search area size
    grid_spacing_km = search_radius_km - 0.180
    grid_spacing = int(grid_spacing_km * 1000)  # Convert km to meters (NZTM units)
    
    # Calculate grid bounds with padding
    start_x = int(min_x // grid_spacing) * grid_spacing
    start_y = int(min_y // grid_spacing) * grid_spacing
    end_x = int(max_x // grid_spacing + 1) * grid_spacing
    end_y = int(max_y // grid_spacing + 1) * grid_spacing
    
    # Generate all grid points within convex hull
    grid_points_nztm = []
    grid_points_latlon = []
    
    y = start_y
    while y <= end_y:
        x = start_x
        while x <= end_x:
            point_nztm = Point(x, y)
            # Check if point is within convex hull
            if hull_polygon.contains(point_nztm):
                grid_points_nztm.append((x, y))
                # Convert to lat/lon for heatmap generation
                lon, lat = transformer_to_latlon.transform(x, y)
                grid_points_latlon.append([lat, lon])
            x += grid_spacing
        y += grid_spacing
    
    total_grid_points = len(grid_points_latlon)
    
    print(f"📐 Generated {total_grid_points} precise {grid_spacing_km:.2f}km grid points within convex hull")
    print(f"📍 Using pre-calculated grid points for efficient coverage")
    
    # Limit to max_tiles if necessary
    if total_grid_points > max_tiles:
        # Use first max_tiles points (could be improved with better selection)
        grid_points_latlon = grid_points_latlon[:max_tiles]
        actual_total = max_tiles
        print(f"📐 Limited to first {max_tiles} grid points (max {max_tiles})")
    else:
        actual_total = total_grid_points
        print(f"📐 Using all {total_grid_points} grid points")
    
    try:
        from sequential_heatmap import generate_grid_heatmaps_from_points
        result = generate_grid_heatmaps_from_points(
            wells_data, 
            grid_points_latlon, 
            search_radius_km,  # use the parameter value
            interpolation_method, 
            polygon_db, 
            soil_polygons, 
            new_clipping_polygon
        )
        
        if isinstance(result, tuple) and len(result) >= 3:
            success_count, stored_heatmap_ids, error_messages = result[0], result[1], result[2]
            
            print(f"📋 FULL GENERATION RESULTS:")
            print(f"   Grid points processed: {actual_total}")
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