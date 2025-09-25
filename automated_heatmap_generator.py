"""
Automated Heatmap Generator
Leverages the proven sequential heatmap system to generate comprehensive coverage
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils import get_distance

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
    
    print("🚨🚨🚨 CRITICAL: generate_automated_heatmaps FUNCTION CALLED! 🚨🚨🚨")
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
    
    # Generate grid points using GEODETIC SPACING (not flat projection)
    # This accounts for Earth's curvature for survey-grade precision
    grid_spacing_km = search_radius_km - 0.180
    
    print(f"🌍 USING GEODETIC GRID GENERATION (Earth-curvature aware)")
    print(f"   Target spacing: {grid_spacing_km:.3f}km between centers")
    print(f"   Using survey-grade precision calculations...")
    
    # Use center of convex hull as starting point for geodetic grid
    hull_center_x = (hull_min_x + hull_max_x) / 2
    hull_center_y = (hull_min_y + hull_max_y) / 2
    hull_center_lon, hull_center_lat = transformer_to_latlon.transform(hull_center_x, hull_center_y)
    
    print(f"   Grid origin: ({hull_center_lat:.6f}, {hull_center_lon:.6f})")
    
    # PRECISE GEODETIC OFFSET CALCULATIONS
    # Same algorithm as manual system for survey-grade accuracy
    MAX_ITERATIONS = 200
    TOLERANCE_KM = 0.0001  # 10cm tolerance
    ADAPTIVE_STEP_SIZE = 0.000001
    
    # Step 1: Calculate precise latitude offset
    target_offset_km = grid_spacing_km
    lat_offset_degrees = target_offset_km / 111.0  # Initial estimate
    best_lat_offset = lat_offset_degrees
    best_lat_error = float('inf')
    
    for i in range(MAX_ITERATIONS):
        test_lat = hull_center_lat - lat_offset_degrees
        actual_distance = get_distance(hull_center_lat, hull_center_lon, test_lat, hull_center_lon)
        error = abs(actual_distance - target_offset_km)
        
        if error < best_lat_error:
            best_lat_offset = lat_offset_degrees
            best_lat_error = error
        
        if error < TOLERANCE_KM:
            break
            
        # Adaptive convergence algorithm
        if error > 0.001:  # > 1 meter error - proportional adjustment
            adjustment_factor = target_offset_km / actual_distance  
            lat_offset_degrees *= adjustment_factor
        else:  # Precision phase - adaptive micro-adjustments
            step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)
            if actual_distance > target_offset_km:
                lat_offset_degrees -= step_size
            else:
                lat_offset_degrees += step_size
    
    lat_offset_degrees = best_lat_offset
    final_lat_distance = get_distance(hull_center_lat, hull_center_lon, hull_center_lat - lat_offset_degrees, hull_center_lon)
    print(f"   Latitude offset: {lat_offset_degrees:.10f}° (achieved: {final_lat_distance:.6f}km)")
    
    print(f"   Latitude offset calculated - will be used for all rows")
    
    # Estimate grid dimensions needed to cover convex hull
    # Convert hull bounds to lat/lon for distance calculations
    hull_width_km = get_distance(hull_center_lat, hull_sw_lon, hull_center_lat, hull_ne_lon)
    hull_height_km = get_distance(hull_sw_lat, hull_center_lon, hull_ne_lat, hull_center_lon)
    
    grid_cols_needed = int(hull_width_km / grid_spacing_km) + 3  # Add padding
    grid_rows_needed = int(hull_height_km / grid_spacing_km) + 3  # Add padding
    
    print(f"   Hull dimensions: {hull_width_km:.1f}km × {hull_height_km:.1f}km")
    print(f"   Grid dimensions needed: {grid_rows_needed} rows × {grid_cols_needed} cols")
    
    # Generate geodetic grid with PER-ROW longitude compensation (same as manual system)
    # This eliminates cumulative longitude offset errors across the north-south extent
    print(f"🔧 CALCULATING ROW-SPECIFIC LONGITUDE OFFSETS:")
    print(f"   Each row will have its own precise longitude offset calculated")
    
    grid_points_latlon = []
    
    # Start from northwest corner to cover hull area
    start_row = -(grid_rows_needed // 2)
    start_col = -(grid_cols_needed // 2)
    
    # Pre-calculate row-specific longitude offsets (same approach as manual system)
    row_longitude_offsets = []
    
    for row in range(grid_rows_needed):
        current_row = start_row + row
        row_lat = hull_center_lat - (current_row * lat_offset_degrees)
        
        # Step 2: Calculate precise longitude offset for THIS ROW'S latitude
        # Use same adaptive convergence as manual system
        lon_offset_initial = target_offset_km / (111.0 * abs(np.cos(np.radians(row_lat))))
        best_lon_offset = lon_offset_initial
        best_lon_error = float('inf')
        
        for i in range(MAX_ITERATIONS):
            test_lon = hull_center_lon + lon_offset_initial
            actual_distance = get_distance(row_lat, hull_center_lon, row_lat, test_lon)
            error = abs(actual_distance - target_offset_km)
            
            if error < best_lon_error:
                best_lon_offset = lon_offset_initial
                best_lon_error = error
            
            if error < TOLERANCE_KM:
                break
                
            # Adaptive longitude precision targeting (same as manual system)
            if error > 0.001:  # > 1 meter error
                adjustment_factor = target_offset_km / actual_distance
                lon_offset_initial *= adjustment_factor
            else:  # Adaptive precision phase
                step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)
                if actual_distance > target_offset_km:
                    lon_offset_initial -= step_size
                else:
                    lon_offset_initial += step_size
        
        row_longitude_offsets.append(best_lon_offset)
        
        # Verify achieved precision for this row
        final_row_distance = get_distance(row_lat, hull_center_lon, row_lat, hull_center_lon + best_lon_offset)
        
        if row < 3 or row == grid_rows_needed - 1:  # Log first few and last row
            print(f"   Row {row:2d} (lat {row_lat:.6f}°): offset {best_lon_offset:.10f}° (achieved: {final_row_distance:.6f}km, error: {abs(final_row_distance-target_offset_km)*1000:.1f}m)")
    
    print(f"   All {len(row_longitude_offsets)} rows calculated with survey-grade precision")
    
    # Generate grid points using row-specific longitude offsets
    for row in range(grid_rows_needed):
        current_row = start_row + row
        row_lat = hull_center_lat - (current_row * lat_offset_degrees)
        row_lon_offset = row_longitude_offsets[row]
        
        for col in range(grid_cols_needed):
            current_col = start_col + col
            row_lon = hull_center_lon + (current_col * row_lon_offset)
            
            # Convert to NZTM to check if within convex hull
            nztm_x, nztm_y = transformer_to_nztm.transform(row_lon, row_lat)
            point_nztm = Point(nztm_x, nztm_y)
            
            # Double validation: within convex hull AND has wells within search radius
            if hull_polygon.contains(point_nztm):
                # Check if this grid point has wells within search radius
                from utils import is_within_square
                wells_in_radius = valid_wells[valid_wells.apply(
                    lambda well: is_within_square(
                        well['latitude'], 
                        well['longitude'], 
                        row_lat, 
                        row_lon, 
                        search_radius_km
                    ), 
                    axis=1
                )]
                
                if len(wells_in_radius) > 0:
                    grid_points_latlon.append([row_lat, row_lon])
                # If no wells within search radius, skip this grid point silently
    
    total_grid_points = len(grid_points_latlon)
    
    print(f"📐 Generated {total_grid_points} precise {grid_spacing_km:.2f}km grid points within convex hull WITH wells in {search_radius_km}km radius")
    
    # COMPREHENSIVE PRECISION VERIFICATION: Test achieved center-to-center distances
    if total_grid_points > 1:
        print(f"🎯 COMPREHENSIVE PRECISION VERIFICATION:")
        print(f"   Testing actual achieved center-to-center distances...")
        
        # Find grid structure for proper adjacent testing
        grid_structure = {}
        for i, (lat, lon) in enumerate(grid_points_latlon):
            # Group by row (latitude) with small tolerance
            row_key = round(lat / lat_offset_degrees) * lat_offset_degrees
            if row_key not in grid_structure:
                grid_structure[row_key] = []
            grid_structure[row_key].append((i, lat, lon))
        
        horizontal_distances = []
        vertical_distances = []
        
        # Test horizontal distances (within rows)
        for row_lat, row_points in grid_structure.items():
            if len(row_points) > 1:
                # Sort by longitude
                row_points.sort(key=lambda x: x[2])  # Sort by lon
                for j in range(len(row_points) - 1):
                    _, lat1, lon1 = row_points[j]
                    _, lat2, lon2 = row_points[j + 1]
                    distance = get_distance(lat1, lon1, lat2, lon2)
                    horizontal_distances.append(distance)
        
        # Test vertical distances (between rows)
        row_keys = sorted(grid_structure.keys(), reverse=True)  # North to south
        for i in range(len(row_keys) - 1):
            row1_points = grid_structure[row_keys[i]]
            row2_points = grid_structure[row_keys[i + 1]]
            
            # Test a few vertical pairs
            min_pairs = min(len(row1_points), len(row2_points), 5)  # Test up to 5 pairs
            for j in range(min_pairs):
                if j < len(row1_points) and j < len(row2_points):
                    _, lat1, lon1 = row1_points[j]
                    _, lat2, lon2 = row2_points[j] 
                    distance = get_distance(lat1, lon1, lat2, lon2)
                    vertical_distances.append(distance)
        
        # Analyze horizontal spacing precision
        if horizontal_distances:
            h_errors = [abs(d - grid_spacing_km) for d in horizontal_distances]
            h_max_error = max(h_errors)
            h_avg_error = sum(h_errors) / len(h_errors)
            
            print(f"   HORIZONTAL (East-West) spacing:")
            print(f"     Tested: {len(horizontal_distances)} adjacent pairs")
            print(f"     Max error: {h_max_error:.6f}km ({h_max_error*1000:.1f}m)")
            print(f"     Avg error: {h_avg_error:.6f}km ({h_avg_error*1000:.1f}m)")
            
            if h_max_error > 0.001:  # > 1m error
                print(f"     ⚠️  HORIZONTAL ERROR: Max {h_max_error*1000:.1f}m exceeds 1m tolerance")
        
        # Analyze vertical spacing precision  
        if vertical_distances:
            v_errors = [abs(d - grid_spacing_km) for d in vertical_distances]
            v_max_error = max(v_errors)
            v_avg_error = sum(v_errors) / len(v_errors)
            
            print(f"   VERTICAL (North-South) spacing:")
            print(f"     Tested: {len(vertical_distances)} adjacent pairs")
            print(f"     Max error: {v_max_error:.6f}km ({v_max_error*1000:.1f}m)")
            print(f"     Avg error: {v_avg_error:.6f}km ({v_avg_error*1000:.1f}m)")
            
            if v_max_error > 0.001:  # > 1m error
                print(f"     ⚠️  VERTICAL ERROR: Max {v_max_error*1000:.1f}m exceeds 1m tolerance")
        
        # Overall assessment
        all_distances = horizontal_distances + vertical_distances
        if all_distances:
            all_errors = [abs(d - grid_spacing_km) for d in all_distances]
            overall_max_error = max(all_errors)
            
            print(f"   OVERALL GRID PRECISION:")
            if overall_max_error < 0.0001:  # < 10cm  
                print("     🏆 SURVEY-GRADE: All distances within 10cm - professional precision achieved")
            elif overall_max_error < 0.001:  # < 1m
                print("     ✅ EXCELLENT: All distances within 1 meter - suitable for geospatial applications") 
            elif overall_max_error < 0.01:  # < 10m
                print("     ✅ VERY GOOD: All distances within 10 meters - good for seamless joining")
            else:
                print(f"     ❌ CRITICAL: Max error {overall_max_error*1000:.1f}m exceeds acceptable tolerance")
                print(f"     This will cause visible raster offset issues - needs refinement")
        else:
            print("   No adjacent grid points found for precision testing")
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