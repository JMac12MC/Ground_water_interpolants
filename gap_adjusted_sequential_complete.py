#!/usr/bin/env python3
"""
Complete gap-adjusted sequential heatmap generation system
This replaces the standard sequential approach with automatic gap measurement and correction
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from measure_heatmap_gaps import get_heatmap_rectangular_bounds, measure_rectangular_edge_gap
from interpolation import generate_geo_json_grid, generate_indicator_kriging_mask
from utils import get_distance, is_within_square

def generate_gap_adjusted_sequential_heatmaps(wells_data, click_point, search_radius, interpolation_method, 
                                             polygon_db, soil_polygons=None, banks_peninsula_coords=None, 
                                             grid_size=None, max_gap_tolerance=0.0001):
    """
    Generate heatmaps sequentially with automatic gap adjustment to ensure seamless joining
    
    Args:
        wells_data: Well dataset
        click_point: Initial click coordinates
        search_radius: Search radius for wells
        interpolation_method: Interpolation method to use
        polygon_db: Database for storage
        soil_polygons: Soil polygon data for clipping
        banks_peninsula_coords: Banks Peninsula exclusion coordinates
        grid_size: Grid dimensions (rows, cols)
        max_gap_tolerance: Maximum allowed gap in km (default 0.0001 km = 10cm)
        
    Returns:
        tuple: (success_count, stored_heatmap_ids, error_messages)
    """
    
    print(f"🎯 STARTING GAP-ADJUSTED SEQUENTIAL HEATMAP GENERATION")
    print(f"   Gap tolerance: {max_gap_tolerance*1000:.1f} meters")
    print(f"   Target spacing: 19.82 km between centers")
    
    # Calculate grid positions using existing precise calculations
    clicked_lat, clicked_lng = click_point
    
    if grid_size is None:
        grid_rows, grid_cols = 2, 3  # Default 2x3 grid
    else:
        grid_rows, grid_cols = grid_size
    
    # Use the same ultra-precise geodetic calculations from sequential_heatmap.py
    target_offset_km = 19.82
    MAX_ITERATIONS = 200
    TOLERANCE_KM = 0.0001
    ADAPTIVE_STEP_SIZE = 0.000001
    
    # Step 1: Ultra-precise latitude offset
    lat_offset_degrees = target_offset_km / 111.0
    best_lat_offset = lat_offset_degrees
    best_lat_error = float('inf')
    
    for i in range(MAX_ITERATIONS):
        test_lat = clicked_lat - lat_offset_degrees
        actual_distance = get_distance(clicked_lat, clicked_lng, test_lat, clicked_lng)
        error = abs(actual_distance - target_offset_km)
        
        if error < best_lat_error:
            best_lat_offset = lat_offset_degrees
            best_lat_error = error
        
        if error < TOLERANCE_KM:
            break
            
        if error > 0.001:
            adjustment_factor = target_offset_km / actual_distance  
            lat_offset_degrees *= adjustment_factor
        else:
            step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)
            if actual_distance > target_offset_km:
                lat_offset_degrees -= step_size
            else:
                lat_offset_degrees += step_size
    
    lat_offset_degrees = best_lat_offset
    south_offset_degrees = lat_offset_degrees
    
    # Step 2: Row-specific longitude offsets
    row_longitude_offsets = []
    for row in range(grid_rows):
        current_lat = clicked_lat - (row * south_offset_degrees)
        
        lon_offset_degrees = target_offset_km / (111.32 * abs(np.cos(np.radians(current_lat))))
        best_lon_offset = lon_offset_degrees
        best_lon_error = float('inf')
        
        for i in range(MAX_ITERATIONS):
            test_lon = clicked_lng + lon_offset_degrees
            actual_distance = get_distance(current_lat, clicked_lng, current_lat, test_lon)
            error = abs(actual_distance - target_offset_km)
            
            if error < best_lon_error:
                best_lon_offset = lon_offset_degrees
                best_lon_error = error
            
            if error < TOLERANCE_KM:
                break
                
            if error > 0.001:
                adjustment_factor = target_offset_km / actual_distance
                lon_offset_degrees *= adjustment_factor
            else:
                step_size = max(ADAPTIVE_STEP_SIZE, error / 10.0)
                if actual_distance > target_offset_km:
                    lon_offset_degrees -= step_size
                else:
                    lon_offset_degrees += step_size
        
        row_longitude_offsets.append(best_lon_offset)
    
    # Step 3: Generate all grid positions
    locations = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            lat = clicked_lat - (row * south_offset_degrees)
            lng = clicked_lng + (col * row_longitude_offsets[row])
            
            if grid_rows == 2 and grid_cols == 3:
                names_2x3 = [
                    ['original', 'east', 'northeast'],
                    ['south', 'southeast', 'far_southeast']
                ]
                name = names_2x3[row][col]
            else:
                name = f"r{row}c{col}"
                if row == 0 and col == 0:
                    name = "original"
            
            locations.append((name, [lat, lng]))
    
    print(f"📍 GRID LAYOUT: {len(locations)} positions in {grid_rows}x{grid_cols} grid")
    for i, (name, coords) in enumerate(locations):
        print(f"   {i+1}. {name.upper()}: ({coords[0]:.6f}, {coords[1]:.6f})")
    
    # Calculate global colormap once
    print(f"🎨 CALCULATING GLOBAL COLORMAP RANGE...")
    global_values = []
    
    for location_name, center_point in locations:
        wells_df_temp = wells_data.copy()
        wells_df_temp['within_square'] = wells_df_temp.apply(
            lambda row: is_within_square(
                row['latitude'], row['longitude'],
                center_point[0], center_point[1], search_radius
            ), axis=1
        )
        filtered_wells_temp = wells_df_temp[wells_df_temp['within_square']]
        
        if len(filtered_wells_temp) > 0:
            if interpolation_method == 'indicator_kriging':
                global_values.extend([0.0, 1.0])
            else:
                if interpolation_method == 'ground_water_level_kriging':
                    values = filtered_wells_temp['ground_water_level'].dropna()
                    if len(values) > 0:
                        global_values.extend(values.tolist())
                else:
                    yield_values = filtered_wells_temp['yield_rate'].dropna()
                    if len(yield_values) > 0:
                        global_values.extend(yield_values.tolist())
    
    if global_values:
        global_min_value = min(global_values)
        global_max_value = max(global_values)
        global_percentiles = np.percentile(global_values, np.linspace(0, 100, num=256))
        percentile_25 = np.percentile(global_values, 25)
        percentile_50 = np.percentile(global_values, 50)
        percentile_75 = np.percentile(global_values, 75)
        print(f"🎨 GLOBAL RANGE: {global_min_value:.2f} to {global_max_value:.2f}")
        print(f"🎨 PERCENTILES: 25th={percentile_25:.2f}, 50th={percentile_50:.2f}, 75th={percentile_75:.2f}")
    else:
        if interpolation_method == 'indicator_kriging':
            global_min_value, global_max_value = 0.0, 1.0
        elif interpolation_method == 'ground_water_level_kriging':
            global_min_value, global_max_value = 0.0, 180.0
        else:
            global_min_value, global_max_value = 0.0, 25.0
        global_percentiles = None
        percentile_25 = percentile_50 = percentile_75 = None
        print(f"🎨 FALLBACK RANGE: {global_min_value:.2f} to {global_max_value:.2f}")
    
    # Store colormap metadata
    colormap_metadata = {
        'global_min': global_min_value,
        'global_max': global_max_value,
        'method': interpolation_method,
        'generated_at': pd.Timestamp.now().isoformat(),
        'percentiles': {
            '25th': percentile_25,
            '50th': percentile_50,
            '75th': percentile_75
        } if percentile_25 is not None else None,
        'total_values': len(global_values)
    }
    
    # Process each location with gap adjustment
    generated_heatmaps = []
    stored_heatmap_ids = []
    error_messages = []
    existing_heatmaps = []  # Track generated heatmaps for gap measurement
    
    for location_idx, (location_name, center_point) in enumerate(locations):
        print(f"\n🎯 PROCESSING LOCATION {location_idx + 1}/{len(locations)}: {location_name.upper()}")
        
        try:
            # Filter wells for this location
            center_lat, center_lon = center_point
            wells_df = wells_data.copy()
            wells_df['within_square'] = wells_df.apply(
                lambda row: is_within_square(
                    row['latitude'], row['longitude'],
                    center_lat, center_lon, search_radius
                ), axis=1
            )
            filtered_wells = wells_df[wells_df['within_square']]
            
            if len(filtered_wells) < 3:
                error_msg = f"Insufficient wells ({len(filtered_wells)}) for {location_name}"
                error_messages.append(error_msg)
                print(f"   ❌ {error_msg}")
                continue
            
            print(f"   ✅ Found {len(filtered_wells)} wells within {search_radius}km")
            
            # Generate heatmap with gap adjustment (if not the first one)
            success = False
            final_center = center_point
            iterations = 0
            
            if location_idx == 0:
                # First heatmap - no gap adjustment needed
                print(f"   📍 FIRST HEATMAP - Using original position")
                success, heatmap_data, final_center, iterations = generate_single_heatmap(
                    filtered_wells, center_point, location_name, interpolation_method,
                    search_radius, soil_polygons, banks_peninsula_coords, colormap_metadata
                )
            else:
                # Subsequent heatmaps - use gap adjustment
                print(f"   🔧 GAP-ADJUSTED GENERATION (tolerance: {max_gap_tolerance*1000:.1f}m)")
                success, heatmap_data, final_center, iterations = generate_gap_adjusted_heatmap(
                    filtered_wells, center_point, location_name, interpolation_method,
                    search_radius, existing_heatmaps, soil_polygons, banks_peninsula_coords,
                    colormap_metadata, max_gap_tolerance
                )
            
            if success and heatmap_data:
                print(f"   ✅ {location_name.upper()}: Generated successfully (iterations: {iterations})")
                
                # Add to existing heatmaps for future gap measurements
                existing_heatmaps.append(heatmap_data)
                
                # Store in database
                if polygon_db:
                    center_lat, center_lon = final_center
                    if location_name == 'original':
                        heatmap_name = f"{interpolation_method}_{center_lat:.3f}_{center_lon:.3f}"
                    else:
                        heatmap_name = f"{interpolation_method}_{location_name}_{center_lat:.3f}_{center_lon:.3f}"
                    
                    # Convert GeoJSON to heatmap data format
                    heatmap_data_points = []
                    geojson_data = heatmap_data['geojson_data']
                    for feature in geojson_data.get('features', []):
                        if 'geometry' in feature and 'properties' in feature:
                            geom = feature['geometry']
                            if geom['type'] == 'Polygon' and len(geom['coordinates']) > 0:
                                coords = geom['coordinates'][0]
                                if len(coords) >= 3:
                                    lat = sum(coord[1] for coord in coords) / len(coords)
                                    lon = sum(coord[0] for coord in coords) / len(coords)
                                    if interpolation_method == 'ground_water_level_kriging':
                                        value = feature['properties'].get('ground_water_level', 0)
                                    else:
                                        value = feature['properties'].get('yield', 0)
                                    heatmap_data_points.append([lat, lon, value])
                    
                    # Store in database
                    stored_heatmap_id = polygon_db.store_heatmap(
                        heatmap_name=heatmap_name,
                        center_lat=float(center_lat),
                        center_lon=float(center_lon),
                        radius_km=search_radius,
                        interpolation_method=interpolation_method,
                        heatmap_data=heatmap_data_points,
                        geojson_data=geojson_data,
                        well_count=len(filtered_wells),
                        colormap_metadata=colormap_metadata
                    )
                    
                    if stored_heatmap_id:
                        stored_heatmap_ids.append((location_name, stored_heatmap_id))
                        print(f"   💾 Stored as ID {stored_heatmap_id}")
                    else:
                        print(f"   ⚠️  Already exists in database")
                
                generated_heatmaps.append((location_name, final_center, heatmap_data['geojson_data'], len(filtered_wells)))
                
            else:
                error_msg = f"Failed to generate {location_name} after {iterations} iterations"
                error_messages.append(error_msg)
                print(f"   ❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"Exception processing {location_name}: {str(e)}"
            error_messages.append(error_msg)
            print(f"   ❌ {error_msg}")
            continue
    
    print(f"\n🏁 GAP-ADJUSTED SEQUENTIAL GENERATION COMPLETE:")
    print(f"   ✅ Successfully generated: {len(generated_heatmaps)} heatmaps")
    print(f"   ❌ Errors: {len(error_messages)}")
    print(f"   💾 Stored in database: {len(stored_heatmap_ids)}")
    
    return len(generated_heatmaps), stored_heatmap_ids, error_messages

def generate_single_heatmap(filtered_wells, center_point, location_name, interpolation_method,
                           search_radius, soil_polygons, banks_peninsula_coords, colormap_metadata):
    """Generate a single heatmap without gap adjustment"""
    try:
        # Generate indicator mask if needed
        indicator_mask = None
        if interpolation_method == 'indicator_kriging':
            try:
                indicator_mask = generate_indicator_kriging_mask(
                    filtered_wells, center_point, search_radius,
                    resolution=100, soil_polygons=soil_polygons, threshold=0.7
                )
            except Exception as e:
                print(f"   Warning: Could not generate indicator mask: {e}")
        
        # Generate the interpolation
        geojson_data = generate_geo_json_grid(
            filtered_wells.copy(),
            center_point,
            search_radius,
            resolution=100,
            method=interpolation_method,
            show_variance=False,
            auto_fit_variogram=True,
            variogram_model='spherical',
            soil_polygons=soil_polygons,
            indicator_mask=indicator_mask,
            banks_peninsula_coords=banks_peninsula_coords
        )
        
        if geojson_data and len(geojson_data.get('features', [])) > 0:
            # Extract rectangular bounds
            bounds = get_heatmap_rectangular_bounds(geojson_data)
            
            heatmap_data = {
                'name': location_name,
                'center': center_point,
                'bounds': bounds,
                'geojson_data': geojson_data,
                'colormap_metadata': colormap_metadata
            }
            
            return True, heatmap_data, center_point, 1
        else:
            return False, None, center_point, 1
            
    except Exception as e:
        print(f"   Error generating single heatmap: {e}")
        return False, None, center_point, 1

def generate_gap_adjusted_heatmap(filtered_wells, initial_center, location_name, interpolation_method,
                                 search_radius, existing_heatmaps, soil_polygons, banks_peninsula_coords,
                                 colormap_metadata, max_gap_tolerance, max_iterations=3):
    """Generate a heatmap with automatic position adjustment to minimize gaps"""
    
    current_center = list(initial_center)
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        print(f"     Iteration {iteration}: Testing position ({current_center[0]:.6f}, {current_center[1]:.6f})")
        
        try:
            # Generate heatmap at current position
            success, heatmap_data, _, _ = generate_single_heatmap(
                filtered_wells, current_center, location_name, interpolation_method,
                search_radius, soil_polygons, banks_peninsula_coords, colormap_metadata
            )
            
            if not success or not heatmap_data:
                print(f"     ❌ Failed to generate interpolation")
                return False, None, current_center, iteration
            
            # Measure gaps with existing adjacent heatmaps
            gaps_measured = []
            max_gap = 0
            worst_gap_info = None
            
            for existing_heatmap in existing_heatmaps:
                # Check if adjacent (within ~25km center distance)
                existing_center = existing_heatmap.get('center', (0, 0))
                center_distance = get_distance(
                    current_center[0], current_center[1],
                    existing_center[0], existing_center[1]
                )
                
                if center_distance <= 25.0:  # Adjacent heatmap
                    existing_bounds = existing_heatmap.get('bounds')
                    if existing_bounds and heatmap_data['bounds']:
                        gap_distance, edge_info = measure_rectangular_edge_gap(
                            heatmap_data['bounds'], existing_bounds,
                            current_center, existing_center
                        )
                        
                        gap_abs = abs(gap_distance)
                        gaps_measured.append({
                            'distance': gap_distance,
                            'abs_distance': gap_abs,
                            'edge_info': edge_info,
                            'existing_name': existing_heatmap.get('name', 'unknown'),
                            'existing_center': existing_center
                        })
                        
                        if gap_abs > max_gap:
                            max_gap = gap_abs
                            worst_gap_info = gaps_measured[-1]
                        
                        print(f"     📏 Gap to {existing_heatmap.get('name', 'unknown')}: {gap_distance:.6f} km ({edge_info})")
            
            if gaps_measured:
                print(f"     📊 Max gap: {max_gap:.6f} km")
                
                # Check if all gaps are within tolerance
                if max_gap <= max_gap_tolerance:
                    print(f"     ✅ ALL GAPS WITHIN TOLERANCE ({max_gap*1000:.1f}m ≤ {max_gap_tolerance*1000:.1f}m)")
                    return True, heatmap_data, current_center, iteration
                else:
                    if iteration >= max_iterations:
                        print(f"     ❌ MAX ITERATIONS REACHED")
                        break
                    
                    # Calculate position adjustment
                    print(f"     ⚠️ GAP TOO LARGE: {max_gap*1000:.1f}m > {max_gap_tolerance*1000:.1f}m")
                    adjustment_lat, adjustment_lon = calculate_position_adjustment(
                        worst_gap_info, current_center[0], current_center[1], heatmap_data['bounds']
                    )
                    
                    print(f"     🔧 Adjusting position: lat {adjustment_lat:.6f}°, lon {adjustment_lon:.6f}°")
                    current_center[0] += adjustment_lat
                    current_center[1] += adjustment_lon
                    continue
            else:
                # No adjacent heatmaps
                print(f"     ✅ NO ADJACENT HEATMAPS - ACCEPTING POSITION")
                return True, heatmap_data, current_center, iteration
                
        except Exception as e:
            print(f"     ❌ Error at iteration {iteration}: {e}")
            return False, None, current_center, iteration
    
    print(f"     ❌ FAILED AFTER {max_iterations} ITERATIONS")
    return False, None, current_center, iteration

def calculate_position_adjustment(gap_info, current_lat, current_lon, current_bounds):
    """Calculate position adjustment to minimize gap"""
    gap_distance = gap_info['distance']
    edge_info = gap_info['edge_info']
    existing_center = gap_info['existing_center']
    
    adjustment_factor = 0.5  # Conservative adjustment
    
    if 'horizontal' in edge_info:
        # Horizontal gap - adjust longitude
        if current_lon < existing_center[1]:
            lon_adjustment = gap_distance * adjustment_factor / (111.32 * abs(np.cos(np.radians(current_lat))))
        else:
            lon_adjustment = -gap_distance * adjustment_factor / (111.32 * abs(np.cos(np.radians(current_lat))))
        lat_adjustment = 0
    elif 'vertical' in edge_info:
        # Vertical gap - adjust latitude
        if current_lat > existing_center[0]:
            lat_adjustment = -gap_distance * adjustment_factor / 111.32
        else:
            lat_adjustment = gap_distance * adjustment_factor / 111.32
        lon_adjustment = 0
    else:
        # Diagonal adjustment
        lat_adjustment = gap_distance * adjustment_factor * 0.5 / 111.32
        lon_adjustment = gap_distance * adjustment_factor * 0.5 / (111.32 * abs(np.cos(np.radians(current_lat))))
        
        if current_lat > existing_center[0]:
            lat_adjustment = -lat_adjustment
        if current_lon > existing_center[1]:
            lon_adjustment = -lon_adjustment
    
    return lat_adjustment, lon_adjustment

if __name__ == "__main__":
    print("Gap-adjusted sequential heatmap generation system")