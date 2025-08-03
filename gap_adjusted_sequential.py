#!/usr/bin/env python3
"""
Gap-adjusted sequential heatmap generation with automatic position correction
"""

import streamlit as st
import pandas as pd
import numpy as np
from measure_heatmap_gaps import get_heatmap_rectangular_bounds, measure_rectangular_edge_gap
from interpolation import generate_geo_json_grid, generate_indicator_kriging_mask
from utils import get_distance

def generate_gap_adjusted_heatmap(wells_data, center_lat, center_lon, heatmap_name, interpolation_method, 
                                 polygon_db, existing_heatmaps=None, soil_polygons=None, banks_peninsula_coords=None,
                                 max_gap_tolerance=0.0001, max_iterations=3):
    """
    Generate a single heatmap with automatic gap adjustment to existing adjacent heatmaps
    
    Args:
        wells_data: Well data for interpolation
        center_lat, center_lon: Initial center position
        heatmap_name: Name for the heatmap
        interpolation_method: Method to use for interpolation
        polygon_db: Database connection for storage
        existing_heatmaps: List of existing heatmap data for gap measurement
        max_gap_tolerance: Maximum allowed gap in km (default 0.0001 km = 10cm)
        max_iterations: Maximum adjustment attempts
        
    Returns:
        tuple: (success, final_heatmap_data, final_center, iterations_used)
    """
    
    current_lat, current_lon = center_lat, center_lon
    iteration = 0
    
    print(f"\n🎯 GENERATING GAP-ADJUSTED HEATMAP: {heatmap_name}")
    print(f"   Initial position: ({current_lat:.6f}, {current_lon:.6f})")
    print(f"   Gap tolerance: {max_gap_tolerance*1000:.1f} meters")
    
    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- ITERATION {iteration} ---")
        print(f"Testing position: ({current_lat:.6f}, {current_lon:.6f})")
        
        # Generate heatmap at current position
        click_point = (current_lat, current_lon)
        try:
            # Use standard interpolation process
            wells_within_radius = []
            search_radius = 20.0  # 20km radius
            
            for _, well in wells_data.iterrows():
                distance = get_distance(current_lat, current_lon, well['NZTM_NORTH'], well['NZTM_EAST'])
                if distance <= search_radius:
                    wells_within_radius.append(well)
            
            if len(wells_within_radius) < 3:
                print(f"❌ Insufficient wells ({len(wells_within_radius)}) at position")
                return False, None, (current_lat, current_lon), iteration
            
            print(f"✅ Found {len(wells_within_radius)} wells within {search_radius}km")
            
            # Generate the interpolation
            wells_df = pd.DataFrame(wells_within_radius)
            
            # Call interpolation function based on method
            if interpolation_method == 'ground_water_level_kriging':
                geojson_data, variance_data, kriging_params = generate_geo_json_grid(
                    wells_df, click_point, search_radius, 
                    interpolation_method='ground_water_level_kriging',
                    soil_polygons=soil_polygons,
                    banks_peninsula_coords=banks_peninsula_coords
                )
            else:
                # Handle other methods as needed
                geojson_data, variance_data, kriging_params = generate_geo_json_grid(
                    wells_df, click_point, search_radius, 
                    interpolation_method=interpolation_method,
                    soil_polygons=soil_polygons,
                    banks_peninsula_coords=banks_peninsula_coords
                )
            
            if not geojson_data or not geojson_data.get('features'):
                print(f"❌ Failed to generate interpolation data")
                return False, None, (current_lat, current_lon), iteration
                
            print(f"✅ Generated heatmap with {len(geojson_data['features'])} features")
            
            # Extract rectangular bounds of new heatmap
            new_bounds = get_heatmap_rectangular_bounds(geojson_data)
            if not new_bounds:
                print(f"❌ Failed to extract bounds from generated heatmap")
                return False, None, (current_lat, current_lon), iteration
                
            print(f"📐 New heatmap bounds: {new_bounds['min_lon']:.6f} to {new_bounds['max_lon']:.6f} (lon)")
            print(f"                      {new_bounds['min_lat']:.6f} to {new_bounds['max_lat']:.6f} (lat)")
            
            # Measure gaps with existing adjacent heatmaps
            if existing_heatmaps and len(existing_heatmaps) > 0:
                gaps_measured = []
                max_gap = 0
                worst_gap_info = None
                
                for existing_heatmap in existing_heatmaps:
                    # Check if adjacent (within ~25km center distance)
                    existing_center = existing_heatmap.get('center', (0, 0))
                    center_distance = get_distance(current_lat, current_lon, existing_center[0], existing_center[1])
                    
                    if center_distance <= 25.0:  # Adjacent heatmap
                        existing_bounds = existing_heatmap.get('bounds')
                        if existing_bounds:
                            gap_distance, edge_info = measure_rectangular_edge_gap(
                                new_bounds, existing_bounds,
                                (current_lat, current_lon), existing_center
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
                            
                            print(f"📏 Gap to {existing_heatmap.get('name', 'unknown')}: {gap_distance:.6f} km ({edge_info})")
                
                if gaps_measured:
                    print(f"📊 Measured {len(gaps_measured)} adjacent gaps, max: {max_gap:.6f} km")
                    
                    # Check if all gaps are within tolerance
                    if max_gap <= max_gap_tolerance:
                        print(f"✅ ALL GAPS WITHIN TOLERANCE ({max_gap*1000:.1f}m ≤ {max_gap_tolerance*1000:.1f}m)")
                        
                        # Store successful heatmap
                        heatmap_data = {
                            'name': heatmap_name,
                            'center': (current_lat, current_lon),
                            'bounds': new_bounds,
                            'geojson_data': geojson_data,
                            'variance_data': variance_data,
                            'kriging_params': kriging_params
                        }
                        
                        return True, heatmap_data, (current_lat, current_lon), iteration
                    
                    else:
                        # Need to adjust position based on worst gap
                        print(f"⚠️ GAP TOO LARGE: {max_gap*1000:.1f}m > {max_gap_tolerance*1000:.1f}m tolerance")
                        
                        if iteration >= max_iterations:
                            print(f"❌ MAXIMUM ITERATIONS REACHED ({max_iterations})")
                            break
                        
                        # Calculate position adjustment based on worst gap
                        adjustment_lat, adjustment_lon = calculate_position_adjustment(
                            worst_gap_info, current_lat, current_lon, new_bounds
                        )
                        
                        print(f"🔧 ADJUSTING POSITION:")
                        print(f"   Gap: {worst_gap_info['distance']:.6f} km ({worst_gap_info['edge_info']})")
                        print(f"   Adjustment: lat {adjustment_lat:.6f}°, lon {adjustment_lon:.6f}°")
                        
                        current_lat += adjustment_lat
                        current_lon += adjustment_lon
                        
                        print(f"   New position: ({current_lat:.6f}, {current_lon:.6f})")
                        continue  # Try again with adjusted position
                        
            else:
                # No existing heatmaps to measure against - accept current position
                print(f"✅ NO EXISTING ADJACENT HEATMAPS - ACCEPTING POSITION")
                
                heatmap_data = {
                    'name': heatmap_name,
                    'center': (current_lat, current_lon),
                    'bounds': new_bounds,
                    'geojson_data': geojson_data,
                    'variance_data': variance_data,
                    'kriging_params': kriging_params
                }
                
                return True, heatmap_data, (current_lat, current_lon), iteration
                
        except Exception as e:
            print(f"❌ Error generating heatmap at iteration {iteration}: {e}")
            return False, None, (current_lat, current_lon), iteration
    
    # If we get here, max iterations reached without success
    print(f"❌ FAILED TO ACHIEVE GAP TOLERANCE AFTER {max_iterations} ITERATIONS")
    return False, None, (current_lat, current_lon), iteration

def calculate_position_adjustment(gap_info, current_lat, current_lon, current_bounds):
    """
    Calculate how to adjust the heatmap position to minimize the gap
    
    Args:
        gap_info: Dictionary with gap measurement info
        current_lat, current_lon: Current center position
        current_bounds: Current heatmap rectangular bounds
        
    Returns:
        tuple: (lat_adjustment, lon_adjustment) in degrees
    """
    
    gap_distance = gap_info['distance']  # Positive = gap, negative = overlap
    edge_info = gap_info['edge_info']
    existing_center = gap_info['existing_center']
    
    # Calculate adjustment direction and magnitude
    # Move to close the gap by approximately half the gap distance
    adjustment_factor = 0.5  # Conservative adjustment
    
    if 'horizontal' in edge_info:
        # Horizontal gap - adjust longitude
        if current_lon < existing_center[1]:  # Current is west of existing
            # Move east to close gap
            lon_adjustment = gap_distance * adjustment_factor / (111.32 * abs(np.cos(np.radians(current_lat))))
        else:  # Current is east of existing
            # Move west to close gap
            lon_adjustment = -gap_distance * adjustment_factor / (111.32 * abs(np.cos(np.radians(current_lat))))
        lat_adjustment = 0
        
    elif 'vertical' in edge_info:
        # Vertical gap - adjust latitude
        if current_lat > existing_center[0]:  # Current is north of existing
            # Move south to close gap
            lat_adjustment = -gap_distance * adjustment_factor / 111.32
        else:  # Current is south of existing
            # Move north to close gap
            lat_adjustment = gap_distance * adjustment_factor / 111.32
        lon_adjustment = 0
        
    else:
        # Diagonal - adjust both (simplified approach)
        lat_adjustment = gap_distance * adjustment_factor * 0.5 / 111.32
        lon_adjustment = gap_distance * adjustment_factor * 0.5 / (111.32 * abs(np.cos(np.radians(current_lat))))
        
        # Adjust direction based on relative positions
        if current_lat > existing_center[0]:
            lat_adjustment = -lat_adjustment
        if current_lon > existing_center[1]:
            lon_adjustment = -lon_adjustment
    
    return lat_adjustment, lon_adjustment

if __name__ == "__main__":
    print("Gap-adjusted sequential heatmap generation module")
    print("Use generate_gap_adjusted_heatmap() function for automatic gap correction")