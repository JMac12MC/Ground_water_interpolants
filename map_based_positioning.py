#!/usr/bin/env python3
"""
Map-Based Positioning System
Uses actual map distance measurements to calculate precise grid positions
"""

import numpy as np
from utils import get_distance
import math

def calculate_map_based_grid_positions(click_point, grid_size=(2, 3), target_distance_km=19.82):
    """
    Calculate grid positions using map-based distance measurements.
    Measures actual 19.82km on the map surface, then converts to coordinates.
    """
    grid_rows, grid_cols = grid_size
    clicked_lat, clicked_lng = click_point
    
    print(f"🗺️ MAP-BASED GRID CALCULATION")
    print(f"📍 Click point: ({clicked_lat:.6f}, {clicked_lng:.6f})")
    print(f"📏 Target distance: {target_distance_km}km")
    
    # Step 1: Calculate precise south offset using map distance
    south_offset_degrees = calculate_precise_south_offset(clicked_lat, clicked_lng, target_distance_km)
    print(f"🧭 South offset: {south_offset_degrees:.8f}° ({target_distance_km}km)")
    
    # Step 2: Calculate east offsets for each row (accounting for latitude changes)
    row_positions = []
    for row in range(grid_rows):
        current_lat = clicked_lat - (row * south_offset_degrees)
        east_offset_degrees = calculate_precise_east_offset(current_lat, clicked_lng, target_distance_km)
        
        row_cols = []
        for col in range(grid_cols):
            current_lng = clicked_lng + (col * east_offset_degrees)
            row_cols.append([current_lat, current_lng])
        
        row_positions.append(row_cols)
        print(f"🧭 Row {row} (lat {current_lat:.6f}): East offset {east_offset_degrees:.8f}° ({target_distance_km}km)")
    
    # Step 3: Generate named positions
    locations = []
    names_2x3 = [
        ['original', 'east', 'northeast'],
        ['south', 'southeast', 'far_southeast']
    ]
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            lat, lng = row_positions[row][col]
            name = names_2x3[row][col] if grid_rows == 2 and grid_cols == 3 else f"r{row}c{col}"
            if row == 0 and col == 0:
                name = "original"
            locations.append((name, [lat, lng]))
    
    # Step 4: Verify distances
    print(f"\n🔍 DISTANCE VERIFICATION:")
    for i, (name, coords) in enumerate(locations):
        print(f"   {i+1}. {name.upper()}: ({coords[0]:.6f}, {coords[1]:.6f})")
        
        # Check adjacent distances
        row = i // grid_cols
        col = i % grid_cols
        
        # Check east neighbor
        if col < grid_cols - 1:
            neighbor_idx = row * grid_cols + (col + 1)
            neighbor_name, neighbor_coords = locations[neighbor_idx]
            distance = get_distance(coords[0], coords[1], neighbor_coords[0], neighbor_coords[1])
            error = abs(distance - target_distance_km)
            print(f"      → {neighbor_name}: {distance:.6f}km (error: {error*1000:.1f}m)")
        
        # Check south neighbor
        if row < grid_rows - 1:
            neighbor_idx = (row + 1) * grid_cols + col
            neighbor_name, neighbor_coords = locations[neighbor_idx]
            distance = get_distance(coords[0], coords[1], neighbor_coords[0], neighbor_coords[1])
            error = abs(distance - target_distance_km)
            print(f"      ↓ {neighbor_name}: {distance:.6f}km (error: {error*1000:.1f}m)")
    
    return locations

def calculate_precise_south_offset(lat, lng, target_km):
    """Calculate precise south offset using iterative map distance measurement"""
    # Initial estimate: 1 degree latitude ≈ 111.32 km
    offset_degrees = target_km / 111.32
    
    # Iterative refinement
    for iteration in range(10):
        test_lat = lat - offset_degrees
        actual_distance = get_distance(lat, lng, test_lat, lng)
        error = abs(actual_distance - target_km)
        
        if error < 0.001:  # 1 meter accuracy
            break
            
        # Adjust based on error
        correction_factor = target_km / actual_distance
        offset_degrees *= correction_factor
    
    return offset_degrees

def calculate_precise_east_offset(lat, lng, target_km):
    """Calculate precise east offset for given latitude using iterative measurement"""
    # Initial estimate accounting for latitude
    cos_lat = abs(math.cos(math.radians(lat)))
    offset_degrees = target_km / (111.32 * cos_lat)
    
    # Iterative refinement
    for iteration in range(10):
        test_lng = lng + offset_degrees
        actual_distance = get_distance(lat, lng, lat, test_lng)
        error = abs(actual_distance - target_km)
        
        if error < 0.001:  # 1 meter accuracy
            break
            
        # Adjust based on error
        correction_factor = target_km / actual_distance
        offset_degrees *= correction_factor
    
    return offset_degrees

def verify_grid_precision(locations, target_distance_km=19.82):
    """Verify the precision of the generated grid"""
    grid_cols = 3  # For 2x3 grid
    total_errors = []
    max_error = 0
    
    print(f"\n📐 GRID PRECISION ANALYSIS:")
    
    for i, (name, coords) in enumerate(locations):
        row = i // grid_cols
        col = i % grid_cols
        
        # Check all adjacent positions
        adjacent_errors = []
        
        # East neighbor
        if col < grid_cols - 1:
            neighbor_idx = row * grid_cols + (col + 1)
            neighbor_coords = locations[neighbor_idx][1]
            distance = get_distance(coords[0], coords[1], neighbor_coords[0], neighbor_coords[1])
            error = abs(distance - target_distance_km)
            adjacent_errors.append(error)
            total_errors.append(error)
            max_error = max(max_error, error)
        
        # South neighbor
        if row < 1:  # Only check south for top row
            neighbor_idx = (row + 1) * grid_cols + col
            neighbor_coords = locations[neighbor_idx][1]
            distance = get_distance(coords[0], coords[1], neighbor_coords[0], neighbor_coords[1])
            error = abs(distance - target_distance_km)
            adjacent_errors.append(error)
            total_errors.append(error)
            max_error = max(max_error, error)
        
        if adjacent_errors:
            avg_error = sum(adjacent_errors) / len(adjacent_errors)
            print(f"   {name.upper()}: Avg error {avg_error*1000:.1f}m")
    
    if total_errors:
        overall_avg_error = sum(total_errors) / len(total_errors)
        print(f"\n📊 OVERALL PRECISION:")
        print(f"   Average error: {overall_avg_error*1000:.1f} meters")
        print(f"   Maximum error: {max_error*1000:.1f} meters")
        print(f"   Total measurements: {len(total_errors)}")
        
        if max_error < 0.001:  # Less than 1 meter
            print(f"   ✅ EXCELLENT: All distances within 1 meter of target")
        elif max_error < 0.01:  # Less than 10 meters
            print(f"   ✅ GOOD: All distances within 10 meters of target")
        else:
            print(f"   ⚠️ NEEDS IMPROVEMENT: Some distances exceed 10 meters")
    
    return max_error, total_errors

if __name__ == "__main__":
    # Test the map-based positioning
    test_point = [-43.315, 172.086]
    locations = calculate_map_based_grid_positions(test_point)
    verify_grid_precision(locations)