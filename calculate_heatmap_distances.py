#!/usr/bin/env python3
"""
Calculate distances between heatmap centers in the 2x3 grid layout
"""

import numpy as np
from utils import get_distance

def calculate_heatmap_distances(click_point):
    """
    Calculate all distances between heatmap centers in the 2x3 grid
    
    Grid Layout:
    [Original] [East] [Northeast]
    [South] [Southeast] [Far_Southeast]
    """
    
    clicked_lat, clicked_lng = click_point
    
    # Use PERFECT 19.82km offset - exactly matching sequential_heatmap.py implementation
    target_offset_km = 19.82
    
    # Step 1: Calculate precise south offset
    km_per_degree_lat = 111.0
    initial_south_degrees = target_offset_km / km_per_degree_lat
    test_south_point = [clicked_lat - initial_south_degrees, clicked_lng]
    actual_south_distance = get_distance(clicked_lat, clicked_lng, test_south_point[0], test_south_point[1])
    south_correction = target_offset_km / actual_south_distance
    south_offset_degrees = initial_south_degrees * south_correction
    
    # Step 2: Calculate east offset for TOP ROW (original latitude)
    top_lat = clicked_lat
    km_per_degree_lon_top = 111.0 * np.cos(np.radians(top_lat))
    initial_east_degrees_top = target_offset_km / km_per_degree_lon_top
    test_east_point_top = [top_lat, clicked_lng + initial_east_degrees_top]
    actual_east_distance_top = get_distance(top_lat, clicked_lng, test_east_point_top[0], test_east_point_top[1])
    east_correction_top = target_offset_km / actual_east_distance_top
    east_offset_degrees_top = initial_east_degrees_top * east_correction_top
    
    # Step 3: Calculate east offset for BOTTOM ROW
    bottom_lat = clicked_lat - south_offset_degrees
    km_per_degree_lon_bottom = 111.0 * np.cos(np.radians(bottom_lat))
    initial_east_degrees_bottom = target_offset_km / km_per_degree_lon_bottom
    test_east_point_bottom = [bottom_lat, clicked_lng + initial_east_degrees_bottom]
    actual_east_distance_bottom = get_distance(bottom_lat, clicked_lng, test_east_point_bottom[0], test_east_point_bottom[1])
    east_correction_bottom = target_offset_km / actual_east_distance_bottom
    east_offset_degrees_bottom = initial_east_degrees_bottom * east_correction_bottom
    
    # Define all six locations using row-specific east offsets for perfect spacing
    locations = {
        'original': [clicked_lat, clicked_lng],
        'east': [clicked_lat, clicked_lng + east_offset_degrees_top],
        'northeast': [clicked_lat, clicked_lng + (2 * east_offset_degrees_top)],
        'south': [clicked_lat - south_offset_degrees, clicked_lng],
        'southeast': [clicked_lat - south_offset_degrees, clicked_lng + east_offset_degrees_bottom],
        'far_southeast': [clicked_lat - south_offset_degrees, clicked_lng + (2 * east_offset_degrees_bottom)]
    }
    
    print("HEATMAP DISTANCE ANALYSIS")
    print("=" * 50)
    print(f"Click point: ({clicked_lat:.6f}, {clicked_lng:.6f})")
    print(f"Target offset: {target_offset_km:.2f} km")
    print()
    
    # Show all heatmap centers
    print("HEATMAP CENTERS:")
    for name, coords in locations.items():
        print(f"  {name.upper():12}: ({coords[0]:8.4f}, {coords[1]:9.4f})")
    print()
    
    # Calculate distances between all pairs
    location_names = list(locations.keys())
    
    print("DISTANCE MATRIX (km):")
    print("=" * 80)
    
    # Header
    header = "FROM \\ TO".ljust(12)
    for name in location_names:
        header += f"{name.upper()[:8]:>10}"
    print(header)
    print("-" * 80)
    
    # Distance matrix
    for from_name in location_names:
        row = f"{from_name.upper()[:8]:12}"
        from_coords = locations[from_name]
        
        for to_name in location_names:
            to_coords = locations[to_name]
            
            if from_name == to_name:
                distance = 0.0
            else:
                distance = get_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
            
            row += f"{distance:10.2f}"
        
        print(row)
    
    print()
    print("ADJACENT HEATMAP DISTANCES:")
    print("=" * 40)
    
    # Horizontal neighbors (same row)
    horizontal_pairs = [
        ('original', 'east'),
        ('east', 'northeast'),
        ('south', 'southeast'), 
        ('southeast', 'far_southeast')
    ]
    
    for from_name, to_name in horizontal_pairs:
        from_coords = locations[from_name]
        to_coords = locations[to_name]
        distance = get_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
        print(f"  {from_name.upper()} ↔ {to_name.upper()}: {distance:.2f} km")
    
    print()
    
    # Vertical neighbors (same column)
    vertical_pairs = [
        ('original', 'south'),
        ('east', 'southeast'),
        ('northeast', 'far_southeast')
    ]
    
    for from_name, to_name in vertical_pairs:
        from_coords = locations[from_name]
        to_coords = locations[to_name]
        distance = get_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
        print(f"  {from_name.upper()} ↕ {to_name.upper()}: {distance:.2f} km")
    
    print()
    
    # Diagonal neighbors
    diagonal_pairs = [
        ('original', 'southeast'),
        ('east', 'south'),
        ('east', 'far_southeast'),
        ('northeast', 'southeast')
    ]
    
    print("DIAGONAL DISTANCES:")
    for from_name, to_name in diagonal_pairs:
        from_coords = locations[from_name]
        to_coords = locations[to_name]
        distance = get_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
        print(f"  {from_name.upper()} ↗ {to_name.upper()}: {distance:.2f} km")

if __name__ == "__main__":
    # Use coordinates from the recently generated heatmaps
    test_click_point = (-43.620, 171.197)  # From the most recent kriging generation
    calculate_heatmap_distances(test_click_point)