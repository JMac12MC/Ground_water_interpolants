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
    
    # Use precise 19.82km offset for perfect grid alignment (same as sequential_heatmap.py)
    target_offset_km = 19.82
    
    # Calculate south offset first (constant for latitude)
    km_per_degree_lat = 111.0
    south_offset_degrees = target_offset_km / km_per_degree_lat
    
    # Calculate longitude offset using AVERAGE latitude of the grid (center of 2x3 grid)
    grid_center_lat = clicked_lat - (south_offset_degrees / 2)  # Halfway between top and bottom rows
    km_per_degree_lon = 111.0 * np.cos(np.radians(grid_center_lat))
    
    # Iteratively refine east offset to get exactly 19.82km spacing
    initial_east_degrees = target_offset_km / km_per_degree_lon
    test_distance = get_distance(clicked_lat, clicked_lng, clicked_lat, clicked_lng + initial_east_degrees)
    correction_factor = target_offset_km / test_distance
    east_offset_degrees = initial_east_degrees * correction_factor
    
    # Define all six locations in 2x3 grid (same as sequential_heatmap.py)
    locations = {
        'original': [clicked_lat, clicked_lng],
        'east': [clicked_lat, clicked_lng + east_offset_degrees],
        'northeast': [clicked_lat, clicked_lng + (2 * east_offset_degrees)],
        'south': [clicked_lat - south_offset_degrees, clicked_lng],
        'southeast': [clicked_lat - south_offset_degrees, clicked_lng + east_offset_degrees],
        'far_southeast': [clicked_lat - south_offset_degrees, clicked_lng + (2 * east_offset_degrees)]
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
    # Use example coordinates from the recent generation
    test_click_point = (-43.665, 171.215)  # From the recent ground water level kriging
    calculate_heatmap_distances(test_click_point)