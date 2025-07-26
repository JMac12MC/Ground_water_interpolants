#!/usr/bin/env python3
"""
Calculate perfect heatmap spacing where ALL adjacent distances are exactly 19.82 km
"""

import numpy as np
from utils import get_distance

def calculate_perfect_grid_spacing(click_lat, click_lng, target_distance_km=19.82):
    """
    Calculate grid positions where ALL adjacent heatmaps are exactly target_distance_km apart
    """
    
    # Step 1: Calculate precise south offset
    km_per_degree_lat = 111.0
    initial_south_degrees = target_distance_km / km_per_degree_lat
    
    # Verify and refine south offset
    test_south_point = [click_lat - initial_south_degrees, click_lng]
    actual_south_distance = get_distance(click_lat, click_lng, test_south_point[0], test_south_point[1])
    south_correction = target_distance_km / actual_south_distance
    south_offset_degrees = initial_south_degrees * south_correction
    
    print(f"South offset calculation:")
    print(f"  Initial: {initial_south_degrees:.8f} degrees")
    print(f"  Actual distance: {actual_south_distance:.4f} km")
    print(f"  Refined: {south_offset_degrees:.8f} degrees")
    
    # Step 2: Calculate east offset for TOP ROW (original latitude)
    top_lat = click_lat
    km_per_degree_lon_top = 111.0 * np.cos(np.radians(top_lat))
    initial_east_degrees_top = target_distance_km / km_per_degree_lon_top
    
    # Verify and refine east offset for top row
    test_east_point_top = [top_lat, click_lng + initial_east_degrees_top]
    actual_east_distance_top = get_distance(top_lat, click_lng, test_east_point_top[0], test_east_point_top[1])
    east_correction_top = target_distance_km / actual_east_distance_top
    east_offset_degrees_top = initial_east_degrees_top * east_correction_top
    
    # Step 3: Calculate east offset for BOTTOM ROW
    bottom_lat = click_lat - south_offset_degrees
    km_per_degree_lon_bottom = 111.0 * np.cos(np.radians(bottom_lat))
    initial_east_degrees_bottom = target_distance_km / km_per_degree_lon_bottom
    
    # Verify and refine east offset for bottom row
    test_east_point_bottom = [bottom_lat, click_lng + initial_east_degrees_bottom]
    actual_east_distance_bottom = get_distance(bottom_lat, click_lng, test_east_point_bottom[0], test_east_point_bottom[1])
    east_correction_bottom = target_distance_km / actual_east_distance_bottom
    east_offset_degrees_bottom = initial_east_degrees_bottom * east_correction_bottom
    
    print(f"\nEast offset calculations:")
    print(f"  Top row (lat {top_lat:.4f}):")
    print(f"    Initial: {initial_east_degrees_top:.8f} degrees")
    print(f"    Actual distance: {actual_east_distance_top:.4f} km")
    print(f"    Refined: {east_offset_degrees_top:.8f} degrees")
    print(f"  Bottom row (lat {bottom_lat:.4f}):")
    print(f"    Initial: {initial_east_degrees_bottom:.8f} degrees")
    print(f"    Actual distance: {actual_east_distance_bottom:.4f} km")
    print(f"    Refined: {east_offset_degrees_bottom:.8f} degrees")
    
    # Step 4: Define all six locations using row-specific east offsets
    locations = {
        'original': [click_lat, click_lng],
        'east': [click_lat, click_lng + east_offset_degrees_top],
        'northeast': [click_lat, click_lng + (2 * east_offset_degrees_top)],
        'south': [click_lat - south_offset_degrees, click_lng],
        'southeast': [click_lat - south_offset_degrees, click_lng + east_offset_degrees_bottom],
        'far_southeast': [click_lat - south_offset_degrees, click_lng + (2 * east_offset_degrees_bottom)]
    }
    
    return locations, south_offset_degrees, east_offset_degrees_top, east_offset_degrees_bottom

def verify_perfect_spacing(locations, target_distance=19.82):
    """
    Verify that all adjacent distances are exactly the target distance
    """
    print(f"\n🎯 VERIFICATION: All distances should be exactly {target_distance:.2f} km")
    print("=" * 60)
    
    print("HEATMAP CENTERS:")
    for name, coords in locations.items():
        print(f"  {name.upper():12}: ({coords[0]:8.4f}, {coords[1]:9.4f})")
    
    print(f"\nADJACENT DISTANCES:")
    
    # Horizontal neighbors (same row)
    horizontal_pairs = [
        ('original', 'east'),
        ('east', 'northeast'),
        ('south', 'southeast'), 
        ('southeast', 'far_southeast')
    ]
    
    all_perfect = True
    for from_name, to_name in horizontal_pairs:
        from_coords = locations[from_name]
        to_coords = locations[to_name]
        distance = get_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
        is_perfect = abs(distance - target_distance) < 0.001
        status = "✅" if is_perfect else "❌"
        print(f"  {status} {from_name.upper()} ↔ {to_name.upper()}: {distance:.4f} km")
        if not is_perfect:
            all_perfect = False
    
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
        is_perfect = abs(distance - target_distance) < 0.001
        status = "✅" if is_perfect else "❌"
        print(f"  {status} {from_name.upper()} ↕ {to_name.upper()}: {distance:.4f} km")
        if not is_perfect:
            all_perfect = False
    
    if all_perfect:
        print(f"\n🎉 SUCCESS: All adjacent heatmaps are exactly {target_distance:.2f} km apart!")
    else:
        print(f"\n⚠️  Some distances are not exactly {target_distance:.2f} km")
    
    return all_perfect

if __name__ == "__main__":
    # Test with the example coordinates
    test_lat, test_lng = -43.665, 171.215
    print(f"Calculating perfect heatmap spacing for point ({test_lat}, {test_lng})")
    
    locations, south_deg, east_top_deg, east_bottom_deg = calculate_perfect_grid_spacing(test_lat, test_lng)
    is_perfect = verify_perfect_spacing(locations)
    
    if is_perfect:
        print(f"\n🎯 IMPLEMENTATION VALUES:")
        print(f"   South offset: {south_deg:.8f} degrees")
        print(f"   East offset (top row): {east_top_deg:.8f} degrees")
        print(f"   East offset (bottom row): {east_bottom_deg:.8f} degrees")