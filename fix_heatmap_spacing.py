#!/usr/bin/env python3
"""
Calculate precise degree offsets that result in exactly 19.82 km spacing
"""

import numpy as np
from utils import get_distance

def calculate_precise_offsets(click_lat, click_lng, target_distance_km=19.82):
    """
    Calculate precise degree offsets that result in exactly target_distance_km spacing
    """
    
    # Initial estimates using standard conversion
    km_per_degree_lon = 111.0 * np.cos(np.radians(click_lat))
    km_per_degree_lat = 111.0
    
    # Calculate initial degree offsets
    initial_east_degrees = target_distance_km / km_per_degree_lon
    initial_south_degrees = target_distance_km / km_per_degree_lat
    
    print(f"Initial estimates:")
    print(f"  East offset: {initial_east_degrees:.6f} degrees")
    print(f"  South offset: {initial_south_degrees:.6f} degrees")
    
    # Test the actual distances and refine
    east_point = [click_lat, click_lng + initial_east_degrees]
    south_point = [click_lat - initial_south_degrees, click_lng]
    
    actual_east_distance = get_distance(click_lat, click_lng, east_point[0], east_point[1])
    actual_south_distance = get_distance(click_lat, click_lng, south_point[0], south_point[1])
    
    print(f"\nActual distances with initial estimates:")
    print(f"  East distance: {actual_east_distance:.4f} km (target: {target_distance_km})")
    print(f"  South distance: {actual_south_distance:.4f} km (target: {target_distance_km})")
    
    # Refine the offsets to get exact distances
    east_correction_factor = target_distance_km / actual_east_distance
    south_correction_factor = target_distance_km / actual_south_distance
    
    precise_east_degrees = initial_east_degrees * east_correction_factor
    precise_south_degrees = initial_south_degrees * south_correction_factor
    
    print(f"\nRefined offsets:")
    print(f"  East offset: {precise_east_degrees:.6f} degrees")
    print(f"  South offset: {precise_south_degrees:.6f} degrees")
    
    # Verify the refined distances
    refined_east_point = [click_lat, click_lng + precise_east_degrees]
    refined_south_point = [click_lat - precise_south_degrees, click_lng]
    
    verified_east_distance = get_distance(click_lat, click_lng, refined_east_point[0], refined_east_point[1])
    verified_south_distance = get_distance(click_lat, click_lng, refined_south_point[0], refined_south_point[1])
    
    print(f"\nVerified distances:")
    print(f"  East distance: {verified_east_distance:.4f} km")
    print(f"  South distance: {verified_south_distance:.4f} km")
    
    return precise_east_degrees, precise_south_degrees

def test_full_grid(click_lat, click_lng):
    """
    Test the full 2x3 grid with precise offsets
    """
    print("=" * 60)
    print("TESTING FULL 2x3 GRID WITH PRECISE OFFSETS")
    print("=" * 60)
    
    east_offset_degrees, south_offset_degrees = calculate_precise_offsets(click_lat, click_lng)
    
    # Define all six locations using precise offsets
    locations = {
        'original': [click_lat, click_lng],
        'east': [click_lat, click_lng + east_offset_degrees],
        'northeast': [click_lat, click_lng + (2 * east_offset_degrees)],
        'south': [click_lat - south_offset_degrees, click_lng],
        'southeast': [click_lat - south_offset_degrees, click_lng + east_offset_degrees],
        'far_southeast': [click_lat - south_offset_degrees, click_lng + (2 * east_offset_degrees)]
    }
    
    print(f"\nHEATMAP CENTERS:")
    for name, coords in locations.items():
        print(f"  {name.upper():12}: ({coords[0]:8.4f}, {coords[1]:9.4f})")
    
    # Test all adjacent distances
    print(f"\nADJACENT DISTANCES:")
    
    # Horizontal neighbors
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
        print(f"  {from_name.upper()} ↔ {to_name.upper()}: {distance:.4f} km")
    
    # Vertical neighbors
    vertical_pairs = [
        ('original', 'south'),
        ('east', 'southeast'),
        ('northeast', 'far_southeast')
    ]
    
    for from_name, to_name in vertical_pairs:
        from_coords = locations[from_name]
        to_coords = locations[to_name]
        distance = get_distance(from_coords[0], from_coords[1], to_coords[0], to_coords[1])
        print(f"  {from_name.upper()} ↕ {to_name.upper()}: {distance:.4f} km")
    
    return east_offset_degrees, south_offset_degrees

if __name__ == "__main__":
    # Test with the example coordinates
    test_lat, test_lng = -43.665, 171.215
    print(f"Testing precise spacing calculation for point ({test_lat}, {test_lng})")
    
    east_deg, south_deg = test_full_grid(test_lat, test_lng)
    
    print(f"\n🎯 FINAL PRECISE OFFSETS:")
    print(f"   East offset: {east_deg:.8f} degrees")
    print(f"   South offset: {south_deg:.8f} degrees")