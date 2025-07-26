#!/usr/bin/env python3

import numpy as np
from utils import get_distance

def fix_precise_spacing(click_lat, click_lng, target_km=19.82):
    """
    Calculate EXACT spacing to achieve perfect 19.82km between ALL adjacent heatmaps
    """
    
    print(f"FIXING HEATMAP SPACING TO EXACT {target_km}km")
    print("=" * 50)
    
    # Step 1: Calculate precise latitude offset
    # 1 degree latitude = 111.0 km (constant everywhere)
    lat_offset_deg = target_km / 111.0
    
    print(f"Latitude offset: {lat_offset_deg:.8f}° = {target_km}km")
    
    # Step 2: Calculate longitude offsets for BOTH rows
    # Top row (original latitude)
    top_lat = click_lat
    km_per_deg_lon_top = 111.0 * abs(np.cos(np.radians(top_lat)))
    lon_offset_top_deg = target_km / km_per_deg_lon_top
    
    # Bottom row (south latitude)  
    bottom_lat = click_lat - lat_offset_deg
    km_per_deg_lon_bottom = 111.0 * abs(np.cos(np.radians(bottom_lat)))
    lon_offset_bottom_deg = target_km / km_per_deg_lon_bottom
    
    print(f"\nLongitude calculations:")
    print(f"  Top row (lat {top_lat:.6f}°): {lon_offset_top_deg:.8f}° = {target_km}km")
    print(f"  Bottom row (lat {bottom_lat:.6f}°): {lon_offset_bottom_deg:.8f}° = {target_km}km")
    
    # Step 3: Define all six positions using row-specific longitude offsets
    positions = {
        'original': [click_lat, click_lng],
        'east': [click_lat, click_lng + lon_offset_top_deg],
        'northeast': [click_lat, click_lng + (2 * lon_offset_top_deg)],
        'south': [click_lat - lat_offset_deg, click_lng],
        'southeast': [click_lat - lat_offset_deg, click_lng + lon_offset_bottom_deg],
        'far_southeast': [click_lat - lat_offset_deg, click_lng + (2 * lon_offset_bottom_deg)]
    }
    
    print(f"\nPROPOSED GRID POSITIONS:")
    for name, (lat, lon) in positions.items():
        print(f"  {name.upper()}: ({lat:.8f}, {lon:.8f})")
    
    # Step 4: Verify all distances
    print(f"\nVERIFICATION - ALL DISTANCES SHOULD BE {target_km:.2f}km:")
    
    # Horizontal distances
    dist_orig_east = get_distance(positions['original'][0], positions['original'][1],
                                 positions['east'][0], positions['east'][1])
    dist_east_northeast = get_distance(positions['east'][0], positions['east'][1],
                                      positions['northeast'][0], positions['northeast'][1])
    dist_south_southeast = get_distance(positions['south'][0], positions['south'][1],
                                       positions['southeast'][0], positions['southeast'][1])
    dist_southeast_far = get_distance(positions['southeast'][0], positions['southeast'][1],
                                     positions['far_southeast'][0], positions['far_southeast'][1])
    
    print("  Horizontal:")
    print(f"    ORIGINAL ↔ EAST: {dist_orig_east:.3f}km")
    print(f"    EAST ↔ NORTHEAST: {dist_east_northeast:.3f}km")
    print(f"    SOUTH ↔ SOUTHEAST: {dist_south_southeast:.3f}km")
    print(f"    SOUTHEAST ↔ FAR_SOUTHEAST: {dist_southeast_far:.3f}km")
    
    # Vertical distances
    dist_orig_south = get_distance(positions['original'][0], positions['original'][1],
                                  positions['south'][0], positions['south'][1])
    dist_east_southeast = get_distance(positions['east'][0], positions['east'][1],
                                      positions['southeast'][0], positions['southeast'][1])
    dist_northeast_far = get_distance(positions['northeast'][0], positions['northeast'][1],
                                     positions['far_southeast'][0], positions['far_southeast'][1])
    
    print("  Vertical:")
    print(f"    ORIGINAL ↕ SOUTH: {dist_orig_south:.3f}km")
    print(f"    EAST ↕ SOUTHEAST: {dist_east_southeast:.3f}km")
    print(f"    NORTHEAST ↕ FAR_SOUTHEAST: {dist_northeast_far:.3f}km")
    
    # Check precision
    all_distances = [dist_orig_east, dist_east_northeast, dist_south_southeast,
                    dist_southeast_far, dist_orig_south, dist_east_southeast, dist_northeast_far]
    
    max_error = max(abs(d - target_km) for d in all_distances)
    print(f"\nMAXIMUM ERROR: {max_error:.4f}km")
    
    if max_error < 0.001:
        print("✅ PERFECT: All distances within 1 meter of target")
    elif max_error < 0.01:
        print("✅ EXCELLENT: All distances within 10 meters of target")
    elif max_error < 0.1:
        print("✅ GOOD: All distances within 100 meters of target")
    else:
        print("⚠️ NEEDS IMPROVEMENT: Distances have significant error")
    
    return lat_offset_deg, lon_offset_top_deg, lon_offset_bottom_deg, positions

if __name__ == "__main__":
    # Test with last click coordinates
    test_lat, test_lng = -44.043, 170.815
    fix_precise_spacing(test_lat, test_lng)