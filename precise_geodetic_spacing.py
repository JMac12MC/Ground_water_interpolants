#!/usr/bin/env python3

import numpy as np
from utils import get_distance

def calculate_precise_offsets(lat, lon, target_distance_km=19.82):
    """
    Calculate precise coordinate offsets using iterative refinement with Haversine distance
    to achieve exactly target_distance_km between adjacent heatmaps
    """
    
    print(f"CALCULATING PRECISE OFFSETS FOR {target_distance_km}km SPACING")
    print("=" * 60)
    
    # Constants
    MAX_ITERATIONS = 10
    TOLERANCE_KM = 0.001  # 1 meter tolerance
    
    # Step 1: Calculate precise latitude offset
    print("1. CALCULATING LATITUDE OFFSET:")
    
    # Initial estimate using 111.0 km per degree
    lat_offset_initial = target_distance_km / 111.0
    
    # Refine using actual distance calculation
    for i in range(MAX_ITERATIONS):
        test_lat = lat - lat_offset_initial
        actual_distance = get_distance(lat, lon, test_lat, lon)
        error = actual_distance - target_distance_km
        
        if abs(error) < TOLERANCE_KM:
            break
            
        # Adjust offset proportionally
        lat_offset_initial *= target_distance_km / actual_distance
    
    lat_offset_precise = lat_offset_initial
    final_lat_distance = get_distance(lat, lon, lat - lat_offset_precise, lon)
    
    print(f"   Final latitude offset: {lat_offset_precise:.10f}°")
    print(f"   Achieved distance: {final_lat_distance:.6f} km (error: {final_lat_distance - target_distance_km:.6f} km)")
    
    # Step 2: Calculate precise longitude offset for TOP ROW
    print("\n2. CALCULATING TOP ROW LONGITUDE OFFSET:")
    
    # Initial estimate using cosine approximation
    lon_offset_top_initial = target_distance_km / (111.0 * abs(np.cos(np.radians(lat))))
    
    # Refine using actual distance calculation
    for i in range(MAX_ITERATIONS):
        test_lon = lon + lon_offset_top_initial
        actual_distance = get_distance(lat, lon, lat, test_lon)
        error = actual_distance - target_distance_km
        
        if abs(error) < TOLERANCE_KM:
            break
            
        # Adjust offset proportionally
        lon_offset_top_initial *= target_distance_km / actual_distance
    
    lon_offset_top_precise = lon_offset_top_initial
    final_top_distance = get_distance(lat, lon, lat, lon + lon_offset_top_precise)
    
    print(f"   Final longitude offset: {lon_offset_top_precise:.10f}°")
    print(f"   Achieved distance: {final_top_distance:.6f} km (error: {final_top_distance - target_distance_km:.6f} km)")
    
    # Step 3: Calculate precise longitude offset for BOTTOM ROW
    print("\n3. CALCULATING BOTTOM ROW LONGITUDE OFFSET:")
    
    bottom_lat = lat - lat_offset_precise
    
    # Initial estimate using cosine approximation at bottom latitude
    lon_offset_bottom_initial = target_distance_km / (111.0 * abs(np.cos(np.radians(bottom_lat))))
    
    # Refine using actual distance calculation
    for i in range(MAX_ITERATIONS):
        test_lon = lon + lon_offset_bottom_initial
        actual_distance = get_distance(bottom_lat, lon, bottom_lat, test_lon)
        error = actual_distance - target_distance_km
        
        if abs(error) < TOLERANCE_KM:
            break
            
        # Adjust offset proportionally
        lon_offset_bottom_initial *= target_distance_km / actual_distance
    
    lon_offset_bottom_precise = lon_offset_bottom_initial
    final_bottom_distance = get_distance(bottom_lat, lon, bottom_lat, lon + lon_offset_bottom_precise)
    
    print(f"   Final longitude offset: {lon_offset_bottom_precise:.10f}°")
    print(f"   Achieved distance: {final_bottom_distance:.6f} km (error: {final_bottom_distance - target_distance_km:.6f} km)")
    
    return lat_offset_precise, lon_offset_top_precise, lon_offset_bottom_precise

def verify_precise_grid(lat, lon, lat_offset, lon_offset_top, lon_offset_bottom, target_km=19.82):
    """
    Verify the precision of the calculated grid spacing
    """
    
    # Define all six positions
    positions = {
        'original': [lat, lon],
        'east': [lat, lon + lon_offset_top],
        'northeast': [lat, lon + (2 * lon_offset_top)],
        'south': [lat - lat_offset, lon],
        'southeast': [lat - lat_offset, lon + lon_offset_bottom],
        'far_southeast': [lat - lat_offset, lon + (2 * lon_offset_bottom)]
    }
    
    print(f"\n4. VERIFICATION OF PRECISE GRID:")
    print("   All distances should be exactly {:.3f} km".format(target_km))
    
    # Calculate all adjacent distances
    distances = {}
    
    # Horizontal distances
    distances['orig_east'] = get_distance(positions['original'][0], positions['original'][1],
                                         positions['east'][0], positions['east'][1])
    distances['east_northeast'] = get_distance(positions['east'][0], positions['east'][1],
                                              positions['northeast'][0], positions['northeast'][1])
    distances['south_southeast'] = get_distance(positions['south'][0], positions['south'][1],
                                               positions['southeast'][0], positions['southeast'][1])
    distances['southeast_far'] = get_distance(positions['southeast'][0], positions['southeast'][1],
                                             positions['far_southeast'][0], positions['far_southeast'][1])
    
    # Vertical distances
    distances['orig_south'] = get_distance(positions['original'][0], positions['original'][1],
                                          positions['south'][0], positions['south'][1])
    distances['east_southeast'] = get_distance(positions['east'][0], positions['east'][1],
                                              positions['southeast'][0], positions['southeast'][1])
    distances['northeast_far'] = get_distance(positions['northeast'][0], positions['northeast'][1],
                                             positions['far_southeast'][0], positions['far_southeast'][1])
    
    print("\n   HORIZONTAL DISTANCES:")
    print(f"     ORIGINAL ↔ EAST: {distances['orig_east']:.6f} km (error: {distances['orig_east'] - target_km:.6f} km)")
    print(f"     EAST ↔ NORTHEAST: {distances['east_northeast']:.6f} km (error: {distances['east_northeast'] - target_km:.6f} km)")
    print(f"     SOUTH ↔ SOUTHEAST: {distances['south_southeast']:.6f} km (error: {distances['south_southeast'] - target_km:.6f} km)")
    print(f"     SOUTHEAST ↔ FAR_SOUTHEAST: {distances['southeast_far']:.6f} km (error: {distances['southeast_far'] - target_km:.6f} km)")
    
    print("\n   VERTICAL DISTANCES:")
    print(f"     ORIGINAL ↕ SOUTH: {distances['orig_south']:.6f} km (error: {distances['orig_south'] - target_km:.6f} km)")
    print(f"     EAST ↕ SOUTHEAST: {distances['east_southeast']:.6f} km (error: {distances['east_southeast'] - target_km:.6f} km)")
    print(f"     NORTHEAST ↕ FAR_SOUTHEAST: {distances['northeast_far']:.6f} km (error: {distances['northeast_far'] - target_km:.6f} km)")
    
    # Calculate maximum error
    all_distances = list(distances.values())
    errors = [abs(d - target_km) for d in all_distances]
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    
    print(f"\n   PRECISION ANALYSIS:")
    print(f"     Maximum error: {max_error:.6f} km ({max_error * 1000:.1f} meters)")
    print(f"     Average error: {avg_error:.6f} km ({avg_error * 1000:.1f} meters)")
    
    if max_error < 0.001:
        print("     ✅ EXCELLENT: All distances within 1 meter of target")
    elif max_error < 0.01:
        print("     ✅ VERY GOOD: All distances within 10 meters of target")
    elif max_error < 0.1:
        print("     ✅ GOOD: All distances within 100 meters of target")
    else:
        print("     ⚠️ NEEDS IMPROVEMENT: Distances have significant error")
    
    return positions, max_error

if __name__ == "__main__":
    # Test with recent coordinates
    test_lat, test_lon = -44.078, 170.837
    
    lat_off, lon_off_top, lon_off_bottom = calculate_precise_offsets(test_lat, test_lon)
    positions, max_error = verify_precise_grid(test_lat, test_lon, lat_off, lon_off_top, lon_off_bottom)
    
    print(f"\n5. RECOMMENDED IMPLEMENTATION:")
    print(f"   lat_offset_degrees = {lat_off:.10f}")
    print(f"   east_offset_degrees_top = {lon_off_top:.10f}")
    print(f"   east_offset_degrees_bottom = {lon_off_bottom:.10f}")