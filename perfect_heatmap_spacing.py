#!/usr/bin/env python3

import numpy as np
from utils import get_distance

def calculate_perfect_grid_coordinates(center_lat, center_lon, target_km=19.82):
    """
    Calculate ultra-precise grid coordinates using advanced geodetic methods
    with individual coordinate pair optimization
    """
    
    print(f"PERFECT GRID COORDINATE CALCULATION (target: {target_km}km)")
    print("=" * 65)
    
    # Enhanced precision parameters
    MAX_ITERATIONS = 100
    TOLERANCE_KM = 0.00005  # 5cm tolerance (0.05 meters)
    MICRO_ADJUSTMENT = 0.000001  # Ultra-fine coordinate adjustments
    
    # Phase 1: Calculate optimal latitude offset with ultra-precision
    print("PHASE 1: ULTRA-PRECISE LATITUDE CALCULATION")
    
    lat_offset = target_km / 111.0  # Initial estimate
    best_lat_offset = lat_offset
    best_lat_error = float('inf')
    
    # Multi-pass refinement with micro-adjustments
    for iteration in range(MAX_ITERATIONS):
        test_lat = center_lat - lat_offset
        actual_distance = get_distance(center_lat, center_lon, test_lat, center_lon)
        error = abs(actual_distance - target_km)
        
        if error < best_lat_error:
            best_lat_offset = lat_offset
            best_lat_error = error
        
        if error < TOLERANCE_KM:
            break
            
        # Dynamic adjustment based on error magnitude
        if error > 0.001:  # > 1 meter error
            adjustment = (target_km / actual_distance)
            lat_offset *= adjustment
        else:  # Fine-tuning phase
            if actual_distance > target_km:
                lat_offset -= MICRO_ADJUSTMENT
            else:
                lat_offset += MICRO_ADJUSTMENT
    
    lat_offset = best_lat_offset
    final_lat_distance = get_distance(center_lat, center_lon, center_lat - lat_offset, center_lon)
    print(f"  Optimized latitude offset: {lat_offset:.15f}°")
    print(f"  Achieved distance: {final_lat_distance:.8f}km (error: {final_lat_distance-target_km:.8f}km)")
    
    # Phase 2: Calculate individual longitude offsets for each coordinate pair
    print("\nPHASE 2: INDIVIDUAL COORDINATE PAIR OPTIMIZATION")
    
    positions = {}
    
    # Calculate each position with individual optimization
    coordinate_specs = [
        ('original', center_lat, center_lon),
        ('east', center_lat, center_lon + 0.248),  # Initial estimate
        ('northeast', center_lat, center_lon + 0.496),  # Initial estimate
        ('south', center_lat - lat_offset, center_lon),
        ('southeast', center_lat - lat_offset, center_lon + 0.248),  # Initial estimate
        ('far_southeast', center_lat - lat_offset, center_lon + 0.496)  # Initial estimate
    ]
    
    # Start with the original position
    positions['original'] = [center_lat, center_lon]
    
    # Calculate EAST position with ultra-precision
    print("  Optimizing EAST position...")
    east_lon = center_lon + (target_km / (111.0 * abs(np.cos(np.radians(center_lat)))))
    
    for iteration in range(MAX_ITERATIONS):
        actual_distance = get_distance(center_lat, center_lon, center_lat, east_lon)
        error = abs(actual_distance - target_km)
        
        if error < TOLERANCE_KM:
            break
            
        if actual_distance > target_km:
            east_lon -= MICRO_ADJUSTMENT
        else:
            east_lon += MICRO_ADJUSTMENT
    
    positions['east'] = [center_lat, east_lon]
    final_east_distance = get_distance(center_lat, center_lon, center_lat, east_lon)
    print(f"    East longitude: {east_lon:.15f}° (distance: {final_east_distance:.8f}km)")
    
    # Calculate NORTHEAST position
    print("  Optimizing NORTHEAST position...")
    northeast_lon = east_lon + (east_lon - center_lon)  # Double the east offset
    
    for iteration in range(MAX_ITERATIONS):
        actual_distance = get_distance(center_lat, east_lon, center_lat, northeast_lon)
        error = abs(actual_distance - target_km)
        
        if error < TOLERANCE_KM:
            break
            
        if actual_distance > target_km:
            northeast_lon -= MICRO_ADJUSTMENT
        else:
            northeast_lon += MICRO_ADJUSTMENT
    
    positions['northeast'] = [center_lat, northeast_lon]
    final_northeast_distance = get_distance(center_lat, east_lon, center_lat, northeast_lon)
    print(f"    Northeast longitude: {northeast_lon:.15f}° (distance: {final_northeast_distance:.8f}km)")
    
    # Calculate SOUTH position (already optimized)
    positions['south'] = [center_lat - lat_offset, center_lon]
    
    # Calculate SOUTHEAST position with bottom-row precision
    print("  Optimizing SOUTHEAST position...")
    south_lat = center_lat - lat_offset
    southeast_lon = center_lon + (target_km / (111.0 * abs(np.cos(np.radians(south_lat)))))
    
    for iteration in range(MAX_ITERATIONS):
        actual_distance = get_distance(south_lat, center_lon, south_lat, southeast_lon)
        error = abs(actual_distance - target_km)
        
        if error < TOLERANCE_KM:
            break
            
        if actual_distance > target_km:
            southeast_lon -= MICRO_ADJUSTMENT
        else:
            southeast_lon += MICRO_ADJUSTMENT
    
    positions['southeast'] = [south_lat, southeast_lon]
    final_southeast_distance = get_distance(south_lat, center_lon, south_lat, southeast_lon)
    print(f"    Southeast longitude: {southeast_lon:.15f}° (distance: {final_southeast_distance:.8f}km)")
    
    # Calculate FAR_SOUTHEAST position
    print("  Optimizing FAR_SOUTHEAST position...")
    far_southeast_lon = southeast_lon + (southeast_lon - center_lon)  # Double the southeast offset
    
    for iteration in range(MAX_ITERATIONS):
        actual_distance = get_distance(south_lat, southeast_lon, south_lat, far_southeast_lon)
        error = abs(actual_distance - target_km)
        
        if error < TOLERANCE_KM:
            break
            
        if actual_distance > target_km:
            far_southeast_lon -= MICRO_ADJUSTMENT
        else:
            far_southeast_lon += MICRO_ADJUSTMENT
    
    positions['far_southeast'] = [south_lat, far_southeast_lon]
    final_far_distance = get_distance(south_lat, southeast_lon, south_lat, far_southeast_lon)
    print(f"    Far southeast longitude: {far_southeast_lon:.15f}° (distance: {final_far_distance:.8f}km)")
    
    return positions

def verify_perfect_spacing(positions, target_km=19.82):
    """
    Verify the ultra-precise spacing with comprehensive distance analysis
    """
    
    print(f"\nPHASE 3: COMPREHENSIVE SPACING VERIFICATION")
    print("=" * 65)
    
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
    
    print("ULTRA-PRECISE DISTANCE MEASUREMENTS:")
    print(f"  HORIZONTAL:")
    print(f"    ORIGINAL → EAST:      {distances['orig_east']:.8f}km (error: {distances['orig_east']-target_km:+.8f}km)")
    print(f"    EAST → NORTHEAST:     {distances['east_northeast']:.8f}km (error: {distances['east_northeast']-target_km:+.8f}km)")
    print(f"    SOUTH → SOUTHEAST:    {distances['south_southeast']:.8f}km (error: {distances['south_southeast']-target_km:+.8f}km)")
    print(f"    SOUTHEAST → FAR:      {distances['southeast_far']:.8f}km (error: {distances['southeast_far']-target_km:+.8f}km)")
    
    print(f"  VERTICAL:")
    print(f"    ORIGINAL → SOUTH:     {distances['orig_south']:.8f}km (error: {distances['orig_south']-target_km:+.8f}km)")
    print(f"    EAST → SOUTHEAST:     {distances['east_southeast']:.8f}km (error: {distances['east_southeast']-target_km:+.8f}km)")
    print(f"    NORTHEAST → FAR:      {distances['northeast_far']:.8f}km (error: {distances['northeast_far']-target_km:+.8f}km)")
    
    # Calculate precision statistics
    all_distances = list(distances.values())
    errors = [abs(d - target_km) for d in all_distances]
    max_error = max(errors)
    avg_error = sum(errors) / len(errors)
    min_distance = min(all_distances)
    max_distance = max(all_distances)
    
    print(f"\nPRECISION ANALYSIS:")
    print(f"  Target distance:    {target_km:.8f}km")
    print(f"  Distance range:     {min_distance:.8f}km to {max_distance:.8f}km")
    print(f"  Maximum error:      {max_error:.8f}km ({max_error*1000:.2f}m)")
    print(f"  Average error:      {avg_error:.8f}km ({avg_error*1000:.2f}m)")
    print(f"  Range variation:    {max_distance-min_distance:.8f}km ({(max_distance-min_distance)*1000:.2f}m)")
    
    # Performance rating
    if max_error < 0.00005:  # < 5cm
        print("  🏆 PERFECT: All distances within 5cm of target")
        rating = "PERFECT"
    elif max_error < 0.0001:  # < 10cm
        print("  🏆 PERFECT: All distances within 10cm of target")
        rating = "PERFECT"
    elif max_error < 0.001:  # < 1m
        print("  ✅ EXCELLENT: All distances within 1 meter of target")
        rating = "EXCELLENT"
    elif max_error < 0.01:  # < 10m
        print("  ✅ VERY GOOD: All distances within 10 meters of target")
        rating = "VERY GOOD"
    else:
        print("  ⚠️ Needs further refinement")
        rating = "NEEDS WORK"
    
    return positions, max_error, rating

if __name__ == "__main__":
    # Test with recent click coordinates
    test_lat, test_lon = -44.071, 170.835
    
    positions = calculate_perfect_grid_coordinates(test_lat, test_lon)
    final_positions, max_error, rating = verify_perfect_spacing(positions)
    
    print(f"\nRESULT SUMMARY:")
    print(f"  Grid rating: {rating}")
    print(f"  Maximum error: {max_error*1000:.2f} meters")
    print(f"  Ready for implementation: {'YES' if max_error < 0.01 else 'NEEDS MORE WORK'}")