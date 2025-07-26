#!/usr/bin/env python3

import numpy as np
from utils import get_distance

def analyze_spacing_errors():
    """
    Analyze the remaining spacing errors to identify systematic issues
    """
    
    # Current coordinates from last generation
    heatmaps = {
        'original': (-44.056, 170.817),
        'east': (-44.056, 171.065),
        'northeast': (-44.056, 171.313),
        'south': (-44.234, 170.817),
        'southeast': (-44.234, 171.066),
        'far_southeast': (-44.234, 171.314)
    }
    
    target_km = 19.82
    
    print("SPACING ERROR ANALYSIS")
    print("=" * 50)
    
    # Calculate all distances and errors
    distances = {}
    errors = {}
    
    # Horizontal distances
    distances['orig_east'] = get_distance(heatmaps['original'][0], heatmaps['original'][1],
                                         heatmaps['east'][0], heatmaps['east'][1])
    distances['east_northeast'] = get_distance(heatmaps['east'][0], heatmaps['east'][1],
                                              heatmaps['northeast'][0], heatmaps['northeast'][1])
    distances['south_southeast'] = get_distance(heatmaps['south'][0], heatmaps['south'][1],
                                               heatmaps['southeast'][0], heatmaps['southeast'][1])
    distances['southeast_far'] = get_distance(heatmaps['southeast'][0], heatmaps['southeast'][1],
                                             heatmaps['far_southeast'][0], heatmaps['far_southeast'][1])
    
    # Vertical distances
    distances['orig_south'] = get_distance(heatmaps['original'][0], heatmaps['original'][1],
                                          heatmaps['south'][0], heatmaps['south'][1])
    distances['east_southeast'] = get_distance(heatmaps['east'][0], heatmaps['east'][1],
                                              heatmaps['southeast'][0], heatmaps['southeast'][1])
    distances['northeast_far'] = get_distance(heatmaps['northeast'][0], heatmaps['northeast'][1],
                                             heatmaps['far_southeast'][0], heatmaps['far_southeast'][1])
    
    # Calculate errors
    for key, dist in distances.items():
        errors[key] = dist - target_km
    
    print("DETAILED ERROR BREAKDOWN:")
    print(f"  HORIZONTAL ERRORS:")
    print(f"    ORIG→EAST: {errors['orig_east']:+.4f}km ({errors['orig_east']*1000:+.0f}m)")
    print(f"    EAST→NE:   {errors['east_northeast']:+.4f}km ({errors['east_northeast']*1000:+.0f}m)")
    print(f"    SOUTH→SE:  {errors['south_southeast']:+.4f}km ({errors['south_southeast']*1000:+.0f}m)")
    print(f"    SE→FAR:    {errors['southeast_far']:+.4f}km ({errors['southeast_far']*1000:+.0f}m)")
    
    print(f"  VERTICAL ERRORS:")
    print(f"    ORIG→SOUTH: {errors['orig_south']:+.4f}km ({errors['orig_south']*1000:+.0f}m)")
    print(f"    EAST→SE:    {errors['east_southeast']:+.4f}km ({errors['east_southeast']*1000:+.0f}m)")
    print(f"    NE→FAR:     {errors['northeast_far']:+.4f}km ({errors['northeast_far']*1000:+.0f}m)")
    
    # Identify patterns
    horizontal_errors = [errors['orig_east'], errors['east_northeast'], errors['south_southeast'], errors['southeast_far']]
    vertical_errors = [errors['orig_south'], errors['east_southeast'], errors['northeast_far']]
    
    avg_horizontal_error = sum(horizontal_errors) / len(horizontal_errors)
    avg_vertical_error = sum(vertical_errors) / len(vertical_errors)
    
    print(f"\nPATTERN ANALYSIS:")
    print(f"  Average horizontal error: {avg_horizontal_error:+.4f}km ({avg_horizontal_error*1000:+.0f}m)")
    print(f"  Average vertical error:   {avg_vertical_error:+.4f}km ({avg_vertical_error*1000:+.0f}m)")
    
    # Check if errors are systematic
    horizontal_consistent = all(abs(e - avg_horizontal_error) < 0.01 for e in horizontal_errors)
    vertical_consistent = all(abs(e - avg_vertical_error) < 0.01 for e in vertical_errors)
    
    print(f"  Horizontal errors consistent: {horizontal_consistent}")
    print(f"  Vertical errors consistent: {vertical_consistent}")
    
    return avg_horizontal_error, avg_vertical_error, errors

def suggest_improvements():
    """
    Suggest specific improvements to achieve sub-10-meter accuracy
    """
    
    avg_h_error, avg_v_error, errors = analyze_spacing_errors()
    
    print(f"\nIMPROVEMENT RECOMMENDATIONS:")
    print("=" * 50)
    
    print("1. SYSTEMATIC ERROR CORRECTION:")
    if abs(avg_v_error) > 0.005:  # > 5 meters
        correction = -avg_v_error
        print(f"   - Apply latitude correction: {correction:+.6f}° to fix {avg_v_error*1000:+.0f}m vertical bias")
    
    if abs(avg_h_error) > 0.005:  # > 5 meters  
        correction = -avg_h_error
        print(f"   - Apply longitude correction: {correction:+.6f}° to fix {avg_h_error*1000:+.0f}m horizontal bias")
    
    print("2. ENHANCED ITERATIVE REFINEMENT:")
    print("   - Reduce tolerance from 1m to 0.1m (0.0001 km)")
    print("   - Increase max iterations from 10 to 50")
    print("   - Use higher precision arithmetic (more decimal places)")
    
    print("3. GEODETIC ACCURACY IMPROVEMENTS:")
    print("   - Use WGS84 ellipsoid instead of spherical approximation")
    print("   - Account for local geoid variations")
    print("   - Apply meridian convergence corrections")
    
    print("4. INDIVIDUAL COORDINATE REFINEMENT:")
    print("   - Refine each coordinate pair independently")
    print("   - Use actual measured distances between existing points")
    print("   - Apply micro-adjustments based on residual errors")
    
    # Calculate potential improvement
    max_current_error = max(abs(e) for e in errors.values())
    print(f"\nCURRENT STATUS:")
    print(f"  Maximum error: {max_current_error:.4f}km ({max_current_error*1000:.0f}m)")
    print(f"  Target for next improvement: < 0.01km (10m)")
    
    return avg_h_error, avg_v_error

if __name__ == "__main__":
    suggest_improvements()