#!/usr/bin/env python3

from utils import get_distance

def check_last_generated_distances():
    """Check distances for the most recently generated heatmaps from the logs"""
    
    # From the logs, the last generated heatmaps had these coordinates:
    heatmaps = {
        'original': (-44.056, 170.817),
        'east': (-44.056, 171.065),
        'northeast': (-44.056, 171.313),
        'south': (-44.234, 170.817),
        'southeast': (-44.234, 171.066),
        'far_southeast': (-44.234, 171.314)
    }
    
    print("LAST GENERATED HEATMAP DISTANCES")
    print("=" * 50)
    print("Heatmap Centers:")
    for name, (lat, lon) in heatmaps.items():
        print(f"  {name.upper()}: ({lat:.3f}, {lon:.3f})")
    
    print("\nADJACENT DISTANCES:")
    print("-" * 30)
    
    # Horizontal distances (same row)
    dist_orig_east = get_distance(heatmaps['original'][0], heatmaps['original'][1],
                                 heatmaps['east'][0], heatmaps['east'][1])
    dist_east_northeast = get_distance(heatmaps['east'][0], heatmaps['east'][1],
                                      heatmaps['northeast'][0], heatmaps['northeast'][1])
    dist_south_southeast = get_distance(heatmaps['south'][0], heatmaps['south'][1],
                                       heatmaps['southeast'][0], heatmaps['southeast'][1])
    dist_southeast_far = get_distance(heatmaps['southeast'][0], heatmaps['southeast'][1],
                                     heatmaps['far_southeast'][0], heatmaps['far_southeast'][1])
    
    print("HORIZONTAL (same row):")
    print(f"  ORIGINAL ↔ EAST: {dist_orig_east:.2f} km")
    print(f"  EAST ↔ NORTHEAST: {dist_east_northeast:.2f} km")
    print(f"  SOUTH ↔ SOUTHEAST: {dist_south_southeast:.2f} km")
    print(f"  SOUTHEAST ↔ FAR_SOUTHEAST: {dist_southeast_far:.2f} km")
    
    # Vertical distances (same column)
    dist_orig_south = get_distance(heatmaps['original'][0], heatmaps['original'][1],
                                  heatmaps['south'][0], heatmaps['south'][1])
    dist_east_southeast = get_distance(heatmaps['east'][0], heatmaps['east'][1],
                                      heatmaps['southeast'][0], heatmaps['southeast'][1])
    dist_northeast_far = get_distance(heatmaps['northeast'][0], heatmaps['northeast'][1],
                                     heatmaps['far_southeast'][0], heatmaps['far_southeast'][1])
    
    print("\nVERTICAL (same column):")
    print(f"  ORIGINAL ↕ SOUTH: {dist_orig_south:.2f} km")
    print(f"  EAST ↕ SOUTHEAST: {dist_east_southeast:.2f} km")
    print(f"  NORTHEAST ↕ FAR_SOUTHEAST: {dist_northeast_far:.2f} km")
    
    print(f"\nTARGET DISTANCE: 19.82 km")
    print("\nCONSISTENCY CHECK:")
    all_distances = [dist_orig_east, dist_east_northeast, dist_south_southeast, 
                    dist_southeast_far, dist_orig_south, dist_east_southeast, dist_northeast_far]
    
    min_dist = min(all_distances)
    max_dist = max(all_distances)
    avg_dist = sum(all_distances) / len(all_distances)
    
    print(f"  Minimum distance: {min_dist:.2f} km")
    print(f"  Maximum distance: {max_dist:.2f} km")
    print(f"  Average distance: {avg_dist:.2f} km")
    print(f"  Range variation: {max_dist - min_dist:.2f} km")
    
    if max_dist - min_dist < 0.1:
        print("✅ EXCELLENT: All distances very consistent (< 0.1km variation)")
    elif max_dist - min_dist < 0.5:
        print("✅ GOOD: Distances mostly consistent (< 0.5km variation)")
    else:
        print("⚠️  ISSUE: Significant distance variation detected")

if __name__ == "__main__":
    check_last_generated_distances()