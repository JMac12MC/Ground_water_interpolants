#!/usr/bin/env python3

from utils import get_distance

# Last generated heatmap coordinates from the logs
heatmaps = [
    {"name": "Original", "lat": -44.036, "lon": 170.810},
    {"name": "East", "lat": -44.036, "lon": 171.058},
    {"name": "Northeast", "lat": -44.036, "lon": 171.306},
    {"name": "South", "lat": -44.215, "lon": 170.810},
    {"name": "Southeast", "lat": -44.215, "lon": 171.059},
    {"name": "Far Southeast", "lat": -44.215, "lon": 171.307}
]

print("=== DISTANCE ANALYSIS FOR LAST GENERATED HEATMAPS ===")
print(f"Center coordinates: ({heatmaps[0]['lat']}, {heatmaps[0]['lon']})")
print()

# Calculate horizontal distances (same latitude)
print("HORIZONTAL DISTANCES (same row):")
# Top row
dist_orig_east = get_distance(heatmaps[0]['lat'], heatmaps[0]['lon'], 
                                   heatmaps[1]['lat'], heatmaps[1]['lon'])
dist_east_northeast = get_distance(heatmaps[1]['lat'], heatmaps[1]['lon'], 
                                        heatmaps[2]['lat'], heatmaps[2]['lon'])

# Bottom row  
dist_south_southeast = get_distance(heatmaps[3]['lat'], heatmaps[3]['lon'], 
                                         heatmaps[4]['lat'], heatmaps[4]['lon'])
dist_southeast_far = get_distance(heatmaps[4]['lat'], heatmaps[4]['lon'], 
                                       heatmaps[5]['lat'], heatmaps[5]['lon'])

print(f"  ORIGINAL ↔ EAST: {dist_orig_east:.2f} km")
print(f"  EAST ↔ NORTHEAST: {dist_east_northeast:.2f} km")
print(f"  SOUTH ↔ SOUTHEAST: {dist_south_southeast:.2f} km")
print(f"  SOUTHEAST ↔ FAR_SOUTHEAST: {dist_southeast_far:.2f} km")

print()
print("VERTICAL DISTANCES (same column):")
# Calculate vertical distances
dist_orig_south = get_distance(heatmaps[0]['lat'], heatmaps[0]['lon'], 
                                    heatmaps[3]['lat'], heatmaps[3]['lon'])
dist_east_southeast = get_distance(heatmaps[1]['lat'], heatmaps[1]['lon'], 
                                        heatmaps[4]['lat'], heatmaps[4]['lon'])
dist_northeast_far = get_distance(heatmaps[2]['lat'], heatmaps[2]['lon'], 
                                       heatmaps[5]['lat'], heatmaps[5]['lon'])

print(f"  ORIGINAL ↕ SOUTH: {dist_orig_south:.2f} km")
print(f"  EAST ↕ SOUTHEAST: {dist_east_southeast:.2f} km")
print(f"  NORTHEAST ↕ FAR_SOUTHEAST: {dist_northeast_far:.2f} km")

print()
print("COORDINATE DIFFERENCES:")
print("Longitude differences (horizontal spacing):")
lon_diff_1 = heatmaps[1]['lon'] - heatmaps[0]['lon']  # East - Original
lon_diff_2 = heatmaps[2]['lon'] - heatmaps[1]['lon']  # Northeast - East
lon_diff_3 = heatmaps[4]['lon'] - heatmaps[3]['lon']  # Southeast - South
lon_diff_4 = heatmaps[5]['lon'] - heatmaps[4]['lon']  # Far - Southeast

print(f"  Original → East: {lon_diff_1:.6f}°")
print(f"  East → Northeast: {lon_diff_2:.6f}°")
print(f"  South → Southeast: {lon_diff_3:.6f}°")
print(f"  Southeast → Far: {lon_diff_4:.6f}°")

print()
print("Latitude differences (vertical spacing):")
lat_diff_1 = heatmaps[3]['lat'] - heatmaps[0]['lat']  # South - Original
lat_diff_2 = heatmaps[4]['lat'] - heatmaps[1]['lat']  # Southeast - East
lat_diff_3 = heatmaps[5]['lat'] - heatmaps[2]['lat']  # Far - Northeast

print(f"  Original → South: {lat_diff_1:.6f}°")
print(f"  East → Southeast: {lat_diff_2:.6f}°")
print(f"  Northeast → Far: {lat_diff_3:.6f}°")

print()
print("CONSISTENCY CHECK:")
horizontal_distances = [dist_orig_east, dist_east_northeast, dist_south_southeast, dist_southeast_far]
vertical_distances = [dist_orig_south, dist_east_southeast, dist_northeast_far]

print(f"Horizontal distance range: {min(horizontal_distances):.2f} - {max(horizontal_distances):.2f} km")
print(f"Vertical distance range: {min(vertical_distances):.2f} - {max(vertical_distances):.2f} km")
print(f"All horizontal distances equal: {len(set([round(d, 2) for d in horizontal_distances])) == 1}")
print(f"All vertical distances equal: {len(set([round(d, 2) for d in vertical_distances])) == 1}")