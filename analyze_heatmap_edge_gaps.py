#!/usr/bin/env python3
"""
Calculate edge-to-edge gaps between stored heatmaps
Each heatmap has a 10km radius clipping boundary from its centroid
"""

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in kilometers"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371

def calculate_edge_gap(lat1, lon1, lat2, lon2, radius1=10.0, radius2=10.0):
    """
    Calculate the gap between the edges of two circular heatmaps
    
    Args:
        lat1, lon1: Center coordinates of first heatmap
        lat2, lon2: Center coordinates of second heatmap  
        radius1, radius2: Radius of each heatmap (default 10km)
    
    Returns:
        Edge-to-edge gap distance in kilometers
    """
    centroid_distance = haversine_distance(lat1, lon1, lat2, lon2)
    edge_gap = centroid_distance - radius1 - radius2
    return edge_gap

# Coordinates from the latest 6 stored heatmaps (with 10km radius each)
heatmaps = {
    "Northeast": (-43.32717570677798, 172.5881412221212),
    "East": (-43.32717570677798, 172.3431123591075), 
    "Center": (-43.32717570677798, 172.09808349609378),
    "Far Southeast": (-43.50542124907107, 172.58958587406715),
    "Southeast": (-43.50542124907107, 172.34383468508048),
    "South": (-43.50542124907107, 172.09808349609378)
}

HEATMAP_RADIUS = 10.0  # km - clipping boundary radius

print("🎯 HEATMAP EDGE-TO-EDGE GAP ANALYSIS")
print("=" * 55)
print(f"📊 Heatmap Configuration:")
print(f"  • Each heatmap radius: {HEATMAP_RADIUS} km")
print(f"  • Total heatmap diameter: {HEATMAP_RADIUS * 2} km")
print(f"  • Grid layout: 2×3 (6 heatmaps total)")

print(f"\n📏 HORIZONTAL GAPS (Same Latitude):")

# Top row horizontal gaps
east_lat, east_lon = heatmaps["East"]
center_lat, center_lon = heatmaps["Center"]
northeast_lat, northeast_lon = heatmaps["Northeast"]

east_center_gap = calculate_edge_gap(east_lat, east_lon, center_lat, center_lon, HEATMAP_RADIUS, HEATMAP_RADIUS)
center_northeast_gap = calculate_edge_gap(center_lat, center_lon, northeast_lat, northeast_lon, HEATMAP_RADIUS, HEATMAP_RADIUS)

print(f"  East ↔ Center gap: {east_center_gap:.3f} km")
print(f"  Center ↔ Northeast gap: {center_northeast_gap:.3f} km")

# Bottom row horizontal gaps
southeast_lat, southeast_lon = heatmaps["Southeast"]
south_lat, south_lon = heatmaps["South"]
far_southeast_lat, far_southeast_lon = heatmaps["Far Southeast"]

southeast_south_gap = calculate_edge_gap(southeast_lat, southeast_lon, south_lat, south_lon, HEATMAP_RADIUS, HEATMAP_RADIUS)
south_far_southeast_gap = calculate_edge_gap(south_lat, south_lon, far_southeast_lat, far_southeast_lon, HEATMAP_RADIUS, HEATMAP_RADIUS)

print(f"  Southeast ↔ South gap: {southeast_south_gap:.3f} km")
print(f"  South ↔ Far Southeast gap: {south_far_southeast_gap:.3f} km")

print(f"\n📏 VERTICAL GAPS (Same Longitude):")

# Vertical gaps between rows
east_southeast_gap = calculate_edge_gap(east_lat, east_lon, southeast_lat, southeast_lon, HEATMAP_RADIUS, HEATMAP_RADIUS)
center_south_gap = calculate_edge_gap(center_lat, center_lon, south_lat, south_lon, HEATMAP_RADIUS, HEATMAP_RADIUS)
northeast_far_southeast_gap = calculate_edge_gap(northeast_lat, northeast_lon, far_southeast_lat, far_southeast_lon, HEATMAP_RADIUS, HEATMAP_RADIUS)

print(f"  East ↔ Southeast gap: {east_southeast_gap:.3f} km")
print(f"  Center ↔ South gap: {center_south_gap:.3f} km")
print(f"  Northeast ↔ Far Southeast gap: {northeast_far_southeast_gap:.3f} km")

# Collect all gaps for analysis
all_gaps = [
    east_center_gap, center_northeast_gap, southeast_south_gap, south_far_southeast_gap,
    east_southeast_gap, center_south_gap, northeast_far_southeast_gap
]

print(f"\n📊 GAP STATISTICS:")
print(f"  Average gap: {sum(all_gaps)/len(all_gaps):.3f} km")
print(f"  Minimum gap: {min(all_gaps):.3f} km")
print(f"  Maximum gap: {max(all_gaps):.3f} km")
print(f"  Gap variation: {max(all_gaps) - min(all_gaps):.3f} km")

print(f"\n🔍 COVERAGE ANALYSIS:")
print(f"  Centroid spacing: 19.820 km")
print(f"  Combined heatmap diameter: {HEATMAP_RADIUS * 2} km")
print(f"  Expected gap: 19.820 - {HEATMAP_RADIUS * 2} = {19.820 - (HEATMAP_RADIUS * 2):.3f} km")

# Check coverage efficiency
expected_gap = 19.820 - (HEATMAP_RADIUS * 2)
actual_avg_gap = sum(all_gaps) / len(all_gaps)
coverage_difference = abs(actual_avg_gap - expected_gap)

print(f"\n✅ COVERAGE VERIFICATION:")
print(f"  Expected edge gap: {expected_gap:.3f} km")
print(f"  Actual average gap: {actual_avg_gap:.3f} km")
print(f"  Difference: {coverage_difference:.3f} km")

if coverage_difference < 0.01:
    print(f"  ✅ PERFECT: Gaps match expected spacing!")
elif coverage_difference < 0.1:
    print(f"  ✅ EXCELLENT: Gaps within 100m of expected!")
else:
    print(f"  ⚠️  ATTENTION: Gap difference > 100m")

print(f"\n" + "=" * 55)