#!/usr/bin/env python3
"""
Quick script to calculate centroid spacing between stored heatmaps
"""

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points on Earth in kilometers"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    return c * r

# Coordinates from the latest 6 stored heatmaps
heatmaps = [
    ("Northeast", -43.32717570677798, 172.5881412221212),
    ("East", -43.32717570677798, 172.3431123591075), 
    ("Center", -43.32717570677798, 172.09808349609378),
    ("Far Southeast", -43.50542124907107, 172.58958587406715),
    ("Southeast", -43.50542124907107, 172.34383468508048),
    ("South", -43.50542124907107, 172.09808349609378)
]

print("🎯 CENTROID SPACING ANALYSIS for Latest 6 Stored Heatmaps")
print("=" * 65)

# Calculate all adjacent distances
distances = []

# Top row (same latitude) - horizontal spacing
top_row = [("East", -43.32717570677798, 172.3431123591075), 
           ("Center", -43.32717570677798, 172.09808349609378),
           ("Northeast", -43.32717570677798, 172.5881412221212)]

print("\n📏 HORIZONTAL SPACING (Same Latitude):")
for i in range(len(top_row) - 1):
    name1, lat1, lon1 = top_row[i]
    name2, lat2, lon2 = top_row[i + 1]
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    distances.append(dist)
    print(f"  {name1} ↔ {name2}: {dist:.3f} km")

# Bottom row (same latitude) - horizontal spacing  
bottom_row = [("Southeast", -43.50542124907107, 172.34383468508048),
              ("South", -43.50542124907107, 172.09808349609378),
              ("Far Southeast", -43.50542124907107, 172.58958587406715)]

for i in range(len(bottom_row) - 1):
    name1, lat1, lon1 = bottom_row[i]
    name2, lat2, lon2 = bottom_row[i + 1]
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    distances.append(dist)
    print(f"  {name1} ↔ {name2}: {dist:.3f} km")

print("\n📏 VERTICAL SPACING (Same Longitude):")
# Vertical spacing - same longitude pairs
vertical_pairs = [
    (("East", -43.32717570677798, 172.3431123591075), ("Southeast", -43.50542124907107, 172.34383468508048)),
    (("Center", -43.32717570677798, 172.09808349609378), ("South", -43.50542124907107, 172.09808349609378)),
    (("Northeast", -43.32717570677798, 172.5881412221212), ("Far Southeast", -43.50542124907107, 172.58958587406715))
]

for (name1, lat1, lon1), (name2, lat2, lon2) in vertical_pairs:
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    distances.append(dist)
    print(f"  {name1} ↔ {name2}: {dist:.3f} km")

print("\n📊 SPACING STATISTICS:")
print(f"  Minimum spacing: {min(distances):.3f} km")
print(f"  Maximum spacing: {max(distances):.3f} km")
print(f"  Average spacing: {sum(distances)/len(distances):.3f} km")
print(f"  Target spacing: 19.820 km")
print(f"  Spacing variation: {max(distances) - min(distances):.3f} km")

# Check if spacing is within acceptable tolerance (±0.01 km = ±10m)
target = 19.820
tolerance = 0.010
all_within_tolerance = all(abs(d - target) <= tolerance for d in distances)

print(f"\n✅ PRECISION CHECK:")
print(f"  Target: {target:.3f} km ± {tolerance:.3f} km tolerance")
if all_within_tolerance:
    print(f"  ✅ ALL spacings within tolerance!")
else:
    print(f"  ❌ Some spacings outside tolerance:")
    for i, dist in enumerate(distances):
        if abs(dist - target) > tolerance:
            error = abs(dist - target)
            print(f"    Distance {i+1}: {dist:.3f} km (error: {error:.3f} km)")

print("\n" + "=" * 65)