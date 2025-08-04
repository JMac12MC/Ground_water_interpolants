#!/usr/bin/env python3
"""
Check the exact spacing between heatmap centroids from stored heatmaps
"""

import psycopg2
import os
from utils import get_distance

def check_centroid_spacing():
    """Check the spacing between all stored heatmap centroids"""
    
    # Connect to database
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found")
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get all stored heatmaps with their center points
        cur.execute("""
            SELECT id, heatmap_name, center_lat, center_lon, interpolation_method
            FROM stored_heatmaps 
            ORDER BY center_lat DESC, center_lon ASC
        """)
        
        results = cur.fetchall()
        
        if not results:
            print("❌ No stored heatmaps found")
            return
            
        print(f"📊 ANALYZING {len(results)} STORED HEATMAP CENTROIDS")
        print("=" * 80)
        
        # Parse heatmap positions
        heatmaps = []
        for row in results:
            heatmap_id, name, center_lat, center_lon, method = row
            heatmaps.append({
                'id': heatmap_id,
                'name': name,
                'lat': center_lat,
                'lon': center_lon,
                'method': method
            })
            print(f"📍 {name}: ({center_lat:.6f}, {center_lon:.6f})")
        
        print("\n🔍 DISTANCE ANALYSIS BETWEEN ADJACENT CENTROIDS:")
        print("=" * 80)
        
        # Calculate distances between all pairs
        distances = []
        for i, h1 in enumerate(heatmaps):
            for j, h2 in enumerate(heatmaps):
                if i != j:
                    distance = get_distance(h1['lat'], h1['lon'], h2['lat'], h2['lon'])
                    distances.append({
                        'from': h1['name'],
                        'to': h2['name'],
                        'distance': distance,
                        'from_coords': (h1['lat'], h1['lon']),
                        'to_coords': (h2['lat'], h2['lon'])
                    })
        
        # Sort by distance to find adjacent pairs
        distances.sort(key=lambda x: x['distance'])
        
        # Find the closest neighbors (likely adjacent heatmaps)
        print("🎯 CLOSEST CENTROID PAIRS (Adjacent Heatmaps):")
        adjacent_distances = []
        seen_pairs = set()
        
        for d in distances:
            # Create a normalized pair key
            pair_key = tuple(sorted([d['from'], d['to']]))
            if pair_key not in seen_pairs and d['distance'] < 30:  # Within reasonable range
                seen_pairs.add(pair_key)
                adjacent_distances.append(d['distance'])
                print(f"  {d['from'][:30]:<30} ↔ {d['to'][:30]:<30} = {d['distance']:.6f} km")
        
        if adjacent_distances:
            min_distance = min(adjacent_distances)
            max_distance = max(adjacent_distances)
            avg_distance = sum(adjacent_distances) / len(adjacent_distances)
            
            print(f"\n📏 SPACING STATISTICS:")
            print(f"  Minimum spacing: {min_distance:.6f} km ({min_distance*1000:.1f} meters)")
            print(f"  Maximum spacing: {max_distance:.6f} km ({max_distance*1000:.1f} meters)")
            print(f"  Average spacing:  {avg_distance:.6f} km ({avg_distance*1000:.1f} meters)")
            print(f"  Spacing range:   {max_distance-min_distance:.6f} km ({(max_distance-min_distance)*1000:.1f} meters)")
            
            # Check if spacing matches expected 19.82km
            expected_spacing = 19.82
            print(f"\n🎯 EXPECTED vs ACTUAL SPACING:")
            print(f"  Expected spacing: {expected_spacing:.6f} km")
            print(f"  Actual spacing:   {avg_distance:.6f} km")
            print(f"  Difference:       {abs(avg_distance-expected_spacing):.6f} km ({abs(avg_distance-expected_spacing)*1000:.1f} meters)")
            
            if abs(avg_distance - expected_spacing) < 0.01:  # Within 10 meters
                print("  ✅ SPACING IS ACCURATE (within 10 meters)")
            else:
                print("  ⚠️  SPACING DEVIATION DETECTED")
                
        # Group by grid pattern recognition
        print(f"\n🗺️  GRID PATTERN ANALYSIS:")
        
        # Sort heatmaps by latitude (north to south), then longitude (west to east)
        grid_heatmaps = sorted(heatmaps, key=lambda x: (-x['lat'], x['lon']))
        
        # Try to identify grid structure
        unique_lats = sorted(list(set(round(h['lat'], 3) for h in heatmaps)), reverse=True)
        unique_lons = sorted(list(set(round(h['lon'], 3) for h in heatmaps)))
        
        print(f"  Unique latitudes:  {len(unique_lats)} ({unique_lats})")
        print(f"  Unique longitudes: {len(unique_lons)} ({unique_lons})")
        print(f"  Grid dimensions:   {len(unique_lats)} rows × {len(unique_lons)} columns")
        
        if len(unique_lats) >= 2:
            lat_spacing = abs(unique_lats[0] - unique_lats[1])
            lat_spacing_km = lat_spacing * 111.0  # Approximate conversion
            print(f"  Latitude spacing:  {lat_spacing:.6f}° = ~{lat_spacing_km:.3f} km")
            
        if len(unique_lons) >= 2:
            lon_spacing = abs(unique_lons[1] - unique_lons[0])
            # Use middle latitude for longitude conversion
            mid_lat = sum(h['lat'] for h in heatmaps) / len(heatmaps)
            lon_spacing_km = lon_spacing * 111.0 * abs(cos(radians(mid_lat)))
            print(f"  Longitude spacing: {lon_spacing:.6f}° = ~{lon_spacing_km:.3f} km")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    from math import cos, radians
    check_centroid_spacing()