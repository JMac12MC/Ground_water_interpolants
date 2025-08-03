#!/usr/bin/env python3
"""
Verify the actual size of displayed heatmaps by measuring their extent
"""

import math
from database import PolygonDatabase

def get_distance_km(lat1, lon1, lat2, lon2):
    """Calculate exact distance between two points using Haversine formula"""
    R = 6371.0  # Earth radius in km
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def analyze_heatmap_geometry(geojson_data):
    """Analyze the actual extent of a heatmap from its GeoJSON geometry"""
    if not geojson_data or 'features' not in geojson_data:
        return None
    
    all_coords = []
    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]  # Exterior ring
            all_coords.extend(coords)
    
    if not all_coords:
        return None
    
    # Find bounds
    lats = [coord[1] for coord in all_coords]
    lons = [coord[0] for coord in all_coords]
    
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    
    # Calculate center
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    
    # Calculate distances from center to edges
    north_dist = get_distance_km(center_lat, center_lon, max_lat, center_lon)
    south_dist = get_distance_km(center_lat, center_lon, min_lat, center_lon)
    east_dist = get_distance_km(center_lat, center_lon, center_lat, max_lon)
    west_dist = get_distance_km(center_lat, center_lon, center_lat, min_lon)
    
    return {
        'center': (center_lat, center_lon),
        'bounds': {'min_lat': min_lat, 'max_lat': max_lat, 'min_lon': min_lon, 'max_lon': max_lon},
        'distances': {
            'north': north_dist,
            'south': south_dist,
            'east': east_dist,
            'west': west_dist
        },
        'width_km': east_dist + west_dist,
        'height_km': north_dist + south_dist,
        'coordinates_count': len(all_coords),
        'features_count': len(geojson_data['features'])
    }

def main():
    print("🔍 VERIFYING ACTUAL HEATMAP SIZES")
    print("=" * 60)
    
    # Connect to database
    heatmap_db = PolygonDatabase()
    
    # Get all stored heatmaps
    try:
        all_heatmaps = heatmap_db.get_all_stored_heatmaps()
        print(f"Found {len(all_heatmaps)} stored heatmaps in database")
        
        for i, heatmap in enumerate(all_heatmaps, 1):
            print(f"\n📊 HEATMAP {i}: {heatmap['heatmap_name']}")
            print(f"   Database ID: {heatmap['id']}")
            print(f"   Center: ({heatmap['center_lat']:.6f}, {heatmap['center_lon']:.6f})")
            
            # Analyze actual geometry
            analysis = analyze_heatmap_geometry(heatmap['geojson_data'])
            
            if analysis:
                print(f"   📐 ACTUAL MEASUREMENTS:")
                print(f"      North edge: {analysis['distances']['north']*1000:.0f} meters")
                print(f"      South edge: {analysis['distances']['south']*1000:.0f} meters") 
                print(f"      East edge:  {analysis['distances']['east']*1000:.0f} meters")
                print(f"      West edge:  {analysis['distances']['west']*1000:.0f} meters")
                print(f"   📏 TOTAL SIZE:")
                print(f"      Width:  {analysis['width_km']*1000:.0f} meters ({analysis['width_km']:.3f} km)")
                print(f"      Height: {analysis['height_km']*1000:.0f} meters ({analysis['height_km']:.3f} km)")
                print(f"   📊 GEOMETRY:")
                print(f"      Features: {analysis['features_count']} triangles")
                print(f"      Coordinates: {analysis['coordinates_count']} points")
                
                # Check if it's actually 10km
                avg_radius = (analysis['distances']['north'] + analysis['distances']['south'] + 
                            analysis['distances']['east'] + analysis['distances']['west']) / 4
                expected_radius = 10.0
                error_meters = abs(avg_radius - expected_radius) * 1000
                
                if error_meters < 50:  # Within 50 meters
                    status = "✅ CORRECT"
                elif error_meters < 200:  # Within 200 meters  
                    status = "⚠️  CLOSE"
                else:
                    status = "❌ INCORRECT"
                
                print(f"   🎯 RADIUS CHECK: {avg_radius:.3f} km (target: 10.0 km) - {status}")
                print(f"      Error: {error_meters:.0f} meters")
            else:
                print(f"   ❌ Could not analyze geometry - no valid data")
    
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
    
    finally:
        pass  # PolygonDatabase doesn't have a close method

if __name__ == "__main__":
    main()