#!/usr/bin/env python3

import re
from database import HeatmapDatabase

def analyze_heatmap_edges():
    """Analyze the edges of stored heatmaps to measure gaps/overlaps"""
    
    db = HeatmapDatabase()
    
    # Get the latest set of heatmaps (last 6)
    recent_heatmaps = db.get_recent_heatmaps(limit=6)
    
    if len(recent_heatmaps) < 6:
        print("Not enough recent heatmaps to analyze")
        return
    
    print("=== HEATMAP EDGE ANALYSIS ===")
    print(f"Analyzing {len(recent_heatmaps)} most recent heatmaps")
    print()
    
    # Group heatmaps by position
    heatmap_coords = {}
    for heatmap in recent_heatmaps:
        name = heatmap['name']
        
        # Extract coordinates from name
        coord_match = re.search(r'(-?\d+\.\d+)_(-?\d+\.\d+)', name)
        if coord_match:
            lat = float(coord_match.group(1))
            lon = float(coord_match.group(2))
            
            # Determine position based on coordinates
            if 'far_southeast' in name:
                position = 'far_southeast'
            elif 'southeast' in name:
                position = 'southeast'
            elif 'northeast' in name:
                position = 'northeast'
            elif 'south' in name:
                position = 'south'
            elif 'east' in name:
                position = 'east'
            else:
                position = 'original'
            
            heatmap_coords[position] = {
                'name': name,
                'lat': lat,
                'lon': lon,
                'id': heatmap['id']
            }
    
    print("HEATMAP POSITIONS:")
    for pos, data in heatmap_coords.items():
        print(f"  {pos.upper()}: ({data['lat']}, {data['lon']})")
    
    print()
    
    # Analyze actual coordinate ranges from GeoJSON data
    print("ANALYZING GEOJSON COORDINATE RANGES:")
    
    for pos, data in heatmap_coords.items():
        geojson_data = db.get_heatmap_geojson(data['id'])
        
        if geojson_data and 'features' in geojson_data:
            lats = []
            lons = []
            
            # Extract all coordinates from triangular features
            for feature in geojson_data['features'][:100]:  # Sample first 100 for speed
                if 'geometry' in feature and 'coordinates' in feature['geometry']:
                    coords = feature['geometry']['coordinates'][0]  # Polygon exterior
                    for coord in coords:
                        lons.append(coord[0])
                        lats.append(coord[1])
            
            if lats and lons:
                lat_range = (min(lats), max(lats))
                lon_range = (min(lons), max(lons))
                
                print(f"  {pos.upper()}:")
                print(f"    Lat range: {lat_range[0]:.6f} to {lat_range[1]:.6f} (span: {lat_range[1]-lat_range[0]:.6f}°)")
                print(f"    Lon range: {lon_range[0]:.6f} to {lon_range[1]:.6f} (span: {lon_range[1]-lon_range[0]:.6f}°)")
    
    print()
    
    # Calculate expected vs actual spacing
    if 'original' in heatmap_coords and 'east' in heatmap_coords:
        orig_lon = heatmap_coords['original']['lon']
        east_lon = heatmap_coords['east']['lon']
        actual_lon_diff = east_lon - orig_lon
        
        print("SPACING ANALYSIS:")
        print(f"ORIGINAL → EAST longitude difference: {actual_lon_diff:.6f}°")
        
        # Calculate expected spacing for seamless joining
        # Each heatmap covers radius_km * 2 = 40km total width
        radius_km = 20
        center_lat = heatmap_coords['original']['lat']
        
        # Convert 40km to degrees longitude at this latitude
        km_per_degree_lon = 111.0 * abs(np.cos(np.radians(center_lat)))
        expected_lon_diff = (radius_km * 2) / km_per_degree_lon
        
        print(f"Expected longitude difference for seamless join: {expected_lon_diff:.6f}°")
        print(f"Difference: {actual_lon_diff - expected_lon_diff:.6f}° ({(actual_lon_diff - expected_lon_diff) * km_per_degree_lon:.3f} km)")
        
        if abs(actual_lon_diff - expected_lon_diff) > 0.001:
            print(f"⚠️  SPACING ISSUE DETECTED: {abs((actual_lon_diff - expected_lon_diff) * km_per_degree_lon):.3f} km gap/overlap")
        else:
            print("✅ Spacing appears correct")

if __name__ == "__main__":
    import numpy as np
    analyze_heatmap_edges()