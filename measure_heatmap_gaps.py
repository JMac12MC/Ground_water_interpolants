#!/usr/bin/env python3
"""
Measure actual gaps between displayed heatmaps by analyzing their geometric boundaries
"""

import json
import math
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
import geopandas as gpd
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371

def get_heatmap_boundary_from_geojson(geojson_data):
    """Extract the boundary polygon from GeoJSON triangular mesh data"""
    try:
        if isinstance(geojson_data, str):
            geojson_data = json.loads(geojson_data)
        
        # Collect all coordinate points from triangular features
        all_points = []
        for feature in geojson_data.get('features', []):
            if feature.get('geometry', {}).get('type') == 'Polygon':
                coords = feature['geometry']['coordinates'][0]  # Exterior ring
                all_points.extend(coords)
        
        if not all_points:
            return None
            
        # Create a GeoDataFrame from points
        gdf = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat in all_points],
            crs='EPSG:4326'
        )
        
        # Get the convex hull (boundary) of all points
        boundary = gdf.unary_union.convex_hull
        return boundary
        
    except Exception as e:
        print(f"Error extracting boundary: {e}")
        return None

def measure_gap_between_boundaries(boundary1, boundary2):
    """Measure the minimum distance between two boundary polygons"""
    try:
        # Find the nearest points between the two boundaries
        point1, point2 = nearest_points(boundary1, boundary2)
        
        # Convert to lat/lon and calculate distance
        lat1, lon1 = point1.y, point1.x
        lat2, lon2 = point2.y, point2.x
        
        gap_distance = haversine_distance(lat1, lon1, lat2, lon2)
        return gap_distance, (lat1, lon1), (lat2, lon2)
        
    except Exception as e:
        print(f"Error measuring gap: {e}")
        return None, None, None

def analyze_displayed_heatmap_gaps(polygon_db):
    """
    Analyze gaps between all currently stored/displayed heatmaps
    Returns detailed gap measurements between adjacent heatmaps
    """
    try:
        # Get all stored heatmaps with their GeoJSON data
        query = """
        SELECT heatmap_name, center_lat, center_lon, geojson_data, created_at
        FROM stored_heatmaps 
        WHERE interpolation_method = 'ground_water_level_kriging'
        ORDER BY created_at DESC 
        LIMIT 10
        """
        
        # Execute query through database connection
        if hasattr(polygon_db, 'pg_engine') and polygon_db.pg_engine:
            result = polygon_db.pg_engine.execute(query).fetchall()
        else:
            print("No database connection available")
            return None
            
        if not result:
            print("No stored heatmaps found")
            return None
            
        print(f"🎯 ANALYZING GAPS BETWEEN {len(result)} DISPLAYED HEATMAPS")
        print("=" * 65)
        
        # Process each heatmap and extract boundaries
        heatmaps = []
        for row in result:
            heatmap_name, center_lat, center_lon, geojson_data, created_at = row
            
            # Extract boundary from GeoJSON data
            boundary = get_heatmap_boundary_from_geojson(geojson_data)
            if boundary:
                heatmaps.append({
                    'name': heatmap_name,
                    'center': (center_lat, center_lon),
                    'boundary': boundary,
                    'created_at': created_at
                })
                print(f"✅ Processed boundary for: {heatmap_name}")
            else:
                print(f"❌ Failed to extract boundary for: {heatmap_name}")
        
        if len(heatmaps) < 2:
            print("Need at least 2 heatmaps to measure gaps")
            return None
            
        print(f"\n📏 MEASURING GAPS BETWEEN ADJACENT HEATMAPS:")
        print("-" * 65)
        
        # Measure gaps between all heatmap pairs
        gap_results = []
        for i, heatmap1 in enumerate(heatmaps):
            for j, heatmap2 in enumerate(heatmaps[i+1:], i+1):
                
                # Check if they are spatially adjacent (within reasonable distance)
                center_distance = haversine_distance(
                    heatmap1['center'][0], heatmap1['center'][1],
                    heatmap2['center'][0], heatmap2['center'][1]
                )
                
                # Only analyze adjacent heatmaps (within ~25km center distance)
                if center_distance <= 25.0:
                    gap_distance, point1, point2 = measure_gap_between_boundaries(
                        heatmap1['boundary'], heatmap2['boundary']
                    )
                    
                    if gap_distance is not None:
                        gap_results.append({
                            'heatmap1': heatmap1['name'],
                            'heatmap2': heatmap2['name'],
                            'center_distance': center_distance,
                            'edge_gap': gap_distance,
                            'gap_points': (point1, point2)
                        })
                        
                        # Display result
                        status = "OVERLAP" if gap_distance < 0 else "GAP"
                        print(f"  {heatmap1['name']} ↔ {heatmap2['name']}:")
                        print(f"    Center distance: {center_distance:.3f} km")
                        print(f"    Edge gap: {gap_distance:.3f} km ({status})")
                        if gap_distance < 0:
                            print(f"    Overlap: {abs(gap_distance):.3f} km")
                        print()
        
        # Summary statistics
        if gap_results:
            gaps = [r['edge_gap'] for r in gap_results]
            print(f"📊 GAP SUMMARY:")
            print(f"  Total adjacent pairs analyzed: {len(gap_results)}")
            print(f"  Average gap: {sum(gaps)/len(gaps):.3f} km")
            print(f"  Minimum gap: {min(gaps):.3f} km")
            print(f"  Maximum gap: {max(gaps):.3f} km")
            print(f"  Gap variation: {max(gaps) - min(gaps):.3f} km")
            
            overlaps = [abs(g) for g in gaps if g < 0]
            actual_gaps = [g for g in gaps if g >= 0]
            
            if overlaps:
                print(f"  Overlapping pairs: {len(overlaps)} (avg overlap: {sum(overlaps)/len(overlaps):.3f} km)")
            if actual_gaps:
                print(f"  Gap pairs: {len(actual_gaps)} (avg gap: {sum(actual_gaps)/len(actual_gaps):.3f} km)")
        
        print("=" * 65)
        return gap_results
        
    except Exception as e:
        print(f"Error analyzing heatmap gaps: {e}")
        return None

if __name__ == "__main__":
    print("This module provides heatmap gap measurement functionality")
    print("Import and use analyze_displayed_heatmap_gaps(polygon_db) function")