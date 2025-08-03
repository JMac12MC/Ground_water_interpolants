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
from sqlalchemy import text
from utils import get_distance

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return c * 6371

def get_heatmap_rectangular_bounds(geojson_data):
    """Extract the rectangular bounds (min/max lat/lon) from GeoJSON triangular mesh data"""
    try:
        if isinstance(geojson_data, str):
            geojson_data = json.loads(geojson_data)
        
        # Collect all coordinate points from triangular features
        min_lon, min_lat = float('inf'), float('inf')
        max_lon, max_lat = float('-inf'), float('-inf')
        
        for feature in geojson_data.get('features', []):
            if feature.get('geometry', {}).get('type') == 'Polygon':
                coords = feature['geometry']['coordinates'][0]  # Exterior ring
                for lon, lat in coords:
                    min_lon = min(min_lon, lon)
                    max_lon = max(max_lon, lon)
                    min_lat = min(min_lat, lat)
                    max_lat = max(max_lat, lat)
        
        if min_lon == float('inf'):
            return None
            
        return {
            'min_lon': min_lon, 'max_lon': max_lon,
            'min_lat': min_lat, 'max_lat': max_lat,
            'width': max_lon - min_lon,
            'height': max_lat - min_lat
        }
        
    except Exception as e:
        print(f"Error extracting rectangular bounds: {e}")
        return None

def measure_rectangular_edge_gap(bounds1, bounds2, center1, center2):
    """
    Measure gap between rectangular heatmap edges
    Returns negative value for overlaps
    """
    # Determine the spatial relationship
    center_lat1, center_lon1 = center1
    center_lat2, center_lon2 = center2
    
    # Calculate differences
    lat_diff = abs(center_lat1 - center_lat2)
    lon_diff = abs(center_lon1 - center_lon2)
    
    # Determine if they're horizontally or vertically adjacent
    if lat_diff < 0.01:  # Same latitude row - horizontal neighbors
        # Measure horizontal gap between east/west edges
        if center_lon1 < center_lon2:  # bounds1 is west of bounds2
            gap_degrees = bounds2['min_lon'] - bounds1['max_lon']
            edge1_desc = f"east edge ({bounds1['max_lon']:.4f})"
            edge2_desc = f"west edge ({bounds2['min_lon']:.4f})"
        else:  # bounds1 is east of bounds2
            gap_degrees = bounds1['min_lon'] - bounds2['max_lon']
            edge1_desc = f"west edge ({bounds1['min_lon']:.4f})"
            edge2_desc = f"east edge ({bounds2['max_lon']:.4f})"
        
        # Convert longitude to km at this latitude
        lat_avg = (center_lat1 + center_lat2) / 2
        gap_km = gap_degrees * 111.32 * abs(math.cos(math.radians(lat_avg)))
        edge_info = f"horizontal ({edge1_desc} to {edge2_desc})"
        
    elif lon_diff < 0.01:  # Same longitude column - vertical neighbors
        # Measure vertical gap between north/south edges
        if center_lat1 > center_lat2:  # bounds1 is north of bounds2
            gap_degrees = bounds2['max_lat'] - bounds1['min_lat']
            edge1_desc = f"south edge ({bounds1['min_lat']:.4f})"
            edge2_desc = f"north edge ({bounds2['max_lat']:.4f})"
        else:  # bounds1 is south of bounds2
            gap_degrees = bounds1['max_lat'] - bounds2['min_lat']
            edge1_desc = f"north edge ({bounds1['max_lat']:.4f})"
            edge2_desc = f"south edge ({bounds2['min_lat']:.4f})"
        
        # Convert latitude to km
        gap_km = gap_degrees * 111.32
        edge_info = f"vertical ({edge1_desc} to {edge2_desc})"
        
    else:
        # Diagonal neighbors - measure minimum edge distance
        # Calculate horizontal and vertical gaps
        h_gap = None
        v_gap = None
        
        # Horizontal gap
        if center_lon1 < center_lon2:  # bounds1 west of bounds2
            h_gap_deg = bounds2['min_lon'] - bounds1['max_lon']
        else:
            h_gap_deg = bounds1['min_lon'] - bounds2['max_lon']
        lat_avg = (center_lat1 + center_lat2) / 2
        h_gap = h_gap_deg * 111.32 * abs(math.cos(math.radians(lat_avg)))
        
        # Vertical gap  
        if center_lat1 > center_lat2:  # bounds1 north of bounds2
            v_gap_deg = bounds2['max_lat'] - bounds1['min_lat']
        else:
            v_gap_deg = bounds1['max_lat'] - bounds2['min_lat']
        v_gap = v_gap_deg * 111.32
        
        # Use minimum distance (closest edge)
        if abs(h_gap) < abs(v_gap):
            gap_km = h_gap
            edge_info = f"horizontal (closest edge)"
        else:
            gap_km = v_gap
            edge_info = f"vertical (closest edge)"
    
    return gap_km, edge_info

def measure_centroid_to_edge_distances(heatmap_data):
    """
    Measure distances from heatmap centroid to each of the 4 edges
    """
    center = heatmap_data.get('center', (0, 0))
    if isinstance(center, list):
        center = tuple(center)
    
    geojson_data = heatmap_data.get('geojson_data')
    name = heatmap_data.get('name', 'unknown')
    
    if not geojson_data:
        return None
    
    # Extract rectangular bounds from GeoJSON
    bounds_dict = get_heatmap_rectangular_bounds(geojson_data)
    if not bounds_dict:
        return None
    
    # Extract bounds: min_lon, min_lat, max_lon, max_lat
    west = bounds_dict['min_lon']
    south = bounds_dict['min_lat']
    east = bounds_dict['max_lon']
    north = bounds_dict['max_lat']
    
    # Calculate distances from centroid to each edge
    distances = {
        'north': get_distance(center[0], center[1], north, center[1]),
        'south': get_distance(center[0], center[1], south, center[1]),
        'east': get_distance(center[0], center[1], center[0], east),
        'west': get_distance(center[0], center[1], center[0], west)
    }
    
    return {
        'name': name,
        'center': center,
        'bounds_dict': bounds_dict,
        'distances': distances
    }

def analyze_centroid_to_edge_distances(polygon_db):
    """
    Analyze centroid-to-edge distances for all stored heatmaps
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
        if hasattr(polygon_db, 'engine') and polygon_db.engine:
            with polygon_db.engine.connect() as connection:
                result = connection.execute(text(query)).fetchall()
        else:
            print("No database connection available")
            return None
            
        if not result:
            print("No stored heatmaps found")
            return None
            
        print(f"📏 ANALYZING CENTROID-TO-EDGE DISTANCES FOR {len(result)} HEATMAPS")
        print("=" * 70)
        
        # Process each heatmap and measure centroid-to-edge distances
        centroid_measurements = []
        for row in result:
            heatmap_name, center_lat, center_lon, geojson_data, created_at = row
            
            heatmap_data = {
                'name': heatmap_name,
                'center': (center_lat, center_lon),
                'geojson_data': geojson_data,
                'created_at': created_at
            }
            
            edge_distances = measure_centroid_to_edge_distances(heatmap_data)
            if edge_distances:
                centroid_measurements.append(edge_distances)
                distances = edge_distances['distances']
                center = edge_distances['center']
                bounds = edge_distances['bounds_dict']
                
                print(f"🎯 {edge_distances['name']}")
                print(f"   Center: ({center[0]:.6f}, {center[1]:.6f})")
                print(f"   Bounds: {bounds['min_lon']:.6f} to {bounds['max_lon']:.6f} (lon)")
                print(f"           {bounds['min_lat']:.6f} to {bounds['max_lat']:.6f} (lat)")
                print(f"   Coverage dimensions: {bounds['width']:.6f}° × {bounds['height']:.6f}°")
                print(f"   Centroid to edges:")
                print(f"     🧭 North: {distances['north']:.3f}km | South: {distances['south']:.3f}km")
                print(f"     🧭 East:  {distances['east']:.3f}km  | West:  {distances['west']:.3f}km")
                
                # Calculate average radius and symmetry
                avg_radius = sum(distances.values()) / 4
                min_radius = min(distances.values())
                max_radius = max(distances.values())
                symmetry_variation = max_radius - min_radius
                
                print(f"   📊 Average radius: {avg_radius:.3f}km")
                print(f"   📊 Radius range: {min_radius:.3f}km to {max_radius:.3f}km")
                print(f"   📊 Symmetry variation: {symmetry_variation:.3f}km")
                
                if symmetry_variation < 0.5:
                    print(f"   ✅ WELL-CENTERED (variation < 0.5km)")
                elif symmetry_variation < 1.0:
                    print(f"   ⚠️ SLIGHTLY OFF-CENTER (variation < 1.0km)")
                else:
                    print(f"   ❌ POORLY CENTERED (variation ≥ 1.0km)")
                
                print()
            else:
                print(f"❌ Failed to measure distances for: {heatmap_name}")
        
        # Summary statistics
        if centroid_measurements:
            all_radii = []
            all_variations = []
            
            for measurement in centroid_measurements:
                distances = measurement['distances']
                avg_radius = sum(distances.values()) / 4
                variation = max(distances.values()) - min(distances.values())
                all_radii.append(avg_radius)
                all_variations.append(variation)
            
            print(f"📊 SUMMARY STATISTICS:")
            print(f"   Total heatmaps analyzed: {len(centroid_measurements)}")
            print(f"   Average radius across all heatmaps: {sum(all_radii)/len(all_radii):.3f}km")
            print(f"   Radius range: {min(all_radii):.3f}km to {max(all_radii):.3f}km")
            print(f"   Average symmetry variation: {sum(all_variations)/len(all_variations):.3f}km")
            
            well_centered = sum(1 for v in all_variations if v < 0.5)
            print(f"   Well-centered heatmaps: {well_centered}/{len(all_variations)} ({well_centered/len(all_variations)*100:.1f}%)")
        
        print("=" * 70)
        return centroid_measurements
        
    except Exception as e:
        print(f"Error analyzing centroid-to-edge distances: {e}")
        return None

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
        if hasattr(polygon_db, 'engine') and polygon_db.engine:
            with polygon_db.engine.connect() as connection:
                result = connection.execute(text(query)).fetchall()
        else:
            print("No database connection available")
            return None
            
        if not result:
            print("No stored heatmaps found")
            return None
            
        print(f"🎯 ANALYZING GAPS BETWEEN {len(result)} DISPLAYED HEATMAPS")
        print("=" * 65)
        
        # Process each heatmap and extract rectangular bounds
        heatmaps = []
        for row in result:
            heatmap_name, center_lat, center_lon, geojson_data, created_at = row
            
            # Extract rectangular bounds from GeoJSON data
            bounds = get_heatmap_rectangular_bounds(geojson_data)
            if bounds:
                heatmaps.append({
                    'name': heatmap_name,
                    'center': (center_lat, center_lon),
                    'bounds': bounds,
                    'created_at': created_at
                })
                print(f"✅ Processed bounds for: {heatmap_name}")
                print(f"    Rectangle: {bounds['min_lon']:.4f} to {bounds['max_lon']:.4f} (lon), {bounds['min_lat']:.4f} to {bounds['max_lat']:.4f} (lat)")
            else:
                print(f"❌ Failed to extract bounds for: {heatmap_name}")
        
        if len(heatmaps) < 2:
            print("Need at least 2 heatmaps to measure gaps")
            return None
            
        print(f"\n📏 MEASURING RECTANGULAR EDGE GAPS BETWEEN ADJACENT HEATMAPS:")
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
                    gap_distance, edge_info = measure_rectangular_edge_gap(
                        heatmap1['bounds'], heatmap2['bounds'],
                        heatmap1['center'], heatmap2['center']
                    )
                    
                    if gap_distance is not None:
                        gap_results.append({
                            'heatmap1': heatmap1['name'],
                            'heatmap2': heatmap2['name'],
                            'center_distance': center_distance,
                            'edge_gap': gap_distance,
                            'edge_info': edge_info
                        })
                        
                        # Display result
                        print(f"  {heatmap1['name']} ↔ {heatmap2['name']}:")
                        print(f"    Center distance: {center_distance:.3f} km")
                        print(f"    Edge measurement: {edge_info}")
                        if gap_distance < 0:
                            print(f"    🔴 OVERLAP: {abs(gap_distance):.3f} km (negative edge distance: {gap_distance:.3f} km)")
                        else:
                            print(f"    🟢 GAP: {gap_distance:.3f} km")
                        print()
        
        # Summary statistics
        if gap_results:
            gaps = [r['edge_gap'] for r in gap_results]
            print(f"📊 GAP AND OVERLAP SUMMARY:")
            print(f"  Total adjacent pairs analyzed: {len(gap_results)}")
            print(f"  Average gap/overlap: {sum(gaps)/len(gaps):.3f} km")
            print(f"  Range: {min(gaps):.3f} to {max(gaps):.3f} km")
            print(f"  Gap variation: {max(gaps) - min(gaps):.3f} km")
            
            overlaps = [g for g in gaps if g < 0]  # Keep negative values for overlaps
            actual_gaps = [g for g in gaps if g >= 0]
            
            if overlaps:
                avg_overlap = sum(abs(o) for o in overlaps) / len(overlaps)
                print(f"  🔴 OVERLAPPING pairs: {len(overlaps)} (avg overlap: {avg_overlap:.3f} km)")
                for i, overlap in enumerate(overlaps):
                    print(f"    • Overlap {i+1}: {abs(overlap):.3f} km")
            if actual_gaps:
                avg_gap = sum(actual_gaps) / len(actual_gaps)
                print(f"  🟢 GAP pairs: {len(actual_gaps)} (avg gap: {avg_gap:.3f} km)")
                for i, gap in enumerate(actual_gaps):
                    print(f"    • Gap {i+1}: {gap:.3f} km")
        
        print("=" * 65)
        return gap_results
        
    except Exception as e:
        print(f"Error analyzing heatmap gaps: {e}")
        return None

if __name__ == "__main__":
    print("This module provides heatmap gap measurement functionality")
    print("Import and use analyze_displayed_heatmap_gaps(polygon_db) function")