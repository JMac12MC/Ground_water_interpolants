#!/usr/bin/env python3
"""
Measure the exact dimensions of generated heatmaps to verify 20km×20km clipping precision
"""

import psycopg2
import os
import json
from utils import get_distance

def measure_heatmap_dimensions():
    """Measure the actual edge-to-edge dimensions of all stored heatmaps"""
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found")
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get all stored heatmaps with their GeoJSON data
        cur.execute("""
            SELECT id, heatmap_name, center_lat, center_lon, geojson_data, radius_km
            FROM stored_heatmaps 
            ORDER BY center_lat DESC, center_lon ASC
        """)
        
        results = cur.fetchall()
        
        if not results:
            print("❌ No stored heatmaps found")
            return
            
        print(f"📏 MEASURING DIMENSIONS OF {len(results)} STORED HEATMAPS")
        print("=" * 80)
        
        all_dimensions = []
        
        for row in results:
            heatmap_id, name, center_lat, center_lon, geojson_data_str, radius_km = row
            
            print(f"\n🔍 ANALYZING: {name}")
            print(f"   Center: ({center_lat:.6f}, {center_lon:.6f})")
            print(f"   Expected: {radius_km*2}km × {radius_km*2}km square")
            
            # Parse GeoJSON data
            try:
                # Handle both string and dict formats
                if isinstance(geojson_data_str, str):
                    geojson_data = json.loads(geojson_data_str)
                else:
                    geojson_data = geojson_data_str
                features = geojson_data.get('features', [])
                
                if not features:
                    print("   ❌ No features found in GeoJSON")
                    continue
                    
                print(f"   Features: {len(features)} triangular polygons")
                
                # Extract all coordinates from all polygons
                all_lats = []
                all_lons = []
                
                for feature in features:
                    if feature.get('geometry', {}).get('type') == 'Polygon':
                        coordinates = feature['geometry']['coordinates'][0]  # Outer ring
                        for coord in coordinates:
                            lon, lat = coord[0], coord[1]
                            all_lats.append(lat)
                            all_lons.append(lon)
                
                if not all_lats or not all_lons:
                    print("   ❌ No coordinates found in polygons")
                    continue
                
                # Find the bounding box (actual heatmap edges)
                min_lat, max_lat = min(all_lats), max(all_lats)
                min_lon, max_lon = min(all_lons), max(all_lons)
                
                print(f"   Bounding box:")
                print(f"     North edge: {max_lat:.8f}°")
                print(f"     South edge: {min_lat:.8f}°")
                print(f"     East edge:  {max_lon:.8f}°")
                print(f"     West edge:  {min_lon:.8f}°")
                
                # Calculate actual dimensions using Haversine distance
                # North-South dimension (height)
                ns_distance = get_distance(min_lat, center_lon, max_lat, center_lon)
                
                # East-West dimension (width) - use center latitude for accurate longitude scaling
                ew_distance = get_distance(center_lat, min_lon, center_lat, max_lon)
                
                # Diagonal distances for verification
                nw_se_diagonal = get_distance(max_lat, min_lon, min_lat, max_lon)
                ne_sw_diagonal = get_distance(max_lat, max_lon, min_lat, min_lon)
                
                print(f"   📏 MEASURED DIMENSIONS:")
                print(f"     Width (E-W):     {ew_distance:.6f} km")
                print(f"     Height (N-S):    {ns_distance:.6f} km")
                print(f"     NW-SE diagonal:  {nw_se_diagonal:.6f} km")
                print(f"     NE-SW diagonal:  {ne_sw_diagonal:.6f} km")
                
                # Expected dimensions
                expected_dimension = radius_km * 2
                expected_diagonal = expected_dimension * (2**0.5)  # sqrt(2)
                
                # Calculate errors
                width_error = abs(ew_distance - expected_dimension)
                height_error = abs(ns_distance - expected_dimension)
                diagonal_error = abs(nw_se_diagonal - expected_diagonal)
                
                print(f"   🎯 PRECISION ANALYSIS:")
                print(f"     Expected: {expected_dimension:.1f}km × {expected_dimension:.1f}km")
                print(f"     Width error:     {width_error:.6f} km ({width_error*1000:.1f}m)")
                print(f"     Height error:    {height_error:.6f} km ({height_error*1000:.1f}m)")
                print(f"     Diagonal error:  {diagonal_error:.6f} km ({diagonal_error*1000:.1f}m)")
                print(f"     Expected diagonal: {expected_diagonal:.6f} km")
                
                # Accuracy percentage
                width_accuracy = (1 - width_error/expected_dimension) * 100
                height_accuracy = (1 - height_error/expected_dimension) * 100
                
                print(f"   ✅ ACCURACY:")
                print(f"     Width accuracy:  {width_accuracy:.4f}%")
                print(f"     Height accuracy: {height_accuracy:.4f}%")
                
                # Store results for summary
                all_dimensions.append({
                    'name': name,
                    'width': ew_distance,
                    'height': ns_distance,
                    'width_error': width_error,
                    'height_error': height_error,
                    'expected': expected_dimension
                })
                
                # Check if within tolerance
                tolerance_m = 50  # 50 meter tolerance
                tolerance_km = tolerance_m / 1000
                
                if width_error <= tolerance_km and height_error <= tolerance_km:
                    print(f"   ✅ WITHIN TOLERANCE ({tolerance_m}m)")
                else:
                    print(f"   ⚠️  EXCEEDS TOLERANCE ({tolerance_m}m)")
                    
            except json.JSONDecodeError:
                print("   ❌ Invalid GeoJSON data")
                continue
            except Exception as e:
                print(f"   ❌ Error processing heatmap: {e}")
                continue
        
        # Summary statistics
        if all_dimensions:
            print(f"\n📊 SUMMARY STATISTICS FOR {len(all_dimensions)} HEATMAPS:")
            print("=" * 80)
            
            width_errors = [d['width_error'] for d in all_dimensions]
            height_errors = [d['height_error'] for d in all_dimensions]
            
            avg_width_error = sum(width_errors) / len(width_errors)
            avg_height_error = sum(height_errors) / len(height_errors)
            max_width_error = max(width_errors)
            max_height_error = max(height_errors)
            
            print(f"  Width errors:")
            print(f"    Average: {avg_width_error:.6f} km ({avg_width_error*1000:.1f}m)")
            print(f"    Maximum: {max_width_error:.6f} km ({max_width_error*1000:.1f}m)")
            
            print(f"  Height errors:")
            print(f"    Average: {avg_height_error:.6f} km ({avg_height_error*1000:.1f}m)")
            print(f"    Maximum: {max_height_error:.6f} km ({max_height_error*1000:.1f}m)")
            
            # Overall precision assessment
            avg_total_error = (avg_width_error + avg_height_error) / 2
            max_total_error = max(max_width_error, max_height_error)
            
            print(f"\n  Overall precision:")
            print(f"    Average dimension error: {avg_total_error:.6f} km ({avg_total_error*1000:.1f}m)")
            print(f"    Maximum dimension error: {max_total_error:.6f} km ({max_total_error*1000:.1f}m)")
            
            # Precision grade
            if max_total_error < 0.010:  # < 10m
                grade = "EXCELLENT (< 10m)"
            elif max_total_error < 0.050:  # < 50m
                grade = "VERY GOOD (< 50m)"
            elif max_total_error < 0.100:  # < 100m
                grade = "GOOD (< 100m)"
            else:
                grade = "NEEDS IMPROVEMENT (> 100m)"
                
            print(f"    Precision grade: {grade}")
            
            # Check consistency
            width_consistency = max(width_errors) - min(width_errors)
            height_consistency = max(height_errors) - min(height_errors)
            
            print(f"\n  Consistency across heatmaps:")
            print(f"    Width variation: {width_consistency:.6f} km ({width_consistency*1000:.1f}m)")
            print(f"    Height variation: {height_consistency:.6f} km ({height_consistency*1000:.1f}m)")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

if __name__ == "__main__":
    measure_heatmap_dimensions()