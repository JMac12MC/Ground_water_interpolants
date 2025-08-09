#!/usr/bin/env python3
"""
COMPLETE POLYGON IMPORT: Load ALL polygon data without any merging or processing
"""
import json
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
from pyproj import Transformer
import pandas as pd
from pathlib import Path

def import_all_polygons_no_merge(esri_json_path, output_geojson_path):
    """Import ALL polygons from ESRI JSON without any merging - preserve everything."""
    
    try:
        print(f"🔄 Loading ESRI JSON from: {esri_json_path}")
        
        # Load the ESRI JSON file
        with open(esri_json_path, 'r') as f:
            esri_data = json.load(f)
        
        if 'features' not in esri_data:
            print("❌ No features found in ESRI JSON file")
            return False
        
        # Setup coordinate transformer: NZTM (EPSG:2193) -> WGS84 (EPSG:4326)
        transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
        
        # Process ALL rings as individual polygons - NO MERGING
        all_polygons = []
        total_rings = 0
        valid_polygons = 0
        
        print(f"📊 Processing {len(esri_data['features'])} ESRI features...")
        
        for feature_idx, esri_feature in enumerate(esri_data['features']):
            if 'geometry' not in esri_feature:
                continue
                
            esri_geometry = esri_feature['geometry']
            
            if 'rings' in esri_geometry:
                rings = esri_geometry['rings']
                print(f"   Feature {feature_idx}: Processing {len(rings)} rings")
                
                # Convert each ring to a separate WGS84 polygon
                for ring_idx, ring in enumerate(rings):
                    try:
                        # Convert NZTM coordinates to WGS84
                        wgs84_coords = []
                        for x, y in ring:
                            lon, lat = transformer.transform(x, y)
                            wgs84_coords.append([lon, lat])
                        
                        # Create polygon if we have enough points
                        if len(wgs84_coords) >= 4:
                            polygon = Polygon(wgs84_coords)
                            
                            if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
                                # Create GeoJSON feature
                                feature = {
                                    "type": "Feature",
                                    "properties": {
                                        "feature_id": feature_idx,
                                        "ring_id": ring_idx,
                                        "points": len(wgs84_coords),
                                        "area_sq_deg": polygon.area,
                                        "is_exterior": ring_idx == 0
                                    },
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [wgs84_coords]
                                    }
                                }
                                
                                all_polygons.append(feature)
                                valid_polygons += 1
                        
                        total_rings += 1
                        
                        # Progress report every 50 rings
                        if total_rings % 50 == 0:
                            print(f"     Processed {total_rings} rings, {valid_polygons} valid polygons...")
                            
                    except Exception as e:
                        print(f"     Ring {ring_idx}: Error - {e}")
                        continue
        
        if not all_polygons:
            print("❌ No valid polygons created!")
            return False
        
        # Create GeoJSON with ALL polygons
        geojson_data = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": all_polygons
        }
        
        # Save to file
        with open(output_geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        print(f"✅ COMPLETE IMPORT FINISHED!")
        print(f"   📊 Total rings processed: {total_rings}")
        print(f"   ✅ Valid polygons created: {valid_polygons}")
        print(f"   📄 Features in GeoJSON: {len(all_polygons)}")
        print(f"   💾 Output file: {output_geojson_path}")
        
        # Calculate total coverage
        total_area = sum(feature['properties']['area_sq_deg'] for feature in all_polygons)
        print(f"   🗺️  Total coverage: {total_area:.8f} square degrees")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in complete import: {e}")
        return False

if __name__ == "__main__":
    # Use the NEWEST polygon file
    input_file = "attached_assets/big_1754735961105.json"
    output_file = "all_polygons_complete_latest.geojson"
    
    print("🚀 Starting COMPLETE polygon import (no merging)...")
    
    # Import all polygons without merging
    success = import_all_polygons_no_merge(input_file, output_file)
    
    if success:
        print("🎉 COMPLETE IMPORT SUCCESSFUL!")
        print(f"   All polygon data preserved in: {output_file}")
        
        # Quick verification
        try:
            gdf = gpd.read_file(output_file)
            print(f"   ✅ Verified: {len(gdf)} polygons loaded")
            print(f"   📍 Coordinate range:")
            print(f"      Longitude: {gdf.bounds.minx.min():.4f} to {gdf.bounds.maxx.max():.4f}")
            print(f"      Latitude: {gdf.bounds.miny.min():.4f} to {gdf.bounds.maxy.max():.4f}")
        except Exception as e:
            print(f"   ⚠️  Could not verify file: {e}")
    else:
        print("❌ Complete import failed")