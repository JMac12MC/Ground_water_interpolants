#!/usr/bin/env python3
"""
COMPREHENSIVE POLYGON IMPORT: Process all 198 rings from ESRI JSON as separate polygon features
"""
import json
import geopandas as gpd
from shapely.geometry import Polygon
import pyproj
from pyproj import Transformer
import pandas as pd
from pathlib import Path

def import_all_rings_as_polygons(esri_json_path, output_geojson_path):
    """Import all rings from ESRI JSON as separate polygon features for comprehensive coverage."""
    
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
        
        # Process all rings as separate polygons
        polygon_features = []
        total_rings = 0
        total_points = 0
        
        print(f"📊 Processing {len(esri_data['features'])} ESRI features...")
        
        for feature_idx, esri_feature in enumerate(esri_data['features']):
            if 'geometry' not in esri_feature:
                continue
                
            esri_geometry = esri_feature['geometry']
            
            if 'rings' in esri_geometry:
                rings = esri_geometry['rings']
                print(f"   Feature {feature_idx}: {len(rings)} rings found")
                
                # Convert each ring to a separate WGS84 polygon
                for ring_idx, ring in enumerate(rings):
                    try:
                        # Convert NZTM coordinates to WGS84
                        wgs84_coords = []
                        for x, y in ring:
                            lon, lat = transformer.transform(x, y)
                            wgs84_coords.append([lon, lat])
                        
                        # Create Shapely polygon
                        if len(wgs84_coords) >= 4:  # Minimum for valid polygon
                            polygon = Polygon(wgs84_coords)
                            
                            if polygon.is_valid and not polygon.is_empty:
                                # Calculate area in square degrees
                                area_deg2 = polygon.area
                                
                                # Create GeoJSON feature
                                feature = {
                                    "type": "Feature",
                                    "properties": {
                                        "feature_id": feature_idx,
                                        "ring_id": ring_idx,
                                        "points_count": len(wgs84_coords),
                                        "area_deg2": area_deg2,
                                        "polygon_type": "exterior" if ring_idx == 0 else "interior/island"
                                    },
                                    "geometry": {
                                        "type": "Polygon",
                                        "coordinates": [wgs84_coords]
                                    }
                                }
                                
                                polygon_features.append(feature)
                                total_rings += 1
                                total_points += len(wgs84_coords)
                                
                                if ring_idx < 3:  # Show details for first few rings
                                    print(f"     Ring {ring_idx}: {len(wgs84_coords)} points, area = {area_deg2:.8f} sq deg")
                            else:
                                print(f"     Ring {ring_idx}: Invalid or empty polygon, skipping")
                        else:
                            print(f"     Ring {ring_idx}: Too few points ({len(wgs84_coords)}), skipping")
                            
                    except Exception as e:
                        print(f"     Ring {ring_idx}: Error processing - {e}")
                        continue
        
        if not polygon_features:
            print("❌ No valid polygons created!")
            return False
        
        # Create comprehensive GeoJSON
        geojson_data = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": polygon_features
        }
        
        # Save to file
        with open(output_geojson_path, 'w') as f:
            json.dump(geojson_data, f, indent=2)
        
        print(f"✅ COMPREHENSIVE IMPORT COMPLETE!")
        print(f"   📈 Total rings processed: {total_rings}")
        print(f"   📍 Total coordinate points: {total_points}")
        print(f"   📄 Features in GeoJSON: {len(polygon_features)}")
        print(f"   💾 Output saved to: {output_geojson_path}")
        
        # Calculate coverage statistics
        total_area = sum(feature['properties']['area_deg2'] for feature in polygon_features)
        avg_area = total_area / len(polygon_features) if polygon_features else 0
        
        print(f"   🗺️  Total coverage: {total_area:.8f} square degrees")
        print(f"   📊 Average polygon size: {avg_area:.8f} square degrees")
        
        return True
        
    except Exception as e:
        print(f"❌ Critical error in comprehensive import: {e}")
        return False

def load_comprehensive_polygon_data(geojson_path):
    """Load all polygon data into GeoPandas for use in the application."""
    
    try:
        if not Path(geojson_path).exists():
            print(f"❌ GeoJSON file not found: {geojson_path}")
            return None
            
        # Load using GeoPandas
        gdf = gpd.read_file(geojson_path)
        
        print(f"✅ Loaded comprehensive polygon data:")
        print(f"   📊 Polygons: {len(gdf)}")
        print(f"   🗺️  Coverage area: {gdf.geometry.area.sum():.8f} square degrees")
        print(f"   📍 Coordinate range:")
        print(f"      Longitude: {gdf.bounds.minx.min():.4f} to {gdf.bounds.maxx.max():.4f}")
        print(f"      Latitude: {gdf.bounds.miny.min():.4f} to {gdf.bounds.maxy.max():.4f}")
        
        return gdf
        
    except Exception as e:
        print(f"❌ Error loading comprehensive polygon data: {e}")
        return None

if __name__ == "__main__":
    # File paths
    input_file = "attached_assets/myDrawing_1754734043555.json"
    output_file = "comprehensive_clipping_polygons.geojson"
    
    print("🚀 Starting comprehensive polygon import process...")
    
    # Import all rings as separate polygons
    success = import_all_rings_as_polygons(input_file, output_file)
    
    if success:
        # Load and verify the data
        polygon_data = load_comprehensive_polygon_data(output_file)
        
        if polygon_data is not None:
            print("🎉 COMPREHENSIVE POLYGON IMPORT SUCCESSFUL!")
            print(f"   Ready to replace Banks Peninsula and soil drainage clipping polygons")
        else:
            print("❌ Failed to load comprehensive polygon data")
    else:
        print("❌ Comprehensive polygon import failed")