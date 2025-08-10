#!/usr/bin/env python3
"""
Process comprehensive polygon data to properly handle holes vs separate polygons.
Uses containment detection rather than size to identify internal holes.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
import json
from pyproj import Proj, transform

def process_comprehensive_polygons():
    """
    Process the ring-structured JSON to create proper polygon with holes.
    
    Algorithm:
    1. Load all rings from the JSON data
    2. Find the largest polygon (main Canterbury Plains area)
    3. Use containment detection to identify internal holes vs separate polygons
    4. Create main polygon with holes subtracted
    5. Keep all external polygons as separate legitimate drainage areas
    """
    
    # Load the ring-structured JSON data
    json_path = "attached_assets/big_1754735961105.json"
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'features' not in data or len(data['features']) == 0:
        print("❌ No features found in JSON data")
        return None
    
    feature = data['features'][0]
    if 'geometry' not in feature or 'rings' not in feature['geometry']:
        print("❌ No rings found in geometry")
        return None
    
    rings = feature['geometry']['rings']
    print(f"🔄 Processing {len(rings)} rings from comprehensive polygon data...")
    
    # Convert from projected coordinates (NZGD2000) to WGS84
    nzgd_proj = Proj(proj='tmerc', lat_0=0, lon_0=173, k=0.9996, x_0=1600000, y_0=10000000, ellps='GRS80')
    wgs84_proj = Proj(proj='latlong', datum='WGS84')
    
    # Convert all rings to polygons
    polygons = []
    for i, ring in enumerate(rings):
        # Convert coordinates
        converted_coords = []
        for coord in ring:
            lon, lat = transform(nzgd_proj, wgs84_proj, coord[0], coord[1])
            converted_coords.append((lon, lat))
        
        # Create polygon
        if len(converted_coords) > 2:  # Need at least 3 points for a valid polygon
            poly = Polygon(converted_coords)
            if poly.is_valid:
                polygons.append({
                    'ring_id': i,
                    'geometry': poly,
                    'area': poly.area
                })
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs='EPSG:4326')
    print(f"✅ Created {len(gdf)} valid polygons from rings")
    
    # Find the largest polygon (main Canterbury Plains area)
    main_idx = gdf.area.idxmax()
    main_poly = gdf.loc[[main_idx]].copy()
    main_geometry = main_poly.iloc[0].geometry
    
    print(f"🗺️ Main polygon (ring {main_poly.iloc[0].ring_id}): {main_geometry.area:.8f} sq degrees")
    
    # Get all other polygons
    others = gdf.drop(main_idx)
    print(f"🔍 Analyzing {len(others)} other polygons for containment...")
    
    # Identify holes using containment detection
    holes = others[others.geometry.within(main_geometry)]
    external_polygons = others[~others.geometry.within(main_geometry)]
    
    print(f"🕳️ Found {len(holes)} internal holes (bedrock exclusions)")
    print(f"🗺️ Found {len(external_polygons)} external polygons (separate drainage areas)")
    
    # Create main polygon with holes subtracted
    if len(holes) > 0:
        holes_union = unary_union(holes.geometry.tolist())
        main_geometry_with_holes = main_geometry.difference(holes_union)
        main_poly.loc[main_idx, 'geometry'] = main_geometry_with_holes
        print(f"✂️ Subtracted {len(holes)} holes from main polygon")
    
    # Combine main polygon (with holes removed) and external polygons
    result_polygons = []
    
    # Add main polygon with holes removed
    result_polygons.append({
        'polygon_type': 'main_with_holes_removed',
        'ring_id': main_poly.iloc[0].ring_id,
        'area_sq_deg': main_poly.iloc[0].geometry.area,
        'geometry': main_poly.iloc[0].geometry
    })
    
    # Add external polygons (separate drainage areas)
    for idx, row in external_polygons.iterrows():
        result_polygons.append({
            'polygon_type': 'external_drainage_area',
            'ring_id': row.ring_id,
            'area_sq_deg': row.area,
            'geometry': row.geometry
        })
    
    # Create final GeoDataFrame
    result_gdf = gpd.GeoDataFrame(result_polygons, crs='EPSG:4326')
    
    print(f"🎯 FINAL RESULT:")
    print(f"   - 1 main polygon with {len(holes)} holes removed")
    print(f"   - {len(external_polygons)} separate external drainage areas")
    print(f"   - Total clipping area: {result_gdf.geometry.area.sum():.8f} sq degrees")
    
    return result_gdf

def save_processed_polygons(gdf, output_path="comprehensive_polygons_processed.geojson"):
    """Save the processed polygons to GeoJSON file"""
    if gdf is not None:
        gdf.to_file(output_path, driver='GeoJSON')
        print(f"💾 Saved processed polygons to {output_path}")
        return True
    return False

if __name__ == "__main__":
    # Process the comprehensive polygons
    result = process_comprehensive_polygons()
    
    if result is not None:
        # Save to file
        save_processed_polygons(result)
        print("✅ Comprehensive polygon processing complete!")
    else:
        print("❌ Failed to process comprehensive polygons")