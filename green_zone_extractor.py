"""
Green Zone Polygon Extractor
Creates boundary polygons around high-probability indicator kriging zones (≥0.7)
"""

import json
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd

def extract_green_zones_from_indicator_heatmaps(polygon_db):
    """
    Extract red/orange zones (<0.7 probability) from stored indicator kriging heatmaps
    and create a boundary polygon around them.
    
    Returns:
        dict: GeoJSON polygon representing the boundary of red/orange zones
    """
    print("🔍 EXTRACTING RED/ORANGE ZONES: Loading indicator kriging heatmaps...")
    
    # Get all stored heatmaps
    all_heatmaps = polygon_db.get_all_stored_heatmaps()
    print(f"🔍 TOTAL HEATMAPS RETRIEVED: {len(all_heatmaps)}")
    
    # Enhanced filtering with multiple pattern checks
    indicator_heatmaps = []
    for hm in all_heatmaps:
        name = hm.get('heatmap_name', '').lower()
        method = hm.get('interpolation_method', '').lower()
        
        # Check multiple patterns for indicator kriging
        if ('indicator' in name or 
            'indicator_kriging' in name or 
            'indicator_kriging' in method or
            name.startswith('indicator_kriging')):
            indicator_heatmaps.append(hm)
    
    print(f"📊 Found {len(indicator_heatmaps)} indicator kriging heatmaps")
    
    # Enhanced debug: Show what we found
    if indicator_heatmaps:
        print("🔍 DETECTED INDICATOR HEATMAPS:")
        for i, hm in enumerate(indicator_heatmaps[:5]):  # Show first 5
            name = hm.get('heatmap_name', 'unknown')
            method = hm.get('interpolation_method', 'unknown')
            has_geojson = bool(hm.get('geojson_data'))
            print(f"  {i+1}. {name} (method: {method}) - GeoJSON: {has_geojson}")
        if len(indicator_heatmaps) > 5:
            print(f"  ... and {len(indicator_heatmaps) - 5} more")
    else:
        print("🔍 DEBUG: No indicator heatmaps found. Showing ALL available heatmaps:")
        for i, hm in enumerate(all_heatmaps[:10]):  # Show first 10
            name = hm.get('heatmap_name', 'unknown')
            method = hm.get('interpolation_method', 'unknown')
            print(f"  {i+1}. {name} (method: {method})")
        if len(all_heatmaps) > 10:
            print(f"  ... and {len(all_heatmaps) - 10} more")
    
    if not indicator_heatmaps:
        print("❌ No indicator kriging heatmaps found")
        return None
    
    # Collect all red/orange zone features (<0.7)
    red_orange_features = []
    total_features = 0
    red_orange_feature_count = 0
    
    for heatmap in indicator_heatmaps:
        geojson_data = heatmap.get('geojson_data')
        if not geojson_data or 'features' not in geojson_data:
            continue
            
        heatmap_name = heatmap.get('heatmap_name', 'unknown')
        features = geojson_data['features']
        total_features += len(features)
        
        heatmap_red_orange_count = 0
        for feature in features:
            if 'properties' in feature:
                # Check for indicator value < 0.7 (red/orange zone)
                value = feature['properties'].get('value', 0)
                if value < 0.7:
                    red_orange_features.append(feature)
                    heatmap_red_orange_count += 1
                    red_orange_feature_count += 1
        
        print(f"  📍 {heatmap_name}: {heatmap_red_orange_count}/{len(features)} red/orange features")
    
    print(f"✅ EXTRACTED: {red_orange_feature_count}/{total_features} total red/orange zone features")
    
    if not red_orange_features:
        print("❌ No red/orange zone features found (<0.7 threshold)")
        return None
    
    # Create polygons from red/orange features
    red_orange_polygons = []
    
    for feature in red_orange_features:
        try:
            coords = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'Polygon':
                # Handle standard polygon
                polygon = Polygon(coords[0])  # First ring is exterior
                if polygon.is_valid:
                    red_orange_polygons.append(polygon)
        except Exception as e:
            continue  # Skip invalid features
    
    print(f"🔷 Created {len(red_orange_polygons)} valid polygons from red/orange features")
    
    if not red_orange_polygons:
        print("❌ No valid polygons created from red/orange features")
        return None
    
    # Create detailed boundaries for red/orange zones (unified when adjacent)
    try:
        # Union overlapping and adjacent areas to create continuous zones
        unified_geometry = unary_union(red_orange_polygons)
        
        # Collect all separate red/orange zone boundaries
        boundary_features = []
        zone_count = 0
        
        if isinstance(unified_geometry, MultiPolygon):
            # Multiple separate zones - create boundary for each
            for i, geom in enumerate(unified_geometry.geoms):
                if geom.area > 0:  # Only include valid geometries
                    # Use actual boundary, detailed shape following red/orange areas
                    boundary_coords = list(geom.exterior.coords)
                    
                    boundary_features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [boundary_coords]
                        },
                        "properties": {
                            "name": f"Red/Orange Zone {i+1}",
                            "description": f"Low/medium probability indicator zone (<0.7)",
                            "zone_type": "low_medium_probability_boundary",
                            "threshold": 0.7,
                            "zone_id": i+1
                        }
                    })
                    zone_count += 1
            
            print(f"🔗 Created {zone_count} separate detailed red/orange zone boundaries")
            
        else:
            # Single unified zone
            boundary_coords = list(unified_geometry.exterior.coords)
            
            boundary_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon", 
                    "coordinates": [boundary_coords]
                },
                "properties": {
                    "name": "Red/Orange Zone 1",
                    "description": f"Low/medium probability indicator zone (<0.7)",
                    "zone_type": "low_medium_probability_boundary", 
                    "threshold": 0.7,
                    "zone_id": 1
                }
            })
            zone_count = 1
            print(f"🔗 Created 1 detailed red/orange zone boundary")
        
        # Create GeoJSON with multiple detailed boundaries
        boundary_geojson = {
            "type": "FeatureCollection",
            "features": boundary_features
        }
        
        print(f"✅ BOUNDARIES CREATED: {len(boundary_features)} detailed polygons for separate red/orange zones")
        return boundary_geojson
        
    except Exception as e:
        print(f"❌ Error creating boundary polygon: {e}")
        return None

def store_green_zone_boundary(polygon_db, boundary_geojson):
    """
    Store the green zone boundary polygon in the database.
    
    Returns:
        int: ID of stored polygon, or None if failed
    """
    if not boundary_geojson:
        return None
        
    try:
        # Store as one comprehensive green zone boundary file (multiple features in one polygon)
        # Calculate overall center point for storage
        all_coords = []
        for feature in boundary_geojson['features']:
            coords = feature['geometry']['coordinates'][0]
            all_coords.extend(coords)
        
        # Store the entire boundary collection as one polygon
        polygon_id = polygon_db.store_polygon(
            name="Red/Orange Zone Boundaries",
            polygon_type="indicator_boundary",
            coordinates=json.dumps(boundary_geojson),  # Store the entire GeoJSON
            metadata={
                "description": f"Detailed boundaries around all low/medium probability indicator zones (<0.7)",
                "threshold": 0.7,
                "zone_type": "red_orange",
                "total_zones": len(boundary_geojson['features']),
                "generated_at": str(np.datetime64('now'))
            }
        )
        
        print(f"💾 STORED: Red/orange zone boundaries as single polygon file with ID {polygon_id}")
        return polygon_id
        
    except Exception as e:
        print(f"❌ Error storing boundary polygon: {e}")
        return None

def display_green_zone_boundary_on_map(folium_map, boundary_geojson):
    """
    Add the green zone boundary to a Folium map.
    """
    if not boundary_geojson or 'features' not in boundary_geojson:
        return False
        
    try:
        import folium
        
        # Add each separate green zone boundary to the map
        zones_added = 0
        
        for feature in boundary_geojson['features']:
            coords = feature['geometry']['coordinates'][0]
            zone_name = feature['properties']['name']
            
            # Convert coordinates to lat/lon format for Folium
            folium_coords = [[coord[1], coord[0]] for coord in coords]
            
            # Add boundary polygon with red color for red/orange zones
            folium.Polygon(
                locations=folium_coords,
                color="#FF4444",  # Red color for red/orange zones
                weight=3,
                opacity=0.8,
                fill=False,  # No fill, just boundary
                popup=f"{zone_name}<br>Low/medium probability zone (<{feature['properties']['threshold']})<br>Detailed boundary following actual red/orange areas"
            ).add_to(folium_map)
            
            zones_added += 1
        
        print(f"🗺️  DISPLAYED: {zones_added} detailed red/orange zone boundaries added to map")
        return True
        
    except Exception as e:
        print(f"❌ Error displaying boundary on map: {e}")
        return False