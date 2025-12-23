#!/usr/bin/env python3

import folium
import geopandas as gpd
from pathlib import Path
import subprocess

def parse_coordinates_file(file_path):
    """
    Parse the coordinate file and extract lat/lon points for polygon creation
    """
    
    coordinates = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Skip header line and process coordinate data
    for line in lines[1:]:  # Skip header
        line = line.strip()
        if not line:
            continue
            
        # Split by tabs and extract coordinate columns
        parts = line.split('\t')
        if len(parts) >= 4:
            try:
                # Extract longitude and latitude, removing degree symbols and directions
                lon_str = parts[2].replace('°E', '').strip()
                lat_str = parts[3].replace('°S', '').strip()
                
                # Convert to float
                longitude = float(lon_str)
                latitude = -float(lat_str)  # Negative because it's South
                
                coordinates.append([latitude, longitude])
                
            except (ValueError, IndexError):
                continue
    
    print(f"Parsed {len(coordinates)} coordinate points from file")
    return coordinates

def create_polygon_geojson(coordinates, name="Banks Peninsula"):
    """
    Create a GeoJSON polygon from coordinates
    """
    
    if not coordinates:
        return None
    
    # Ensure polygon is closed (first point = last point)
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    
    # Convert to [lon, lat] format for GeoJSON
    geojson_coords = [[coord[1], coord[0]] for coord in coordinates]
    
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": name,
                    "description": f"Polygon with {len(coordinates)-1} vertices"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [geojson_coords]
                }
            }
        ]
    }
    
    return geojson

def add_polygon_to_map(folium_map, coordinates, name="Banks Peninsula", color="red", weight=2, opacity=0.8, fill_opacity=0.3):
    """
    Add polygon to existing Folium map
    """
    
    if not coordinates:
        return folium_map
    
    # Ensure polygon is closed
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])
    
    # Add polygon to map
    folium.Polygon(
        locations=coordinates,
        color=color,
        weight=weight,
        opacity=opacity,
        fill=True,
        fillColor=color,
        fillOpacity=fill_opacity,
        popup=folium.Popup(f"<b>{name}</b><br>{len(coordinates)-1} vertices", max_width=200),
        tooltip=name
    ).add_to(folium_map)
    
    # Calculate center point for map centering
    center_lat = sum(coord[0] for coord in coordinates[:-1]) / (len(coordinates) - 1)
    center_lon = sum(coord[1] for coord in coordinates[:-1]) / (len(coordinates) - 1)
    
    print(f"Added polygon '{name}' to map")
    print(f"Polygon center: ({center_lat:.6f}, {center_lon:.6f})")
    print(f"Coordinate bounds: Lat {min(coord[0] for coord in coordinates):.6f} to {max(coord[0] for coord in coordinates):.6f}")
    print(f"                   Lon {min(coord[1] for coord in coordinates):.6f} to {max(coord[1] for coord in coordinates):.6f}")
    
    return folium_map, (center_lat, center_lon)

def load_new_clipping_polygon():
    """Load the comprehensive clipping polygon data (all 197 rings as separate polygons)."""
    
    try:
        # Check if complete polygon file exists (no merging)
        complete_file = "all_polygons_no_merge.geojson"
        if Path(complete_file).exists():
            # Load complete polygon data
            print(f"Loading ALL polygons from {complete_file}")
            gdf = gpd.read_file(complete_file)
            
            if not gdf.empty:
                print(f"✅ Loaded ALL {len(gdf)} polygons (no merging)")
                print(f"   Coverage area: {gdf.geometry.area.sum():.8f} square degrees")
                print(f"   Coordinate range: {gdf.bounds.minx.min():.4f} to {gdf.bounds.maxx.max():.4f} longitude")
                print(f"                    {gdf.bounds.miny.min():.4f} to {gdf.bounds.maxy.max():.4f} latitude")
                return gdf
        
        # If complete file doesn't exist, create it from the new polygon file
        new_file = "attached_assets/big_1754735278782.json"
        if Path(new_file).exists():
            print(f"Creating complete polygon data from {new_file}")
            
            # Run complete import process (no merging)
            result = subprocess.run(['python3', 'import_complete_polygons.py'], 
                                   capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                print("✅ Complete import finished successfully")
                # Load the newly created complete file
                if Path(complete_file).exists():
                    gdf = gpd.read_file(complete_file)
                    print(f"✅ Loaded ALL {len(gdf)} polygons after import")
                    return gdf
            else:
                print(f"❌ Complete import failed: {result.stderr}")
        
        # Fallback to old files if available
        old_file = "comprehensive_clipping_polygons.geojson"
        if Path(old_file).exists():
            print(f"Using fallback polygon data from {old_file}")
            gdf = gpd.read_file(old_file)
            return gdf
        
        print("❌ No polygon data available")
        return None
        
    except Exception as e:
        print(f"❌ Error loading comprehensive clipping polygon: {e}")
        return None

if __name__ == "__main__":
    # Test the coordinate parsing
    file_path = "attached_assets/banks peninsula_1753603323297.txt"
    
    coordinates = parse_coordinates_file(file_path)
    
    if coordinates:
        print(f"\nFirst 5 coordinates:")
        for i, coord in enumerate(coordinates[:5]):
            print(f"  {i+1}: ({coord[0]:.6f}, {coord[1]:.6f})")
        
        print(f"\nLast 5 coordinates:")
        for i, coord in enumerate(coordinates[-5:]):
            print(f"  {len(coordinates)-4+i}: ({coord[0]:.6f}, {coord[1]:.6f})")
        
        # Create GeoJSON
        geojson = create_polygon_geojson(coordinates, "Banks Peninsula")
        if geojson:
            print(f"\nGeoJSON polygon created successfully")
            print(f"Polygon area coverage: approximately {len(coordinates)} vertices")
    else:
        print("No valid coordinates found in file")