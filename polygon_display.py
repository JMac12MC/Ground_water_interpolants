#!/usr/bin/env python3

import pandas as pd
import re
import folium
import json
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
        # Check if comprehensive polygon file exists
        comprehensive_file = "comprehensive_clipping_polygons.geojson"
        if Path(comprehensive_file).exists():
            # Load comprehensive polygon data
            print(f"Loading comprehensive clipping polygons from {comprehensive_file}")
            gdf = gpd.read_file(comprehensive_file)
            
            if not gdf.empty:
                print(f"✅ Loaded {len(gdf)} comprehensive clipping polygons")
                print(f"   Coverage area: {gdf.geometry.area.sum():.8f} square degrees")
                print(f"   Coordinate range: {gdf.bounds.minx.min():.4f} to {gdf.bounds.maxx.max():.4f} longitude")
                print(f"                    {gdf.bounds.miny.min():.4f} to {gdf.bounds.maxy.max():.4f} latitude")
                return gdf
        
        # If comprehensive file doesn't exist, create it
        original_file = "attached_assets/myDrawing_1754734043555.json"
        if Path(original_file).exists():
            print(f"Creating comprehensive polygon data from {original_file}")
            
            # Run comprehensive import process
            result = subprocess.run(['python3', 'import_all_polygon_rings.py'], 
                                   capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                print("✅ Comprehensive import completed successfully")
                # Load the newly created comprehensive file
                if Path(comprehensive_file).exists():
                    gdf = gpd.read_file(comprehensive_file)
                    print(f"✅ Loaded {len(gdf)} comprehensive polygons after import")
                    return gdf
            else:
                print(f"❌ Comprehensive import failed: {result.stderr}")
        
        print("❌ No polygon data available - need comprehensive or processed polygon file")
        return None
        
    except Exception as e:
        print(f"❌ Error loading comprehensive clipping polygon: {e}")
        return None

def load_banks_peninsula_coords():
    """LEGACY: Load Banks Peninsula coordinates from text file (kept for backward compatibility)."""
    
    try:
        file_path = "attached_assets/banks peninsula_1753603323297.txt"
        if Path(file_path).exists():
            coordinates = parse_coordinates_file(file_path)
            if coordinates:
                print(f"✅ Loaded Banks Peninsula coordinates: {len(coordinates)} points")
                return coordinates
        
        print("❌ Banks Peninsula coordinate file not found")
        return None
        
    except Exception as e:
        print(f"❌ Error loading Banks Peninsula coordinates: {e}")
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