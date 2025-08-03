#!/usr/bin/env python3
"""
Create a test scenario with overlapping heatmaps to demonstrate overlap detection
"""

import os
import sys
sys.path.append('.')

from database import PolygonDatabase
from sqlalchemy import create_engine, text
import json

def create_overlapping_test_heatmaps():
    """Create test heatmaps with intentional overlaps for demonstration"""
    
    # Simple overlapping GeoJSON polygons for testing
    overlap_test_data = [
        {
            'heatmap_name': 'test_overlap_1',
            'center_lat': -43.400,
            'center_lon': 172.200,
            'interpolation_method': 'test_overlap',
            'geojson_data': {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [172.15, -43.35],
                                [172.25, -43.35],
                                [172.25, -43.45],
                                [172.15, -43.45],
                                [172.15, -43.35]
                            ]]
                        },
                        "properties": {"value": 10}
                    }
                ]
            }
        },
        {
            'heatmap_name': 'test_overlap_2',
            'center_lat': -43.400,
            'center_lon': 172.250,
            'interpolation_method': 'test_overlap',
            'geojson_data': {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[
                                [172.20, -43.35],  # Overlaps with first polygon
                                [172.30, -43.35],
                                [172.30, -43.45],
                                [172.20, -43.45],
                                [172.20, -43.35]
                            ]]
                        },
                        "properties": {"value": 15}
                    }
                ]
            }
        }
    ]
    
    DATABASE_URL = os.getenv('DATABASE_URL')
    if DATABASE_URL:
        engine = create_engine(DATABASE_URL)
        
        for test_heatmap in overlap_test_data:
            # Insert test heatmap
            insert_query = text("""
                INSERT INTO stored_heatmaps 
                (heatmap_name, center_lat, center_lon, interpolation_method, geojson_data, created_at)
                VALUES (:name, :lat, :lon, :method, :geojson, NOW())
                ON CONFLICT (heatmap_name) DO UPDATE SET
                geojson_data = EXCLUDED.geojson_data,
                created_at = NOW()
            """)
            
            with engine.connect() as conn:
                conn.execute(insert_query, {
                    'name': test_heatmap['heatmap_name'],
                    'lat': test_heatmap['center_lat'],
                    'lon': test_heatmap['center_lon'],
                    'method': test_heatmap['interpolation_method'],
                    'geojson': json.dumps(test_heatmap['geojson_data'])
                })
                conn.commit()
        
        print("✅ Created 2 test heatmaps with intentional overlap")
        print("   • test_overlap_1: Rectangle from 172.15-172.25, -43.35 to -43.45")
        print("   • test_overlap_2: Rectangle from 172.20-172.30, -43.35 to -43.45")
        print("   • Expected overlap: ~5.5 km in the 172.20-172.25 zone")
        
        return True
    else:
        print("❌ No DATABASE_URL available")
        return False

if __name__ == "__main__":
    create_overlapping_test_heatmaps()