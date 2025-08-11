#!/usr/bin/env python3
"""
Test script to debug single heatmap generation
"""

import pandas as pd
from data_loader import load_sample_data
from database import PolygonDatabase
from sequential_heatmap import generate_single_heatmap
import json

def test_single_heatmap():
    print("🧪 TESTING SINGLE HEATMAP GENERATION")
    
    # Load wells data (use the FULL dataset like the app does)
    print("📊 Loading wells data...")
    from data_loader import load_nz_govt_data
    wells_data = load_nz_govt_data()
    print(f"   Wells loaded: {len(wells_data)}")
    
    # Initialize database
    print("🗄️ Initializing database...")
    polygon_db = PolygonDatabase()
    
    # Test coordinates (Canterbury center)
    test_lat, test_lon = -43.5, 172.0
    print(f"📍 Test coordinates: ({test_lat}, {test_lon})")
    
    # Generate single heatmap
    print("🎯 Generating single heatmap...")
    result = generate_single_heatmap(
        wells_data=wells_data,
        center_point=[test_lat, test_lon],
        search_radius=20,
        interpolation_method='ground_water_level_kriging',
        polygon_db=polygon_db,
        soil_polygons=None,
        new_clipping_polygon=None,
        heatmap_id_prefix="test"
    )
    
    print(f"📋 RESULT: {result}")
    
    if result and len(result) >= 2:
        success, heatmap_id = result
        print(f"   Success: {success}")
        print(f"   Heatmap ID: {heatmap_id}")
        
        if success:
            print("✅ Single heatmap generation SUCCESSFUL")
        else:
            print(f"❌ Single heatmap generation FAILED: {heatmap_id}")
    else:
        print("❌ Invalid result format")

if __name__ == "__main__":
    test_single_heatmap()