
#!/usr/bin/env python3
"""
Utility to check the status of regional interpolation data
"""

import os
import json
from generate_regional_gwl import load_regional_interpolation

def main():
    print("Canterbury Regional Groundwater Level Data Status")
    print("=" * 50)
    
    # Check if file exists
    data_file = os.path.join("sample_data", "canterbury_gwl_interpolation.json")
    
    if os.path.exists(data_file):
        file_size = os.path.getsize(data_file) / (1024 * 1024)  # MB
        print(f"✅ Regional data file exists: {data_file}")
        print(f"📁 File size: {file_size:.1f} MB")
        
        # Try to load and validate
        try:
            geojson = load_regional_interpolation()
            if geojson:
                num_features = len(geojson.get('features', []))
                print(f"🗺️  Contains {num_features:,} interpolated polygons")
                
                # Calculate some basic stats
                if num_features > 0:
                    values = []
                    for feature in geojson['features']:
                        values.append(feature['properties']['yield'])
                    
                    print(f"📊 Value range: {min(values):.2f} to {max(values):.2f}")
                    print(f"📊 Average value: {sum(values)/len(values):.2f}")
                
                print("\n✅ Regional data is ready to use!")
                print("Enable 'Use pre-computed Canterbury region' in the app.")
            else:
                print("❌ Failed to load regional data (file may be corrupted)")
        except Exception as e:
            print(f"❌ Error loading regional data: {e}")
    else:
        print("❌ Regional data file not found")
        print(f"Expected location: {data_file}")
        print("\nTo generate regional data, run:")
        print("python generate_regional_data.py")

if __name__ == "__main__":
    main()
