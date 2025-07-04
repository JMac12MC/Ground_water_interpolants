
#!/usr/bin/env python3
"""
Check the status of regional interpolation data in the database
"""

import os
from database import PolygonDatabase

def main():
    print("Database Regional Interpolation Status")
    print("=" * 50)
    
    try:
        db = PolygonDatabase()
        
        # List all regional interpolations
        interpolations = db.list_regional_interpolations()
        
        if interpolations:
            print(f"✅ Found {len(interpolations)} regional interpolation(s) in database:")
            print()
            
            for interp in interpolations:
                print(f"🗺️  Region: {interp['region_name']}")
                print(f"   Type: {interp['interpolation_type']}")
                print(f"   Features: {interp['feature_count']:,}")
                print(f"   Created: {interp['created_at']}")
                print(f"   Updated: {interp['updated_at']}")
                print()
            
            # Try to load Canterbury ground water level specifically
            canterbury_gwl = db.get_regional_interpolation('canterbury', 'ground_water_level')
            if canterbury_gwl:
                print("✅ Canterbury groundwater level interpolation is ready!")
                print("You can now use 'Use pre-computed Canterbury region' in the app.")
            else:
                print("❌ Canterbury groundwater level interpolation not found.")
                print("Run 'python generate_regional_data.py' to generate it.")
        else:
            print("❌ No regional interpolations found in database.")
            print("Run 'python generate_regional_data.py' to generate Canterbury data.")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        print("Make sure your DATABASE_URL environment variable is set correctly.")

if __name__ == "__main__":
    main()
