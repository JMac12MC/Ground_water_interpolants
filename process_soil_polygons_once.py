
#!/usr/bin/env python3
"""
One-time script to process and store soil polygons in the database.
Run this once after creating your PostgreSQL database.
"""

import os
import sys
from database_setup import create_database_tables, load_and_merge_soil_polygons, load_soil_polygons_from_database

def main():
    """Process soil polygons once and store in database"""
    
    # Check if DATABASE_URL is set
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL environment variable not set.")
        print("Please create a PostgreSQL database in Replit first:")
        print("1. Open a new tab and type 'Database'")
        print("2. Click 'Create a database'")
        print("3. Select PostgreSQL")
        return False
    
    print("🔧 Starting one-time soil polygon processing...")
    
    # Step 1: Create database tables
    print("\n📋 Step 1: Creating database tables...")
    if not create_database_tables():
        print("❌ Failed to create database tables")
        return False
    print("✅ Database tables created successfully")
    
    # Step 2: Check if polygons already exist
    print("\n🔍 Step 2: Checking for existing polygons...")
    existing_polygons = load_soil_polygons_from_database()
    if existing_polygons is not None and len(existing_polygons) > 0:
        print(f"⚠️  Found {len(existing_polygons)} existing polygon groups in database")
        response = input("Do you want to reprocess and overwrite them? (y/N): ").strip().lower()
        if response != 'y':
            print("✅ Using existing polygons. Processing cancelled.")
            return True
    
    # Step 3: Process and merge polygons
    print("\n🔄 Step 3: Processing and merging soil polygons...")
    print("This may take a few minutes...")
    
    if not load_and_merge_soil_polygons():
        print("❌ Failed to process soil polygons")
        return False
    
    # Step 4: Verify the results
    print("\n✅ Step 4: Verifying results...")
    final_polygons = load_soil_polygons_from_database()
    if final_polygons is not None and len(final_polygons) > 0:
        total_area = final_polygons['area_sqkm'].sum() if 'area_sqkm' in final_polygons.columns else 0
        print(f"🎉 SUCCESS! Processed and stored {len(final_polygons)} merged polygon groups")
        print(f"📊 Total coverage area: {total_area:.1f} km²")
        
        if 'DRAINAGE' in final_polygons.columns:
            print("\n📋 Drainage types found:")
            for drainage_type in final_polygons['DRAINAGE'].unique():
                count = len(final_polygons[final_polygons['DRAINAGE'] == drainage_type])
                area = final_polygons[final_polygons['DRAINAGE'] == drainage_type]['area_sqkm'].sum()
                print(f"  • {drainage_type}: {area:.1f} km²")
        
        print("\n✅ Polygon processing complete! Your app will now load polygons instantly from the database.")
        return True
    else:
        print("❌ Failed to verify stored polygons")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 You can now run your app - polygons will load instantly from the database!")
        sys.exit(0)
    else:
        print("\n💔 Processing failed. Check the error messages above.")
        sys.exit(1)
