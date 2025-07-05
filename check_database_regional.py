#!/usr/bin/env python3
"""
Check the status of regional interpolations in the database
"""

from database import PolygonDatabase

def main():
    print("Database Regional Interpolation Status")
    print("=" * 50)

    try:
        db = PolygonDatabase()

        # Check sub-interpolations progress
        sub_count = db.count_sub_interpolations('canterbury', 'ground_water_level')
        print(f"Sub-interpolations completed: {sub_count}")

        if sub_count > 0:
            # Estimate total grid points (this is approximate)
            # Canterbury bounds: lat_range ≈ 2.5°, lon_range ≈ 3°
            # At 10km spacing: ~28 lat points × ~37 lon points ≈ 1036 total points
            estimated_total = 1036
            progress = (sub_count / estimated_total) * 100 if estimated_total > 0 else 0
            print(f"Estimated progress: {progress:.1f}% ({sub_count}/{estimated_total})")
            print()

            # Show recent completions
            sub_interpolations = db.list_sub_interpolations('canterbury', 'ground_water_level')
            if sub_interpolations:
                print("Recent sub-interpolations:")
                recent = sorted(sub_interpolations, key=lambda x: x['created_at'], reverse=True)[:5]
                for sub in recent:
                    print(f"  Grid {sub['grid_idx']}: {sub['feature_count']} features at ({sub['center_lat']:.3f}, {sub['center_lon']:.3f}) - {sub['created_at']}")
                print()

        # List final regional interpolations
        interpolations = db.list_regional_interpolations()

        if interpolations:
            print(f"✅ Found {len(interpolations)} final regional interpolations:")
            print()
            for interp in interpolations:
                print(f"Region: {interp['region_name']}")
                print(f"Type: {interp['interpolation_type']}")
                print(f"Features: {interp['feature_count']:,}")
                print(f"Created: {interp['created_at']}")
                print(f"Updated: {interp['updated_at']}")
                print("-" * 30)
        else:
            if sub_count > 0:
                print("📊 Sub-interpolations in progress...")
                print("Final regional interpolation will be created when all sub-regions are complete.")
            else:
                print("❌ No regional interpolations found in database.")
                print("Run 'python generate_regional_data.py' to generate Canterbury data.")

    except Exception as e:
        print(f"Error listing regional interpolations: {e}")
        print("❌ No regional interpolations found in database.")
        print("Run 'python generate_regional_data.py' to generate Canterbury data.")

if __name__ == "__main__":
    main()