
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
            # Estimate total grid points for Canterbury
            # Canterbury bounds: lat_range = 2.5°, lon_range = 3.0°
            # Grid spacing: 10km ≈ 0.09° lat, 0.12° lon (approximate)
            # Estimated grid: ~28 lat × ~25 lon = ~700 points
            lat_range = 2.5  # degrees
            lon_range = 3.0  # degrees
            grid_spacing_deg = 10 / 111.0  # 10km in degrees
            
            estimated_lat_points = int(lat_range / grid_spacing_deg) + 1
            estimated_lon_points = int(lon_range / grid_spacing_deg) + 1
            estimated_total = estimated_lat_points * estimated_lon_points
            
            progress = (sub_count / estimated_total) * 100
            print(f"Estimated progress: {progress:.1f}% ({sub_count}/{estimated_total})")
            print(f"Grid size estimate: {estimated_lat_points} × {estimated_lon_points} points")
            print()

            # Show recent completions
            sub_interpolations = db.list_sub_interpolations('canterbury', 'ground_water_level')
            if sub_interpolations:
                print("Recent sub-interpolations (last 5):")
                recent = sorted(sub_interpolations, key=lambda x: x['created_at'], reverse=True)[:5]
                for sub in recent:
                    created_time = sub['created_at'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(sub['created_at'], 'strftime') else str(sub['created_at'])
                    print(f"  Grid {sub['grid_idx']}: {sub['feature_count']} features at ({sub['center_lat']:.3f}, {sub['center_lon']:.3f}) - {created_time}")
                print()

                # Show statistics
                total_features = sum(sub['feature_count'] for sub in sub_interpolations)
                avg_features = total_features / len(sub_interpolations)
                print(f"📊 Statistics:")
                print(f"  Total features: {total_features:,}")
                print(f"  Average per sub-region: {avg_features:.1f}")
                print()

        # List final regional interpolations
        interpolations = db.list_regional_interpolations()

        if interpolations:
            print(f"✅ Found {len(interpolations)} final regional interpolations:")
            print()
            for interp in interpolations:
                created_time = interp['created_at'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(interp['created_at'], 'strftime') else str(interp['created_at'])
                updated_time = interp['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if hasattr(interp['updated_at'], 'strftime') else str(interp['updated_at'])
                
                print(f"🌍 Region: {interp['region_name']}")
                print(f"📊 Type: {interp['interpolation_type']}")
                print(f"🗺️  Features: {interp['feature_count']:,}")
                print(f"📅 Created: {created_time}")
                print(f"🔄 Updated: {updated_time}")
                print("-" * 30)
                
            print("✅ Regional interpolation is complete and ready to use!")
            print("💡 Enable 'Use pre-computed Canterbury region' in the main app.")
        else:
            if sub_count > 0:
                remaining = estimated_total - sub_count
                print("📊 Sub-interpolations in progress...")
                print(f"⏳ Approximately {remaining} grid points remaining")
                print("🔄 Final regional interpolation will be created when all sub-regions are complete.")
                print()
                print("💡 To continue processing, run: python generate_regional_data.py")
            else:
                print("❌ No regional interpolations found in database.")
                print("🚀 Run 'python generate_regional_data.py' to generate Canterbury data.")

    except Exception as e:
        print(f"❌ Error checking database status: {e}")
        print("💡 Make sure the database is properly configured.")
        print("🚀 Try running 'python generate_regional_data.py' to start fresh.")

if __name__ == "__main__":
    main()
