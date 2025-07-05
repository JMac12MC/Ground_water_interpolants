
#!/usr/bin/env python3
"""
Script to generate the pre-computed Canterbury regional groundwater level interpolation.
Uses resumable, database-backed processing that can be interrupted and resumed.
"""

import sys
import os
from generate_regional_gwl import save_regional_interpolation
from check_database_regional import main as check_status

def main():
    print("=" * 60)
    print("Canterbury Regional Groundwater Level Interpolation Generator")
    print("=" * 60)
    print()
    
    print("This script will:")
    print("- Load all available wells data for Canterbury region")
    print("- Create a grid of interpolation points with 10km spacing")
    print("- Generate kriging interpolations for 15km radius around each point")
    print("- Store each sub-interpolation in database (resumable)")
    print("- Merge all sub-regions into a single regional surface")
    print("- Save the result as a GeoJSON file for fast loading")
    print()
    print("✅ Key Features:")
    print("  • Resumable: Can be interrupted and restarted")
    print("  • Progress Tracking: Shows completion percentage")
    print("  • Fault Tolerant: Failed points can be retried")
    print("  • High Resolution: 15km radius with 5km overlap")
    print()
    
    # Check current status
    print("Current Status:")
    print("-" * 30)
    check_status()
    print()
    
    response = input("Continue with regional interpolation generation? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    print("\n🚀 Starting regional interpolation generation...")
    print("💡 This process can be interrupted with Ctrl+C and resumed later")
    print()
    
    try:
        output_file = save_regional_interpolation(store_in_database=True)
        if output_file:
            print(f"\n✅ Regional interpolation successfully generated!")
            print(f"💾 Stored in database and saved to: {output_file}")
            print("\n🎉 Success! You can now:")
            print("  • Use 'Use pre-computed Canterbury region' in the main app")
            print("  • Check status anytime with: python check_database_regional.py")
            print("  • Resume processing if interrupted")
        else:
            print("\n❌ Failed to generate regional interpolation.")
            print("💡 You can retry by running this script again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user.")
        print("💡 Progress has been saved. Run this script again to resume.")
        return 1
    except Exception as e:
        print(f"\n❌ Error generating regional interpolation: {e}")
        print("💡 You can retry by running this script again.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
