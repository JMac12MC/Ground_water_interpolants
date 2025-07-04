
#!/usr/bin/env python3
"""
Script to generate the pre-computed Canterbury regional groundwater level interpolation.
Run this once to create the regional dataset that can be loaded by the main app.
"""

import sys
import os
from generate_regional_gwl import save_regional_interpolation

def main():
    print("=" * 60)
    print("Canterbury Regional Groundwater Level Interpolation Generator")
    print("=" * 60)
    print()
    
    print("This script will:")
    print("- Load all available wells data for Canterbury region")
    print("- Create a grid of interpolation points with 10km spacing")
    print("- Generate kriging interpolations for 15km radius around each point")
    print("- Merge all sub-regions into a single regional surface")
    print("- Save the result as a GeoJSON file for fast loading")
    print()
    
    response = input("This process may take 30-60 minutes. Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return
    
    print("\nStarting regional interpolation generation...")
    
    try:
        output_file = save_regional_interpolation(store_in_database=True)
        if output_file:
            print(f"\n✅ Regional interpolation successfully generated!")
            print(f"💾 Stored in database and saved to: {output_file}")
            print("\nYou can now use the 'Use pre-computed Canterbury region' option")
            print("in the main application for instant loading of regional data.")
        else:
            print("\n❌ Failed to generate regional interpolation.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error generating regional interpolation: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
