
#!/usr/bin/env python3
"""
Utility script to pre-compute heatmaps for the groundwater application.
Run this script to generate yield and depth heatmaps from all available well data.
"""

import sys
import time
from precompute_heatmaps import HeatmapPreprocessor

def main():
    print("=== Groundwater Heatmap Preprocessing ===")
    print("This will process all well data and generate pre-computed heatmaps.")
    print("This may take several minutes to complete.")
    
    # Confirm before proceeding
    response = input("Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Preprocessing cancelled.")
        return
    
    start_time = time.time()
    
    try:
        preprocessor = HeatmapPreprocessor()
        preprocessor.run_full_preprocessing()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"⏱️  Total time: {duration:.1f} seconds")
        print("\nYour app will now load heatmaps instantly!")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        print("Please check your database connection and well data availability.")
        sys.exit(1)

if __name__ == "__main__":
    main()
