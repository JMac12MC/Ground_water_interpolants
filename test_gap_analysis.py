#!/usr/bin/env python3
"""
Test the gap analysis functionality directly
"""

import os
import sys
sys.path.append('.')

from database import PolygonDatabase
from measure_heatmap_gaps import analyze_displayed_heatmap_gaps

def test_gap_analysis():
    """Test gap analysis with debug output"""
    try:
        # Create database connection
        db = PolygonDatabase()
        
        print("🔍 Testing gap analysis...")
        result = analyze_displayed_heatmap_gaps(db)
        
        if result:
            print(f"✅ Analysis complete: {len(result)} pairs analyzed")
            for gap in result:
                print(f"  {gap['heatmap1']} ↔ {gap['heatmap2']}: {gap['edge_gap']:.3f} km gap")
        else:
            print("❌ No results returned")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gap_analysis()