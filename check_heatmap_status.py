
#!/usr/bin/env python3
"""
Check if pre-computed heatmaps are available in the database.
"""

from database import PolygonDatabase
import pandas as pd

def check_heatmap_status():
    print("=== Heatmap Status Check ===")
    
    try:
        db = PolygonDatabase()
        
        if not db.pg_engine:
            print("❌ PostgreSQL connection not available")
            print("   Heatmap preprocessing requires PostgreSQL")
            return False
        
        # Check yield heatmap
        try:
            yield_count = pd.read_sql("SELECT COUNT(*) as count FROM yield_heatmap", db.pg_engine)
            yield_points = yield_count.iloc[0]['count']
            print(f"✅ Yield heatmap: {yield_points:,} points")
        except:
            print("❌ Yield heatmap: Not available")
            yield_points = 0
        
        # Check depth heatmap  
        try:
            depth_count = pd.read_sql("SELECT COUNT(*) as count FROM depth_heatmap", db.pg_engine)
            depth_points = depth_count.iloc[0]['count']
            print(f"✅ Depth heatmap: {depth_points:,} points")
        except:
            print("❌ Depth heatmap: Not available")
            depth_points = 0
        
        if yield_points > 0 and depth_points > 0:
            print("\n🚀 Heatmaps are ready! Your app will run in high-performance mode.")
            return True
        else:
            print("\n⚠️  Heatmaps not ready. Run 'python run_preprocessing.py' to generate them.")
            return False
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        return False

if __name__ == "__main__":
    check_heatmap_status()
