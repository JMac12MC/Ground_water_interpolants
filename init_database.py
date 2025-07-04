
#!/usr/bin/env python3
"""
Initialize database tables for regional interpolations
"""

import os
from database import PolygonDatabase

def main():
    print("Initializing database tables...")
    
    try:
        db = PolygonDatabase()
        
        # Create the regional_interpolations table explicitly
        with db.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS regional_interpolations (
                    id SERIAL PRIMARY KEY,
                    region_name VARCHAR(100) NOT NULL,
                    interpolation_type VARCHAR(50) NOT NULL,
                    geojson_data JSON NOT NULL,
                    feature_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(region_name, interpolation_type)
                )
            """))
            conn.commit()
            print("✅ regional_interpolations table created successfully!")
            
        # Test the table
        interpolations = db.list_regional_interpolations()
        print(f"✅ Database initialized. Found {len(interpolations)} existing interpolations.")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    from sqlalchemy import text
    exit(main())
