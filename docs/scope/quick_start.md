# DeepSight Implementation Quick-Start Guide

**For:** Developers starting Phase 1 implementation  
**Date:** 2026-01-06

---

## Quick Setup (5 Minutes)

### Prerequisites

```bash
# Verify Python version
python --version  # Should be 3.11+

# Verify dependencies
pip list | grep -E "streamlit|folium|pykrige|geopandas"
```

### Clone and Explore

```bash
cd /home/runner/work/Ground_water_interpolants/Ground_water_interpolants

# Explore current structure
tree -L 2 -I '__pycache__|*.pyc'

# Check current app
streamlit run app.py  # See current implementation
```

---

## Understanding the Current System (15 Minutes)

### Key Files to Review

1. **app.py** (3659 lines) - Main Streamlit interface
   - Lines 1-100: Imports and session state setup
   - Lines 100-800: Sidebar controls and data loading
   - Lines 1000-2000: Map rendering and heatmap display
   - Lines 2500-3000: Stored heatmap rendering
   
2. **interpolation.py** (5541 lines) - Core algorithms
   - Lines 1-100: Coordinate transformation helpers
   - Lines 500-1500: Ordinary Kriging implementation
   - Lines 1500-2500: Indicator Kriging
   - Lines 3000-4000: Regression Kriging & QRF
   
3. **database.py** (953 lines) - Data persistence
   - Lines 1-100: Database connection setup
   - Lines 200-500: Heatmap storage/retrieval
   - Lines 500-800: Polygon management

### Current Data Flow

```
User Click on Map
    ↓
Filter Wells (data_loader.py)
    ↓
Transform to NZTM2000 (interpolation.py)
    ↓
Run Kriging (PyKrige)
    ↓
Generate Grid (100m resolution)
    ↓
Create GeoJSON triangles (Delaunay)
    ↓
Store in Database (database.py)
    ↓
Render on Map (Folium)
```

**Problem:** This entire flow happens synchronously on every click (7-45 seconds)

---

## Phase 1 Implementation Plan

### Week 1: Foundation

#### Day 1-2: Create Modular Structure

**Task 1: Set up new package structure**

```bash
# Create new directories
mkdir -p interpolation/kriging interpolation/ml
mkdir -p tiles
mkdir -p scripts
mkdir -p tests/interpolation tests/tiles

# Create __init__.py files
touch interpolation/__init__.py
touch interpolation/kriging/__init__.py
touch interpolation/ml/__init__.py
touch tiles/__init__.py
```

**Task 2: Create base interpolator interface**

Create `interpolation/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class BaseInterpolator(ABC):
    """Base class for all interpolation methods"""
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Fit interpolation model to data (in NZTM2000 coordinates)"""
        pass
    
    @abstractmethod
    def predict(self, grid_x: np.ndarray, grid_y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict values at grid points
        
        Returns:
            (predictions, variance) - variance is optional
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> dict:
        """Return model parameters and fit statistics"""
        pass
```

**Task 3: Extract existing kriging code**

Create `interpolation/kriging/ordinary.py` by copying relevant code from `interpolation.py`:

```python
# Copy lines 500-1000 from interpolation.py (approximately)
# Wrap in OrdinaryKrigingInterpolator class
# Implement BaseInterpolator interface
```

**Verification:**
```python
# Test in Python REPL
from interpolation.kriging.ordinary import OrdinaryKrigingInterpolator
import numpy as np

# Create test data
x = np.array([0, 1000, 2000, 0, 1000, 2000])
y = np.array([0, 0, 0, 1000, 1000, 1000])
z = np.array([1.0, 2.0, 1.5, 1.2, 2.5, 1.8])

# Test fit
interpolator = OrdinaryKrigingInterpolator()
interpolator.fit(x, y, z)

# Test predict
grid_x = np.linspace(0, 2000, 10)
grid_y = np.linspace(0, 1000, 10)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

predictions, variance = interpolator.predict(grid_xx, grid_yy)
print(f"Predictions shape: {predictions.shape}")
print(f"Variance shape: {variance.shape}")
```

#### Day 3-4: Grid Generation

**Task 4: Create grid utilities**

Create `interpolation/grid.py`:

```python
import numpy as np
from typing import Tuple, Dict
from pyproj import Transformer

class GridGenerator:
    """Generate regular grids in NZTM2000"""
    
    def __init__(self, resolution_meters: float = 100.0):
        self.resolution = resolution_meters
        # Standard transformers (always_xy=True is critical!)
        self.to_nztm = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
        self.to_wgs84 = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    
    def create_grid_from_bounds(self, bounds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create regular grid from WGS84 bounds
        
        Args:
            bounds: {'north': lat, 'south': lat, 'east': lon, 'west': lon}
            
        Returns:
            (grid_x, grid_y) in NZTM2000 meters
        """
        # Transform bounds to NZTM2000
        west_m, south_m = self.to_nztm.transform(bounds['west'], bounds['south'])
        east_m, north_m = self.to_nztm.transform(bounds['east'], bounds['north'])
        
        # Create regular grid
        x_coords = np.arange(west_m, east_m, self.resolution)
        y_coords = np.arange(south_m, north_m, self.resolution)
        
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        
        return grid_x, grid_y
    
    def grid_to_wgs84(self, grid_x: np.ndarray, grid_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert grid from NZTM2000 to WGS84"""
        return self.to_wgs84.transform(grid_x, grid_y)
```

**Verification:**
```python
from interpolation.grid import GridGenerator

# Test with Canterbury bounds
bounds = {
    'north': -43.0,
    'south': -43.5,
    'east': 172.5,
    'west': 172.0
}

grid_gen = GridGenerator(resolution_meters=100.0)
grid_x, grid_y = grid_gen.create_grid_from_bounds(bounds)

print(f"Grid shape: {grid_x.shape}")
print(f"Grid X range: {grid_x.min():.0f} to {grid_x.max():.0f} meters")
print(f"Grid Y range: {grid_y.min():.0f} to {grid_y.max():.0f} meters")

# Convert back to WGS84
grid_lons, grid_lats = grid_gen.grid_to_wgs84(grid_x, grid_y)
print(f"Lon range: {grid_lons.min():.4f} to {grid_lons.max():.4f}")
print(f"Lat range: {grid_lats.min():.4f} to {grid_lats.max():.4f}")
```

#### Day 5: Integration Test

**Task 5: End-to-end test with real data**

Create `scripts/test_phase1.py`:

```python
"""
Phase 1 integration test

Tests the new modular interpolation system with real Canterbury well data
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_loader import load_nz_govt_data
from interpolation.kriging.ordinary import OrdinaryKrigingInterpolator
from interpolation.grid import GridGenerator
import numpy as np
import time

def main():
    print("Phase 1 Integration Test")
    print("=" * 50)
    
    # Load real well data
    print("\n1. Loading Canterbury well data...")
    wells_data = load_nz_govt_data()
    print(f"   Loaded {len(wells_data)} wells")
    
    # Filter to small test area
    test_bounds = {
        'north': -43.4,
        'south': -43.6,
        'east': 172.7,
        'west': 172.5
    }
    
    wells_in_area = wells_data[
        (wells_data['latitude'] >= test_bounds['south']) &
        (wells_data['latitude'] <= test_bounds['north']) &
        (wells_data['longitude'] >= test_bounds['west']) &
        (wells_data['longitude'] <= test_bounds['east'])
    ]
    
    print(f"   Found {len(wells_in_area)} wells in test area")
    
    # Transform well coordinates to NZTM2000
    print("\n2. Transforming coordinates to NZTM2000...")
    grid_gen = GridGenerator()
    
    well_x = []
    well_y = []
    well_z = []
    
    for _, well in wells_in_area.iterrows():
        if not np.isnan(well.get('yield_rate', np.nan)):
            x, y = grid_gen.to_nztm.transform(well['longitude'], well['latitude'])
            well_x.append(x)
            well_y.append(y)
            well_z.append(well['yield_rate'])
    
    well_x = np.array(well_x)
    well_y = np.array(well_y)
    well_z = np.array(well_z)
    
    print(f"   Using {len(well_z)} wells with yield data")
    
    # Create interpolation grid
    print("\n3. Creating 100m resolution grid...")
    grid_x, grid_y = grid_gen.create_grid_from_bounds(test_bounds)
    print(f"   Grid shape: {grid_x.shape} ({grid_x.size:,} points)")
    
    # Fit kriging model
    print("\n4. Fitting Ordinary Kriging model...")
    start_time = time.time()
    
    interpolator = OrdinaryKrigingInterpolator(variogram_model='spherical')
    interpolator.fit(well_x, well_y, well_z)
    
    fit_time = time.time() - start_time
    print(f"   Fit completed in {fit_time:.2f} seconds")
    print(f"   Model metadata: {interpolator.get_metadata()}")
    
    # Predict on grid
    print("\n5. Interpolating...")
    start_time = time.time()
    
    predictions, variance = interpolator.predict(grid_x, grid_y)
    
    predict_time = time.time() - start_time
    print(f"   Prediction completed in {predict_time:.2f} seconds")
    print(f"   Predictions range: {np.nanmin(predictions):.2f} to {np.nanmax(predictions):.2f}")
    
    # Summary
    print("\n" + "=" * 50)
    print("PHASE 1 TEST SUMMARY")
    print("=" * 50)
    print(f"✅ Modular interpolation: Working")
    print(f"✅ Grid generation: Working")
    print(f"✅ Coordinate transformation: Working")
    print(f"✅ Kriging interpolation: Working")
    print(f"⏱️  Total time: {fit_time + predict_time:.2f} seconds")
    print(f"📊 Grid points: {grid_x.size:,}")
    print(f"📍 Wells used: {len(well_z)}")
    print("\nPhase 1 foundation is working! ✨")

if __name__ == "__main__":
    main()
```

**Run test:**
```bash
python scripts/test_phase1.py
```

**Expected output:**
```
Phase 1 Integration Test
==================================================

1. Loading Canterbury well data...
   Loaded 5234 wells

2. Filtering to test area...
   Found 127 wells in test area
   Using 98 wells with yield data

3. Creating 100m resolution grid...
   Grid shape: (223, 223) (49,729 points)

4. Fitting Ordinary Kriging model...
   Fit completed in 1.23 seconds
   Model metadata: {'variogram_model': 'spherical', ...}

5. Interpolating...
   Prediction completed in 3.45 seconds
   Predictions range: 0.15 to 4.67

==================================================
PHASE 1 TEST SUMMARY
==================================================
✅ Modular interpolation: Working
✅ Grid generation: Working
✅ Coordinate transformation: Working
✅ Kriging interpolation: Working
⏱️  Total time: 4.68 seconds
📊 Grid points: 49,729
📍 Wells used: 98

Phase 1 foundation is working! ✨
```

---

### Week 2: Tile Cutting

#### Day 6-8: Implement Tile Cutter

**Task 6: Install mercantile library**

```bash
pip install mercantile
```

**Task 7: Create tile cutter**

Create `tiles/cutter.py`:

```python
import numpy as np
from PIL import Image
import io
from typing import Dict, Tuple, Optional, Callable
import mercantile

class TileCutter:
    """Cut interpolated grids into map tiles"""
    
    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size
    
    def grid_to_tiles(self,
                     grid_values: np.ndarray,
                     grid_lons: np.ndarray,
                     grid_lats: np.ndarray,
                     zoom_level: int,
                     colormap_func: Callable[[float], Tuple[int, int, int, int]]) -> Dict[Tuple[int, int, int], bytes]:
        """
        Convert interpolated grid to tiles at specified zoom level
        
        Args:
            grid_values: 2D array of interpolated values
            grid_lons: 2D array of longitude coordinates
            grid_lats: 2D array of latitude coordinates
            zoom_level: Target zoom level
            colormap_func: Function mapping value -> (R, G, B, A)
            
        Returns:
            Dict of {(z, x, y): tile_image_bytes}
        """
        tiles = {}
        
        # Determine tile coverage
        min_lon, max_lon = np.nanmin(grid_lons), np.nanmax(grid_lons)
        min_lat, max_lat = np.nanmin(grid_lats), np.nanmax(grid_lats)
        
        # Get tile coordinates that cover this area
        ul_tile = mercantile.tile(min_lon, max_lat, zoom_level)
        lr_tile = mercantile.tile(max_lon, min_lat, zoom_level)
        
        print(f"Generating tiles for zoom {zoom_level}: "
              f"x={ul_tile.x} to {lr_tile.x}, "
              f"y={ul_tile.y} to {lr_tile.y}")
        
        # Generate each tile
        tile_count = 0
        for x in range(ul_tile.x, lr_tile.x + 1):
            for y in range(ul_tile.y, lr_tile.y + 1):
                tile_img = self._create_tile(
                    x, y, zoom_level,
                    grid_values, grid_lons, grid_lats,
                    colormap_func
                )
                
                if tile_img is not None:
                    tiles[(zoom_level, x, y)] = tile_img
                    tile_count += 1
        
        print(f"Generated {tile_count} tiles at zoom {zoom_level}")
        return tiles
    
    def _create_tile(self,
                    x: int, y: int, z: int,
                    grid_values: np.ndarray,
                    grid_lons: np.ndarray,
                    grid_lats: np.ndarray,
                    colormap_func: Callable) -> Optional[bytes]:
        """Create single tile image"""
        # Get tile bounds
        tile_bounds = mercantile.bounds(x, y, z)
        
        # Simple implementation: nearest-neighbor sampling
        # TODO: Improve with proper resampling
        
        tile_img = Image.new('RGBA', (self.tile_size, self.tile_size), (0, 0, 0, 0))
        pixels = tile_img.load()
        
        # Sample grid at tile pixel locations
        has_data = False
        for px in range(self.tile_size):
            for py in range(self.tile_size):
                # Convert pixel to lat/lon
                lon = tile_bounds.west + (px / self.tile_size) * (tile_bounds.east - tile_bounds.west)
                lat = tile_bounds.north - (py / self.tile_size) * (tile_bounds.north - tile_bounds.south)
                
                # Find nearest grid point (simple approach)
                lon_diff = np.abs(grid_lons - lon)
                lat_diff = np.abs(grid_lats - lat)
                dist = lon_diff + lat_diff
                
                nearest_idx = np.unravel_index(np.nanargmin(dist), dist.shape)
                value = grid_values[nearest_idx]
                
                if not np.isnan(value):
                    pixels[px, py] = colormap_func(value)
                    has_data = True
        
        if not has_data:
            return None
        
        # Convert to PNG bytes
        img_buffer = io.BytesIO()
        tile_img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
```

**Task 8: Create simple colormap**

Add to `tiles/cutter.py`:

```python
def create_yield_colormap(vmin: float, vmax: float):
    """Create colormap function for yield values"""
    
    def colormap(value: float) -> Tuple[int, int, int, int]:
        """Map yield value to RGBA color"""
        if np.isnan(value):
            return (0, 0, 0, 0)  # Transparent
        
        # Normalize to 0-1
        normalized = (value - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
        
        # Simple blue-to-red gradient
        r = int(255 * normalized)
        g = int(128 * (1 - abs(2 * normalized - 1)))
        b = int(255 * (1 - normalized))
        a = 180  # 70% opacity
        
        return (r, g, b, a)
    
    return colormap
```

**Verification:**
```python
from tiles.cutter import TileCutter, create_yield_colormap
import numpy as np

# Create test grid (10km x 10km)
grid_gen = GridGenerator(resolution_meters=100.0)
test_bounds = {'north': -43.4, 'south': -43.5, 'east': 172.6, 'west': 172.5}

grid_x, grid_y = grid_gen.create_grid_from_bounds(test_bounds)
grid_lons, grid_lats = grid_gen.grid_to_wgs84(grid_x, grid_y)

# Create fake interpolation data
grid_values = np.random.rand(*grid_x.shape) * 5.0  # Random yields 0-5 L/s

# Generate tiles
tile_cutter = TileCutter()
colormap = create_yield_colormap(0.0, 5.0)

tiles = tile_cutter.grid_to_tiles(
    grid_values, grid_lons, grid_lats,
    zoom_level=14,
    colormap_func=colormap
)

print(f"Generated {len(tiles)} tiles")
print(f"Tile coordinates: {list(tiles.keys())[:5]}")  # Show first 5
```

#### Day 9-10: Tile Storage

**Task 9: Create tile storage system**

Create `tiles/storage.py`:

```python
import os
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

class TileStorage:
    """Manage tile storage with versioning"""
    
    def __init__(self, base_path: str = "./tiles"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store_tile(self,
                   z: int, x: int, y: int,
                   tile_data: bytes,
                   layer: str = "default",
                   version: str = "latest") -> str:
        """
        Store tile with versioning
        
        Directory structure: tiles/{layer}/{version}/{z}/{x}/{y}.png
        """
        tile_dir = self.base_path / layer / version / str(z) / str(x)
        tile_dir.mkdir(parents=True, exist_ok=True)
        
        tile_path = tile_dir / f"{y}.png"
        
        with open(tile_path, 'wb') as f:
            f.write(tile_data)
        
        return str(tile_path)
    
    def get_tile(self,
                 z: int, x: int, y: int,
                 layer: str = "default",
                 version: str = "latest") -> Optional[bytes]:
        """Retrieve tile if it exists"""
        tile_path = self.base_path / layer / version / str(z) / str(x) / f"{y}.png"
        
        if not tile_path.exists():
            return None
        
        with open(tile_path, 'rb') as f:
            return f.read()
    
    def tile_exists(self, z: int, x: int, y: int, layer: str = "default", version: str = "latest") -> bool:
        """Check if tile exists"""
        tile_path = self.base_path / layer / version / str(z) / str(x) / f"{y}.png"
        return tile_path.exists()
    
    def store_metadata(self, layer: str, version: str, metadata: dict) -> None:
        """Store metadata about tile set"""
        meta_dir = self.base_path / layer / version
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        meta_path = meta_dir / "metadata.json"
        
        # Add timestamp
        metadata['stored_at'] = datetime.now().isoformat()
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_metadata(self, layer: str, version: str) -> Optional[dict]:
        """Retrieve metadata about tile set"""
        meta_path = self.base_path / layer / version / "metadata.json"
        
        if not meta_path.exists():
            return None
        
        with open(meta_path, 'r') as f:
            return json.load(f)
    
    def list_versions(self, layer: str) -> list:
        """List available versions for a layer"""
        layer_path = self.base_path / layer
        
        if not layer_path.exists():
            return []
        
        return [d.name for d in layer_path.iterdir() if d.is_dir()]
```

**Verification:**
```python
from tiles.storage import TileStorage

storage = TileStorage(base_path="./test_tiles")

# Store a test tile
test_tile_data = b"PNG_IMAGE_DATA_HERE"
path = storage.store_tile(14, 12345, 23456, test_tile_data, layer="yield", version="test_v1")
print(f"Stored at: {path}")

# Retrieve it
retrieved = storage.get_tile(14, 12345, 23456, layer="yield", version="test_v1")
print(f"Retrieved: {len(retrieved)} bytes")

# Store metadata
metadata = {
    'region': 'canterbury',
    'zoom_level': 14,
    'num_tiles': 1,
    'interpolation_method': 'ordinary_kriging'
}
storage.store_metadata("yield", "test_v1", metadata)

# Retrieve metadata
meta = storage.get_metadata("yield", "test_v1")
print(f"Metadata: {meta}")
```

---

### Week 3-4: Batch Processing & API

(Continue with Task 10-15...)

---

## Common Issues & Solutions

### Issue 1: Coordinate Transformation Errors

**Symptom:** Grid appears in wrong location or inverted

**Solution:** Always use `always_xy=True` in Transformer
```python
# CORRECT
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
x, y = transformer.transform(lon, lat)  # lon first!

# WRONG
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2193")  # Missing always_xy
x, y = transformer.transform(lat, lon)  # Wrong order!
```

### Issue 2: Kriging Takes Too Long

**Symptom:** Interpolation >30 seconds for small grid

**Solutions:**
1. Reduce grid resolution (100m → 200m)
2. Limit number of wells (spatial filtering)
3. Use `enable_plotting=False` in PyKrige
4. Consider using universal kriging with trend

### Issue 3: Tiles Not Appearing

**Checklist:**
- [ ] Tile coordinates correct? (use mercantile.bounds to debug)
- [ ] Colormap returning valid RGBA? (check alpha channel)
- [ ] Tile path correct? (check storage.tile_exists)
- [ ] CORS enabled on tile server?

---

## Testing Checklist

Before moving to next phase:

- [ ] Modular interpolation works with real data
- [ ] Grid generation produces correct bounds
- [ ] Coordinate transformation validated
- [ ] Tile cutting generates valid PNGs
- [ ] Tile storage/retrieval working
- [ ] End-to-end test passes
- [ ] Code documented
- [ ] Unit tests written

---

## Next Steps

After Phase 1 completion:

1. Week 5: Create FastAPI tile server
2. Week 6: Update Streamlit to use tiles
3. Week 7: Pre-compute Canterbury tiles
4. Week 8: Performance testing

---

## Resources

- **Architecture Doc:** `docs/scope/architecture_summary.md`
- **Gap Analysis:** `docs/scope/gap_analysis.md`
- **Full Plan:** `docs/scope/refactoring_plan.md`
- **PyKrige Docs:** https://geostat-framework.readthedocs.io/projects/pykrige/
- **Mercantile Docs:** https://github.com/mapbox/mercantile

---

## Getting Help

**Stuck on something?**

1. Check existing code in `interpolation.py` for reference
2. Review PyKrige examples
3. Test with small data first (10 wells, 1km x 1km grid)
4. Add print statements to debug data flow

**Good luck!** 🚀
