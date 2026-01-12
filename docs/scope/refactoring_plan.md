# DeepSight Refactoring Plan

**Version:** 1.0  
**Date:** 2026-01-06  
**Status:** Draft for Review

---

## Overview

This document provides a **detailed, actionable refactoring plan** to transform the current Streamlit prototype into a production-ready DeepSight system that meets all architectural requirements.

---

## Refactoring Strategy

### Approach: **Parallel Implementation with Gradual Migration**

Rather than a "big bang" rewrite, we'll:
1. Build new components alongside existing code
2. Gradually migrate features to new architecture
3. Maintain backward compatibility during transition
4. Validate each phase before proceeding

### Core Principles

- ✅ **No Breaking Changes** - Old functionality continues working
- ✅ **Incremental Value** - Each phase delivers measurable improvements
- ✅ **Testable Milestones** - Clear success criteria per phase
- ✅ **Reversible Decisions** - Can roll back if issues arise

---

## Phase 1: Tile Generation Infrastructure (Weeks 1-4)

### 1.1 Create Modular Interpolation System

**Goal:** Extract interpolation logic into reusable, testable modules

#### Current State Analysis
- `interpolation.py` is 5541 lines with mixed concerns
- Functions directly called from Streamlit UI
- Hard to test individual methods
- No separation of grid generation vs visualization

#### Refactoring Tasks

**Task 1.1.1: Create base interpolation interface**
```python
# File: interpolation/base.py

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class BaseInterpolator(ABC):
    """Base class for all interpolation methods"""
    
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Fit the interpolation model to data"""
        pass
    
    @abstractmethod
    def predict(self, 
                grid_x: np.ndarray, 
                grid_y: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict values at grid points
        
        Returns:
            predictions: np.ndarray of predicted values
            variance: Optional[np.ndarray] of prediction variance
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> dict:
        """Return metadata about the fitted model"""
        pass
```

**Task 1.1.2: Implement OrdinaryKriging wrapper**
```python
# File: interpolation/kriging/ordinary.py

from ..base import BaseInterpolator
from pykrige.ok import OrdinaryKriging as PyKrigeOK
import numpy as np

class OrdinaryKrigingInterpolator(BaseInterpolator):
    """Wrapper for PyKrige OrdinaryKriging with standardized interface"""
    
    def __init__(self, variogram_model='spherical', verbose=False, enable_plotting=False):
        self.variogram_model = variogram_model
        self.verbose = verbose
        self.enable_plotting = enable_plotting
        self._model = None
        
    def fit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Fit kriging model in NZTM2000 coordinates"""
        self._model = PyKrigeOK(
            x, y, z,
            variogram_model=self.variogram_model,
            verbose=self.verbose,
            enable_plotting=self.enable_plotting
        )
        
    def predict(self, grid_x: np.ndarray, grid_y: np.ndarray):
        """Predict on regular grid"""
        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # PyKrige expects 1D arrays
        predictions, variance = self._model.execute(
            'grid', grid_x, grid_y
        )
        
        return predictions, variance
    
    def get_metadata(self) -> dict:
        """Return variogram parameters and fit statistics"""
        if self._model is None:
            return {}
            
        return {
            'variogram_model': self.variogram_model,
            'variogram_parameters': self._model.variogram_model_parameters,
            'anisotropy_scaling': self._model.anisotropy_scaling,
            'anisotropy_angle': self._model.anisotropy_angle
        }
```

**Task 1.1.3: Create grid generation utilities**
```python
# File: interpolation/grid.py

import numpy as np
from typing import Tuple, Dict
from pyproj import Transformer

class GridGenerator:
    """Utilities for generating interpolation grids in NZTM2000"""
    
    def __init__(self, resolution_meters: float = 100.0):
        """
        Args:
            resolution_meters: Grid cell size in meters (default 100m)
        """
        self.resolution = resolution_meters
        self.transformer_to_nztm = Transformer.from_crs(
            "EPSG:4326", "EPSG:2193", always_xy=True
        )
        self.transformer_to_wgs84 = Transformer.from_crs(
            "EPSG:2193", "EPSG:4326", always_xy=True
        )
    
    def create_grid_from_bounds(self, 
                                bounds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create regular grid in NZTM2000 from WGS84 bounds
        
        Args:
            bounds: dict with 'north', 'south', 'east', 'west' in WGS84
            
        Returns:
            grid_x, grid_y: 2D arrays in NZTM2000 meters
        """
        # Convert bounds to NZTM2000
        west_m, south_m = self.transformer_to_nztm.transform(
            bounds['west'], bounds['south']
        )
        east_m, north_m = self.transformer_to_nztm.transform(
            bounds['east'], bounds['north']
        )
        
        # Create regular grid at specified resolution
        x_coords = np.arange(west_m, east_m, self.resolution)
        y_coords = np.arange(south_m, north_m, self.resolution)
        
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        
        return grid_x, grid_y
    
    def grid_to_wgs84(self, 
                      grid_x: np.ndarray, 
                      grid_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert grid from NZTM2000 to WGS84"""
        lons, lats = self.transformer_to_wgs84.transform(grid_x, grid_y)
        return lons, lats
```

**Testing Requirements:**
- Unit tests for each interpolator class
- Grid generation validation (check bounds, resolution)
- Coordinate transformation accuracy tests

---

### 1.2 Implement Tile Cutting System

**Goal:** Convert interpolated grids to map tiles at multiple zoom levels

#### Design Decisions

**Tile Format:** PNG with color mapping (visualization tiles, not data tiles)  
**Zoom Levels:** z10 (regional) to z16 (local detail)  
**Tile Size:** 256×256 pixels (standard)  
**CRS:** WebMercator (EPSG:3857) for web compatibility

#### Refactoring Tasks

**Task 1.2.1: Create tile cutting utilities**
```python
# File: tiles/cutter.py

import numpy as np
from PIL import Image
import io
from typing import Tuple, Optional
import mercantile  # Tile math library

class TileCutter:
    """Cut interpolated grids into map tiles"""
    
    def __init__(self, tile_size: int = 256):
        self.tile_size = tile_size
    
    def grid_to_tiles(self,
                     grid_values: np.ndarray,
                     grid_lons: np.ndarray,
                     grid_lats: np.ndarray,
                     zoom_level: int,
                     colormap_func: callable) -> dict:
        """
        Convert interpolated grid to tiles at specified zoom level
        
        Args:
            grid_values: 2D array of interpolated values
            grid_lons: 2D array of longitude coordinates
            grid_lats: 2D array of latitude coordinates
            zoom_level: Target zoom level (10-16)
            colormap_func: Function to map values to RGBA colors
            
        Returns:
            dict: {(z, x, y): tile_image_bytes}
        """
        tiles = {}
        
        # Determine tile bounds from grid
        min_lon, max_lon = np.min(grid_lons), np.max(grid_lons)
        min_lat, max_lat = np.min(grid_lats), np.max(grid_lats)
        
        # Get tile coordinates that cover this area
        ul_tile = mercantile.tile(min_lon, max_lat, zoom_level)
        lr_tile = mercantile.tile(max_lon, min_lat, zoom_level)
        
        # Iterate over tiles
        for x in range(ul_tile.x, lr_tile.x + 1):
            for y in range(ul_tile.y, lr_tile.y + 1):
                tile_img = self._create_tile(
                    x, y, zoom_level,
                    grid_values, grid_lons, grid_lats,
                    colormap_func
                )
                
                if tile_img is not None:
                    tiles[(zoom_level, x, y)] = tile_img
        
        return tiles
    
    def _create_tile(self, 
                    x: int, y: int, z: int,
                    grid_values: np.ndarray,
                    grid_lons: np.ndarray,
                    grid_lats: np.ndarray,
                    colormap_func: callable) -> Optional[bytes]:
        """Create single tile image"""
        # Get tile bounds in lat/lon
        tile_bounds = mercantile.bounds(x, y, z)
        
        # Extract grid data within tile bounds
        mask = (
            (grid_lons >= tile_bounds.west) &
            (grid_lons <= tile_bounds.east) &
            (grid_lats >= tile_bounds.south) &
            (grid_lats <= tile_bounds.north)
        )
        
        if not mask.any():
            return None  # No data in this tile
        
        # Create tile image
        tile_img = self._render_tile(
            grid_values, grid_lons, grid_lats,
            mask, tile_bounds, colormap_func
        )
        
        # Convert to PNG bytes
        img_buffer = io.BytesIO()
        tile_img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def _render_tile(self, 
                    grid_values: np.ndarray,
                    grid_lons: np.ndarray,
                    grid_lats: np.ndarray,
                    mask: np.ndarray,
                    tile_bounds: mercantile.LngLatBbox,
                    colormap_func: callable) -> Image.Image:
        """Render grid data to tile image"""
        # TODO: Implement efficient grid-to-image rendering
        # This is a placeholder for the actual implementation
        img = Image.new('RGBA', (self.tile_size, self.tile_size), (0, 0, 0, 0))
        return img
```

**Task 1.2.2: Create tile storage system**
```python
# File: tiles/storage.py

import os
from pathlib import Path
from typing import Optional, Dict
import json

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
        
        Args:
            z, x, y: Tile coordinates
            tile_data: PNG image bytes
            layer: Layer name (e.g., 'yield', 'depth', 'probability')
            version: Version identifier (timestamp or semantic version)
            
        Returns:
            tile_path: Path where tile was stored
        """
        # Create directory structure: tiles/{layer}/{version}/{z}/{x}/{y}.png
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
    
    def store_metadata(self,
                      layer: str,
                      version: str,
                      metadata: dict) -> None:
        """Store metadata about tile set"""
        meta_dir = self.base_path / layer / version
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        meta_path = meta_dir / "metadata.json"
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_metadata(self,
                     layer: str,
                     version: str) -> Optional[dict]:
        """Retrieve metadata about tile set"""
        meta_path = self.base_path / layer / version / "metadata.json"
        
        if not meta_path.exists():
            return None
        
        with open(meta_path, 'r') as f:
            return json.load(f)
```

**Testing Requirements:**
- Tile coordinate calculations
- Tile storage and retrieval
- Version management
- Metadata consistency

---

### 1.3 Create Batch Processing Pipeline

**Goal:** Enable offline tile generation for entire regions

#### Refactoring Tasks

**Task 1.3.1: Create batch processing script**
```python
# File: scripts/generate_tiles.py

"""
Batch tile generation script

Usage:
    python scripts/generate_tiles.py --region canterbury --zoom 10-14 --layer yield
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from interpolation.kriging.ordinary import OrdinaryKrigingInterpolator
from interpolation.grid import GridGenerator
from tiles.cutter import TileCutter
from tiles.storage import TileStorage
from data_loader import load_nz_govt_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_tiles_for_region(region_bounds: dict,
                              layer_type: str,
                              zoom_levels: list,
                              version: str = None):
    """Generate tiles for a region"""
    
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting tile generation for {layer_type} at zoom {zoom_levels}")
    logger.info(f"Version: {version}")
    
    # Load well data
    logger.info("Loading well data...")
    wells_data = load_nz_govt_data()
    
    # Filter to region
    wells_in_region = wells_data[
        (wells_data['latitude'] >= region_bounds['south']) &
        (wells_data['latitude'] <= region_bounds['north']) &
        (wells_data['longitude'] >= region_bounds['west']) &
        (wells_data['longitude'] <= region_bounds['east'])
    ]
    
    logger.info(f"Found {len(wells_in_region)} wells in region")
    
    # Create grid
    grid_gen = GridGenerator(resolution_meters=100.0)
    grid_x, grid_y = grid_gen.create_grid_from_bounds(region_bounds)
    
    logger.info(f"Created grid: {grid_x.shape}")
    
    # Fit interpolation model
    logger.info("Fitting interpolation model...")
    interpolator = OrdinaryKrigingInterpolator(variogram_model='spherical')
    
    # TODO: Transform well coordinates to NZTM2000
    # TODO: Call interpolator.fit()
    
    # Predict on grid
    logger.info("Interpolating...")
    predictions, variance = interpolator.predict(grid_x, grid_y)
    
    # Convert grid to WGS84 for tile cutting
    grid_lons, grid_lats = grid_gen.grid_to_wgs84(grid_x, grid_y)
    
    # Generate tiles at each zoom level
    tile_cutter = TileCutter()
    tile_storage = TileStorage()
    
    for zoom in zoom_levels:
        logger.info(f"Generating tiles at zoom {zoom}...")
        
        tiles = tile_cutter.grid_to_tiles(
            predictions, grid_lons, grid_lats,
            zoom_level=zoom,
            colormap_func=lambda v: colormap_yield(v)  # TODO: Define colormap
        )
        
        logger.info(f"Generated {len(tiles)} tiles at zoom {zoom}")
        
        # Store tiles
        for (z, x, y), tile_data in tiles.items():
            tile_storage.store_tile(z, x, y, tile_data, layer_type, version)
    
    # Store metadata
    metadata = {
        'region': region_bounds,
        'layer_type': layer_type,
        'zoom_levels': zoom_levels,
        'version': version,
        'generated_at': datetime.now().isoformat(),
        'num_wells': len(wells_in_region),
        'interpolator': interpolator.get_metadata()
    }
    
    tile_storage.store_metadata(layer_type, version, metadata)
    
    logger.info(f"Tile generation complete! Version: {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate map tiles from interpolation")
    parser.add_argument("--region", required=True, help="Region name (e.g., canterbury)")
    parser.add_argument("--layer", required=True, help="Layer type (yield, depth, probability)")
    parser.add_argument("--zoom", required=True, help="Zoom levels (e.g., 10-14)")
    parser.add_argument("--version", help="Version identifier (default: timestamp)")
    
    args = parser.parse_args()
    
    # Define region bounds (TODO: Load from config)
    REGIONS = {
        'canterbury': {
            'north': -42.5,
            'south': -44.5,
            'east': 173.5,
            'west': 171.0
        }
    }
    
    region_bounds = REGIONS.get(args.region)
    if not region_bounds:
        logger.error(f"Unknown region: {args.region}")
        sys.exit(1)
    
    # Parse zoom levels
    if '-' in args.zoom:
        min_zoom, max_zoom = map(int, args.zoom.split('-'))
        zoom_levels = list(range(min_zoom, max_zoom + 1))
    else:
        zoom_levels = [int(args.zoom)]
    
    generate_tiles_for_region(
        region_bounds,
        args.layer,
        zoom_levels,
        args.version
    )
```

**Testing Requirements:**
- End-to-end tile generation for small test area
- Verify tile coordinates are correct
- Check metadata completeness
- Validate tile quality

---

### 1.4 Deliverables for Phase 1

✅ **Code Deliverables:**
- Modular interpolation system (`interpolation/` package)
- Tile cutting utilities (`tiles/` package)
- Batch processing script (`scripts/generate_tiles.py`)
- Unit tests for all new modules

✅ **Documentation:**
- API documentation for new modules
- Batch processing guide
- Tile storage structure documentation

✅ **Validation:**
- Generate tiles for small test area (10km × 10km)
- Verify tiles render correctly in web map
- Compare tile output to current GeoJSON output

---

## Phase 2: Tile Serving & Migration (Weeks 5-8)

### 2.1 Create Tile Serving API

**Goal:** Build FastAPI service to serve pre-computed tiles

#### Refactoring Tasks

**Task 2.1.1: Create FastAPI tile server**
```python
# File: api/tile_server.py

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from tiles.storage import TileStorage
import logging

app = FastAPI(title="DeepSight Tile Server", version="1.0.0")

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_methods=["GET"],
    allow_headers=["*"],
)

tile_storage = TileStorage()
logger = logging.getLogger(__name__)


@app.get("/tiles/{layer}/{version}/{z}/{x}/{y}.png")
async def get_tile(layer: str, version: str, z: int, x: int, y: int):
    """
    Serve a map tile
    
    Args:
        layer: Layer name (yield, depth, probability)
        version: Version identifier or 'latest'
        z: Zoom level
        x: Tile X coordinate
        y: Tile Y coordinate
    """
    tile_data = tile_storage.get_tile(z, x, y, layer, version)
    
    if tile_data is None:
        raise HTTPException(status_code=404, detail="Tile not found")
    
    return Response(
        content=tile_data,
        media_type="image/png",
        headers={
            "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            "Content-Type": "image/png"
        }
    )


@app.get("/layers")
async def list_layers():
    """List available layers and versions"""
    # TODO: Implement layer catalog
    return {
        "layers": ["yield", "depth", "probability"],
        "versions": ["latest", "20260106"]
    }


@app.get("/metadata/{layer}/{version}")
async def get_layer_metadata(layer: str, version: str):
    """Get metadata about a layer"""
    metadata = tile_storage.get_metadata(layer, version)
    
    if metadata is None:
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    return metadata


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Task 2.1.2: Update Streamlit app to use tile endpoint**
```python
# File: app_with_tiles.py (new parallel version)

import streamlit as st
import folium
from streamlit_folium import st_folium

# Tile server configuration
TILE_SERVER_URL = "http://localhost:8000"

st.set_page_config(page_title="DeepSight (Tile Mode)", layout="wide")

# Create map
m = folium.Map(
    location=[-43.5, 172.5],  # Canterbury center
    zoom_start=10
)

# Add tile layer from our server
folium.TileLayer(
    tiles=f"{TILE_SERVER_URL}/tiles/yield/latest/{{z}}/{{x}}/{{y}}.png",
    attr="DeepSight",
    name="Yield",
    overlay=True,
    control=True
).add_to(m)

# Display map
st_folium(m, width=1200, height=600)

st.success("🎉 Now using tile-based rendering!")
```

---

### 2.2 Pre-compute Canterbury Tiles

**Goal:** Generate complete tile set for Canterbury region

#### Tasks

1. **Run batch generation**
   ```bash
   python scripts/generate_tiles.py --region canterbury --layer yield --zoom 10-14
   python scripts/generate_tiles.py --region canterbury --layer depth --zoom 10-14
   python scripts/generate_tiles.py --region canterbury --layer probability --zoom 10-14
   ```

2. **Validate output**
   - Check tile counts at each zoom level
   - Verify no missing tiles
   - Test tile loading in browser

3. **Document storage requirements**
   - Calculate total storage per layer
   - Estimate costs for object storage (S3)

---

### 2.3 Performance Testing

**Goal:** Validate that tile system meets performance requirements

#### Test Scenarios

1. **Initial map load (<1.5s)**
   - Measure time from page load to first tile displayed
   - Test with cold cache and warm cache
   - Target: <1.5s on 4G connection

2. **Tile fetching (<200ms p95)**
   - Measure tile request latency
   - Test concurrent requests
   - Target: 200ms at 95th percentile

3. **Pan/zoom performance**
   - Measure responsiveness during map interaction
   - Verify no blocking operations
   - Check request cancellation works

---

### 2.4 Deliverables for Phase 2

✅ **Services:**
- Tile serving API (FastAPI)
- Complete Canterbury tile set (3 layers × 5 zoom levels)

✅ **Migration:**
- Parallel Streamlit app using tiles
- Performance comparison (old vs new)
- Migration decision based on results

✅ **Documentation:**
- API documentation
- Deployment guide
- Performance test results

---

## Phase 3: Next.js Frontend (Weeks 9-16)

### 3.1 Next.js Setup

**Goal:** Create modern React-based frontend

#### Technology Stack

- **Framework:** Next.js 14 (App Router)
- **Mapping:** MapLibre GL JS
- **Scientific Viz:** deck.gl
- **State Management:** React Context + TanStack Query
- **Styling:** Tailwind CSS

#### Project Structure
```
deepsight-web/
├── app/
│   ├── page.tsx                 # Home page
│   ├── map/
│   │   └── page.tsx             # Main map view
│   └── api/
│       └── query/route.ts       # Point query API
├── components/
│   ├── Map/
│   │   ├── MapView.tsx          # MapLibre integration
│   │   ├── TileLayer.tsx        # Tile layer management
│   │   └── WellMarkers.tsx      # Well visualization
│   ├── Controls/
│   │   ├── LayerSelector.tsx    # Layer controls
│   │   └── OpacitySlider.tsx    # Transparency control
│   └── Query/
│       ├── PointQuery.tsx       # Click-to-evaluate
│       └── UncertaintyDisplay.tsx # Uncertainty viz
├── lib/
│   ├── tiles.ts                 # Tile fetching logic
│   ├── query.ts                 # Query service client
│   └── colormap.ts              # Color mapping
└── public/
    └── styles/                  # Custom styles
```

---

### 3.2 MapLibre Integration

**Task 3.2.1: Create map component**
```typescript
// File: components/Map/MapView.tsx

'use client';

import { useRef, useEffect } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

interface MapViewProps {
  initialCenter: [number, number];
  initialZoom: number;
  tileServerUrl: string;
}

export function MapView({ initialCenter, initialZoom, tileServerUrl }: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);

  useEffect(() => {
    if (!mapContainer.current) return;

    // Initialize map
    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          'osm': {
            type: 'raster',
            tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
            tileSize: 256,
          },
          'deepsight-yield': {
            type: 'raster',
            tiles: [`${tileServerUrl}/tiles/yield/latest/{z}/{x}/{y}.png`],
            tileSize: 256,
          }
        },
        layers: [
          {
            id: 'osm',
            type: 'raster',
            source: 'osm',
          },
          {
            id: 'yield',
            type: 'raster',
            source: 'deepsight-yield',
            paint: {
              'raster-opacity': 0.7
            }
          }
        ]
      },
      center: initialCenter,
      zoom: initialZoom,
    });

    // Add navigation controls
    map.current.addControl(new maplibregl.NavigationControl());

    return () => {
      map.current?.remove();
    };
  }, []);

  return (
    <div 
      ref={mapContainer} 
      className="w-full h-full"
    />
  );
}
```

---

### 3.3 Click-to-Evaluate

**Task 3.3.1: Implement point query**
```typescript
// File: components/Query/PointQuery.tsx

'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';

interface PointQueryResult {
  yield: number;
  yield_variance: number;
  depth: number;
  depth_variance: number;
  probability: number;
  nearby_wells: Array<{
    id: string;
    distance_km: number;
    yield: number;
  }>;
}

async function queryPoint(lat: number, lon: number): Promise<PointQueryResult> {
  const response = await fetch(
    `/api/query?lat=${lat}&lon=${lon}`
  );
  
  if (!response.ok) {
    throw new Error('Query failed');
  }
  
  return response.json();
}

export function PointQuery({ lat, lon }: { lat: number, lon: number }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['point-query', lat, lon],
    queryFn: () => queryPoint(lat, lon),
    staleTime: 5 * 60 * 1000, // Cache for 5 minutes
  });

  if (isLoading) {
    return <div className="p-4">Loading...</div>;
  }

  if (error) {
    return <div className="p-4 text-red-500">Error loading data</div>;
  }

  if (!data) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-4">
      <h3 className="text-lg font-bold mb-2">Location Analysis</h3>
      
      <div className="space-y-2">
        <div>
          <span className="font-semibold">Expected Yield:</span>{' '}
          {data.yield.toFixed(2)} L/s ± {Math.sqrt(data.yield_variance).toFixed(2)}
        </div>
        
        <div>
          <span className="font-semibold">Depth to Groundwater:</span>{' '}
          {data.depth.toFixed(1)} m ± {Math.sqrt(data.depth_variance).toFixed(1)}
        </div>
        
        <div>
          <span className="font-semibold">Success Probability:</span>{' '}
          {(data.probability * 100).toFixed(0)}%
        </div>
      </div>
      
      <div className="mt-4">
        <h4 className="font-semibold mb-2">Nearby Wells ({data.nearby_wells.length})</h4>
        <ul className="text-sm space-y-1">
          {data.nearby_wells.map(well => (
            <li key={well.id}>
              {well.distance_km.toFixed(1)} km: {well.yield.toFixed(2)} L/s
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
```

---

### 3.4 Deliverables for Phase 3

✅ **Frontend:**
- Next.js application with MapLibre GL
- Tile-based rendering
- Click-to-evaluate functionality (<500ms)
- Uncertainty visualization

✅ **Backend:**
- Point query API
- Nearby wells service
- Metadata API

✅ **Performance:**
- <1.5s initial map paint
- <500ms click-to-evaluate
- Support 100+ concurrent users

---

## Testing Strategy

### Unit Tests
- Interpolation algorithms
- Grid generation
- Tile cutting logic
- Coordinate transformations

### Integration Tests
- End-to-end tile generation
- API endpoints
- Data pipeline

### Performance Tests
- Load testing (100+ concurrent users)
- Latency testing (p50, p95, p99)
- Stress testing (large datasets)

### Test Coverage Targets
- Unit tests: >80% coverage
- Integration tests: All critical paths
- Performance tests: All requirements validated

---

## Risk Mitigation

### High-Risk Items

1. **Tile generation performance**
   - Risk: Takes too long for large regions
   - Mitigation: Parallel processing, optimize algorithms
   
2. **Storage costs**
   - Risk: Tile storage becomes expensive
   - Mitigation: Start with single region, implement compression

3. **Breaking changes**
   - Risk: New system incompatible with existing workflows
   - Mitigation: Parallel implementation, gradual migration

### Contingency Plans

- If tile system underperforms: Fall back to on-demand interpolation
- If storage too expensive: Implement on-demand tile generation with caching
- If migration problematic: Continue maintaining Streamlit app in parallel

---

## Success Metrics

### Phase 1
- ✅ Tile generation working
- ✅ <1s per tile generation time
- ✅ Batch processing functional

### Phase 2
- ✅ Tile API working
- ✅ <200ms tile serving latency
- ✅ Complete Canterbury coverage

### Phase 3
- ✅ Next.js app functional
- ✅ <1.5s initial map paint
- ✅ <500ms click-to-evaluate
- ✅ 100+ concurrent user support

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1 | Weeks 1-4 | Tile generation pipeline |
| Phase 2 | Weeks 5-8 | Tile serving + Canterbury coverage |
| Phase 3 | Weeks 9-16 | Next.js frontend + production deployment |
| **Total** | **16 weeks** | **Production DeepSight system** |

---

## Next Steps

1. ✅ Review and approve this refactoring plan
2. Set up development environment
3. Create Phase 1 task board
4. Begin Task 1.1.1 (base interpolation interface)
5. Daily standups to track progress

---

**Document Status:** Draft for Review  
**Next Review:** After Phase 1 kickoff  
**Maintained By:** Development Team
