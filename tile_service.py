"""
Server-side XYZ tile service for rendering triangular heatmap polygons to PNG tiles
Preserves full triangular detail without WebSocket payload limitations
"""

import io
import math
import threading
import time
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import json

from PIL import Image, ImageDraw, ImageFont
import pyproj
import mercantile
from rtree import index
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
import uvicorn


class TileService:
    """Server-side tile service for rendering heatmap triangular polygons to PNG tiles"""
    
    def __init__(self, port: int = 8001):
        self.port = port
        self.app = FastAPI()
        self.heatmap_data: Dict[str, Any] = {}  # Store heatmap GeoJSON data
        self.spatial_indices: Dict[str, index.Index] = {}  # R-tree spatial indices per heatmap
        self.color_function = None  # Global color mapping function
        self.server_thread = None
        
        # Web Mercator projection - use modern pyproj syntax
        self.web_mercator = pyproj.CRS.from_epsg(3857)
        self.wgs84 = pyproj.CRS.from_epsg(4326)
        self.transformer = pyproj.Transformer.from_crs(self.wgs84, self.web_mercator, always_xy=True)
        
        # Setup FastAPI routes
        self.setup_routes()
        
    def setup_routes(self):
        """Setup FastAPI routes for tile serving"""
        
        @self.app.get("/tiles/{layer_id}/{z}/{x}/{y}.png")
        async def get_tile(layer_id: str, z: int, x: int, y: int):
            """Generate PNG tile for specific layer at z/x/y coordinates"""
            try:
                # Validate parameters
                if z < 0 or z > 18:  # Reasonable zoom range
                    raise HTTPException(status_code=400, detail="Invalid zoom level")
                
                # Generate tile
                tile_png = self.generate_tile(layer_id, z, x, y)
                
                if tile_png is None:
                    # Return transparent tile if no data
                    transparent_tile = self.create_transparent_tile()
                    return Response(content=transparent_tile, media_type="image/png")
                
                return Response(content=tile_png, media_type="image/png")
                
            except Exception as e:
                print(f"Error generating tile {layer_id}/{z}/{x}/{y}: {e}")
                # Return transparent tile on error
                transparent_tile = self.create_transparent_tile()
                return Response(content=transparent_tile, media_type="image/png")
    
    def add_heatmap(self, layer_id: str, geojson_data: Dict, color_func=None):
        """Add heatmap data and build spatial index"""
        print(f"🔧 TILE SERVICE: Adding heatmap {layer_id} with {len(geojson_data.get('features', []))} features")
        
        # Store heatmap data
        self.heatmap_data[layer_id] = geojson_data
        if color_func:
            self.color_function = color_func
        
        # Build spatial index
        self.build_spatial_index(layer_id, geojson_data)
        
    def build_spatial_index(self, layer_id: str, geojson_data: Dict):
        """Build R-tree spatial index for fast tile queries"""
        idx = index.Index()
        
        for i, feature in enumerate(geojson_data.get('features', [])):
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                
                # Calculate feature bounds
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]
                
                bounds = (min(lons), min(lats), max(lons), max(lats))  # (minx, miny, maxx, maxy)
                idx.insert(i, bounds)
        
        self.spatial_indices[layer_id] = idx
        print(f"🌐 SPATIAL INDEX: Built index for {layer_id} with {len(geojson_data.get('features', []))} features")
    
    @lru_cache(maxsize=1000)
    def generate_tile(self, layer_id: str, z: int, x: int, y: int) -> Optional[bytes]:
        """Generate PNG tile for specified tile coordinates"""
        
        if layer_id not in self.heatmap_data:
            return None
            
        # Get tile bounds in WGS84
        tile_bounds = mercantile.bounds(x, y, z)
        
        # Query spatial index for intersecting features
        geojson_data = self.heatmap_data[layer_id]
        spatial_idx = self.spatial_indices.get(layer_id)
        
        if spatial_idx is None:
            return None
        
        # Find features that intersect with tile bounds
        intersecting_features = list(spatial_idx.intersection(
            (tile_bounds.west, tile_bounds.south, tile_bounds.east, tile_bounds.north)
        ))
        
        if not intersecting_features:
            return None
        
        # Create tile image (256x256 standard tile size)
        tile_size = 256
        img = Image.new('RGBA', (tile_size, tile_size), (0, 0, 0, 0))  # Transparent background
        draw = ImageDraw.Draw(img)
        
        # Transform coordinates and draw polygons
        for feature_idx in intersecting_features:
            feature = geojson_data['features'][feature_idx]
            
            if feature['geometry']['type'] == 'Polygon':
                coords = feature['geometry']['coordinates'][0]
                
                # Transform coordinates to tile pixel space
                pixel_coords = []
                for lon, lat in coords:
                    # Convert to Web Mercator using modern pyproj
                    x_merc, y_merc = self.transformer.transform(lon, lat)
                    
                    # Convert to tile pixel coordinates
                    pixel_x = int((x_merc - mercantile.xy_bounds(x, y, z).left) / 
                                (mercantile.xy_bounds(x, y, z).right - mercantile.xy_bounds(x, y, z).left) * tile_size)
                    pixel_y = int((mercantile.xy_bounds(x, y, z).top - y_merc) / 
                                (mercantile.xy_bounds(x, y, z).top - mercantile.xy_bounds(x, y, z).bottom) * tile_size)
                    
                    pixel_coords.append((pixel_x, pixel_y))
                
                # Get feature value and color
                feature_value = feature['properties'].get('value', 0)
                
                if self.color_function:
                    color_rgb = self.color_function(feature_value)
                    if isinstance(color_rgb, str):
                        # Convert hex to RGB
                        color_rgb = color_rgb.lstrip('#')
                        color_rgb = tuple(int(color_rgb[i:i+2], 16) for i in (0, 2, 4))
                    
                    # Add alpha for transparency (70% opacity)
                    color_rgba = color_rgb + (180,)
                else:
                    # Default color
                    color_rgba = (100, 150, 200, 180)
                
                # Draw polygon
                if len(pixel_coords) >= 3:  # Need at least 3 points for polygon
                    draw.polygon(pixel_coords, fill=color_rgba, outline=color_rgba)
        
        # Convert to PNG bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def create_transparent_tile(self) -> bytes:
        """Create a transparent 256x256 PNG tile"""
        img = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
    
    def start_server(self):
        """Start the tile server in a separate thread"""
        if self.server_thread is None or not self.server_thread.is_alive():
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            print(f"🚀 TILE SERVER: Starting on port {self.port}")
            
            # Wait a moment for server to start
            time.sleep(2)
    
    def _run_server(self):
        """Internal method to run the uvicorn server"""
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning"  # Reduce log noise
        )
    
    def clear_cache(self):
        """Clear the tile cache"""
        self.generate_tile.cache_clear()
        print("🧹 TILE CACHE: Cleared tile cache")


# Global tile service instance
tile_service = None

def get_tile_service() -> TileService:
    """Get or create the global tile service instance"""
    global tile_service
    if tile_service is None:
        tile_service = TileService()
        tile_service.start_server()
    return tile_service