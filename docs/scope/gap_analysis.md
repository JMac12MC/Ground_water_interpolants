# DeepSight Gap Analysis: Current vs Target Architecture

**Date:** 2026-01-06  
**Purpose:** Identify gaps between current implementation and target architecture defined in `architecture_summary.md`

---

## Executive Summary

The current groundwater interpolation application is a **Streamlit-based prototype** that demonstrates core interpolation capabilities but requires significant architectural refactoring to meet the production requirements outlined in the DeepSight architecture.

**Key Findings:**
- ✅ **Strengths:** Working interpolation engine, database integration, multiple kriging methods
- ⚠️ **Critical Gaps:** No tile-based delivery, synchronous on-demand computation, monolithic architecture
- 🎯 **Priority:** Transition from prototype to production-ready tile-first architecture

---

## 1. Current Architecture Analysis

### 1.1 Current Components

| Component | Technology | Purpose | Status |
|-----------|-----------|---------|--------|
| **Frontend** | Streamlit + Folium | Interactive map interface | ✅ Working |
| **Interpolation** | PyKrige + scikit-learn | Spatial interpolation | ✅ Working |
| **Database** | PostgreSQL + PostGIS | Data storage | ⚠️ Mixed use |
| **Data Loader** | Pandas + GeoPandas | Data ingestion | ✅ Working |
| **Visualization** | GeoJSON + HeatMap | Map rendering | ⚠️ Performance issues |

### 1.2 Current Data Flow

```
User Click → Filter Wells → Interpolate (Kriging) → Generate GeoJSON → Render on Map
    ↓              ↓                ↓                    ↓                ↓
 <500ms         <1s            5-30s               1-5s            1-10s
```

**Total Response Time:** 7-45 seconds (far exceeds <1.5s requirement)

### 1.3 Current Interpolation Methods

The application currently implements:
- ✅ Ordinary Kriging (yield)
- ✅ Indicator Kriging (probability)
- ✅ Regression Kriging (ML + geostatistics)
- ✅ Quantile Regression Forest (uncertainty)
- ✅ Depth to groundwater interpolation
- ✅ Ground water level kriging

---

## 2. Gap Analysis by Architectural Principle

### 2.1 Tile-First Delivery ❌ **CRITICAL GAP**

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| All outputs as z/x/y tiles | GeoJSON polygons rendered directly | **No tile system** |
| Tile storage and caching | In-memory session state | **No persistent tiles** |
| Viewport-based requests | Entire heatmap loaded | **No viewport filtering** |
| Request cancellation | No cancellation logic | **Missing** |

**Impact:** Cannot scale beyond regional datasets, no CDN delivery possible

**Refactoring Required:**
1. Implement tile generation from interpolated grids
2. Create tile storage structure (filesystem or object storage)
3. Build tile serving API
4. Add client-side tile fetching with viewport awareness

### 2.2 Offline Compute, Online Read ⚠️ **MAJOR GAP**

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Pre-computed tiles | On-demand interpolation per click | **Synchronous computation** |
| Async processing | Blocking operations in Streamlit | **No async queue** |
| Read-only runtime | Live kriging calculations | **Runtime computation** |

**Impact:** 7-45s response times, cannot handle concurrent users

**Refactoring Required:**
1. Create offline batch processing pipeline
2. Implement job queue (Celery + Redis)
3. Build tile pre-computation system
4. Separate compute workers from web servers

**Current Workaround:** `automated_heatmap_generator.py` pre-computes some heatmaps, but:
- Stores as GeoJSON, not tiles
- No versioning or provenance
- No invalidation strategy
- Limited to small grids (2×3 or 10×10)

### 2.3 Metric Compute, Web Display ⚠️ **PARTIAL IMPLEMENTATION**

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Modeling in metric CRS (NZTM2000) | ✅ Uses NZTM2000 for kriging | **Good** |
| 100m × 100m true ground grid | Variable grid based on search radius | **Inconsistent resolution** |
| WebMercator tile pyramids | Direct WGS84 rendering | **No tile pyramids** |
| No browser resampling | Browser-side rendering | **Implicit resampling** |

**Impact:** Inconsistent spatial resolution, no zoom-level optimization

**Refactoring Required:**
1. Standardize to fixed 100m grid in NZTM2000
2. Generate multi-resolution tile pyramids (z10-z16)
3. Pre-compute all zoom levels offline

### 2.4 Explicit Uncertainty ⚠️ **PARTIAL IMPLEMENTATION**

| Requirement | Current State | Gap |
|-------------|--------------|-----|
| Probability as first-class output | Indicator kriging available | ✅ **Implemented** |
| Depth with uncertainty | DTW kriging, no variance layer | **Missing variance tiles** |
| Yield with confidence | Available but not displayed | **Missing confidence UI** |
| Confidence context | Not shown to users | **Missing visualization** |

**Impact:** Users cannot assess reliability of predictions

**Refactoring Required:**
1. Generate variance/uncertainty tiles alongside prediction tiles
2. Create UI components for uncertainty display
3. Add data coverage indicators

---

## 3. Component-Level Analysis

### 3.1 app.py (3659 lines) 🔴 **MONOLITHIC**

**Current Issues:**
- Single 3659-line file handling UI, logic, and data
- Session state management complexity
- Tight coupling between components
- Hard to test or maintain

**Recommended Refactoring:**
```
app.py (main entry point, <200 lines)
├── pages/
│   ├── map_view.py          # Main map interface
│   ├── data_explorer.py     # Well data exploration
│   └── admin.py             # System administration
├── components/
│   ├── map_controls.py      # Sidebar controls
│   ├── heatmap_layer.py     # Map layer management
│   └── well_markers.py      # Well visualization
└── services/
    ├── tile_service.py      # Tile fetching
    ├── query_service.py     # Point queries
    └── metadata_service.py  # Layer catalog
```

### 3.2 interpolation.py (5541 lines) 🔴 **MONOLITHIC**

**Current Issues:**
- Single file with all interpolation methods
- Mixed concerns: kriging, ML, visualization, clipping
- Difficult to extend or test individual methods

**Recommended Refactoring:**
```
interpolation/
├── __init__.py
├── base.py                  # Base interpolator interface
├── kriging/
│   ├── ordinary.py          # OK implementation
│   ├── indicator.py         # IK implementation
│   └── regression.py        # RK implementation
├── ml/
│   ├── random_forest.py     # QRF implementation
│   └── ensemble.py          # ML ensemble methods
├── grid.py                  # Grid generation utilities
├── variogram.py             # Variogram modeling
└── uncertainty.py           # Uncertainty quantification
```

### 3.3 database.py (953 lines) ⚠️ **MIXED RESPONSIBILITIES**

**Current Issues:**
- Combines polygon storage with heatmap storage
- No separation between read and write operations
- Lacks proper transaction management

**Recommended Refactoring:**
```
database/
├── __init__.py
├── models.py                # SQLAlchemy models
├── repositories/
│   ├── wells.py             # Well data access
│   ├── tiles.py             # Tile metadata
│   ├── polygons.py          # Polygon storage
│   └── models.py            # Model provenance
└── migrations/              # Alembic migrations
```

### 3.4 data_loader.py (777 lines) ✅ **REASONABLE**

**Current State:** Well-structured data ingestion
- Multiple data sources (API, CSV, sample)
- Validation and preprocessing
- Coordinate transformation

**Minor Improvements Needed:**
- Add streaming for large datasets
- Implement data versioning
- Add data quality metrics

---

## 4. Performance Analysis

### 4.1 Current Performance Bottlenecks

| Operation | Current Time | Target Time | Gap |
|-----------|-------------|-------------|-----|
| **Initial map load** | 5-15s | <1.5s | 🔴 **10x slower** |
| **Click-to-evaluate** | 7-45s | <500ms | 🔴 **90x slower** |
| **Heatmap generation** | 10-30s per tile | Pre-computed | 🔴 **Blocking** |
| **Well data fetch** | 1-3s | <100ms | 🔴 **30x slower** |

### 4.2 Scalability Issues

| Metric | Current Limit | Target Capacity | Gap |
|--------|--------------|-----------------|-----|
| **Concurrent users** | 1-2 | 100+ | 🔴 **50x gap** |
| **Dataset size** | ~5000 wells | Country-scale (50k+) | 🔴 **10x gap** |
| **Map area** | Single region | Multiple regions | 🔴 **Limited** |
| **Tile cache** | Session-only | Persistent CDN | 🔴 **No persistence** |

---

## 5. Technology Stack Gaps

### 5.1 Frontend

| Component | Current | Target | Migration Path |
|-----------|---------|--------|----------------|
| **Framework** | Streamlit | Next.js | Phased migration |
| **Mapping** | Folium/Leaflet | MapLibre GL | New implementation |
| **Scientific viz** | None | deck.gl | New addition |
| **State management** | Session state | React Context/Redux | Architecture change |

### 5.2 Backend

| Component | Current | Target | Migration Path |
|-----------|---------|--------|----------------|
| **Web framework** | Streamlit | FastAPI/Flask | New API layer |
| **Task queue** | None | Celery + Redis | New addition |
| **Tile serving** | None | TileServer/custom | New service |
| **API gateway** | None | Nginx/Traefik | Infrastructure |

### 5.3 Storage

| Component | Current | Target | Migration Path |
|-----------|---------|--------|----------------|
| **Tile storage** | None | S3/MinIO | New addition |
| **Cache** | Session state | Redis | Implementation needed |
| **Well data** | PostgreSQL | PostgreSQL + PostGIS | Optimization needed |
| **Model artifacts** | None | Versioned storage | New system |

---

## 6. Priority Refactoring Roadmap

### Phase 1: Foundation (Weeks 1-4) 🎯 **START HERE**

**Goal:** Establish tile-first architecture without disrupting current functionality

1. **Create tile generation pipeline**
   - Extract grid generation from `interpolation.py`
   - Implement z/x/y tile cutting from grids
   - Add tile storage (filesystem first, then object storage)
   - Build tile metadata tracking

2. **Separate compute from serving**
   - Create standalone batch processing script
   - Implement job queue (simple file-based first)
   - Build progress tracking
   - Add result storage

3. **Add basic tile serving**
   - Create FastAPI tile endpoint
   - Implement tile fetching by z/x/y
   - Add CORS support
   - Basic caching headers

**Deliverable:** Parallel tile system running alongside current Streamlit app

### Phase 2: Migration (Weeks 5-8)

**Goal:** Transition primary visualization to tile-based system

4. **Implement viewport-based tile loading**
   - Update Folium integration to use tile endpoint
   - Add tile fetching logic
   - Implement request cancellation
   - Add loading indicators

5. **Pre-compute core datasets**
   - Run batch processing for Canterbury region
   - Generate all zoom levels (z10-z16)
   - Create tile cache
   - Add versioning

6. **Performance optimization**
   - Add Redis caching
   - Implement CDN headers
   - Optimize database queries
   - Add connection pooling

**Deliverable:** Production-ready tile-based system for Canterbury

### Phase 3: Modernization (Weeks 9-16)

**Goal:** Build new Next.js frontend with advanced features

7. **Next.js foundation**
   - Set up Next.js project
   - Integrate MapLibre GL
   - Implement basic tile rendering
   - Add deck.gl for scientific layers

8. **Advanced features**
   - Click-to-evaluate with <500ms response
   - Uncertainty visualization
   - Point query service
   - Nearby wells lookup

9. **Production deployment**
   - Set up CDN (CloudFront/Cloudflare)
   - Implement monitoring
   - Add error tracking
   - Performance testing

**Deliverable:** Full production DeepSight system

---

## 7. Risk Assessment

### High-Risk Areas

1. **Data migration** - Converting stored GeoJSON heatmaps to tiles
   - **Mitigation:** Parallel systems during transition
   
2. **Performance regression** - Ensuring tile system is actually faster
   - **Mitigation:** Load testing before cutover
   
3. **Breaking changes** - Users accustomed to current interface
   - **Mitigation:** Phased rollout, feature flags

### Medium-Risk Areas

4. **Model provenance** - Tracking lineage of pre-computed tiles
   - **Mitigation:** Implement versioning from day 1
   
5. **Storage costs** - Tile storage at multiple zoom levels
   - **Mitigation:** Start with single region, monitor costs

---

## 8. Success Metrics

### Phase 1 Success Criteria
- ✅ Tile generation working for sample area
- ✅ <1s tile generation per tile at z14
- ✅ Tile serving endpoint operational
- ✅ Parallel system running without issues

### Phase 2 Success Criteria
- ✅ Initial map paint <1.5s
- ✅ Tile fetching <200ms p95
- ✅ Canterbury region fully tiled
- ✅ No synchronous kriging during map interaction

### Phase 3 Success Criteria
- ✅ Click-to-evaluate <500ms p95
- ✅ Support 100+ concurrent users
- ✅ Country-scale data capacity
- ✅ All tiles served from CDN

---

## 9. Conclusion

The current implementation demonstrates strong domain expertise in groundwater interpolation but requires fundamental architectural refactoring to meet production requirements. The gap is **significant but addressable** through phased refactoring.

**Key Recommendation:** Start with Phase 1 (tile generation pipeline) while maintaining current Streamlit app. This allows:
- Immediate value delivery (pre-computed tiles improve performance)
- Risk mitigation (parallel systems)
- Learning opportunity (validate tile approach before full migration)

**Estimated Timeline:** 16 weeks to full production system
**Estimated Effort:** 2-3 full-time developers

---

**Next Steps:**
1. Review and approve this gap analysis
2. Prioritize Phase 1 tasks
3. Set up development environment for tile system
4. Begin implementation of tile generation pipeline
