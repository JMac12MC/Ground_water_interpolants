# DeepSight Architecture Summary

## Purpose of this Document
This document provides a concise, implementation-oriented summary of the DeepSight system architecture. It exists to guide refactoring, PR planning, and AI-assisted development.

It is derived from the canonical DeepSight scope document and does not introduce new requirements. Where conflicts exist, the scope document takes precedence.

This summary focuses on system boundaries, data flow, performance constraints, and non-negotiable architectural decisions.

---

## System Overview
DeepSight is a decision-support platform for groundwater drilling that converts heterogeneous well and environmental data into probabilistic, spatially explicit groundwater intelligence.

The system produces:
- Probability of encountering usable groundwater
- Depth-to-groundwater (DTW) surfaces with uncertainty
- Expected yield estimates
- Confidence and data-coverage indicators

The platform is designed to support country-scale datasets while delivering instant, interactive map experiences.

---

## Core Architectural Principles

### Tile-First Delivery
- Raster data is never served as country-scale or region-scale grids.
- All spatial outputs are delivered as map tiles (z/x/y).
- Tiles are the unit of storage, caching, delivery, and invalidation.

### Offline Compute, Online Read
- All heavy modelling and interpolation occurs offline or asynchronously.
- Runtime services never generate rasters.
- The online system only reads precomputed tiles and metadata.

### Metric Compute, Web Display
- All modelling is performed in an appropriate metric CRS on a true ground grid (e.g., 100m × 100m).
- Display uses WebMercator tile pyramids derived from the compute grid.
- No ad hoc resampling is performed in the browser.

### Explicit Uncertainty
- Uncertainty is a first-class output.
- Probability, depth, yield, and confidence are distinct layers.
- No single value is presented without accompanying confidence context.

---

## High-Level Data Flow

1. Public and private well datasets are ingested and normalised.
2. Wells are assigned to aquifers and modelling zones.
3. Feature engineering produces physically meaningful covariates.
4. Models are trained per zone:
   - Probability: indicator kriging + ML ensemble
   - DTW: regression kriging with residual kriging
5. Monte Carlo runs propagate uncertainty.
6. Outputs are written directly as versioned tile sets.
7. Tiles are served via CDN to the frontend.
8. Point queries return values, uncertainty, and nearby context.

---

## Runtime System Components

### Frontend
- Next.js application
- MapLibre GL for base mapping
- deck.gl (optional/advanced) for GPU-accelerated scientific layers

Responsibilities:
- Tile fetching and rendering
- Viewport-only requests with cancellation
- Progressive refinement
- Click-to-evaluate interactions

### Backend Services
- Tile service (static tile delivery)
- Query service (point evaluation, nearby wells)
- Metadata/catalog service (releases, layers, provenance)

Responsibilities:
- Low-latency reads only
- No raster generation at runtime

### Storage
- Object storage for raster and vector tiles
- PostgreSQL/PostGIS for wells, metadata, and geometry
- Redis for hot query caching

---

## Performance Constraints (Hard Requirements)

- Initial map paint: < 1.5 seconds
- Click-to-evaluate response: < 500 ms
- Browser requests only viewport tiles (+ small prefetch margin)
- Stale tile requests are cancelled during pan/zoom
- Tile cache is bounded with LRU eviction
- Country-scale rasters are never transmitted

These constraints are enforced through architecture, not optimisation.

---

## Versioning and Provenance

- All outputs are versioned and immutable.
- Tiles are stored under versioned namespaces.
- Model versions are traceable to:
  - input datasets
  - model configuration
  - training metadata
- Old releases remain accessible for audit and rollback.

---

## Explicit Non-Responsibilities

The system does not:
- Generate rasters on demand
- Provide guarantees of water availability
- Replace hydrogeological judgement
- Store full-resolution rasters in the database
- Perform real-time physical groundwater simulation

---

## Guidance for Refactoring and PRs

- Refactors must respect tile-first delivery.
- Performance guarantees must not be weakened.
- Scientific outputs must remain traceable and auditable.
- Large changes should be split into PR-sized steps.
- Tests and performance checks should be added before optimisation.

This document should be read before undertaking architectural or performance-related changes.
