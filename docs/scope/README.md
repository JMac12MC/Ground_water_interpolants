# DeepSight Development Plan - Executive Summary

**Project:** Groundwater Interpolation Application (DeepSight)  
**Date:** 2026-01-06  
**Status:** Planning Complete - Ready for Implementation

---

## Overview

This document provides a comprehensive plan to transform the current Streamlit-based groundwater interpolation prototype into a production-ready **DeepSight** system that meets all architectural requirements defined in `architecture_summary.md`.

---

## Document Structure

### 1. Architecture Summary (`architecture_summary.md`)
- **Purpose:** Defines the target production architecture
- **Key Requirements:**
  - Tile-first delivery (z/x/y map tiles)
  - Offline compute, online read (pre-computed tiles)
  - Metric compute (NZTM2000 100m grid), web display (WebMercator)
  - Explicit uncertainty as first-class output
  - Performance: <1.5s initial load, <500ms click-to-evaluate

### 2. Gap Analysis (`gap_analysis.md`)
- **Purpose:** Identifies differences between current and target state
- **Key Findings:**
  - Current: Streamlit prototype with on-demand interpolation (7-45s)
  - Gaps: No tile system, synchronous compute, monolithic code
  - Risk: Cannot scale beyond regional datasets
- **Recommendation:** Phased refactoring starting with tile generation

### 3. Refactoring Plan (`refactoring_plan.md`)
- **Purpose:** Detailed implementation guide for all phases
- **Timeline:** 16 weeks, 3 phases
- **Key Deliverables:**
  - Phase 1 (Weeks 1-4): Tile generation infrastructure
  - Phase 2 (Weeks 5-8): Tile serving + Canterbury coverage
  - Phase 3 (Weeks 9-16): Next.js frontend + production deployment

### 4. Quick-Start Guide (`quick_start.md`)
- **Purpose:** Practical day-by-day implementation guide for developers
- **Contents:**
  - 5-minute setup instructions
  - Week 1 detailed tasks with code examples
  - Verification steps and testing checklist
  - Common issues and solutions

---

## Current System Analysis

### What's Working ✅

1. **Interpolation Engine**
   - Multiple kriging methods (OK, IK, RK)
   - Machine learning integration (QRF)
   - Proper coordinate transformations (NZTM2000)
   
2. **Data Management**
   - PostgreSQL + PostGIS storage
   - Well data ingestion from Environment Canterbury
   - Polygon clipping for exclusion zones

3. **Visualization**
   - Interactive Folium maps
   - GeoJSON rendering
   - Multiple layer types

### Critical Issues 🔴

1. **Performance**
   - 7-45 second response times (target: <1.5s)
   - Synchronous computation blocks UI
   - Cannot handle concurrent users

2. **Architecture**
   - Monolithic 3659-line app.py
   - No tile system (sends entire rasters)
   - No separation of compute/serving
   - No caching or CDN delivery

3. **Scalability**
   - Limited to ~5000 wells (target: 50k+)
   - Single-region focused
   - No multi-user support

---

## Implementation Strategy

### Approach: Parallel Implementation

Build new components alongside existing code rather than "big bang" rewrite:

```
Current Streamlit App          New Tile System
        ↓                            ↓
   (Keep running)           (Build in parallel)
        ↓                            ↓
   Gradually migrate features  →  Validate
        ↓                            ↓
   Phase out old system      Full production
```

**Benefits:**
- No disruption to current usage
- Can validate new system before migration
- Reversible if issues arise
- Incremental value delivery

---

## Phase Breakdown

### Phase 1: Tile Generation (Weeks 1-4) 🎯 **START HERE**

**Goal:** Create infrastructure to generate map tiles from interpolated grids

**Key Tasks:**
1. Extract interpolation into modular classes (`interpolation/` package)
2. Create grid generation utilities (100m NZTM2000 grids)
3. Implement tile cutting (z/x/y from grids)
4. Build tile storage system (versioned, metadata tracking)
5. Create batch processing script

**Deliverables:**
- Working tile generation pipeline
- Unit tests for all modules
- Documentation
- End-to-end test with small area

**Success Criteria:**
- ✅ Generate tiles for 10km × 10km test area
- ✅ <1s per tile generation time
- ✅ Tiles render correctly in web map
- ✅ Parallel to existing Streamlit app (no disruption)

### Phase 2: Tile Serving (Weeks 5-8)

**Goal:** Serve pre-computed tiles and migrate primary visualization

**Key Tasks:**
1. Build FastAPI tile server
2. Pre-compute Canterbury region (all layers, zoom 10-14)
3. Update Streamlit to use tile endpoint
4. Add Redis caching
5. Performance testing

**Deliverables:**
- Tile serving API
- Complete Canterbury tile set
- Parallel Streamlit app using tiles
- Performance comparison report

**Success Criteria:**
- ✅ Initial map load <1.5s
- ✅ Tile fetching <200ms p95
- ✅ Canterbury fully covered
- ✅ No synchronous kriging during map interaction

### Phase 3: Next.js Frontend (Weeks 9-16)

**Goal:** Build modern React-based frontend with advanced features

**Key Tasks:**
1. Set up Next.js + MapLibre GL
2. Implement tile-based rendering
3. Add click-to-evaluate (<500ms)
4. Build uncertainty visualization
5. Create point query service
6. Production deployment with CDN

**Deliverables:**
- Production Next.js application
- MapLibre GL tile rendering
- Point query API
- Uncertainty visualization
- Deployed to CDN

**Success Criteria:**
- ✅ <1.5s initial map paint
- ✅ <500ms click-to-evaluate response
- ✅ Support 100+ concurrent users
- ✅ Country-scale data capacity

---

## Technology Stack

### Current → Target Migration

| Component | Current | Target | Timeline |
|-----------|---------|--------|----------|
| **Frontend** | Streamlit | Next.js + MapLibre GL | Phase 3 |
| **Mapping** | Folium/Leaflet | MapLibre GL + deck.gl | Phase 3 |
| **Backend** | Streamlit server | FastAPI | Phase 2 |
| **Interpolation** | Monolithic module | Modular classes | Phase 1 |
| **Tile System** | None | z/x/y tiles | Phase 1-2 |
| **Caching** | Session state | Redis + CDN | Phase 2-3 |
| **Queue** | None | Celery + Redis | Phase 3 |
| **Storage** | PostgreSQL | PostgreSQL + S3 | Phase 2-3 |

---

## Resource Requirements

### Team

**Minimum:** 2-3 full-time developers for 16 weeks

**Skills Needed:**
- Python (PyKrige, FastAPI, GeoPandas)
- TypeScript/React (Next.js, MapLibre GL)
- Spatial data processing
- Database optimization (PostgreSQL/PostGIS)

### Infrastructure

**Development:**
- PostgreSQL database (existing)
- Redis instance (new)
- Object storage (S3 or MinIO)
- Development servers

**Production:**
- Web servers (Next.js hosting)
- Tile API servers (FastAPI)
- CDN (CloudFront/Cloudflare)
- Redis cluster
- PostgreSQL primary + replica

**Estimated Costs:**
- Development: ~$200/month
- Production: ~$500-1000/month (scales with usage)

---

## Risk Management

### High Risks

**1. Performance Regression**
- **Risk:** New tile system slower than expected
- **Mitigation:** Load test before cutover, parallel systems
- **Contingency:** Optimize or revert to hybrid approach

**2. Data Migration Issues**
- **Risk:** Converting stored GeoJSON heatmaps to tiles
- **Mitigation:** Keep both formats during transition
- **Contingency:** Regenerate from source data

**3. User Adoption**
- **Risk:** Users prefer current Streamlit interface
- **Mitigation:** Phased rollout, feature parity check
- **Contingency:** Maintain both interfaces

### Medium Risks

**4. Storage Costs**
- **Risk:** Tile storage more expensive than expected
- **Mitigation:** Start single region, monitor costs
- **Contingency:** Implement compression, on-demand generation

**5. Timeline Slippage**
- **Risk:** 16-week timeline not met
- **Mitigation:** Weekly progress reviews, clear milestones
- **Contingency:** Extend Phase 2, delay Phase 3

---

## Success Metrics

### Technical Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Initial map load | 5-15s | <1.5s | Time to first tile |
| Click-to-evaluate | 7-45s | <500ms | API response time |
| Concurrent users | 1-2 | 100+ | Load testing |
| Dataset size | ~5k wells | 50k+ | Scalability test |
| Tile latency | N/A | <200ms p95 | CDN monitoring |

### Business Metrics

- **User satisfaction:** Survey after Phase 3 launch
- **Usage growth:** Track active users monthly
- **Data coverage:** Expand beyond Canterbury
- **API adoption:** External integrations using tile API

---

## Next Steps

### Immediate Actions (This Week)

1. **Review & Approve** this plan
2. **Set up dev environment** (create `interpolation/` package)
3. **Assign developers** to Phase 1 tasks
4. **Create task board** (GitHub Projects or similar)
5. **Schedule kickoff** meeting

### Week 1 Start

1. **Day 1:** Create modular structure, base interpolator interface
2. **Day 2:** Implement OrdinaryKrigingInterpolator wrapper
3. **Day 3:** Create grid generation utilities
4. **Day 4:** Test coordinate transformations
5. **Day 5:** End-to-end integration test

### Communication

- **Daily standups:** 15 minutes, progress & blockers
- **Weekly reviews:** Demo progress, adjust plan
- **Stakeholder updates:** Biweekly email summary
- **Documentation:** Update as we build

---

## Key Decisions

### Architectural Decisions

✅ **Tile-first approach** - All outputs as map tiles  
✅ **Parallel implementation** - Build alongside existing system  
✅ **Phased migration** - Gradual cutover, not "big bang"  
✅ **Next.js frontend** - Modern web framework for production  
✅ **FastAPI backend** - Python-based tile/query services

### Technical Decisions

✅ **100m grid resolution** - Balance between detail and performance  
✅ **NZTM2000 for computation** - True metric CRS  
✅ **WebMercator for display** - Standard web tiles  
✅ **PNG tiles** - Visualization tiles, not data tiles  
✅ **Versioned storage** - Enable rollback and A/B testing

---

## Conclusion

This plan provides a **clear, actionable roadmap** to transform the groundwater interpolation prototype into a production-ready DeepSight system. The phased approach minimizes risk while delivering incremental value.

**Key Advantages:**
- ✅ Addresses all critical architectural gaps
- ✅ Maintains current functionality during transition
- ✅ Delivers performance improvements early (Phase 2)
- ✅ Scalable to country-scale datasets
- ✅ Clear success criteria and milestones

**The foundation is solid.** The interpolation science is working. Now we need the right architecture to deliver it at scale.

**Next Step:** Approve plan and begin Phase 1, Week 1 implementation.

---

**Document Index:**
- 📋 This summary: `docs/scope/README.md`
- 🎯 Target architecture: `docs/scope/architecture_summary.md`
- 📊 Gap analysis: `docs/scope/gap_analysis.md`
- 🔧 Detailed refactoring plan: `docs/scope/refactoring_plan.md`
- 🚀 Quick-start guide: `docs/scope/quick_start.md`
- 🧭 DTW execution plan: `docs/scope/dtw_development_plan.md`

**Status:** ✅ Planning Complete - Ready for Implementation
