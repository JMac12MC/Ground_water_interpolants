

# **DeepSight — Groundwater Intelligence Platform**

## **1\. Context**

DeepSight is a decision-support service for farmers, businesses, and local authorities planning to drill groundwater wells.

Drilling a well can cost **$50,000–$150,000+**, yet many drilling decisions are still guided by intuition, divining, or incomplete local knowledge. Meanwhile, **large volumes of public borehole and groundwater data already exist**, collected by councils, regulators, and neighbouring landowners.

DeepSight transforms this fragmented data into **probability-based, spatially explicit groundwater intelligence**, allowing users to understand:

* Where groundwater is likely to exist  
* How deep it is  
* How much water can be expected  
* How confidence and uncertainty vary spatially and seasonally

Example public dataset:  
Canterbury Regional Council well database  
[https://opendata.canterburymaps.govt.nz/datasets/ecan::wells-bores-existing/explore](https://opendata.canterburymaps.govt.nz/datasets/ecan::wells-bores-existing/explore)

---

## **2\. Project Overview**

### **Elevator Pitch**

*Before you spend $100,000 drilling a well, wouldn’t you want to understand the likelihood of finding groundwater?*  
DeepSight uses real borehole and environmental data to model where to drill, how deep to go, and how much water to expect — **reducing risk and saving tens of thousands of dollars per decision**.

### **App Name**

**DeepSight**

### **Problem Being Solved**

Farmers and organisations currently lack:

* Clear probability estimates of hitting usable groundwater  
* Reliable depth-to-water predictions  
* Understanding of seasonal variability  
* Insight into neighbouring well interference and long-term trends

This leads to:

* Wells drilled in poor locations  
* Wells drilled too shallow or too deep  
* Expensive failures or underperforming wells  
* Hesitation to drill when groundwater *does* exist

### **High-Level Goal**

Build a web-based groundwater intelligence platform that:

* Quantifies **probability, depth, yield, and uncertainty**  
* Visualises spatial and seasonal groundwater behaviour  
* Enables users to make **risk-aware drilling decisions**  
* Is grounded in **hydrogeologically accepted methods**

Success \= users can confidently answer:

*“Is it worth drilling here, how deep should I go, and what water can I realistically expect?”*

DeepSight is decision-support infrastructure for groundwater professionals.  
 It is not:

* a consumer mapping product  
* a guarantee of drilling success  
* a generic AI prediction engine

DeepSight exists to reduce risk, not eliminate it, by making uncertainty explicit, spatially grounded, and professionally defensible.

DeepSight explicitly chooses *not* to grow if professional trust is compromised. Growth without trust is treated as a failure mode.

---

## **3\. Users & Use Cases**

### **Target Users**

* **Farmers** (irrigation, stock water)  
* **Local authorities** (drinking water supply)  
* **Businesses & drillers** (commercial and industrial use)

  ### **Farmers & Landowners (Decision Makers)**

* **As a farmer**, I want to see where groundwater is most likely on my property so I can identify the best place to drill.  
* **As a farmer**, I want to understand the probability of finding usable groundwater and how that probability changes across my land.  
* **As a farmer**, I want to know the expected drilling depth and water yield so I can estimate drilling cost versus benefit.  
* **As a farmer**, I want to understand the confidence and uncertainty of these predictions so I know how much risk I am taking.  
* **As a farmer**, I want to see how groundwater behaves seasonally and during dry periods so I can assess long-term reliability.  
* **As a farmer**, I want all of this combined into a clear cost–risk–reward summary for a specific drilling location, without needing to interpret technical models.  
  ---

  ### **Drillers & Groundwater Consultants (Professional Users)**

* **As a driller**, I want to quickly assess where groundwater is *unlikely* to be found so I can avoid failed wells.  
* **As a driller**, I want defensible depth-to-water and yield estimates that I can confidently stand behind when advising clients.  
* **As a driller**, I want to understand uncertainty and data coverage so I can set realistic expectations with landowners.  
* **As a driller**, I want to compare multiple potential drilling locations to choose the option with the highest likelihood of success.  
* **As a driller**, I want professional-grade reports that I can share with clients to justify drilling recommendations.  
* **As a driller**, I want access to higher-accuracy local models that incorporate my own private data without exposing it to competitors.  
  ---

  ### **Local Authorities, Utilities & Regulators (Planning & Oversight)**

* **As a local authority**, I want region-wide groundwater depth and probability surfaces to support water supply planning.  
* **As a regulator**, I want transparent, auditable groundwater models with clear uncertainty to support permitting decisions.  
* **As a council**, I want to understand long-term and seasonal groundwater behaviour to assess sustainability and drought risk.  
* **As a utility**, I want to identify areas of low interference risk and stable supply for future infrastructure investment.  
  ---

  ### **Data Contributors (Accuracy Partners)**

* **As a data contributor**, I want to add private bore, monitoring, or test-pumping data to improve model accuracy in my area.  
* **As a data contributor**, I want to retain ownership and control over my data and decide who can benefit from it.  
* **As a data contributor**, I want to clearly see how my data improves predictions and reduces uncertainty locally.  
* **As a data contributor**, I want access to enhanced accuracy layers as a direct benefit of sharing data.  
  ---

  ### **Cross-User Outcome**

* **As any user**, I want DeepSight to clearly communicate *what is known, what is uncertain, and why*, so I can make informed, defensible groundwater decisions appropriate to my level of risk and responsibility.  
  ---

  ### **Why This Matters**

This expanded user-story set reflects the reality that:

* **Farmers trigger decisions**  
* **Drillers influence outcomes**  
* **Authorities govern sustainability**  
* **Data contributors drive accuracy**

DeepSight succeeds by serving *all four roles* — not just end landowners.

---

## **4\. Key Workflows**

1. User opens the map and views **regional groundwater depth and probability surfaces**  
2. User zooms to their property  
3. User clicks a point of interest  
4. DeepSight generates a **location-specific report** including:  
   * Probability of usable groundwater  
   * Expected depth range (seasonal min/max)  
   * Expected yield range  
   * Uncertainty indicators  
   * “Nearby wells and *potential* interference (distance-based heuristic; full modelling in Phase 2)”  
   * Geological context like how many aquifers are below that location, there depth and the material that needs to be drilled through

---

## **5\. Core Features & Scientific Models**

### **5.1 Is There Groundwater? (Probability of Encountering Usable Water)**

Primary question:  
 “If I drill here, what is the probability that I will encounter usable groundwater?”

This is the gateway layer for all further analysis. DeepSight models groundwater presence as a probabilistic spatial process, not a binary classification.

#### 5.1.1 Definition of Groundwater Presence

Groundwater is considered present where a well drilled to a reasonable depth:

* Intersects a saturated zone  
* Produces extractable water  
* Is supported by nearby producing wells in the same aquifer context

Shallow dry wells do not automatically imply absence of deeper groundwater.

#### 5.1.2 Modelling Approach

DeepSight uses an ensemble framework:

* Indicator kriging as the spatial baseline  
* Machine learning classifiers (Random Forest, Gradient Boosting) using geological and recharge covariates  
* Monte Carlo bootstrapping to quantify uncertainty

Outputs include:

* Mean probability of groundwater  
* Confidence and uncertainty classification

Validation uses spatial cross-validation only with calibration metrics (Brier score, reliability curves).

#### ---

#### 5.1.3 Data Inputs for Groundwater Presence

Depth-to-groundwater estimates (from 5.2, **only in regions already classified as likely groundwater-bearing**).

Primary well-derived inputs

* Well outcome: producing vs dry  
* Final drilled depth  
* Notes indicating refusal, partial yields, or abandonment  
* Screen presence (where available)

Contextual inputs

* Aquifer / geological unit  
* Proximity to recharge features  
* Neighbouring well outcomes  
* Depth-to-groundwater estimates (from 5.2, used conditionally)

All wells are spatially and geologically contextualised before modelling.

#### ---

#### 5.1.4 Pre-Processing & Domain Rules (Critical)

To avoid systematic bias, DeepSight applies hydrogeologically motivated rules before modelling.

Dry well handling

A dry well does not automatically imply absence of groundwater.

A dry well is down-weighted or excluded if:

* a producing well exists within \~300 m at greater depth, or  
* triangulation of the nearest 3 producing wells indicates deeper groundwater

This prevents:

* shallow drilling failures  
* poor drilling decisions  
   from incorrectly suppressing deeper aquifer probability.

Aquifer-aware labelling

* Presence/absence is evaluated per aquifer system  
* A dry shallow aquifer does not imply a dry deep aquifer

#### ---

#### 5.1.5 Modelling Approaches (Ensemble, Not Single Model)

DeepSight uses multiple complementary models to estimate groundwater probability.

No single model is trusted everywhere.

##### A. Indicator Kriging (Spatial Baseline

Method

* Convert wells to binary outcomes (1 \= producing, 0 \= dry)  
* Compute variograms of the indicator variable  
* Interpolate to a continuous probability surface

Why it is used

* Well-established in hydrogeology  
* Purely spatial and highly interpretable  
* Produces calibrated probability estimates

Limitations

* Does not use explanatory covariates  
* Can underperform in complex geology

Indicator kriging provides the spatial backbone of probability estimates.  
---

##### B. Machine Learning Probability Models (Covariate-Aware)

Models

* Random Forest Classifier  
* Gradient Boosting Classifier

Inputs

* Geological and aquifer unit  
* Distance to recharge features  
* Recharge proxies  
* Depth-to-groundwater estimates where stable  
* Surface elevation and DEM-derived features are only retained if they improve spatially cross-validated RMSE within a modelling zone

Outputs

* Probability of encountering groundwater  
* Calibrated probabilities using isotonic or Platt scaling

Why used

* Captures nonlinear relationships  
* Performs well in heterogeneous geology  
* Complements purely spatial models

---

##### C. Monte Carlo Ensemble Probability Modelling (Confidence Engine)

To avoid false precision, DeepSight treats probability itself as uncertain.

Process

* Train \~100 model realisations  
* Each run:  
  * bootstraps wells  
  * perturbs training data  
  * refits models  
* Predict probability for each run

Outputs

* Mean probability of groundwater  
* Variance / uncertainty  
* Confidence classification

This enables maps such as:

* High probability, high confidence  
* Moderate probability, low confidence  
* Low probability, high confidence

---

#### **5.1.6 Model Combination Strategy**

Final groundwater probability is derived by combining:

* Indicator kriging output (spatial continuity)  
* Machine-learning probability output (covariate signal)  
* Monte Carlo ensemble statistics (uncertainty)

Combination weights are region-specific and governed by spatial cross-validated performance. No model dominates universally.  
---

#### **5.1.7 Validation & Accuracy Assessment**

Groundwater presence models are validated using:

* Spatial cross-validation only  
* Region or aquifer hold-outs

Metrics include:

* Brier score  
* Log loss  
* Calibration curves  
* Contextual confusion matrices

Accuracy is interpreted in terms of decision risk, recognising that false positives and false negatives have different real-world costs.  
---

#### **5.1.8 User-Facing Outputs**

For any selected location, DeepSight provides:

* Probability of encountering groundwater (%)  
* Confidence level (high / medium / low)  
* Nearby producing and dry wells used in inference  
* Plain-language explanation of dominant drivers

Probability outputs are explicitly separated from:

* Depth-to-groundwater estimates (Section 5.2)  
* Yield estimates (Section 5.3)

---

#### **5.1.9 Defensibility & Role in the Platform**

This approach:

* Aligns with hydrogeological best practice  
* Avoids misleading binary classifications  
* Explicitly communicates uncertainty  
* Handles sparse and biased well datasets  
* Reflects real drilling outcomes

Section 5.1 acts as the gateway layer for DeepSight:

* It filters poor locations early  
* Frames drilling risk  
* Underpins trust in depth and yield modelling

---

### **5.2 Depth to Groundwater (DTW) Modelling Specification**

**DeepSight – Production-grade, zone-aware DTW mapping**

---

#### **5.2.1 Purpose & scope**

Produce **defensible, spatially valid Depth to Groundwater (DTW)** prediction surfaces with **quantified uncertainty**, suitable for:

* drilling risk assessment  
* seasonal reliability analysis  
* regulator-facing reporting  
* downstream tiling and visualization in DeepSight

The model must:

* respect **hydrogeologic boundaries**  
* avoid spatial leakage  
* degrade gracefully in sparse data zones  
* remain interpretable and auditable

---

#### **5.2.2 Target variable**

##### **5.2.2.1 Definition**

Depth to groundwater (DTW), in meters below ground surface:

`DTW_m = ground_surface_elev_m − static_water_level_elev_m`

Sign convention:

* Larger DTW \= deeper water table

---

##### **5.2.2.2 Seasonal targets (if time series available – recommended)**

For each well with sufficient observations:

* **DTW\_low** \= P90 DTW (late summer / high risk)  
* **DTW\_high** \= P10 DTW (late winter / recharge peak)  
* **DTW\_med** \= P50 DTW (expected)

Primary decision surface:

**DTW\_low** (seasonal low / worst-case)

Each target is modelled **independently** using the same pipeline.

---

#### **5.2.3 Modelling zones (non-negotiable)**

##### **5.2.3.1 Zone definition**

Zones are defined by **aquifer / hydrostratigraphic boundaries**.

Inputs:

`zones.gpkg`  
`- zone_id`  
`- geometry`

Rules:

* Every well belongs to **exactly one zone**  
* **No model spans zones**  
* All training, CV, variograms, and outputs are zone-scoped

Rationale:

* Controls non-stationarity  
* Prevents cross-aquifer leakage  
* Improves interpretability and CV honesty

---

#### **5.2.4 Input data & canonical data model**

##### **5.2.4.1 Wells (point data)**

`wells.parquet`  
`- well_id (string)`  
`- x, y (float, projected CRS)`  
`- ground_surface_elev_m (float)`  
`- static_water_level_elev_m (float)`  
`- well_depth_m (float)`  
`- screen_top_m (float)`  
`- screen_bottom_m (float)`  
`- aquifer_unit (category)`  
`- soil_perm_class (category)`

Derived per well:

`DTW_m`  
`screen_thickness_m = screen_bottom_m − screen_top_m`  
`screen_midpoint_elev_m`

---

##### **5.2.4.2 Recharge / river features**

`recharge_features.gpkg`  
`- geometry (LineString / Polygon)`  
`- feature_type (river_losing, fan, recharge_zone)`  
`- stage_elev_m (nullable)`

Only **hydraulically connected / losing** features are used.

---

##### **5.2.4.3 Raster covariates**

`soil_perm_index.tif        (ordinal / numeric)`  
`recharge_potential.tif     (0–1)`  
`dem.tif                    (optional; gated by CV)`

Rules:

* Common CRS  
* Zone-consistent resolution  
* Explicit nodata handling

---

#### **5.2.5 Feature engineering (deterministic)**

All features are computed **per well, per zone**.

---

##### **5.2.5.1 Recharge distance features**

For each well:

**Horizontal distance**

`d_xy = distance to nearest recharge feature (m)`

**Vertical distance**

`d_z = screen_midpoint_elev_m − recharge_surface_elev_m`

**3D hydraulic distance**

`d_3D = sqrt(d_xy^2 + (α × d_z)^2)`

Defaults:

`α = 1.0`  
`d_xy capped at 95th percentile per zone`

Transforms:

`log_d_xy  = log1p(d_xy)`  
`log_d_3D  = log1p(d_3D)`

---

##### **5.2.5.2 Aquifer / storage proxies**

* Aquifer unit (one-hot encoded)  
* Screen thickness (proxy for saturated thickness / storage)  
* Aquifer thickness proxy (if separate from screen thickness)

---

##### **5.2.5.3 Soil & recharge surface**

* Soil permeability class (categorical → one-hot)  
* Recharge potential index (0–1)

---

##### **5.2.5.4 DEM & DEM-derived features (conditional)**

Examples:

* elevation  
* slope  
* curvature  
* TPI  
* flow accumulation

**Inclusion rule (per zone):**

DEM-derived features are **only retained** if they improve **spatial CV RMSE** by ≥ **2%** and do not introduce bias (Section 5.2.9).

---

#### **5.2.6 Model architecture (authoritative choice)**

##### **5.2.6.1 Core model**

**Regression Kriging with nonlinear trend**

`DTW(x) = ML_trend(x) + spatial_residual(x)`

Where:

* `ML_trend` \= Gradient Boosted Trees (primary) or Random Forest (fallback)  
* `spatial_residual` \= ordinary kriging of residuals

This is the **best-performing, defensible approach** for your inputs.

---

##### **5.2.6.2 Trend model**

Primary:

`LightGBM / XGBoost regressor`  
`loss = L2`

Fallback:

`RandomForestRegressor`

Characteristics:

* Handles nonlinear distance effects  
* Handles categorical geology/soil  
* Learns interactions (e.g. distance × aquifer)

---

##### **5.2.6.3 Residual kriging**

* Residuals computed on **training wells only**  
* Variogram fitted **per zone**  
* Ordinary kriging onto output grid

Residual kriging is **disabled automatically** if it worsens spatial CV RMSE.

---

#### **5.2.7 Spatial cross-validation (mandatory)**

##### **5.2.7.1 CV method**

**Blocked spatial cross-validation**

Defaults:

`block_size_m = max(5 km, 3 × median well spacing in zone)`  
`n_folds = 5`

Algorithm:

1. Tile zone extent into square blocks  
2. Assign wells to blocks  
3. Hold out entire blocks per fold

---

##### **5.2.7.2 Metrics (per fold)**

* RMSE (primary)  
* MAE  
* Bias (mean error)

Aggregate:

`mean ± std across folds`

---

#### **5.2.8 Feature gating (DEM rule)**

For each zone:

1. Train model **without DEM features**  
2. Train model **with DEM features**  
3. Compare spatial CV RMSE

Keep DEM features only if:

`RMSE_with_DEM ≤ RMSE_without_DEM × 0.98`

This prevents DEM imprinting in Canterbury-style hydraulically flat systems.

---

#### **5.2.9 Residual variogram specification**

##### **5.2.9.1 Library**

* `GSTools` or `scikit-gstat`

##### **5.2.9.2 Allowed models**

* Exponential  
* Spherical  
* Matérn

##### **5.2.9.3 Fitting rules**

* Fit all allowed models  
* Select lowest AIC  
* If unstable or degenerate:

  * Fallback to exponential \+ nugget

---

#### **5.2.10 Uncertainty modelling**

##### **5.2.10.1 Spatial uncertainty**

* Kriging variance of residuals → `σ_krige(x)`

##### **5.2.10.2 Trend uncertainty (choose one)**

Preferred:

* Quantile GBM (P10 / P50 / P90)

Fallback:

* Ensemble spread (bootstrapped RF)

---

##### **5.2.10.3 Confidence classification (UX-facing)**

Per grid cell:

* **High**: low σ\_krige \+ strong CV performance  
* **Medium**  
* **Low**: sparse wells, high σ\_krige, or fallback mode

---

#### **5.2.11 Outputs (contractual)**

Per zone:

`DTW_trend.tif`  
`DTW_residual.tif`  
`DTW_final.tif`  
`DTW_uncertainty.tif`  
`confidence_class.tif`

Metadata JSON (same basename):

`{`  
  `"zone_id": "...",`  
  `"n_wells": 123,`  
  `"cv_rmse": 2.4,`  
  `"cv_mae": 1.8,`  
  `"features_used": [...],`  
  `"dem_included": false,`  
  `"model_type": "GBM + residual kriging",`  
  `"timestamp": "..."`  
`}`

All outputs:

* Defined CRS  
* Defined resolution  
* Suitable for tiling / COG conversion

---

#### **5.2.12 Failure & fallback rules (agent-critical)**

* **Too few wells (\< N\_min)**  
   → trend only, low confidence  
* **Variogram fit fails**  
   → trend only, inflate uncertainty  
* **Residual kriging worsens CV RMSE**  
   → disable kriging for that zone

All fallbacks must be **logged and surfaced**.

---

#### **5.2.13 Expected PR decomposition (for Copilot)**

Copilot should be able to split this into:

1. Data ingestion & validation  
2. Feature engineering (distance \+ raster sampling)  
3. Zone partitioning  
4. Spatial CV framework  
5. Trend model training  
6. DEM feature gating  
7. Residual variogram & kriging  
8. Uncertainty estimation  
9. Output rasterization & metadata  
10. Orchestration / CLI / pipeline runner

---

#### **5.2.14 Authoritative modelling position**

This spec intentionally:

* Rejects pure kriging  
* Rejects pure ML  
* Rejects co-kriging for deterministic covariates

It implements **best-practice DTW modelling** consistent with modern hydrogeologic literature and real-world decision risk.

Reference \- https://hess.copernicus.org/articles/23/4603/2019/

---

### **5.3 Groundwater Yield**

**Primary question:**  
 “How much water can I extract?”

**Approaches include:**

* Spatial interpolation of tested yields  
* ML-based yield prediction using screen length, aquifer type, depth, and geology  
* Ensemble uncertainty estimation

**Yield Confidence Classification Rule**

Yield outputs are explicitly classified as **Decision-Grade** or **Advisory** based on local data support, not just model fit. A yield estimate is considered Decision-Grade only when minimum evidence is met within the relevant zone (e.g., sufficient density of pumping tests or reliable yield records, acceptable spatial CV stability, and low ensemble spread). Where data are sparse or heterogeneous, DeepSight still provides a yield estimate but labels it **Advisory**, increases uncertainty bounds, and applies confidence-adjusted pricing and report messaging accordingly.

This rule prevents yield from being interpreted as equally reliable everywhere and aligns with how drilling professionals evaluate supply estimates in practice.

---

### **5.4 Aquifer Geometry & Geology**

* Identification of extractable aquifers  
* Estimation of aquifer depth ranges  
* Grouping of sparse aquifers where required  
* Geological cross-sections with uncertainty envelopes

---

### **5.5 Seasonal & Neighbour Interference Effects**

* Seasonal high and low groundwater surfaces  
* Drawdown cones around producing wells  
* Neighbour interference risk estimation  
* Long-term trend analysis

---

## **6\. Data Sources**

#### **6.1 Well and Bore Data (Primary)**

Used for probability, DTW, and yield modelling.

* Well location (lat/long or projected coordinates)  
* Total drilled depth  
* Static water level or water level observations used to infer DTW  
* Producing vs dry outcome  
* Screen intervals (top and bottom depths where available)  
* Well construction metadata (casing, diameter where available)  
   **Typical sources:** council well registries, national groundwater portals, open data catalogues.

#### **6.2 Lithology and Geology From Wells (High Value)**

Used for aquifer assignment, covariates, and cross sections.

* Lithology logs (material vs depth)  
* Interpreted stratigraphy where available  
* Aquifer unit labels derived from:  
  * well logs, and/or  
  * regional geological/aquifer map overlays  
     **Note:** Where lithology is only free-text, it is normalized into standard classes (e.g., gravel, sand, silt, clay, basalt).

#### **6.3 Yield and Pumping Test Data (Yield Layer)**

Used for expected yield modelling and calibration.

* Reported yield (L/s or m3/day)  
* Pumping test outcomes and duration (if present)  
* Drawdown and recovery (Phase 2 for interference modelling)  
   **Sources:** well completion reports, driller test sheets, council consent documents.

#### **6.4 Monitoring Bore Time Series (Seasonal and Trend Layers)**

Used for seasonal min/max DTW, long-term trends, and model validation.

* Water level time series (timestamp, level)  
* Metadata: sensor type, datum, measurement frequency  
* Quality flags if provided  
   **Sources:** council monitoring networks, utility monitoring networks, restricted datasets (permissioned).

#### **6.5 Rivers and Recharge Feature Datasets (Covariates and Physical Structure)**

Used to derive recharge proximity and hydraulic context predictors.

* River centerlines (braided rivers and major tributaries)  
* River stage surfaces where available (optional)  
* Recharge zones (foothills, alluvial fans, known recharge polygons)  
* Watershed boundaries (optional)  
   **Derived features generated from these datasets**  
* Horizontal distance to nearest river (m)  
* Horizontal distance to recharge zone boundary (m)  
* Vertical distance between well screen elevation and recharge reference elevation (m) where river stage or potentiometric surfaces exist  
* 3D hydraulic distance combining horizontal and vertical distance

#### **6.6 Geological and Aquifer Maps (Model Zoning and Covariates)**

Used to assign modelling zones and constrain prediction domains.

* Aquifer extents and grouping  
* Faults/barriers where mapped  
* Geological units (surface and subsurface interpretations)  
* Aquifer thickness proxies (where mapped)

#### **6.7 Soil and Permeability Classification (Recharge Proxy)**

Used primarily for shallow systems and recharge potential indices.

* Soil type polygons or raster classification  
* Drainage/permeability class  
* Land surface infiltration potential proxies (if available)

#### **6.8 Elevation Data (Conditional Covariate)**

Used only where it improves spatial cross-validated performance.

* Surface elevation and DEM-derived features are only retained if they improve spatially cross-validated RMSE within a modelling zone  
* Derived slope/curvature (optional)  
   **Important:** In regions like Canterbury where DTW is often weakly correlated with elevation, DEM is treated as optional and is dropped if it reduces accuracy.

Update Frequency

* Static for historical wells  
* Periodic refresh as new wells are published

#### **6.9 Data Contracts**

 Include:

* Canonical well record schema (required vs optional fields)  
* Units \+ normalisation rules (L/s vs m³/day, depth datum assumptions, etc.)  
* CRS rules:  
  * storage CRS(s)  
  * compute CRS (metric)  
  * display CRS (WebMercator tiles)

* Null handling rules (e.g., missing screen depth, missing static WL)  
* QA flags taxonomy (what flags exist and how they affect modelling inclusion)

## **7\. Backend Logic & Processing**

### **Core Libraries**

* PyKrige / GSTools  
* Scikit-learn  
* NumPy / SciPy  
* GeoPandas / Rasterio  
* PostGIS

**“Model & Tile Versioning Rules”**

* Semantic versioning for:  
  * model logic  
  * covariates  
    data refresh

* Immutable tile URLs per version

* Report must reference:  
  * model version  
  * data cut date

**Model Registry & Provenance**  
Define:

* `release_id` format (code\_version \+ data\_cut \+ config\_version) (you already describe this—codify fields)  
   Water design Scope doc (7)  
* `model_run_id` per zone and per layer  
* hashes:  
  * input dataset hash  
  * training set hash per zone  
* stored artifacts per run:  
  * fitted variogram params  
  * regression model params  
  * calibration method  
  * spatial CV configuration  
* lineage links:  
  * tileset manifest → model run → input datasets

---

## **8\. Frontend & Visualization**

**Screens / pages** (reference to your prototypes)

* [Interactive prototypes](https://www.figma.com/design/GxldRzDNlSjC5Ud1h0VpwP/Untitled?node-id=0-1&t=RWNThfYa8Li0Xmhl-1)

To build trust and give an immediate impression of the solution a map that visually shows the depth to ground water for the entire region, so the initial experience is very intuitive and the user can see where ground water is and the likely depth

![][image1]

If the user clicks on the map they get a report for that location that opens in a new page

### **Rendering Strategy**

* **Next.js (React) frontend**  
* **MapLibre GL JS** for base mapping  
* **deck.gl** for GPU-accelerated scientific overlays

This stack supports millions of polygons at interactive frame rates and avoids DOM bottlenecks.

---

## **9\. System Architecture**

* Frontend: Next.js \+ MapLibre \+ deck.gl  
* Backend: Python (FastAPI)  
* Database: Postgres \+ PostGIS  
* Object storage: tiled rasters and vector tiles  
* Compute: batch and on-demand model execution  
* Hosting: Docker-based cloud deployment

[Mermaid diagram](https://www.mermaidchart.com/app/projects/717f3ce3-b67e-43cf-b8bc-4431c84f47ed/diagrams/adac1eb8-a346-4c8f-9ba0-bedcebabe35e/share/invite/eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkb2N1bWVudElEIjoiYWRhYzFlYjgtYTM0Ni00YzhmLTliYTAtYmVkY2ViYWJlMzVlIiwiYWNjZXNzIjoiRWRpdCIsImlhdCI6MTc2NzY4MzE3N30.NCZhYAv94B8A2ahl-erkzD_jiIyoTXUeklpNBYrRjwk)

---

### Diagram Legend (Data Tiers and What Users See)

* **Baseline (Public)**: Uses public well registry, public geology/soil maps, and public river/recharge datasets. Visible to all users under standard access.  
* **Enhanced (Private Scoped)**: Baseline plus an organization’s private drilling or monitoring data. Visible only to that organization (and explicitly authorized users).  
* **Professional Shared (Opt-in Aggregation)**: Aggregated enhancements from multiple contributors where contracts allow derived use. Visible to licensed professional users in participating regions.  
* **Institutional**: Full access to restricted datasets and advanced governance/audit needs (councils/utilities), often including monitoring networks and planning layers.

In the diagram, tiering is enforced by **AuthN/AuthZ \+ Tiers** controlling access to tile namespaces in object storage/CDN.

### Explicit Non-Responsibilities (Runtime Constraints)

To ensure performance and scalability, DeepSight enforces the following runtime constraints:

* No kriging, regression-kriging, variogram fitting, or Monte Carlo ensembles are executed as part of a standard user map interaction.  
* No full-resolution rasters are generated on user click; all map layers are served as **precomputed tiles** (value tiles preferred; raster fallback only).  
* User clicks request only: precomputed tiles, summary statistics, and a report assembled from stored model artifacts.  
* On-demand compute is permitted only for **scoped reruns** (affected zones only) and is queued, cached, and released as a new model version (never silently overwriting an existing release).

### Model & Tile Versioning and Release Rules

All model outputs and tiles are versioned so that maps and reports are reproducible and auditable. Every published surface (probability, DTW, yield, uncertainty) is associated with a **Model Release ID** composed of three components: **code version**, **data cut**, and **configuration version** (covariates, zoning, variogram settings, calibration choices). Tile URLs are immutable per release (new releases generate new tile namespaces), and the UI can safely cache tiles because a “release” never changes once published.

Releases are promoted using governance gates: **spatial cross-validation metrics must meet minimum thresholds** (e.g., RMSE for DTW, Brier/calibration for probability, yield confidence class rules) and the release is recorded with provenance (input dataset versions, preprocessing rules applied, and model parameters). Location reports always include: **release ID**, **data cut date**, and **confidence tier** so professional users can cite outputs in client-facing documents.

Model execution produces two artifacts: (1) **model surfaces and statistics** (mean, P10/P50/P90, uncertainty metrics) and (2) **tile products** for fast delivery (value tiles primary). Both are stored with the same **release ID** and are served through versioned namespaces so frontend caching is safe and rollbacks are straightforward. Any change to covariates, zoning, preprocessing rules, calibration, or training data triggers a new release; releases are promoted only after passing spatial validation gates and are never modified after publication.

### Private Data Isolation Mechanism

Private-data-enhanced models run in **isolated model namespaces** that are separate from baseline public releases. Private datasets are stored with explicit ownership and access controls, and model runs that incorporate private data produce tiles and artifacts under **separate tile prefixes** (e.g., `enhanced/{org_id}/{release_id}/...`) that are only accessible to authorized users. Baseline public models are never overwritten or contaminated; instead, the system supports parallel releases (Baseline vs Enhanced) so contributors receive local accuracy uplift without exposing raw data or derived advantages to competitors

### **Quality Gates (Agent Contracts)** Two parts:

### **A) Golden science outputs**

* Fixed fixture dataset(s) per region/zone  
* Expected outputs with tolerances:  
  * DTW: sampled grid values \+ mean/quantiles  
  * Probability: Brier score range \+ calibration curve error bounds  
* Rule: PRs that change modelling must update fixtures *explicitly* and explain why.

### **B) Performance budgets (enforced)**

* API: point query p95 \< X ms  
* Tiles: time-to-first paint \< 1.5s (from your doc)  
   Water design Scope doc (7)  
* Tile payload cap per view: \< X MB  
* Browser: max memory for tile cache, LRU settings  
* Rule: perf regression \> 25% fails CI

### **Tile Specification** Include:

* Tile coordinate scheme (XYZ WebMercator)  
* Zoom mapping to ground resolution (“100m truth grid” → which max zoom)  
* Raster tiles:  
  * format choice (WebP vs PNG)  
  * nodata encoding (alpha=0 etc.)  
* Value tiles:  
  * chosen encoding (RGB terrain-style vs uint16 quantized)  
  * value range, scaling, precision  
  * nodata sentinel  
* Manifest fields:  
  * layer\_name, unit, min/max, percentiles, colormap id, release\_id, created\_at  
* Seam handling rule (buffer \+ crop) (you mention tile-first compute; make it explicit)  
   Water design Scope doc (7)

### **AuthZ & Tenant Isolation Model**  Define:

* entities: `org`, `user`, `role`, `tier`, `release_namespace`

* enforcement:  
  * tile prefix access (CDN signed URLs / gateway auth)  
  * query scoping by release namespace

* audit events:  
  * “who accessed which release/layer in which region”

  ---

### **Frontend**

**Purpose:** Instant first paint, smooth pan/zoom, progressive refinement, and interactive analysis.

**Technology**

* Web framework: Next.js (or equivalent)  
* Map rendering: **MapLibre GL JS**  
* GPU overlays (optional/advanced): **deck.gl**

**Tile rendering types**

* **Raster tiles** (WebP/PNG) — pre-colored pixels for fastest initial UX  
* **Value tiles** (numeric tiles):  
  * RGB-encoded value images (terrain-RGB style), or  
  * Quantized binary numeric tiles  
* **Vector tiles (MVT)** — wells, boundaries, contours

**How tiles are displayed**

* Raster tiles: drawn directly as textures  
* Value tiles:  
  * decoded in a **WebGL fragment shader**  
  * color ramps, thresholds, and opacity applied **on the GPU**  
* Vector tiles: GPU-rendered points/lines with clustering

**UX & performance guarantees**

* \< 1.5 s first paint using low-zoom tiles  
* Progressive refinement as higher-resolution tiles stream in  
* Viewport-only tile requests \+ adaptive prefetch  
* Cancellation of stale requests during fast pan/zoom  
* Bounded in-memory tile cache with LRU eviction

**User interaction**

* Click-to-evaluate (DTW, probability, yield, confidence)  
* Toggleable confidence / coverage overlays  
* Seamless region switching (NZ / AU / USA)  
  ---

### **Backend**

**Purpose:** Low-latency tile delivery and fast analytical queries, with no runtime raster generation.

**API Gateway**

* Authentication, rate limiting, logging  
* Routes requests to tile, query, and metadata services

**Core services**

1. **Tile Service**  
   * Serves static tiles (`z/x/y`) via CDN  
   * Supports:  
     * raster tiles (WebP/PNG)  
     * value tiles (RGB-encoded or binary)  
     * vector tiles (MVT)  
   * Optionally serves **PMTiles** or **COG-backed dynamic tiles**

2. **Query Service**  
   * Point query (lat/lon → predictions \+ confidence)  
   * Nearest-well lookup  
   * Region and layer metadata

3. **Catalog & Metadata Service**  
   * Model versions, tile versions  
   * Update timestamps  
   * Attribution and licensing text

4. **Ingestion & Normalization Service**  
   * Imports wells from NZ, AU, US public and private sources  
   * Normalizes schema, units, CRS, QA flags  
   * Triggers incremental tile recomputation

   ---

### **Database**

**Purpose:** Store and query **millions of wells and all metadata** — not large rasters.

**Primary database**

* **PostgreSQL \+ PostGIS**  
* Stores:  
  * well geometries and attributes  
  * QA flags and timestamps  
  * region/state/basin polygons  
  * model and tile metadata  
  * licensing and provenance

**Scaling strategies**

* GiST spatial indexing  
* Partitioning by region/state (critical for US scale)  
* Thin tables for map rendering; full records fetched on demand

**Caching**

* **Redis** for:  
  * point query results  
  * nearest-well queries  
  * hot metadata

**Explicit non-responsibility**

* The database **does not store** full 100 m rasters or heatmap pixels. Raster tiles are stored in object storage and served via CDN; Postgres stores only wells and metadata  
  ---

### **Raster & Tile Storage**

**Purpose:** Efficient storage and ultra-fast delivery of massive heatmap outputs.

**Tile types stored**

* Raster tiles: pre-colored pixels (WebP/PNG)  
* Value tiles:  
  * RGB-encoded numeric tiles  
  * Quantized binary numeric tiles  
* Vector tiles: Mapbox Vector Tiles (MVT)

**Storage layer**

* Object storage (S3 / GCS / Azure Blob)

**Delivery**

* Global CDN (CloudFront / Cloudflare / Fastly)  
* Immutable, versioned URLs for aggressive caching

**Optional packaging**

* **PMTiles** for single-file national/regional tile archives  
* **COGs with overviews** for dynamic tiling workflows

**Why this scales**

* Browser fetches only 10–50 tiles per view  
* CDN edge caching serves most traffic  
* No country-scale raster is ever loaded or transmitted  
  ---

### **Compute**

**Purpose:** Generate and maintain 100 m truth-grid heatmaps at national scale.

**Core pattern: tile-first compute**

* Interpolate/model **per tile with buffer**  
* Crop to tile bounds to prevent seams  
* Write directly to tile storage

**Distributed execution**

* Worker framework: **Ray** or **Dask**  
* Job queue: managed queue (SQS / PubSub / RabbitMQ)  
* Pipeline orchestration: Airflow / Dagster / Prefect

**Model execution**

* Baseline models for rapid coverage (IDW / RBF)  
* High-accuracy models (Regression Kriging / hybrid ML)  
* Confidence mask computed per tile (distance, density, uncertainty)

**Incremental updates**

* Identify tiles impacted by new wells/covariates  
* Recompute only those tiles  
* Version outputs for rollback

**Hardware**

* CPU-optimized nodes for spatial modeling  
* GPU optional (not required for kriging-based workflows)  
  ---

  ### **Hosting**

**Purpose:** Global low-latency delivery with elastic compute and cost control.

**Frontend**

* CDN-backed hosting (Vercel, Cloudflare Pages, Netlify)

**APIs**

* Containerized services on Cloud Run / ECS / Fargate  
* Kubernetes optional at later scale

**Tiles**

* Object storage \+ CDN (critical path for UX)

**Compute**

* Separate autoscaling worker pool  
* Spot/preemptible instances where possible

**Observability & operations**

* Centralized logging  
* Metrics dashboards (latency, cache hit rate, failures)  
* Alerting on pipeline and tile generation errors

**Security**

* Auth for premium layers  
* Rate limiting at CDN and gateway  
* Full audit trail of data and model versions  
  ---

  ### **Why this architecture meets your requirements**

* **Geographic scale:** Millions of wells and country-scale 100 m grids are feasible because tiles—not rasters—are the unit of storage and delivery.  
* **Performance:** Maps feel instant because only a handful of cached tiles are loaded.  
* **Flexibility:** Value tiles enable dynamic styling without reprocessing data.  
* **Accuracy:** All modeling is done in metric CRSs on a true 100 m grid.  
* **Trust:** Confidence masks and provenance are first-class outputs.

---

## **10\. Produce high-resolution map views**

### **1.1 Output products (what you must generate)**

**R1. Map layers (minimum)**

* **Depth to groundwater (DTW):** continuous surface (meters)  
* **Probability of groundwater:** 0–1 (or 0–100%) surface  
* **Expected yield (if available):** continuous surface (e.g., L/s or m³/day)  
* **Uncertainty / confidence:** continuous surface (e.g., prediction std dev, or quantile width)  
* **Data coverage / reliability mask:** categorical or continuous “trust” layer

**R2. Multi-scale pyramid outputs**  
 Each layer must be available as a multi-resolution pyramid with zoom levels supporting:

* country / continent overview (coarse)  
* regional (medium)  
* local farm-scale detail (fine)

“Fine” must include your **100 m × 100 m ground grid** as the highest required resolution (or finer, but not required).

**R3. Tile-based deliverables**  
 Output must be consumable as a tile set (`z/x/y`) for fast web display.

Supported delivery formats (at least one):

* Pre-rendered raster tiles (WebP/PNG) per layer, per zoom  
* Value tiles (RGB-encoded or quantized values) for GPU colorization in-browser  
* Optional: COG \+ tiler (dynamic tiles) if fewer stored files are preferred

**Acceptance criteria**

* Given a bbox and a zoom, the system can return the exact set of tiles needed to render all enabled layers.

**Incremental Recompute Algorithm**  
Define:

* What triggers recompute (new wells, updated covariates, zoning changes)  
* Impacted tiles detection:  
  * spatial index lookup of tiles intersecting buffer radius  
  * per-zone impacted set  
* Rules:  
  * recompute by zone or tile depending on model type  
  * release creation rules (never overwrite; new release namespace)  
* Observability requirements per tile (you mention them—define table fields) 

**API Contracts”**  
 Minimum endpoints:

* `GET /tiles/{release}/{layer}/{z}/{x}/{y}`  
* `GET /query/point?lat=..&lon=..&release=..`  
* `GET /catalog/releases`  
* `GET /catalog/layers?release=..`  
* `GET /wells/nearby?lat=..&lon=..&radius=..`

Include response shape \+ caching headers behavior.

---

### **1.4 Performance \+ orchestration requirements (production scale)**

**R11. Throughput**

* Support \~5,000 wells per interpolation tile  
* Generate **1,000s–100,000s** of tiles across zoom levels

**R12. Parallelism**

* Tile jobs must be embarrassingly parallel (Dask / Ray / queue workers)  
* Workers must be stateless

**R13. Incremental updates**

* Identify impacted tiles  
* Recompute only affected tiles  
* Version outputs for rollback

**R14. Storage \+ CDN**

* Tiles stored in object storage  
* Served via CDN with long-lived caching and versioned URLs

**R15. Observability**  
 Track per-tile:

* model used  
* input counts  
* compute time  
* error state  
* uncertainty summary

Maintain dashboards for failures and hotspots.

**Acceptance criteria**

* Partial rebuilds complete without full reprocessing  
* Tiles remain fast and cacheable under load  
  ---

## **11\. Display maps with a great user experience**

### **2.1 “Windy.com-like” UX goals (non-functional)**

**D1. Instant first paint**

* \< 1.5 seconds to meaningful map  
* Load basemap \+ low-zoom tiles \+ UI skeleton

**D2. Smooth interaction**

* 60 fps target on modern laptops  
* Graceful degradation on weaker devices

**D3. Progressive refinement**

* Show coarse tiles immediately  
* Refine as higher-zoom tiles arrive  
* Never block interaction

**D4. Predictable performance**

* Never download country-scale rasters  
* Request only viewport \+ small prefetch margin  
  ---

  ### **2.2 Rendering requirements**

**D5. WebGL renderer**

* MapLibre GL / Mapbox GL / deck.gl  
* Support stacked layers, opacity, blend modes

**D6. Layer types**

* Continuous surfaces: raster tiles or value-tile shaders  
* Probability: raster or value tiles with thresholding  
* Wells: point layer with clustering \+ tooltips  
* Optional: contours (low density)

**D7. Color ramps \+ legends**

* Units displayed  
* Global \+ local min/max  
* Percentile clipping toggle

**D8. No-data transparency**

* Transparent or hatched areas for coverage gaps

**Acceptance criteria**

* Layer toggles are instant  
* Legends always match rendered data  
  ---

  ### **2.3 Interaction requirements**

**D9. Click-to-evaluate**  
 Returns:

* predicted DTW / probability / yield  
* confidence score  
* nearest wells (count, distances)  
* “why” breakdown (top drivers if ML)

**D10. Drilldown panel**

* values \+ confidence  
* nearby wells list  
* cross-section (later)  
* downloadable report (later)

**D11. Region switching**

* Seamless NZ / AU / USA switching  
* Region-specific extents and layers

**Acceptance criteria**

* Click response \< 500 ms  
* Region switch keeps UI state, swaps tiles  
  ---

  ### **2.4 Performance engineering requirements (browser)**

**D12. Tile request management**

* Prioritize center-viewport tiles  
* Cancel stale requests  
* Adaptive prefetch

**D13. Memory management**

* Tile cache limits  
* LRU eviction

**D14. Mobile fallback**

* Reduced effects  
* Optional max-zoom cap  
* Prefer pre-colored tiles

**Acceptance criteria**

* No memory bloat or pan stutter  
  ---

### **2.5 Trust, transparency, and safety UX (critical)**

**D15. Confidence-first design**

* Confidence always visible  
* Data-density overlay toggle

**D16. Explain limitations**

* Clear inline disclaimers  
* Nudge users when uncertainty is high

**D17. Versioning & provenance**

* Model version  
* Data freshness  
* Source attribution

**Acceptance criteria**

* Users can clearly distinguish reliable vs speculative areas  
  ---

  ### Suggested MVP scope (ship fast, correctly)

  #### MVP v1

* Pre-colored raster tiles (WebP) for DTW \+ Probability  
* Wells overlay \+ click query  
* Confidence mask overlay  
* NZ (Canterbury) → one AU state → one US state

  #### MVP v2

* Value tiles \+ GPU color ramps  
* Incremental tile recompute \+ versioning  
* Uncertainty layer

---

## **12\. Non-Functional Requirements**

* Performance: sub-second map interaction  
* Scalability: regional to national datasets  
* Security: read-only public data, permissioned private layers  
* Reliability: cached model outputs  
* Compliance: regional data usage regulations

---

## **13\. MVP Definition**

### **V1 (Decision-Grade Core)**

* Probability of groundwater  
* Depth to groundwater (seasonal min/max)  
* Expected yield  
* Uncertainty surfaces  
* Regression kriging and indicator kriging

  ### **V2 (Advanced Hydro Features)**

* Drawdown cones  
* Well interference modelling  
* Flow lines and contours  
* Geological cross-sections  
* Long-term groundwater trends

---

## **14\. Risks & Assumptions**

### **Risks**

* Sparse data in some regions  
* Variable public data quality  
* Misinterpretation of uncertainty

  ### **Assumptions**

* Users accept probabilistic outputs  
* Public datasets remain accessible  
* Decision support is valued over guarantees

## **15\. Commercial Model, Pricing & Market Strategy**

DeepSight is positioned as **decision-grade infrastructure for groundwater planning**, not a consumer mapping product. Its commercial model reflects how groundwater decisions are actually made, who carries risk, and who influences outcomes.

---

### 15.1 Market Reality & Buying Dynamics

Groundwater drilling decisions have three defining characteristics:

1. **High irreversible cost**  
    ($50k–$150k+ per well)  
2. **Asymmetric risk**  
    One failed well can outweigh years of small savings  
3. **Delegated expertise**  
    Farmers rely on drillers and consultants to interpret risk

As a result, DeepSight is designed to monetise:

* *decisions*, not browsing  
* *professional usage*, not casual curiosity  
* *confidence and defensibility*, not raw data access

---

### 15.2 Primary Customer Segments (Who Actually Pays)

#### 1\. Drilling Companies & Groundwater Consultants (Primary Revenue Driver)

**This is the most important commercial insight from the recent chats.**

Drillers are:

* repeat users  
* geographically mobile  
* highly incentivised to improve success rates  
* reputationally exposed when wells fail

Contrary to initial intuition:

Drill failures **hurt drillers more than farmers** in the long run.

Failed wells:

* reduce referrals  
* create disputes  
* damage local reputation  
* increase regulatory scrutiny

**Why drillers pay**

* To choose *where not to drill* as much as where to drill  
* To justify decisions to clients  
* To reduce failure rates  
* To differentiate from competitors using intuition or divining

**How they buy**

* Annual or multi-year **professional licenses**  
* Access to multiple inventory layers  
* Commercial usage rights (client-facing)

**This is where predictable revenue lives.**

---

#### 2\. Farmers & Landowners (Transaction-Based Buyers)

Farmers:

* pay only when a drilling decision is imminent  
* do not want ongoing subscriptions  
* want *answers*, not tools

**Why they pay**

* To decide *whether* to drill  
* To decide *where* and *how deep*  
* To understand *risk before committing capital*

**Buying behaviour**

* One-off or short-term access  
* Will pay hundreds of dollars to reduce a six-figure risk  
* Value clarity over technical depth

Farmers are best served as:

* pay-per-report users  
* or indirect beneficiaries via driller usage

---

#### 3\. Councils, Utilities & Institutions (Secondary, High-Value)

These users:

* value transparency, auditability, and uncertainty  
* move slowly  
* require procurement processes  
* pay well once adopted

They are **not the MVP target**, but represent:

* long-term contracts  
* regional-scale deployments  
* strong credibility signals

---

### 15.3 Inventory-Layer Monetisation (Key Commercial Lever)

DeepSight’s defensibility and revenue scale come from **layered inventory**, not user count.

Each layer answers a more expensive question.

| Inventory Layer | Question Answered | Commercial Value |
| ----- | ----- | ----- |
| Depth to groundwater | “How deep do I drill?” | Core cost driver |
| Probability of groundwater | “Will this work at all?” | Risk framing |
| Yield estimate | “Will it meet demand?” | Operational viability |
| Seasonal min/max | “Will it fail in summer?” | Reliability |
| Uncertainty surfaces | “How confident is this?” | Trust & liability |
| Interference / drawdown | “Will neighbours affect me?” | Advanced / V2 |
| Cross-sections & geology | “Why is this happening?” | Professional justification |

Users pay more as they move **closer to action and accountability**.

---

### 15.4 Pricing Model 

#### Tier A — Driller / Consultant License (Core Revenue)

**Target**

* Drilling companies  
* Hydro consultants

**Includes**

* Full regional access  
* All V1 inventory layers  
* Client-facing reports  
* Commercial usage rights

**Indicative pricing**

* NZ / AU: $5,000 – $15,000 per year per company  
* USA (state-based): $10,000 – $25,000+ per year

Only a **small fraction of drillers** need to adopt for strong revenue.

---

#### Tier B — Property Decision Report (Farmer-Facing)

**Target**

* Farmers about to drill

**Includes**

* One location  
* DTW (seasonal min/max)  
* Probability \+ uncertainty  
* Yield range  
* Neighbouring wells context

**Indicative pricing**

* $200 – $500 per report

This is intentionally:

* cheaper than a drilling mistake  
* expensive enough to signal seriousness

---

#### Tier C — Regional / Institutional Access (Phase 2\)

**Target**

* Councils  
* Utilities  
* Regulators

**Includes**

* Regional models  
* Scenario analysis  
* Long-term trend layers  
* Audit documentation

**Pricing**

* Custom / contract-based

---

### 15.5 Adoption Expectations

This is **not** a mass-market SaaS.

Expected adoption:

* 5–15% of drillers in a region  
* Low churn (workflow embedded)  
* Farmers mostly arrive via drillers or just-in-time decisions

DeepSight succeeds if:

It becomes a *standard pre-drilling check* among professionals.

---

### 15.6 Competitive & AI Defensibility 

DeepSight is defensible because:

* It encodes **domain rules** (dry-well handling, aquifer zoning)  
* It uses **spatial validation**, not generic ML metrics  
* It exposes **uncertainty explicitly**  
* It is regionally tuned (not globally generic)  
* It produces outputs that can be **stood behind professionally**

Generic AI tools:

* can explain groundwater concepts  
* cannot produce **decision-grade, spatially validated, auditable outputs**

This makes DeepSight **infrastructure**, not content.

---

### 15.7 Commercial Success Criteria

DeepSight is commercially successful when:

* Drillers reference it when advising clients  
* Reports are attached to drilling quotes  
* Failed wells decrease in covered regions  
* “Did you check DeepSight?” becomes a normal question

At that point, pricing power follows naturally.

## **16\. Private Data Integration & Accuracy Uplift Model**

DeepSight recognises that **the highest-quality groundwater intelligence often exists outside public datasets**.  
 Local authorities, drilling companies, and consultants frequently hold **private or semi-private data** that can materially improve model accuracy for specific areas.

Rather than treating this as a data acquisition problem, DeepSight treats it as a **commercial accuracy upgrade**.

---

### **16.1 Types of Private Data Held by Local Bodies & Drillers**

Potential contributors often hold data such as:

**Drillers**

* Failed wells not reported publicly  
* Precise static water levels  
* Screen intervals and test yields  
* Pumping test results  
* Temporal measurements during drilling  
* Anecdotal “near-miss” locations

**Local Authorities / Utilities**

* Monitoring bore time series  
* Aquifer pressure data  
* Long-term trend analyses  
* Regulatory-only datasets  
* Recharge estimates and flow models

This data is often:

* spatially dense  
* regionally specific  
* unavailable to competitors  
* extremely valuable for local accuracy

---

### **16.2 Why Contributors Are Incentivised to Share Data**

Contributors do **not** want to “give away” their data — but they *do* want better outcomes.

Key incentives:

1. **Improved Local Accuracy**

   * Their data improves predictions **where they operate**  
   * Fewer failed wells  
   * Better decision outcomes

2. **Competitive Advantage**

   * Contributors receive access to **enhanced local models**  
   * Non-contributors see only baseline public-data models

3. **Data Control & Privacy**

   * Data is:

     * permissioned  
     * region-scoped  
     * never exposed raw to other users

4. **Reduced Liability**

   * Better models → fewer disputes  
   * Clear uncertainty communication

---

### **16.3 How Private Data Is Integrated (Technically & Commercially)**

Private data is integrated as a **model augmentation**, not a global overwrite.

**Key principles**

* Data is:

  * isolated by contributor  
  * spatially scoped  
  * versioned  
* Models are:

  * recalibrated locally  
  * never “polluted” globally

**Integration workflow**

1. Contributor uploads or authorises data  
2. Data is validated and normalised  
3. Models are re-run for affected zones  
4. Accuracy metrics (RMSE, uncertainty) are re-evaluated  
5. Enhanced layers are generated

---

### **16.4 Tiered Accuracy Model (Critical Commercial Mechanism)**

DeepSight operates with **explicit accuracy tiers**:

| Model Tier | Data Sources | Who Sees It |
| ----- | ----- | ----- |
| Baseline | Public datasets only | All users |
| Enhanced | Public \+ contributor private data | Contributor only |
| Professional Shared | Aggregated private data (opt-in) | Licensed professionals |
| Institutional | Full datasets | Councils / utilities |

This ensures:

* Contributors benefit first  
* Data sharing never erodes competitive position  
* Accuracy becomes a **paid feature**

---

### **16.5 Commercialisation of Data Contribution**

Private data integration is **not free** — it is a premium capability.

**Pricing mechanisms**

* Included in higher-tier professional licenses  
* Additional fee for:  
  * model re-runs  
  * accuracy uplift  
  * private-layer hosting  
* Long-term contracts for councils and utilities

This reframes the value proposition from:

“Give us your data”

to:

“Pay to unlock higher-accuracy models using your data”

---

### **16.6 Data Governance & Trust**

To make this viable, DeepSight enforces strict governance:

* Raw private data is never exposed to other users  
* Models clearly label:  
  * data sources  
  * confidence levels  
  * coverage extent  
* Contributors retain ownership of their data  
* Data can be withdrawn or time-limited

Trust is treated as a **core product feature**, not a legal afterthought.

---

### **16.7 Strategic Impact**

This model creates a powerful flywheel:

* More contributors → better local accuracy  
* Better accuracy → stronger professional adoption  
* Professional adoption → more data contribution

Importantly:

DeepSight improves fastest **where it is already trusted**, not everywhere at once.

This aligns incentives without relying on altruism or open data.

---

### **16.8 Why This Is Defensible**

This approach is hard to copy because it requires:

* Technical ability to isolate models spatially  
* Robust uncertainty handling  
* Strong governance  
* Trust from professionals

Generic AI tools cannot replicate this without deep domain integration.

---

#### Summary (Why This Matters)

This model:

* unlocks hidden data value  
* creates premium accuracy tiers  
* strengthens professional lock-in  
* improves outcomes without compromising data ownership

It turns **data asymmetry into a feature, not a problem**.

---

## **17\. Confidence-Adjusted Pricing, Revenue Model & Valuation**

### **17.1 Confidence-Adjusted Pricing (Critical Mechanism)**

DeepSight pricing is explicitly tied to **decision confidence**, not just feature access.

Groundwater decisions carry asymmetric risk:

* A confident recommendation can justify action  
* A low-confidence result must be cheaper, or advisory only

DeepSight therefore prices **certainty**, not pixels.

#### **Confidence Tiers (User-Facing)**

| Confidence Tier | Characteristics | User Meaning |
| ----- | ----- | ----- |
| High | Dense wells, consistent depths, stable geology | Decision-grade |
| Moderate | Mixed data density, some disagreement | Advisory |
| Low | Sparse or conflicting data | Risk screening only |

#### **Pricing Implication**

* **High confidence outputs justify higher pricing**  
* **Low confidence outputs are explicitly discounted**  
* Users are never charged “decision-grade prices” for speculative areas

This aligns:

* Trust  
* Liability  
* Willingness to pay

---

### **17.2 Pricing by User Type (With Confidence Adjustment)**

#### **Tier A — Driller / Consultant License**

Base price covers **access**  
 Effective value scales with **confidence availability in their operating regions**

* High-confidence regions → higher ROI → strong renewal  
* Low-confidence regions → advisory usage only

This naturally limits churn.

#### **Tier B — Property Decision Report (Farmer)**

Report pricing is **confidence-aware**:

| Confidence Outcome | Indicative Price |
| ----- | ----- |
| High confidence | $400–$500 |
| Moderate confidence | $250–$350 |
| Low confidence | $100–$200 (screening only) |

This prevents:

* Overcharging in uncertain regions  
* Perceived “map selling”

---

### **17.3 Revenue Ramp to $2.5M (Conservative)**

**Value Alignment**: Professionals (drillers/consultants) stand to gain the most—defensible reports reduce reputational risk, avoid failed wells (saving $50k+ per avoidance), and enable premium client charging. A $5k–$7k fee delivers quick ROI if it prevents even one moderate failure or wins extra jobs. This is conservative compared to the scope's higher targets ($10k–$25k in USA) but realistic for launch.  
**Market Benchmarks**:

* Closest analog: DrillerDB (US water well drilling ops software with AI depth estimation) uses a "simple monthly subscription" (implied affordable, likely $200–$500/month or \~$2.4k–$6k/year, including unlimited users/support).  
* Oil/gas drilling reporting tools (e.g., On Demand Well Operations) start at \~$750/month (\~$9k/year), but that's for larger ops.  
* Broader B2B agtech/environmental SaaS: Professional tools often $3k–$15k/year (e.g., precision ag platforms like Farmers Edge or environmental GIS add-ons). Hydrogeology software (e.g., Groundwater Vistas) is often perpetual license (\~$5k–$10k one-time) with annual maintenance, but SaaS shifts to recurring.  
* General B2B SaaS trends (2026): Mid-market tools average $5k–$20k/year for specialized decision-support, with hybrids blending subscription \+ usage.

**Regional Starter Adjustment**: Begin lower in NZ/AU ($4k–$6k NZD/AUD equiv.) where drillers may be smaller operations, then tier up for USA ($8k+). This matches the scope's regional variance ($5k–$15k NZ/AU vs. higher US).

**Launch Strategy**: Start at the lower end ($5k–$6k) for beta/early users to drive adoption and testimonials. Include perks like unlimited reports, private data integration, and support to justify value.

To encourage uptake:

* **Basic Professional**: $4,800/year (\~$400/month) – Core maps, probability/depth/yield reports (limited credits), basic uncertainty.  
* **Standard Professional** (Recommended Start): $6,000–$7,200/year – Unlimited reports, seasonal layers, custom branding for client sharing.  
* **Premium**: $9,000+ – Advanced (V2 interference), priority support, private data uplifts.

Add a 20–30% discount for annual prepay or multi-year commitments. Pilot with 10–20 drillers at introductory rates to validate.

This range balances accessibility (avoiding sticker shock in a conservative industry) with profitability, setting up upsells as the data flywheel improves accuracy. Test via surveys or A/B offers—aim for 3–5x perceived ROI to hit 10% adoption quickly.

**Market realities temper projections:**

* NZ has \~50–100 active water well drilling companies (based on federation listings and directories showing dozens of registered firms).  
* Australia has several hundred (directories list 100+ specialists).  
* US groundwater drilling is vast but fragmented (thousands of licensed drillers, e.g., California/Texas alone have massive well counts).  
* Total addressable professional users in initial markets: \~400–600 drillers/consultants.  
* B2B SaaS benchmarks (2025 data): Median Year 1 revenue for early-stage startups is low ($100k–$500k), with growth slowing to 28–50% YoY as ARR scales. Agtech/environmental niches often start slower due to conservative adoption.  
* Closest analog (DrillerDB): Affordable monthly subscriptions (\~$79–$199/month), suggesting room for premium pricing if value (probabilistic intelligence) proves out.  
* Customer Acquisition Cost (CAC): \~$1,000–$3,000 per professional customer in agtech/B2B (higher for field-based sales; lower with inbound/freemium).

#### **Assumptions (Explicit & Conservative)**

* Region example: NZ \+ 1 AU state \+ 1 US state  
* Drillers in scope: \~400  
* Adoption rate: **10%** (40 companies)  
* Farmer reports are mostly **pull-through**, not primary sales

#### **Revised Revenue**

#### **Year 1 (validation)**

* 10 drillers × $5k \= $50k  
* 150 reports × $300 \= $45k  
* **Total: \~$95k**

#### **Year 2 (early PMF)**

* 25 drillers × $5.5k \= $137k  
* 450 reports × $325 \= $146k  
* **Total: \~$283k**

#### **Year 3 (regional credibility)**

* 40 drillers × $6.5k \= $260k  
* 900 reports × $375 \= $338k  
* **Total: \~$600k**

#### **Year 4 (controlled expansion)**

* 120 drillers × $8k \= $960k  
* 2,400 reports × $425 \= $1.02M  
* 3 enterprise / institutional contracts × $180k \= $540k  
* **Total: \~$2.52M**

$2.5M ARR is achievable by Year 4 through moderate geographic expansion, increased professional ARPU as users adopt higher tiers, maturation of report pull-through, and a small number (3–5) of regional enterprise contracts — without requiring mass-market farmer adoption.

---

### **17.4 Valuation (Conservative Multiples)**

DeepSight is **decision-grade infrastructure**, not content SaaS.

Appropriate multiples:

* 4–6× ARR (conservative, infrastructure-adjacent)  
* Higher if institutional contracts dominate

At $2.5M ARR:

| Multiple | Valuation |
| ----- | ----- |
| 4× | $10M |
| 5× | $12.5M |
| 6× | $15M |

This excludes:

* Strategic value to drilling firms  
* Data moat compounding  
* Expansion to additional regions

---

### **17.5 Failure Case — What Breaks This**

DeepSight fails if **any one** of the following occurs:

1. **Uncertainty is hidden or softened**  
   * Leads to mistrust  
   * Leads to reputational damage

2. **Drillers do not adopt**  
   * Farmer-only sales do not scale  
   * Product becomes a “map curiosity”

3. **Private data governance fails**  
   * Loss of trust  
   * Immediate professional churn

4. **Models are not spatially validated**  
   * RMSE looks good, outcomes are bad  
   * Professionals stop trusting outputs

5. **Pricing ignores confidence**  
   * Users feel overcharged in poor data regions  
   * Perception of pseudo-science risk

The current architecture and model choices explicitly mitigate these risks — but they must remain **product rules**, not marketing language.

**Operational Failure Modes**  
 Examples:

* tiles missing at some zoom → fallback to lower zoom, show warning  
* query service degraded → cached result or “temporarily unavailable”  
* sparse data region → force “screening only” UX and pricing tier  
* modelling job fails mid-run → release is not promoted; old release remains

## **18\. Report**

*(Primary orientation section)*

#### **Purpose**

Provide a clear framing of the decision and its key risks, without oversimplification.

This section allows users to understand:

* what the analysis suggests  
* where uncertainty lies  
* what requires further consideration

DeepSight intentionally sacrifices coverage and confidence in low-data regions rather than producing misleading outputs.  
 Areas with insufficient data are:

* flagged  
* discounted  
* sometimes deliberately *not recommended*

The absence of guidance is treated as a valid, decision-relevant outcome.

#### **Contents**

**Headline guidance (plain language)**  
 Example:

“Relocating approximately 200–300 m east of the originally proposed site is associated with lower overall drilling risk based on available data.”

**Decision status indicator**

* 🟢 Lower risk relative to nearby alternatives  
* 🟡 Moderate risk / uncertainty present  
* 🔴 High risk based on current information

**Key summary metrics**

* Likelihood of encountering usable groundwater (%)  
* Expected drilling depth range (P10–P90)  
* Yield adequacy (qualitative: adequate / marginal / uncertain)  
* Interference risk (qualitative: Low / Moderate / High / Unknown)  
* Primary risk drivers (e.g. seasonal variability, sparse data, neighbour abstraction pressure)

**Anchor reference**

* Selected analysis location  
* Distance and direction from any original proposed site

#### **Why this section exists**

* It orients the reader  
* It frames trade-offs  
* It avoids false certainty

---

### **18.1 Confidence & Data Transparency**

*(Trust and expectation management)*

#### **Purpose**

Explain how confident the assessment is and why.

#### **Contents**

**Confidence score**

* Numeric score (e.g. 78 / 100\)  
* Qualitative label: High / Moderate / Low

**Drivers of confidence (plain-language bullets)**  
 Examples:

* Density of nearby wells  
* Consistency of groundwater depths  
* Geological uniformity  
* Agreement between independent models  
* Clarity of neighbour well context (where metadata exists)

**Data support inset**

* Small map showing data density or coverage

**Explicit limitation statement**

“This assessment is probabilistic and reflects available data and modelled uncertainty. It does not guarantee drilling outcomes.”

#### **Why this section exists**

* Prevents over-interpretation  
* Protects professional credibility  
* Justifies pricing based on confidence

---

### **18.2 Local Groundwater Overview**

*(From point selection to spatial context)*

#### **Purpose**

Show how groundwater conditions vary across the local area, not just at a single point.

#### **Contents**

* Map of the local context envelope (parcel or defined buffer)  
* Groundwater probability surface (contextual)  
* Highlighted recommended drilling area(s), if identified  
* Currently selected anchor location  
* Neighbour well context (nearby wells and local density, shown visually)

#### **Key rule**

This section provides context, not conclusions.

---

### **18.3. Recommended Drilling Areas**

*(Guidance, not prescriptions)*

#### **Concept**

DeepSight identifies recommended drilling areas where overall drilling risk is lower relative to nearby alternatives, based on multiple factors considered jointly.

#### **Internal basis (not exposed numerically)**

* Probability of groundwater  
* Model confidence / uncertainty  
* Yield adequacy  
* Seasonal reliability  
* Interference risk screening (proximity and density of neighbouring wells, where data exists)

Depth to groundwater is treated as a cost and design consideration, not a suitability driver, in regions where deeper systems may be more reliable.

#### **User-facing representation**

* One or two softly shaded areas  
* Simple labels (e.g. “More reliable area”)

If no area meets minimum reliability thresholds, no recommended area is shown.  
 This absence is an intentional and meaningful result.

#### **Explanation of Recommended Areas**

Whenever a recommended drilling area is shown, the report must clearly explain why that area is favoured relative to nearby alternatives.

This explanation is provided through:

* spatial context on the map  
* cross sections intersecting the recommended area  
* a short, fixed-format plain-language summary

The explanation must reference:

* relative groundwater likelihood  
* yield adequacy  
* seasonal stability  
* model confidence  
* neighbour interference screening (e.g. fewer nearby high-use wells)

Numerical optimisation scores or factor weightings are intentionally not exposed.

If no area can be clearly justified as lower risk, no recommended area is shown, and this outcome is explicitly stated.

#### **Why this is the right balance**

* answers the user’s “why”  
* keeps the UX simple  
* reinforces trust through visuals  
* avoids overclaiming  
* aligns with a cross-section-first design

In short:

* The shaded area draws attention  
* The cross section explains it  
* The bullets confirm it

---

### **18.4 Cross Sections & Subsurface Explanation**

*(Primary explanatory mechanism)*

#### **Purpose**

Explain why conditions vary spatially and why certain areas are favoured.

#### **Automatic generation**

For every selected location, DeepSight generates two cross sections automatically:

* **Cross Section A:** aligned with the strongest local change in depth and/or confidence  
* **Cross Section B:** perpendicular context section

Both sections:

* span the local context envelope  
* are anchored at the selected location

#### **Cross section contents**

Each cross section visualises, in a single view:

* Median depth to groundwater  
* Seasonal depth envelope (min / max)  
* Prediction uncertainty  
* Simplified aquifer / geological context  
* Nearby wells intersecting or adjacent to the section  
* Neighbour well markers and local abstraction context (where metadata exists)

Where applicable, seasonal behaviour and potential neighbour influence are reflected visually through:

* widening seasonal envelopes  
* changes in depth consistency  
* clustering or absence of supporting wells

This avoids the need for separate, abstract sections for seasonality or interference.

---

### **18.5 Cross Section Interaction & Update Workflow**

*(Explicit UX requirement)*

#### **Click-to-regenerate behaviour**

Clicking a new location on the map:

* re-anchors the analysis  
* regenerates both cross sections automatically  
* updates recommended drilling areas  
* updates the report content

Users do not manage or “reset” cross sections — clicking again is the reset.

#### **Interactive adjustment**

* Cross sections are always visible as lines on the map  
* Each section has draggable endpoints  
* Dragging a section:  
  * updates the cross-section view immediately  
  * updates supporting summaries in the report

This makes cross sections:

* easy to generate  
* easy to reinterpret  
* spatially obvious at all times

---

### **18.6. Alternatives & Trade-Offs**

*(Comparison, not optimisation)*

#### **Purpose**

Help users understand relative risk differences, not to optimise numerically.

#### **Contents**

**Spatial alternatives table**, comparing:

* selected anchor  
* nearby alternatives  
* original proposed site (if applicable)

Metrics are comparative and rounded:

* probability (approximate)  
* depth range  
* confidence  
* interference risk (Low / Moderate / High / Unknown)  
* qualitative overall risk

**Interpretive text**  
 Example:

“Locations 200–300 m east show lower uncertainty, more stable seasonal behaviour, and lower neighbour-interference screening risk compared to the original site.”

This section supports judgment rather than replacing it.

---

### **18.7. Summary & Next Considerations**

*(Close with clarity, not instruction)*

#### **Contents**

* Summary of key findings  
* Where conditions appear more or less reliable  
* Interference considerations (e.g. nearby abstraction, data gaps)  
* Primary uncertainties to consider  
* What additional information could change the assessment

**Validity statement**

“This assessment reflects data available as of \[date\] and should be revisited if new wells or pumping information becomes available.”

---

### **Golden Rule (Explicit)**

The report must support careful decision-making even when uncertainty remains, without creating false confidence or unnecessary complexity.

---

### **Why this structure is intentional**

* It respects the seriousness of the decision  
* It avoids unvetted financial or operational prescriptions  
* It uses cross sections as the unifying explanation for:  
  * depth  
  * yield adequacy  
  * seasonality  
  * neighbour interference context

It keeps the UX simple:  
 **click → see → adjust → understand**

Ff

## Links

### [https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/03\_variogram/03\_directional\_2d.html](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/examples/03_variogram/03_directional_2d.html)

### [https://www.researchgate.net/figure/Main-aquifer-types-of-Canterbury-region-inset-figure-shows-New-Zealand-include-coastal\_fig1\_363320994](https://www.researchgate.net/figure/Main-aquifer-types-of-Canterbury-region-inset-figure-shows-New-Zealand-include-coastal_fig1_363320994)

[https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR023437](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR023437)

[https://www.researchgate.net/publication/353245804\_Spatial\_Interpolation\_for\_the\_Distribution\_of\_Groundwater\_Level\_in\_an\_Area\_of\_Complex\_Geology\_Using\_Widely\_Available\_GIS\_Tools](https://www.researchgate.net/publication/353245804_Spatial_Interpolation_for_the_Distribution_of_Groundwater_Level_in_an_Area_of_Complex_Geology_Using_Widely_Available_GIS_Tools)

[https://lida.leeds.ac.uk/research-projects/harnessing-ai-for-groundwater-predictions/](https://lida.leeds.ac.uk/research-projects/harnessing-ai-for-groundwater-predictions/)

[https://www.fulcrumapp.com/blog/ai-driven-groundwater-monitoring-exploring-cutting-edge-technologies/](https://www.fulcrumapp.com/blog/ai-driven-groundwater-monitoring-exploring-cutting-edge-technologies/)

chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/[https://www.ijraset.com/best-journal/ai-enabled-water-well-predictor](https://www.ijraset.com/best-journal/ai-enabled-water-well-predictor)

[https://github.com/peterson-tim-j/HydroSight/blob/master/README.md](https://github.com/peterson-tim-j/HydroSight/blob/master/README.md)

[https://farmonaut.com/precision-farming/ai-enabled-water-well-predictor-ai-crop-yield-prediction\#ai-enabled-water-well-predictor](https://farmonaut.com/precision-farming/ai-enabled-water-well-predictor-ai-crop-yield-prediction#ai-enabled-water-well-predictor)

[https://peterson-tim-j.github.io/HydroSight/](https://peterson-tim-j.github.io/HydroSight/)

[https://hess.copernicus.org/articles/23/4603/2019/](https://hess.copernicus.org/articles/23/4603/2019/)  
Denmark example

[https://www.sciencedirect.com/science/article/pii/S1674987118301488](https://www.sciencedirect.com/science/article/pii/S1674987118301488)  
Potential groundwater recharge zones within New Zealand

### Open papers with groundwater maps \+ performance

1. **Spatiotemporal Regression Kriging — Arapahoe Aquifer (Water Resources Research, 2019\)**  
    What you’ll get:

* Groundwater level mapping with a **spatiotemporal regression-kriging** framework

* Strong emphasis on **prediction performance** and improving confidence in unsampled locations/times  
   Why it’s good for you:

* Very close to what DeepSight needs (spatial \+ temporal).

https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018wr023437?utm\_source=chatgpt.com

2. **Kriging with External Drift vs Collocated Cokriging for water-table mapping (2006-ish era, classic comparison)**  
    What you’ll get:

* A head-to-head comparison: **external drift (RK/KED)** vs **cokriging**

* Water-table surface development \+ method comparison (usually includes maps and CV)  
   Why it’s good:

* Directly answers “RK vs co-kriging” in a water-table mapping setting.

https://www.researchgate.net/publication/249845566\_Kriging\_with\_an\_external\_drift\_versus\_collocated\_cokriging\_for\_water\_table\_mapping

3. **Improvement of groundwater level prediction in sparsely gauged areas — Residual Kriging / KED (2013)**  
    What you’ll get:

* Explicit framing of **Residual Kriging (RK)** / **KED** for groundwater levels

* Focus on **improving prediction under sparse monitoring** (very relevant to NZ)  
   Why it’s good:

* Method \+ validation focus; commonly includes mapped surfaces \+ error metrics.

https://www.sciencedirect.com/science/article/abs/pii/S0309170812002229?utm\_source=chatgpt.com

4. **On the kriging of water table elevations using collateral information from a DEM (2001)**  
    What you’ll get:

* Groundwater/water-table mapping using **KED with DEM collateral information**

* A classic reference used in many later groundwater-KED papers  
   Why it’s good:

* Shows how “secondary information” is incorporated (even if Canterbury DEM is weak, the method is clear).

https://www.researchgate.net/publication/222574362\_On\_the\_kriging\_of\_water\_table\_elevations\_using\_collateral\_information\_from\_a\_digital\_elevation\_model

5. **Regression Kriging groundwater case study (Adyar River Basin) — seasonal pre/post monsoon**  
    What you’ll get:

* Groundwater level data used to build **regression kriging models**

* Reports **cross-validation performance** (R² values stated) and typically includes seasonal surfaces  
   Why it’s good:

* Demonstrates RK in a seasonal context (mirrors your “seasonal high/low” requirement).

https://www.researchgate.net/publication/200043307\_Kriging\_of\_Groundwater\_Levels\_-\_A\_Case\_Study

### Groundwater “depth” example (often depth/capacity rather than DTW)

6. **Using Regression Kriging to analyze groundwater according to depth and capacity of wells (Sulaimani, Iraq)**  
    What you’ll get:

* Regression kriging applied to **well depth** (and capacity), with resulting **prediction maps**

* Descriptive spatial interpretation of high/low zones  
   Why it’s good:

* Not DTW, but it’s a clear “groundwater \+ RK \+ maps” example to crib structure from.

https://www.researchgate.net/publication/334273896\_Using\_Regression\_Kriging\_to\_Analyze\_Groundwater\_According\_to\_Depth\_and\_Capacity\_of\_Wells?utm\_source=chatgpt.com

## **Tutorials with code you can run (to reproduce maps)**

7. **PyKrige RegressionKriging example (Python) — map outputs**

* Working RK implementation (trend model \+ kriged residuals)

* Produces prediction surfaces you can adapt to groundwater DTW/heads by swapping inputs/covariates  
   (If you’re using Python heavily, this is the quickest “see a map today” route.)

https://cran.r-project.org/web/packages/npsp/vignettes/krigstat.html?utm\_source=chatgpt.com

8. **“About regression-kriging: From equations to case studies” (Hengl et al., 2007\)**

* The most-cited, practical RK reference

* Explains workflow \+ case studies (not always groundwater, but the mechanics and evaluation pattern are exactly what you need)

researchgate.net/publication/223155790\_About\_regression-kriging\_From\_equations\_to\_case\_studies?utm\_source=chatgpt.com

### How to use these for Canterbury DTW (fast path)

If your goal is *DTW surfaces with maps \+ performance*:

* Use **(1) or (5)** as your “seasonality framing”

* Use **(2) \+ (3)** as your “why RK beats cokriging most of the time” evidence

* Use **(7)** as the runnable code base (swap covariates to your Canterbury set)

[image1]: <data:image/png;base64,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>