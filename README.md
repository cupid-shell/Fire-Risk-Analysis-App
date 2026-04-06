# Fire Risk Analysis Tool (FRAT)

A scientifically grounded geospatial fire risk assessment platform for urban settlements, built on OpenStreetMap data and published research standards. FRAT computes a multi-factor composite risk score across a spatial grid, visualises results interactively, and supports scenario planning, uncertainty quantification, and export for downstream GIS workflows.

---

## Scientific Basis

Each component of the risk model is anchored to a published standard or peer-reviewed reference:

| Risk Factor | Normalization Anchor | Reference |
|---|---|---|
| Building Density | 30 buildings / 50×50 m cell = critical | UN-Habitat Urban Indicators, 2016 |
| Fire Station Access | 240 s travel time = maximum risk | NFPA 1710, Standard for Career Fire Departments |
| Water Source Proximity | 500 m = maximum risk | ISO/TR 13387-1, Fire Safety Engineering |
| Building Height | 10 floors = maximum risk | SFPE Handbook of Fire Protection Engineering |
| Hazard Proximity | Inverse-square decay from 50–2000 m | UNDRR Global Risk Assessment Framework |
| Weight Derivation | AHP Consistency Ratio < 0.10 | Saaty, T.L. (1980), The Analytic Hierarchy Process |
| Risk Classification | Jenks Natural Breaks | Jenks, G.F. (1967), The Data Model Concept in Statistical Mapping |
| Uncertainty | Weight perturbation, n = 300 | Saltelli et al. (2008), Global Sensitivity Analysis |
| Spatial Autocorrelation | Moran's I, Queen contiguity | Moran, P.A.P. (1950), Biometrika |
| Aggregation | Weighted sum / Geometric mean | UNDRR GRAF, 2022 |

---

## Key Features

### Risk Model
- **5-factor composite score** — Density, Access, Water, Height, Hazard
- **Absolute threshold normalization** — scores are calibrated against NFPA/ISO/UN-Habitat anchors, not relative to local min/max, so results are comparable across different cities and analyses
- **Gross Floor Area (GFA)** — density accounts for building volume, not just footprint count
- **Building combustibility** — material type (concrete, wood, metal, mixed) and use (fuel station, warehouse, residential) scored using SFPE Handbook fire load values
- **Road travel time** — access risk uses actual network travel time via `osmnx` speed/time attributes, not straight-line distance
- **Hazard type weighting** — fuel stations (×3.0), hospitals (×2.0), marketplaces (×1.5), schools (×1.0)
- **Road width modifier** — narrow roads increase density and access risk
- **Wind direction modifier** — downwind cells receive up to +20% risk boost

### Scientific Validation
- **Analytic Hierarchy Process (AHP)** — derive weights from pairwise factor comparisons; Consistency Ratio displayed and flagged if CR ≥ 0.10 (Saaty threshold)
- **Monte Carlo uncertainty** — 300 simulations with ±10% weight perturbation; outputs mean risk, standard deviation, and coefficient of variation per cell
- **Moran's I spatial autocorrelation** — detects spatial clustering of high-risk zones (libpysal Queen contiguity + esda)
- **Jenks Natural Breaks** — statistically optimal risk band thresholds; falls back to quantile if data is insufficient
- **OSM data completeness flag** — each grid cell receives a completeness score; low-confidence cells are flagged in the summary

### Interface
- **Sidebar inputs** — location (name or coordinates), radius, weight sliders with live normalisation display, AHP matrix, advanced options
- **8 result tabs** — Summary · Maps · Interactive · Hotspots · Export · History · Batch · Compare
- **Interactive map layer controls** — toggle Risk Grid, Risk Heatmap, Road Risk Overlay, Fire Stations, Water Sources, Distance Rings, and Satellite Base independently via Streamlit checkboxes
- **Hypothetical station simulation** — place a new fire station at any coordinate and instantly see the impact on access risk
- **Scenario comparison** — run two analyses and compare their risk distributions side-by-side
- **Batch analysis** — analyse multiple locations in sequence from the UI
- **Shareable URL** — analysis parameters are encoded in query strings

### Export
| Format | Use Case |
|---|---|
| GeoJSON | QGIS, ArcGIS, web mapping |
| CSV | Statistical analysis, Excel |
| Shapefile (.zip) | Desktop GIS |
| KMZ | Google Earth |
| HTML map | Fullscreen interactive sharing |
| HTML report | Printable summary with maps |

---

## Architecture

```
app.py                      ← Streamlit frontend (UI, tabs, session state)
fire_risk_analyzer.py       ← Backend (all spatial analysis functions)
requirements.txt            ← Python dependencies
history/                    ← Auto-saved GeoJSON + metadata per analysis run
```

**Data flow:**
```
OSM (parallel fetch) → Buildings → Height / Combustibility / Occupancy
                     → Road network → Travel time to fire stations
                     → Water sources → Proximity risk
                     → Hazard points → Weighted proximity decay
                            ↓
                    50×50 m grid → Composite risk (weighted sum or geometric mean)
                            ↓
                    Jenks bands → Monte Carlo → Moran's I → Interactive map
```

---

## Setup

### Requirements
- Python 3.9+
- Anaconda or Miniconda (recommended)

### Installation

```bash
conda create --name frat_env python=3.9
conda activate frat_env
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Usage

1. **Enter a location** — place name (e.g. `Korail, Dhaka`) or latitude/longitude coordinates
2. **Set radius** — 500–2500 m search area around the point
3. **Adjust weights** — drag sliders for Density, Access, Water, Height, Hazard; the display shows live normalised percentages
4. **AHP (optional)** — open the AHP expander to derive weights scientifically from pairwise comparisons
5. **Advanced Options (optional)** — filter road types, enable wind modifier, place a hypothetical fire station, choose weighted-sum vs geometric-mean aggregation, enable Monte Carlo
6. **Click Analyse** — results appear across 8 tabs

### Risk Bands

| Band | Score Range | Interpretation |
|---|---|---|
| Low | 0.00 – 0.25 | Within acceptable risk tolerance |
| Medium | 0.25 – 0.50 | Monitor; consider targeted interventions |
| High | 0.50 – 0.75 | Infrastructure review recommended |
| Critical | 0.75 – 1.00 | Immediate prioritisation required |

---

## Dependencies

| Package | Purpose |
|---|---|
| `osmnx` | OSM road network and building feature download |
| `geopandas` | Spatial data manipulation |
| `folium` + `streamlit-folium` | Interactive map rendering |
| `networkx` | Graph routing for travel time |
| `scipy` | Spatial interpolation |
| `jenkspy` | Jenks Natural Breaks classification |
| `libpysal` + `esda` | Spatial weights and Moran's I |
| `branca` | Folium colormap |
| `geopy` | Place name geocoding |
| `simplekml` | KMZ export |

---

## Limitations

- OSM data quality varies by region; rural or unmapped areas may have sparse building/road coverage
- Travel time estimates use OSM speed tags or road-type defaults; they are not real-time traffic-aware
- The 50×50 m grid cell size is fixed; very small study areas may produce few cells
- Combustibility scoring relies on OSM building material/use tags, which are often absent in developing-country datasets

---

## Author

Avishek Adhikari
avishek.jidpus@gmail.com
