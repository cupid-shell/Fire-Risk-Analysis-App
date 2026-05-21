"""
FRAT – Fire Risk Analysis Tool  |  FastAPI endpoint
Run with:  uvicorn api:app --reload
Docs at:   http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import geopandas as gpd
from shapely.geometry import Point

from fire_risk_analyzer import (
    run_analysis_pipeline,
    DEFAULT_ROAD_TYPES,
)

app = FastAPI(
    title="Fire Risk Analysis API",
    description="Geospatial fire-risk scoring for urban settlements using OpenStreetMap data.",
    version="2.0",
)

class AnalysisRequest(BaseModel):
    latitude:        float = Field(..., example=23.774, description="Latitude of the analysis centre")
    longitude:       float = Field(..., example=90.405, description="Longitude of the analysis centre")
    radius_m:        int   = Field(1000, ge=100, le=5000, description="Search radius in metres")
    density_weight:  float = Field(0.30, ge=0, le=1)
    access_weight:   float = Field(0.25, ge=0, le=1)
    water_weight:    float = Field(0.20, ge=0, le=1)
    height_weight:   float = Field(0.10, ge=0, le=1)
    hazard_weight:   float = Field(0.15, ge=0, le=1)
    wind_direction:  Optional[float] = Field(None, description="Wind direction in degrees FROM (0=N, 90=E)")
    road_types:      Optional[List[str]] = Field(None, description="List of road types to include")
    aggregation_method: Optional[str] = Field("weighted_sum", description="weighted_sum or geometric_mean")
    run_uncertainty: Optional[bool] = Field(False, description="Run Monte Carlo uncertainty analysis")

class ZoneSummary(BaseModel):
    lat:        float
    lon:        float
    risk_score: float
    risk_band:  str
    n_buildings: int

class AnalysisResponse(BaseModel):
    avg_risk:     float
    max_risk:     float
    n_buildings:  int
    n_stations:   int
    n_water:      int
    n_hazards:    int
    critical_zones: int
    high_zones:   int
    medium_zones: int
    low_zones:    int
    data_completeness_warning: bool
    moran_i:      Optional[float] = None
    moran_p_value: Optional[float] = None
    moran_significance: Optional[str] = None
    uncertainty_mean_risk: Optional[float] = None
    uncertainty_avg_std: Optional[float] = None
    recommendations: List[str]
    top_5_hotspots: List[ZoneSummary]

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "FRAT Fire Risk Analysis API"}

@app.post("/analyze", response_model=AnalysisResponse, summary="Run fire risk analysis")
def analyze(req: AnalysisRequest):
    try:
        location_point = (req.latitude, req.longitude)

        # Build weights dict
        weights = {
            "density": req.density_weight, "access": req.access_weight,
            "water":   req.water_weight,   "height": req.height_weight,
            "hazard":  req.hazard_weight,
        }
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            raise HTTPException(status_code=400, detail="All weights are zero — cannot run analysis.")

        road_types = req.road_types if req.road_types else DEFAULT_ROAD_TYPES
        agg_method = req.aggregation_method if req.aggregation_method else "weighted_sum"

        # Execute unified pipeline
        res = run_analysis_pipeline(
            location_point=location_point,
            search_distance=req.radius_m,
            weights=weights,
            road_types=road_types,
            wind_direction=req.wind_direction,
            aggregation_method=agg_method,
            run_uncertainty=req.run_uncertainty or False,
            generate_maps=False
        )

        frg = res["final_risk_grid"]
        mc_grid = res["mc_grid"]
        moran_res = res["moran_result"]

        bc = frg['risk_band'].value_counts()
        top5 = frg.nlargest(5, 'final_risk').to_crs("EPSG:4326")

        # Determine completeness warning
        completeness_warning = bool((frg['completeness_score'] < 0.5).any()) if 'completeness_score' in frg.columns else False

        # Extract Moran's I details
        m_i = moran_res.get('moran_i') if moran_res and 'error' not in moran_res else None
        m_p = moran_res.get('p_value') if moran_res and 'error' not in moran_res else None
        m_sig = moran_res.get('significance') if moran_res and 'error' not in moran_res else None

        # Extract MC details
        mc_mean = float(mc_grid['risk_mean'].mean()) if mc_grid is not None else None
        mc_std = float(mc_grid['risk_std'].mean()) if mc_grid is not None else None

        return AnalysisResponse(
            avg_risk=round(float(frg['final_risk'].mean()), 4),
            max_risk=round(float(frg['final_risk'].max()),  4),
            n_buildings=res["n_buildings"],
            n_stations=res["n_stations"],
            n_water=res["n_water"],
            n_hazards=res["n_hazards"],
            critical_zones=int(bc.get('Critical', 0)),
            high_zones=int(bc.get('High',     0)),
            medium_zones=int(bc.get('Medium',   0)),
            low_zones=int(bc.get('Low',      0)),
            data_completeness_warning=completeness_warning,
            moran_i=m_i,
            moran_p_value=m_p,
            moran_significance=m_sig,
            uncertainty_mean_risk=mc_mean,
            uncertainty_avg_std=mc_std,
            recommendations=res["recs"],
            top_5_hotspots=[
                ZoneSummary(
                    lat=round(float(row.geometry.centroid.y), 5),
                    lon=round(float(row.geometry.centroid.x), 5),
                    risk_score=round(float(row['final_risk']), 4),
                    risk_band=row['risk_band'],
                    n_buildings=int(row['n_buildings']),
                ) for _, row in top5.iterrows()
            ],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
