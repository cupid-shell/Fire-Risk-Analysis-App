"""
FRAT – Fire Risk Analysis Tool  |  FastAPI endpoint
Run with:  uvicorn api:app --reload
Docs at:   http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import geopandas as gpd
from shapely.geometry import Point

from fire_risk_analyzer import (
    get_geospatial_data,
    calculate_density_grid,
    calculate_height_risk,
    calculate_occupancy_modifier,
    calculate_hazard_risk,
    calculate_travel_risk,
    calculate_water_risk,
    calculate_composite_risk,
    calculate_road_width_modifier,
    apply_wind_modifier,
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
    critical_zones: int
    high_zones:   int
    medium_zones: int
    low_zones:    int
    top_5_hotspots: list[ZoneSummary]

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "FRAT Fire Risk Analysis API"}

@app.post("/analyze", response_model=AnalysisResponse, summary="Run fire risk analysis")
def analyze(req: AnalysisRequest):
    try:
        location_point = (req.latitude, req.longitude)
        gdf = gpd.GeoDataFrame(geometry=[Point(req.longitude, req.latitude)], crs="EPSG:4326")
        target_crs = gdf.estimate_utm_crs()

        weights = {
            "density": req.density_weight, "access": req.access_weight,
            "water":   req.water_weight,   "height": req.height_weight,
            "hazard":  req.hazard_weight,
        }
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        graph, buildings, accessible_roads, water_sources, fire_stations = get_geospatial_data(
            location_point, req.radius_m, target_crs, DEFAULT_ROAD_TYPES
        )
        if graph is None or buildings.empty:
            raise HTTPException(status_code=404, detail="No buildings or road network found at this location.")

        # Fetch hazards separately
        try:
            import osmnx as ox
            hz = ox.features_from_point(location_point,
                {"amenity": ["fuel", "hospital", "school"]}, dist=req.radius_m)
            hazards = hz.to_crs(target_crs)
        except Exception:
            hazards = gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)

        density_grid = calculate_density_grid(buildings)
        density_grid = calculate_road_width_modifier(density_grid, accessible_roads)
        bh  = calculate_height_risk(buildings)
        bho = calculate_occupancy_modifier(bh)
        bt  = calculate_travel_risk(bho, fire_stations, graph)
        bw  = calculate_water_risk(bt, water_sources)
        ba  = calculate_hazard_risk(bw, hazards)
        frg = calculate_composite_risk(density_grid, ba, weights)
        if req.wind_direction is not None:
            frg = apply_wind_modifier(frg, req.wind_direction)

        bc = frg['risk_band'].value_counts()
        top5 = frg.nlargest(5, 'final_risk').to_crs("EPSG:4326")

        return AnalysisResponse(
            avg_risk=round(float(frg['final_risk'].mean()), 4),
            max_risk=round(float(frg['final_risk'].max()),  4),
            n_buildings=len(buildings),
            n_stations=len(fire_stations) if not fire_stations.empty else 0,
            n_water=len(water_sources)   if not water_sources.empty  else 0,
            critical_zones=int(bc.get('Critical', 0)),
            high_zones=int(bc.get('High',     0)),
            medium_zones=int(bc.get('Medium',   0)),
            low_zones=int(bc.get('Low',      0)),
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
