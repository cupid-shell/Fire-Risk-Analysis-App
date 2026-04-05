import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import folium
import folium.plugins
import branca.colormap as cm
import networkx as nx

# Prevent osmnx from hanging indefinitely on slow/unresponsive OSM servers
ox.settings.requests_timeout = 180

DEFAULT_ROAD_TYPES = ['primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified']

# ── IMPROVEMENT 1: Absolute threshold normalization (NFPA/ISO anchors) ────────
# NFPA 1710: fire stations should reach any point within 4 min travel time (240s)
# ISO/TR 13387: hydrant spacing ≤ 150m in dense urban areas; ≥ 500m = max water risk
# UN-Habitat: > 120 buildings per hectare = critically dense; cell is 50x50m = 0.25ha → 30 buildings = critical
ACCESS_MAX_S      = 240.0   # seconds travel time — beyond this = full access risk (NFPA 1710, 4 min)
WATER_MAX_M       = 500.0   # metres to nearest water — beyond this = full water risk (ISO/TR 13387)
DENSITY_MAX       = 30.0    # buildings per 50×50 m cell — at or above this = full density risk (UN-Habitat)
HAZARD_MAX_M      = 50.0    # metres to hazard — at or closer than this = full hazard risk
HAZARD_SAFE_M     = 2000.0  # metres — beyond this = zero hazard risk
HEIGHT_MAX_FLOORS = 10.0    # floors — at or above this = full height risk

# Legacy alias kept for any callers that may reference the old name
ACCESS_MAX_M = ACCESS_MAX_S


def _clip_norm(series, lo, hi):
    """Linearly scales series from [lo, hi] to [0, 1], clamped."""
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return ((series - lo) / (hi - lo)).clip(0, 1)


# ── IMPROVEMENT 2: AHP weight derivation with Consistency Ratio check ─────────
def ahp_weights(pairwise_matrix):
    """
    Derives normalised weights from a 5×5 AHP pairwise comparison matrix.
    Returns (weights_dict, consistency_ratio).
    Saaty random index values for n=1..10:
    RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    CR < 0.10 is considered acceptable (Saaty, 1980).
    """
    A = np.array(pairwise_matrix, dtype=float)
    n = A.shape[0]
    # Column-normalise
    col_sums = A.sum(axis=0)
    norm = A / col_sums
    # Priority vector = row means
    weights = norm.mean(axis=1)
    # Consistency check
    Aw = A @ weights
    lambda_max = (Aw / weights).mean()
    CI = (lambda_max - n) / (n - 1)
    RI_values = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
    RI = RI_values[n - 1]
    CR = CI / RI if RI > 0 else 0.0
    factor_names = ['density', 'access', 'water', 'height', 'hazard']
    return dict(zip(factor_names, weights)), CR


def get_geospatial_data(location_point, distance, target_crs, road_types=None):
    """
    Downloads and projects all necessary data, handling cases where features are not found.
    """
    if road_types is None:
        road_types = DEFAULT_ROAD_TYPES

    print(f"Fetching data within {distance}m of {location_point}...")
    try:
        graph = ox.graph_from_point(location_point, dist=distance, network_type='all')
        # IMPROVEMENT 3: add travel time attributes
        graph = ox.add_edge_speeds(graph)
        graph = ox.add_edge_travel_times(graph)
        graph_proj = ox.project_graph(graph, to_crs=target_crs)
        edges = ox.graph_to_gdfs(graph_proj, nodes=False)
        accessible_roads = edges[edges['highway'].isin(road_types)]
    except Exception as e:
        print(f"Could not download road network. Error: {e}")
        return [None] * 5

    try:
        tags = {"building": True}
        buildings = ox.features_from_point(location_point, tags, dist=distance)
        buildings_proj = buildings.to_crs(target_crs)
    except Exception:
        print("Warning: No buildings found in the area. Creating empty buildings dataset.")
        buildings_proj = gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)

    try:
        water_tags = {"natural": "water", "amenity": "fire_hydrant"}
        water_sources = ox.features_from_point(location_point, water_tags, dist=distance)
        water_sources_proj = water_sources.to_crs(target_crs)
    except Exception:
        print("Warning: No water sources found in the area. Creating empty water sources dataset.")
        water_sources_proj = gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)

    try:
        station_tags = {"amenity": "fire_station"}
        fire_stations = ox.features_from_point(location_point, station_tags, dist=distance)
        fire_stations_proj = fire_stations.to_crs(target_crs)
    except Exception:
        print("Warning: No fire stations found in the area. Creating empty fire stations dataset.")
        fire_stations_proj = gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)

    print("Data fetching complete!")
    return graph_proj, buildings_proj, accessible_roads, water_sources_proj, fire_stations_proj


# ── IMPROVEMENT 4: GFA + Building material combustibility ─────────────────────
def calculate_density_grid(buildings_proj, cell_size=50):
    if buildings_proj.empty:
        print("No buildings to process for density grid.")
        return gpd.GeoDataFrame(columns=['geometry', 'n_buildings'], crs=buildings_proj.crs)
    print("Calculating density grid...")
    xmin, ymin, xmax, ymax = buildings_proj.total_bounds
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            x1, y1 = x0 - cell_size, y0 + cell_size
            grid_cells.append(Polygon([(x0, y0), (x0, y1), (x1, y1), (x1, y0)]))
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=buildings_proj.crs)

    # Compute footprint area and GFA
    b = buildings_proj.copy()
    b['footprint_area'] = b.geometry.area
    if 'levels' in b.columns:
        b['gfa'] = b['footprint_area'] * pd.to_numeric(b['levels'], errors='coerce').fillna(1)
    elif 'building:levels' in b.columns:
        b['gfa'] = b['footprint_area'] * pd.to_numeric(b['building:levels'], errors='coerce').fillna(1)
    else:
        b['gfa'] = b['footprint_area']

    buildings_centroids = b.copy()
    buildings_centroids['geometry'] = buildings_centroids.geometry.centroid
    buildings_centroids = buildings_centroids.reset_index(drop=True)

    merged = gpd.sjoin(buildings_centroids, grid, how='left', predicate='within')
    merged['n_buildings'] = 1

    agg_dict = {'n_buildings': 'sum', 'gfa': 'sum'}
    if 'combustibility' in merged.columns:
        agg_dict['combustibility'] = 'mean'

    building_counts = merged.groupby('index_right').agg(agg_dict).reset_index()
    building_counts = building_counts.rename(columns={
        'gfa': 'total_gfa',
        'combustibility': 'avg_combustibility',
    })

    grid = grid.merge(building_counts, left_index=True, right_on='index_right', how='left')
    grid = grid.drop(columns=['index_right'])
    grid['n_buildings'] = grid['n_buildings'].fillna(0)
    grid['total_gfa'] = grid['total_gfa'].fillna(0)

    print("Density calculation complete!")
    return grid


def calculate_combustibility(buildings_proj):
    """
    Assigns a combustibility score per building based on OSM building:material tag.
    Higher score = more combustible.
    Reference: SFPE Handbook of Fire Protection Engineering.
    """
    MATERIAL_SCORES = {
        'wood': 1.0, 'timber': 1.0, 'bamboo': 1.0,
        'brick': 0.4, 'concrete': 0.2, 'stone': 0.1,
        'metal': 0.3, 'steel': 0.2, 'glass': 0.3,
    }
    b = buildings_proj.copy()
    mat_col = 'building:material' if 'building:material' in b.columns else None
    if mat_col:
        b['combustibility'] = b[mat_col].str.lower().map(MATERIAL_SCORES).fillna(0.5)
    else:
        b['combustibility'] = 0.5  # unknown = medium

    # Also factor in building use type
    USE_SCORES = {
        'industrial': 0.9, 'warehouse': 0.9, 'retail': 0.7,
        'commercial': 0.6, 'residential': 0.5, 'apartments': 0.5,
        'hotel': 0.6, 'school': 0.6, 'hospital': 0.5,
        'church': 0.4, 'government': 0.3, 'yes': 0.5,
    }
    if 'building' in b.columns:
        use_score = b['building'].str.lower().map(USE_SCORES).fillna(0.5)
        b['combustibility'] = (b['combustibility'] * 0.6 + use_score * 0.4)

    return b


# ── IMPROVEMENT 3: Travel TIME instead of travel DISTANCE ─────────────────────
def calculate_travel_risk(buildings_proj, fire_stations_proj, graph_proj, extra_station=None):
    print("Calculating travel time risk from fire stations...")
    stations = fire_stations_proj.copy()

    if extra_station is not None:
        target_crs = stations.crs if not stations.empty else buildings_proj.crs
        extra_pt = gpd.GeoDataFrame(
            geometry=[Point(extra_station[1], extra_station[0])],
            crs="EPSG:4326"
        ).to_crs(target_crs)
        stations = pd.concat([stations, extra_pt], ignore_index=True) if not stations.empty else extra_pt

    if stations.empty:
        print("No fire stations found. Assigning high-risk travel time.")
        buildings_with_risk = buildings_proj.copy()
        buildings_with_risk['travel_time'] = 480  # 8 minutes = well beyond NFPA 4-min standard
        return buildings_with_risk

    fire_station_points = stations.copy()
    fire_station_points['geometry'] = fire_station_points.geometry.centroid
    station_nodes = ox.nearest_nodes(graph_proj, fire_station_points.geometry.x, fire_station_points.geometry.y)
    building_nodes = ox.nearest_nodes(graph_proj, buildings_proj.geometry.centroid.x, buildings_proj.geometry.centroid.y)
    travel_times = []
    for b_node in building_nodes:
        try:
            path_time = min([nx.shortest_path_length(graph_proj, b_node, s_node, weight='travel_time') for s_node in np.atleast_1d(station_nodes)])
            travel_times.append(path_time)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            travel_times.append(480)
    buildings_with_risk = buildings_proj.copy()
    buildings_with_risk['travel_time'] = travel_times
    print("Travel risk calculation complete!")
    return buildings_with_risk


def calculate_water_risk(buildings_proj, water_sources_proj):
    print("Calculating water proximity risk...")
    if water_sources_proj.empty:
        buildings_with_risk = buildings_proj.copy()
        buildings_with_risk['distance_to_water'] = 1000
        return buildings_with_risk
    combined_water = water_sources_proj.union_all()
    buildings_with_risk = buildings_proj.copy()
    buildings_with_risk['distance_to_water'] = buildings_with_risk.geometry.apply(lambda geom: geom.distance(combined_water))
    print("Water risk calculation complete!")
    return buildings_with_risk


# ── IMPROVEMENT 10: Weighted hazard scoring by type ───────────────────────────
def calculate_hazard_risk(buildings_proj, hazards_proj):
    """
    Proximity-weighted hazard risk using type-specific danger multipliers.
    Score = Σ(multiplier_i / max(distance_i, 10)²) — inverse square decay.

    Multipliers based on fire hazard potential:
    - fuel/gas station: 3.0 (flammable fuel storage, SFPE Handbook)
    - hospital: 2.0 (critical infrastructure, hazardous materials)
    - marketplace: 1.5 (dense combustible goods, high occupancy)
    - school: 1.0 (high occupancy, evacuation complexity)

    Reference: SFPE Handbook of Fire Protection Engineering, 5th Ed.
    """
    b = buildings_proj.copy()
    if hazards_proj is None or hazards_proj.empty:
        b['hazard_score'] = 0.0
        b['distance_to_hazard'] = 2000.0
        return b

    HAZARD_WEIGHTS = {
        'fuel':        3.0,
        'hospital':    2.0,
        'marketplace': 1.5,
        'school':      1.0,
    }

    hazards = hazards_proj.copy()
    hazards['geometry'] = hazards.geometry.centroid

    # Get amenity type
    if 'amenity' in hazards.columns:
        hazards['h_weight'] = hazards['amenity'].map(HAZARD_WEIGHTS).fillna(1.0)
    else:
        hazards['h_weight'] = 1.0

    def _hazard_score(geom):
        pt = geom.centroid
        total = 0.0
        min_dist = float('inf')
        for _, h in hazards.iterrows():
            d = max(pt.distance(h.geometry), 10.0)
            total += h['h_weight'] / (d ** 2) * 10000  # scale factor
            min_dist = min(min_dist, d)
        return total, min_dist if min_dist != float('inf') else 2000.0

    scores = b.geometry.apply(_hazard_score)
    b['hazard_score'] = scores.apply(lambda x: x[0])
    b['distance_to_hazard'] = scores.apply(lambda x: x[1])

    return b


def calculate_occupancy_modifier(buildings_proj):
    """Estimates occupancy from floor count × footprint area. Used to boost density risk."""
    b = buildings_proj.copy()
    levels = b['levels'] if 'levels' in b.columns else pd.Series(1.0, index=b.index)
    b['occupancy_proxy'] = (levels * b.geometry.area / 25.0).clip(lower=1)
    return b


def calculate_road_width_modifier(density_grid, accessible_roads):
    """Cells with fewer-lane roads get a narrow-road access penalty."""
    grid = density_grid.copy()
    grid['avg_lanes'] = 2.0  # safe default for all cells
    if accessible_roads is None or accessible_roads.empty:
        return grid
    try:
        def _parse_lanes(v):
            if isinstance(v, list): v = v[0]
            try: return float(v)
            except: return 1.0

        lanes_vals = accessible_roads['lanes'].apply(_parse_lanes) \
            if 'lanes' in accessible_roads.columns \
            else pd.Series(1.0, index=accessible_roads.index)

        road_pts = gpd.GeoDataFrame(
            {'lanes_num': lanes_vals.values},
            geometry=accessible_roads.geometry.centroid,
            crs=accessible_roads.crs
        ).reset_index(drop=True)

        # Join road centroids to grid polygons using only the geometry column
        # so geopandas creates a clean 'index_right' mapped to grid.index
        grid_geom = grid[['geometry']].copy()
        joined = gpd.sjoin(road_pts, grid_geom, how='inner', predicate='within')

        if 'index_right' in joined.columns and not joined.empty:
            avg = joined.groupby('index_right')['lanes_num'].mean()
            grid.loc[avg.index, 'avg_lanes'] = avg.values
    except Exception as e:
        print(f"Road width modifier skipped: {e}")
    return grid


def apply_wind_modifier(risk_grid, wind_direction_deg):
    """
    Applies a directional wind multiplier to final_risk.
    Cells downwind of the area centroid get up to +20% risk boost.
    wind_direction_deg: compass bearing the wind is blowing FROM (0=N, 90=E, 180=S, 270=W).
    """
    import math
    if wind_direction_deg is None:
        return risk_grid
    grid = risk_grid.copy()
    cx = grid.centroid.x.mean()
    cy = grid.centroid.y.mean()
    # Direction wind blows TOWARD (opposite of FROM)
    wind_rad = math.radians((wind_direction_deg + 180) % 360)
    dx = math.sin(wind_rad)
    dy = math.cos(wind_rad)
    # Dot product of (cell - centroid) with wind vector gives downwind score
    grid['_cx'] = grid.centroid.x - cx
    grid['_cy'] = grid.centroid.y - cy
    dot = grid['_cx'] * dx + grid['_cy'] * dy
    max_dot = dot.abs().max()
    if max_dot > 0:
        wind_factor = (dot / max_dot).clip(0, 1) * 0.20  # up to +20%
        grid['final_risk'] = (grid['final_risk'] + wind_factor * grid['final_risk']).clip(0, 1)
    grid = grid.drop(columns=['_cx', '_cy'])
    return grid


def calculate_height_risk(buildings_proj):
    """Extracts building:levels from OSM tags and returns per-building height risk."""
    b = buildings_proj.copy()
    if 'building:levels' in b.columns:
        b['levels'] = pd.to_numeric(b['building:levels'], errors='coerce').fillna(1)
    else:
        b['levels'] = 1
    return b


# ── IMPROVEMENT 5: Jenks Natural Breaks risk bands ────────────────────────────
def classify_risk_bands(grid, n_classes=4):
    """
    Classifies final_risk into bands using Jenks Natural Breaks (Fisher-Jenks),
    a statistically optimal classification that minimises within-class variance.
    Falls back to fixed quantile thresholds if insufficient data.
    Reference: Jenks & Caspall (1971), Cartographica.
    """
    scores = grid['final_risk'].dropna().values

    if len(scores) >= n_classes * 2:
        try:
            import jenkspy
            breaks = jenkspy.jenks_breaks(scores, n_classes=n_classes)
            # breaks has n_classes+1 values: [min, break1, break2, break3, max]
            def _band_jenks(s):
                if s <= breaks[1]: return 'Low'
                if s <= breaks[2]: return 'Medium'
                if s <= breaks[3]: return 'High'
                return 'Critical'
            grid = grid.copy()
            grid['risk_band'] = grid['final_risk'].apply(_band_jenks)
            grid['jenks_breaks'] = str([round(b, 3) for b in breaks])
            return grid
        except ImportError:
            pass  # fall back below

    # Fallback: quantile-based (better than fixed thresholds)
    q25, q50, q75 = grid['final_risk'].quantile([0.25, 0.5, 0.75])

    def _band_q(s):
        if s >= q75: return 'Critical'
        if s >= q50: return 'High'
        if s >= q25: return 'Medium'
        return 'Low'

    grid = grid.copy()
    grid['risk_band'] = grid['final_risk'].apply(_band_q)
    return grid


# ── IMPROVEMENT 9: OSM data completeness flag ─────────────────────────────────
def assess_data_completeness(buildings_proj, density_grid):
    """
    Estimates OSM data completeness per grid cell.
    Uses building footprint coverage ratio as a proxy:
    if a cell has buildings but very small total footprint area relative to
    the cell area, data may be incomplete.

    Returns grid with 'completeness_score' (0=low confidence, 1=high confidence)
    and 'data_warning' flag.
    """
    grid = density_grid.copy()
    cell_area = 50 * 50  # 2500 m² per cell

    # Compute total footprint area per cell
    try:
        b = buildings_proj.copy()
        b['footprint_area'] = b.geometry.area
        b_centroids = b.copy()
        b_centroids['geometry'] = b_centroids.geometry.centroid
        b_centroids = b_centroids.reset_index(drop=True)
        grid_geom = grid[['geometry']].copy()
        joined = gpd.sjoin(b_centroids[['geometry', 'footprint_area']], grid_geom, how='left', predicate='within')
        if 'index_right' in joined.columns:
            total_fp = joined.groupby('index_right')['footprint_area'].sum()
            grid['total_footprint_area'] = 0.0
            grid.loc[total_fp.index, 'total_footprint_area'] = total_fp.values
        else:
            grid['total_footprint_area'] = 0.0
    except Exception:
        grid['total_footprint_area'] = 0.0

    # Coverage ratio: what fraction of the cell is covered by building footprints?
    grid['coverage_ratio'] = (grid['total_footprint_area'] / cell_area).clip(0, 1)

    # Completeness heuristic:
    # - Cell with buildings but coverage < 1% → possibly sparse OSM data
    # - Cell with 0 buildings but in the middle of the study area → possibly missing data
    # - Well-mapped urban areas typically have 5-40% building coverage
    grid['completeness_score'] = 1.0  # default: high confidence

    # Low coverage in populated cells suggests missing data
    low_coverage = (grid['n_buildings'] > 0) & (grid['coverage_ratio'] < 0.01)
    grid.loc[low_coverage, 'completeness_score'] = 0.4

    # Zero buildings in interior cells (surrounded by cells with buildings) may be unmapped
    # Simple proxy: cells with 0 buildings get moderate uncertainty
    grid.loc[grid['n_buildings'] == 0, 'completeness_score'] = 0.6

    grid['data_warning'] = grid['completeness_score'] < 0.5

    return grid


# ── IMPROVEMENT 8: Moran's I spatial autocorrelation ─────────────────────────
def calculate_spatial_autocorrelation(risk_grid):
    """
    Computes Moran's I statistic for spatial autocorrelation of final_risk scores.
    A positive Moran's I indicates spatial clustering of high-risk zones.
    Reference: Moran (1950), Biometrika; Anselin (1995) for LISA.
    Returns dict with moran_i, p_value, interpretation.
    """
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran

        grid = risk_grid.copy()
        # Build spatial weights matrix (Queen contiguity)
        w = Queen.from_dataframe(grid, silence_warnings=True)
        w.transform = 'r'  # row-standardise

        moran = Moran(grid['final_risk'], w)

        if moran.p_sim < 0.01:
            sig = "highly significant (p < 0.01)"
        elif moran.p_sim < 0.05:
            sig = "significant (p < 0.05)"
        else:
            sig = "not significant (p ≥ 0.05)"

        if moran.I > 0.3:
            interp = "Strong spatial clustering — high-risk zones form concentrated hotspot clusters."
        elif moran.I > 0.1:
            interp = "Moderate spatial clustering — some tendency for high-risk zones to neighbour each other."
        elif moran.I > 0:
            interp = "Weak spatial clustering."
        else:
            interp = "Spatial dispersion — high-risk zones are scattered rather than clustered."

        return {
            'moran_i': round(float(moran.I), 4),
            'p_value': round(float(moran.p_sim), 4),
            'significance': sig,
            'interpretation': interp,
            'z_score': round(float(moran.z_sim), 3),
        }
    except ImportError:
        return {'error': 'Install libpysal and esda: pip install libpysal esda'}
    except Exception as e:
        return {'error': str(e)}


# ── IMPROVEMENT 6: Weighted geometric mean aggregation + IMPROVEMENT 1 norms ──
def calculate_composite_risk(density_grid, buildings_with_all_risks, weights, aggregation='weighted_sum'):
    print("Calculating composite risk score...")
    grid = density_grid.reset_index(drop=True)

    # Determine which travel column is present (travel_time is new, travel_distance legacy)
    travel_col = 'travel_time' if 'travel_time' in buildings_with_all_risks.columns else 'travel_distance'

    if buildings_with_all_risks.empty or travel_col not in buildings_with_all_risks.columns:
        grid['avg_travel_time'] = 0
        grid['avg_distance_water'] = 0
    else:
        buildings_centroids = buildings_with_all_risks.copy()
        buildings_centroids['geometry'] = buildings_centroids.geometry.centroid
        buildings_centroids = buildings_centroids.reset_index(drop=True)
        merged = gpd.sjoin(buildings_centroids, grid, how='left', predicate='within')
        agg_cols = [travel_col, 'distance_to_water']
        for col in ['levels', 'distance_to_hazard', 'occupancy_proxy', 'hazard_score', 'combustibility']:
            if col in buildings_centroids.columns:
                agg_cols.append(col)
        avg_risks_in_grid = merged.groupby('index_right')[agg_cols].mean()
        grid = grid.join(avg_risks_in_grid)
        grid = grid.reset_index(drop=True)  # prevent duplicate-label errors from join

    # Rename for unified reference
    if travel_col in grid.columns:
        grid = grid.rename(columns={travel_col: 'avg_travel_time'})
    if 'distance_to_water' in grid.columns:
        grid = grid.rename(columns={'distance_to_water': 'avg_distance_water'})
    if 'hazard_score' in grid.columns:
        grid = grid.rename(columns={'hazard_score': 'avg_hazard_score'})
    if 'combustibility' in grid.columns:
        grid = grid.rename(columns={'combustibility': 'avg_combustibility'})

    for col in ['avg_travel_time', 'avg_distance_water']:
        if col not in grid.columns:
            grid[col] = 0
        grid[col] = grid[col].fillna(0)

    # IMPROVEMENT 1: Absolute threshold normalization
    # Density risk — prefer GFA if available
    if 'total_gfa' in grid.columns and grid['total_gfa'].sum() > 0:
        grid['density_risk'] = _clip_norm(grid['total_gfa'], 0, DENSITY_MAX * 50 * 50)  # GFA in m²
    else:
        grid['density_risk'] = _clip_norm(grid['n_buildings'], 0, DENSITY_MAX)

    # Occupancy modifier — boosts density risk by up to 50% for high-occupancy buildings
    if 'occupancy_proxy' in grid.columns:
        grid['occupancy_proxy'] = grid['occupancy_proxy'].fillna(1)
        occ_norm = _clip_norm(grid['occupancy_proxy'], grid['occupancy_proxy'].min(), grid['occupancy_proxy'].max())
        grid['density_risk'] = (grid['density_risk'] * (1 + occ_norm * 0.5)).clip(0, 1)

    # IMPROVEMENT 4: Apply combustibility modifier
    if 'avg_combustibility' in grid.columns:
        grid['density_risk'] = (grid['density_risk'] * (1 + grid['avg_combustibility'].fillna(0.5) * 0.5)).clip(0, 1)

    # Access risk — use absolute threshold (NFPA 1710: 240 seconds)
    grid['access_risk'] = _clip_norm(grid['avg_travel_time'], 0, ACCESS_MAX_S)

    # Road width modifier — narrow roads boost access risk by up to 30%
    if 'avg_lanes' in grid.columns:
        grid['avg_lanes'] = grid['avg_lanes'].fillna(2)
        lane_penalty = _clip_norm(1.0 / grid['avg_lanes'].clip(lower=0.5), 0, 2)
        grid['access_risk'] = (grid['access_risk'] * (1 + lane_penalty * 0.3)).clip(0, 1)

    # Water risk — use absolute threshold (ISO/TR 13387: 500m)
    grid['water_risk'] = _clip_norm(grid['avg_distance_water'], 0, WATER_MAX_M)

    # Height risk — use reference of 10 floors = max
    if 'levels' in grid.columns and weights.get('height', 0) > 0:
        grid['levels'] = grid['levels'].fillna(1)
        grid['height_risk'] = _clip_norm(grid['levels'], 1, HEIGHT_MAX_FLOORS)
    else:
        grid['height_risk'] = 0.0

    # IMPROVEMENT 10: Hazard risk — use weighted score if available, else distance-based
    if 'avg_hazard_score' in grid.columns and weights.get('hazard', 0) > 0:
        p95 = grid['avg_hazard_score'].quantile(0.95)
        if p95 > 0:
            grid['hazard_risk'] = _clip_norm(grid['avg_hazard_score'].fillna(0), 0, p95)
        else:
            grid['hazard_risk'] = 0.0
    elif 'distance_to_hazard' in grid.columns and weights.get('hazard', 0) > 0:
        grid['distance_to_hazard'] = grid['distance_to_hazard'].fillna(HAZARD_SAFE_M)
        grid['hazard_risk'] = _clip_norm(-grid['distance_to_hazard'], -HAZARD_SAFE_M, -HAZARD_MAX_M)
    else:
        grid['hazard_risk'] = 0.0

    # IMPROVEMENT 6: Aggregation method
    if aggregation == 'geometric_mean':
        # Weighted geometric mean: product of (risk_i ^ weight_i)
        # Handles factor interactions — poor performance on any single factor pulls down total
        risk_factors = {
            'density': grid['density_risk'],
            'access':  grid['access_risk'],
            'water':   grid['water_risk'],
            'height':  grid['height_risk'],
            'hazard':  grid['hazard_risk'],
        }
        log_sum = sum(
            weights.get(k, 0) * np.log(v.clip(lower=1e-6))
            for k, v in risk_factors.items()
        )
        grid['final_risk'] = np.exp(log_sum).clip(0, 1)
    else:
        # Default: weighted arithmetic sum
        grid['final_risk'] = (
            grid['density_risk'] * weights.get('density', 0) +
            grid['access_risk']  * weights.get('access',  0) +
            grid['water_risk']   * weights.get('water',   0) +
            grid['height_risk']  * weights.get('height',  0) +
            grid['hazard_risk']  * weights.get('hazard',  0)
        ).clip(0, 1)

    # IMPROVEMENT 5: Jenks Natural Breaks risk bands
    grid = classify_risk_bands(grid)

    return grid


# ── IMPROVEMENT 7: Monte Carlo uncertainty bounds ─────────────────────────────
def monte_carlo_uncertainty(density_grid, buildings_with_all_risks, weights, n_simulations=300, weight_std=0.10):
    """
    Runs n_simulations of composite risk with weights perturbed by Gaussian noise
    (std = weight_std fraction of each weight). Returns per-cell mean and std dev of final_risk.
    This quantifies how sensitive risk scores are to weight uncertainty.
    Reference: Saltelli et al. (2008), Global Sensitivity Analysis.
    """
    from copy import deepcopy

    all_risks = []
    factor_names = list(weights.keys())
    w_array = np.array([weights[k] for k in factor_names])

    for _ in range(n_simulations):
        # Perturb weights with Gaussian noise, renormalise
        noise = np.random.normal(0, weight_std, size=len(w_array))
        w_perturbed = np.abs(w_array + noise * w_array)
        w_sum = w_perturbed.sum()
        if w_sum > 0:
            w_perturbed /= w_sum
        w_dict = dict(zip(factor_names, w_perturbed))

        sim_grid = calculate_composite_risk(
            deepcopy(density_grid), deepcopy(buildings_with_all_risks), w_dict
        )
        all_risks.append(sim_grid['final_risk'].values)

    all_risks = np.array(all_risks)  # shape: (n_simulations, n_cells)
    result_grid = density_grid.copy()
    result_grid['risk_mean'] = all_risks.mean(axis=0)
    result_grid['risk_std']  = all_risks.std(axis=0)
    result_grid['risk_cv']   = (result_grid['risk_std'] / result_grid['risk_mean'].clip(lower=1e-6)).clip(0, 1)
    # High CV = unstable/uncertain zone; low CV = stably classified
    return result_grid[['geometry', 'risk_mean', 'risk_std', 'risk_cv']]


def save_footprints_map(buildings, graph, filepath):
    print(f"Saving building footprints map to {filepath}...")
    fig, ax = ox.plot_graph(graph, show=False, close=True, bgcolor='#060606', edge_color='grey', edge_linewidth=0.2, node_size=0)
    if not buildings.empty:
        buildings.plot(ax=ax, color='cyan', alpha=0.7)
    ax.set_title('Building Footprints', color='white')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='#060606')
    plt.close(fig)

def save_roads_map(accessible_roads, graph, filepath):
    print(f"Saving road network map to {filepath}...")
    fig, ax = ox.plot_graph(graph, show=False, close=True, bgcolor='#060606', edge_color='#333333', edge_linewidth=0.5, node_size=0)
    if not accessible_roads.empty:
        accessible_roads.plot(ax=ax, color='yellow', linewidth=1)
    ax.set_title('Accessible Road Network', color='white')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='#060606')
    plt.close(fig)

def generate_static_risk_map(grid, graph):
    print("Generating static risk map...")
    points_for_interpolation = grid[grid['n_buildings'] > 0]
    if len(points_for_interpolation) >= 4:
        x, y, z = points_for_interpolation.centroid.x, points_for_interpolation.centroid.y, points_for_interpolation['final_risk']
        xmin, ymin, xmax, ymax = ox.graph_to_gdfs(graph, nodes=False).total_bounds
        grid_x, grid_y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        try:
            grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic', fill_value=0)
            fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0, bgcolor='#060606', edge_color='white', edge_linewidth=0.2)
            cax = ax.imshow(grid_z.T, extent=[xmin, xmax, ymin, ymax], origin='lower', cmap='magma', alpha=0.9)
            cbar = fig.colorbar(cax, ax=ax); cbar.set_label('Composite Risk Score', color='white')
            cbar.ax.yaxis.set_tick_params(color='white'); plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            ax.set_title('Composite Fire Risk Map (Interpolated)', color='white')
        except Exception as e:
            print(f"Interpolation failed: {e}, falling back to grid plot.")
            fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0, bgcolor='#060606', edge_color='white', edge_linewidth=0.1)
            grid.plot(column='final_risk', ax=ax, cmap='magma', alpha=0.7, legend=True)
            ax.set_title('Composite Fire Risk Map (Grid)', color='white')
    else:
        print("Not enough data for smooth interpolation. Generating a grid map instead.")
        fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0, bgcolor='#060606', edge_color='white', edge_linewidth=0.1)
        grid.plot(column='final_risk', ax=ax, cmap='magma', alpha=0.7, legend=True)
        ax.set_title('Composite Fire Risk Map (Grid)', color='white')
    ax.axis('off')
    fig.savefig('final_risk_map.png', dpi=300, bbox_inches='tight', pad_inches=0, facecolor='#060606')
    plt.close(fig)

def generate_interactive_risk_map(grid, fire_stations_proj=None, water_sources_proj=None, extra_station=None, accessible_roads=None):
    print("Generating interactive risk map...")
    grid_wgs84 = grid.to_crs("EPSG:4326")
    map_center = [grid_wgs84.centroid.y.mean(), grid_wgs84.centroid.x.mean()]

    # tiles=None so BOTH base layers appear properly in LayerControl
    m = folium.Map(location=map_center, zoom_start=15, tiles=None)

    # Base layer 1 – Street map (default / shown first)
    folium.TileLayer('CartoDB positron', name='Street Map', overlay=False).add_to(m)

    # Base layer 2 – Satellite
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
        overlay=False,
    ).add_to(m)

    colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=grid['final_risk'].min(), vmax=grid['final_risk'].max())
    colormap.caption = 'Composite Fire Risk Score'
    tooltip_fields = ['n_buildings', 'final_risk', 'risk_band'] if 'risk_band' in grid_wgs84.columns else ['n_buildings', 'final_risk']
    tooltip_aliases = ['Buildings:', 'Risk Score:', 'Risk Band:'] if 'risk_band' in grid_wgs84.columns else ['Buildings:', 'Risk Score:']
    style_function = lambda x: {'fillColor': colormap(x['properties']['final_risk']), 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7}
    risk_layer = folium.FeatureGroup(name="Risk Grid")
    folium.GeoJson(grid_wgs84, style_function=style_function, tooltip=folium.features.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases)).add_to(risk_layer)
    risk_layer.add_to(m)

    # Water source markers
    if water_sources_proj is not None and not water_sources_proj.empty:
        water_layer = folium.FeatureGroup(name="Water Sources")
        water_wgs84 = water_sources_proj.to_crs("EPSG:4326")
        for _, row in water_wgs84.iterrows():
            pt = row.geometry.centroid
            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=5,
                color='blue',
                fill=True,
                fill_color='#3399ff',
                fill_opacity=0.8,
                tooltip="Water Source"
            ).add_to(water_layer)
        water_layer.add_to(m)

    # Fire station markers
    if fire_stations_proj is not None and not fire_stations_proj.empty:
        station_layer = folium.FeatureGroup(name="Fire Stations (OSM)")
        stations_wgs84 = fire_stations_proj.to_crs("EPSG:4326")
        for _, row in stations_wgs84.iterrows():
            pt = row.geometry.centroid
            folium.Marker(
                location=[pt.y, pt.x],
                icon=folium.Icon(color='red', icon='fire', prefix='fa'),
                tooltip="Fire Station"
            ).add_to(station_layer)
        station_layer.add_to(m)

    # Hypothetical station marker
    if extra_station is not None:
        hyp_layer = folium.FeatureGroup(name="Hypothetical Station")
        folium.Marker(
            location=[extra_station[0], extra_station[1]],
            icon=folium.Icon(color='orange', icon='fire', prefix='fa'),
            tooltip="Hypothetical Fire Station"
        ).add_to(hyp_layer)
        hyp_layer.add_to(m)

    # Distance rings around all stations
    all_station_coords = []
    if fire_stations_proj is not None and not fire_stations_proj.empty:
        s_wgs84 = fire_stations_proj.to_crs("EPSG:4326")
        for _, row in s_wgs84.iterrows():
            pt = row.geometry.centroid
            all_station_coords.append((pt.y, pt.x))
    if extra_station is not None:
        all_station_coords.append((extra_station[0], extra_station[1]))

    if all_station_coords:
        ring_layer = folium.FeatureGroup(name="Response Distance Rings")
        ring_colors = {500: '#00cc44', 1000: '#ffaa00', 1500: '#ff4444'}
        for lat, lon in all_station_coords:
            for radius_m, color in ring_colors.items():
                folium.Circle(
                    location=[lat, lon],
                    radius=radius_m,
                    color=color,
                    fill=False,
                    weight=1.5,
                    dash_array='6',
                    tooltip=f"{radius_m}m response zone"
                ).add_to(ring_layer)
        ring_layer.add_to(m)

    # Heatmap layer
    heat_data = [
        [row.geometry.centroid.y, row.geometry.centroid.x, row['final_risk']]
        for _, row in grid_wgs84[grid_wgs84['n_buildings'] > 0].iterrows()
    ]
    if heat_data:
        heatmap_layer = folium.FeatureGroup(name="Risk Heatmap (smooth)")
        folium.plugins.HeatMap(heat_data, radius=20, blur=15, min_opacity=0.3).add_to(heatmap_layer)
        heatmap_layer.add_to(m)

    # Road risk overlay
    if accessible_roads is not None and not accessible_roads.empty:
        try:
            roads_wgs84 = accessible_roads.to_crs("EPSG:4326").copy()
            grid_pts = grid_wgs84[['geometry', 'final_risk']].copy()
            grid_pts['geometry'] = grid_pts.geometry.centroid
            road_sample = roads_wgs84.iloc[::max(1, len(roads_wgs84)//400)]  # cap at ~400 segments
            road_mids = road_sample.copy()
            road_mids['geometry'] = road_mids.geometry.centroid
            joined = gpd.sjoin_nearest(road_mids[['geometry']], grid_pts, how='left')
            road_sample = road_sample.copy()
            road_sample['road_risk'] = joined['final_risk'].values
            road_cmap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=0, vmax=1)
            road_layer = folium.FeatureGroup(name="Road Risk Overlay")
            for _, row in road_sample.iterrows():
                risk = float(row.get('road_risk') or 0)
                try:
                    coords = [(pt[1], pt[0]) for pt in row.geometry.coords]
                    folium.PolyLine(coords, color=road_cmap(risk), weight=3, opacity=0.75,
                                   tooltip=f"Road Risk: {risk:.3f}").add_to(road_layer)
                except Exception:
                    pass
            road_layer.add_to(m)
        except Exception as e:
            print(f"Road overlay skipped: {e}")

    m.add_child(colormap)
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    # Save HTML (used for the download button)
    m.save('interactive_risk_map.html')

    # Return the map object so Streamlit can render it natively via st_folium
    return m

def main(place_name, location_point, search_distance, weights, road_types=None, extra_station=None, wind_direction=None):
    """Orchestrates the entire analysis and map generation. Returns the final risk grid."""
    gdf_wgs84 = gpd.GeoDataFrame(geometry=[Point(location_point[1], location_point[0])], crs="EPSG:4326")
    target_crs = gdf_wgs84.estimate_utm_crs()
    graph, buildings, accessible_roads, water_sources, fire_stations = get_geospatial_data(location_point, search_distance, target_crs, road_types)
    if graph is None or buildings.empty:
        raise ValueError("No buildings or road network found. Cannot generate analysis.")
    buildings_with_heights = calculate_height_risk(buildings)
    buildings_with_comb = calculate_combustibility(buildings_with_heights)
    buildings_with_occ = calculate_occupancy_modifier(buildings_with_comb)
    density_grid = calculate_density_grid(buildings_with_occ)
    density_grid = assess_data_completeness(buildings_with_occ, density_grid)
    buildings_with_travel_risk = calculate_travel_risk(buildings_with_occ, fire_stations, graph, extra_station)
    buildings_with_water = calculate_water_risk(buildings_with_travel_risk, water_sources)
    buildings_with_all_risks = calculate_hazard_risk(buildings_with_water, None)
    density_grid = calculate_road_width_modifier(density_grid, accessible_roads)
    final_risk_grid = calculate_composite_risk(density_grid, buildings_with_all_risks, weights)
    final_risk_grid = apply_wind_modifier(final_risk_grid, wind_direction)
    save_footprints_map(buildings, graph, 'building_footprints.png')
    save_roads_map(accessible_roads, graph, 'road_network.png')
    generate_static_risk_map(final_risk_grid, graph)
    generate_interactive_risk_map(final_risk_grid, fire_stations, water_sources, extra_station, accessible_roads)
    return final_risk_grid

if __name__ == "__main__":
    test_place = "Korail, Dhaka"
    test_point = (23.774, 90.405)
    search_dist = 1000
    test_weights = {"density": 0.33, "access": 0.33, "water": 0.34}
    main(test_place, test_point, search_dist, test_weights)
