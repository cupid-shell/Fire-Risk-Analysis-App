import streamlit as st
from geopy.geocoders import Nominatim
import streamlit.components.v1 as components
from streamlit_folium import st_folium as _st_folium
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import json, os, io, zipfile, datetime
from concurrent.futures import ThreadPoolExecutor

from fire_risk_analyzer import (
    calculate_density_grid,
    calculate_height_risk,
    calculate_occupancy_modifier,
    calculate_combustibility,
    calculate_hazard_risk,
    calculate_travel_risk,
    calculate_water_risk,
    calculate_composite_risk,
    calculate_road_width_modifier,
    apply_wind_modifier,
    assess_data_completeness,
    calculate_spatial_autocorrelation,
    monte_carlo_uncertainty,
    ahp_weights,
    save_footprints_map,
    save_roads_map,
    generate_static_risk_map,
    generate_interactive_risk_map,
    DEFAULT_ROAD_TYPES,
    ACCESS_MAX_S,
    WATER_MAX_M,
    DENSITY_MAX,
    HAZARD_MAX_M,
    HAZARD_SAFE_M,
    HEIGHT_MAX_FLOORS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="FRAT – Fire Risk Analysis Tool",
    page_icon="🔥",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tighten sidebar padding */
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
/* Metric card tweaks */
div[data-testid="metric-container"] {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 8px;
    padding: 10px 14px;
}
/* Tab font size */
button[data-baseweb="tab"] { font-size: 0.9rem; }
/* Welcome card */
.welcome-card {
    background: linear-gradient(135deg, #1e1e2e 0%, #2a1a2e 100%);
    border: 1px solid #ff4444;
    border-radius: 12px;
    padding: 32px 40px;
    text-align: center;
    margin-top: 60px;
}
.welcome-card h2 { color: #ff6666; margin-bottom: 8px; }
.welcome-card p  { color: #aaa; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ─────────────────────────────────────────────────────
_defaults = {
    'maps_generated': False, 'final_risk_grid': None, 'location_query': '',
    'scenario_a': None, 'scenario_b': None,
    'last_location_point': None, 'last_accessible_roads': None,
    'last_weights': None, 'last_n_buildings': 0,
    'last_n_stations': 0, 'last_n_water': 0, 'last_n_hazards': 0,
    'last_recs': [], 'folium_map': None,
    'mc_grid': None, 'moran_result': None,
    'use_ahp': False, 'ahp_w': None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── URL params ─────────────────────────────────────────────────────────────────
_qp = st.query_params
_default_name   = _qp.get("loc",    "Korail, Dhaka")
_default_radius = int(_qp.get("r",  1000))
_default_method = _qp.get("method", "Search by Name")

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔥 FRAT")
    st.caption("Fire Risk Analysis Tool")

    # Guide
    with st.expander("👋 How to use FRAT", expanded=False):
        st.markdown("""
**4 simple steps:**

1. **Enter a location** — place name or coordinates.
2. **Set radius & weights** — area size and factor importance.
3. **Advanced Options** *(optional)* — road filter, wind, hypothetical station.
4. **Click Analyse** — results appear in tabs on the right.

**Risk bands:**
🟢 Low · 🟡 Medium · 🔴 High · ⛔ Critical
        """)

    st.markdown("---")

    # ── Location input ─────────────────────────────────────────────────────────
    st.markdown("#### 📍 Location")
    input_method = st.radio("Input method:", ('Search by Name', 'Enter Coordinates'),
                            index=0 if _default_method == 'Search by Name' else 1,
                            label_visibility="collapsed")
    location_query = ""
    location_point = None

    if input_method == 'Search by Name':
        location_query = st.text_input("Place name", _default_name, placeholder="e.g. Korail, Dhaka")
    else:
        _dlat = float(_qp.get("lat", 23.774))
        _dlon = float(_qp.get("lon", 90.405))
        lc1, lc2 = st.columns(2)
        with lc1:
            lat = st.number_input("Latitude",  value=_dlat, min_value=-90.0,  max_value=90.0,
                                  step=0.001, format="%.5f")
        with lc2:
            lon = st.number_input("Longitude", value=_dlon, min_value=-180.0, max_value=180.0,
                                  step=0.001, format="%.5f")
        location_point = (lat, lon)
        location_query = f"{lat:.5f}, {lon:.5f}"

    search_dist = st.slider("Search radius (m)", 500, 2500, _default_radius, 50)

    st.markdown("---")

    # ── Weight sliders ─────────────────────────────────────────────────────────
    st.markdown("#### ⚖ Risk Factor Weights")
    st.caption("Adjust importance of each factor (auto-normalised to 100%).")

    density_weight = st.slider("🏘 Density",  0, 100, 30,
                               help="Building density and occupancy load.")
    access_weight  = st.slider("🚒 Access",   0, 100, 25,
                               help="Fire station travel distance via road network.")
    water_weight   = st.slider("💧 Water",    0, 100, 20,
                               help="Proximity to water sources and hydrants.")
    height_weight  = st.slider("🏗 Height",   0, 100, 10,
                               help="Building floor count from OSM data.")
    hazard_weight  = st.slider("⚠ Hazard",   0, 100, 15,
                               help="Proximity to gas stations, hospitals, schools.")

    _total_w = density_weight + access_weight + water_weight + height_weight + hazard_weight
    if _total_w > 0:
        st.caption(
            f"Normalised → "
            f"D:{density_weight/_total_w:.0%} "
            f"A:{access_weight/_total_w:.0%} "
            f"W:{water_weight/_total_w:.0%} "
            f"H:{height_weight/_total_w:.0%} "
            f"Z:{hazard_weight/_total_w:.0%}"
        )
    else:
        st.warning("All weights are zero.")

    # ── IMPROVEMENT 2: AHP Weight Derivation ──────────────────────────────────
    with st.expander("🔬 AHP Weight Derivation (Scientific)"):
        st.caption("Analytic Hierarchy Process (Saaty, 1980). Compare each pair of factors. 1=equal, 3=moderately more, 5=strongly more, 7=very strongly, 9=extremely more important.")
        use_ahp = st.checkbox("Use AHP-derived weights instead of sliders")
        if use_ahp:
            factors = ['Density', 'Access', 'Water', 'Height', 'Hazard']
            n = 5
            ahp_matrix = [[1.0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    val = st.select_slider(
                        f"{factors[i]} vs {factors[j]}",
                        options=[1/9, 1/7, 1/5, 1/3, 1.0, 3.0, 5.0, 7.0, 9.0],
                        value=1.0,
                        format_func=lambda x: f"{x:.2g}",
                        key=f"ahp_{i}_{j}"
                    )
                    ahp_matrix[i][j] = val
                    ahp_matrix[j][i] = 1.0 / val

            ahp_w, cr = ahp_weights(ahp_matrix)

            if cr < 0.10:
                st.success(f"Consistency Ratio: {cr:.3f} (acceptable < 0.10)")
            else:
                st.warning(f"Consistency Ratio: {cr:.3f} — revise comparisons (should be < 0.10)")

            st.caption("Derived weights: " + " | ".join(f"{k.capitalize()}: {v:.1%}" for k, v in ahp_w.items()))
            st.session_state.use_ahp = True
            st.session_state.ahp_w = ahp_w
        else:
            st.session_state.use_ahp = False
            st.session_state.ahp_w = None

    st.markdown("---")

    # ── Advanced Options ───────────────────────────────────────────────────────
    ALL_ROAD_TYPES = ['primary','secondary','tertiary','residential','service',
                      'unclassified','trunk','living_street','pedestrian','track']

    with st.expander("⚙ Advanced Options"):
        st.markdown("**Road Type Filter**")
        selected_road_types = st.multiselect("Included road types:", ALL_ROAD_TYPES,
                                             default=DEFAULT_ROAD_TYPES)

        st.markdown("**Wind Direction**")
        st.caption("Downwind cells get up to +20% risk boost.")
        enable_wind = st.checkbox("Enable wind modifier")
        wind_direction = None
        if enable_wind:
            wind_direction = st.slider("Wind from (°)", 0, 359, 0)
            _compass = {0:"N",45:"NE",90:"E",135:"SE",180:"S",225:"SW",270:"W",315:"NW"}
            _near = min(_compass, key=lambda k: abs(k - wind_direction))
            st.caption(f"Wind from: **{_compass[_near]}**")

        # IMPROVEMENT 6: Risk Aggregation Method
        st.markdown("**Risk Aggregation Method**")
        aggregation_method = st.radio(
            "Method:",
            ['weighted_sum', 'geometric_mean'],
            format_func=lambda x: '+ Weighted Sum (additive)' if x == 'weighted_sum' else 'x Geometric Mean (synergistic)',
            help="Geometric mean penalises poor performance on any single factor more strongly than the additive model. Reference: UNDRR Global Risk Assessment Framework."
        )

        # IMPROVEMENT 7: Monte Carlo uncertainty
        run_uncertainty = st.checkbox("Run Monte Carlo uncertainty analysis (slower, ~300 simulations)", value=False)

        st.markdown("**Hypothetical Fire Station**")
        add_station = st.checkbox("Simulate extra station")
        extra_station = None
        if add_station:
            sc1, sc2 = st.columns(2)
            with sc1:
                s_lat = st.number_input("Lat", value=23.774, min_value=-90.0, max_value=90.0,
                                        step=0.001, format="%.5f", key="s_lat")
            with sc2:
                s_lon = st.number_input("Lon", value=90.405, min_value=-180.0, max_value=180.0,
                                        step=0.001, format="%.5f", key="s_lon")
            extra_station = (s_lat, s_lon)
            st.caption(f"Station at ({s_lat:.5f}, {s_lon:.5f})")

    st.markdown("---")

    # ── Action buttons ─────────────────────────────────────────────────────────
    analyse_clicked = st.button("🔍 Analyse Location", type="primary", use_container_width=True)

    if st.button("↺ Reset", use_container_width=True, help="Clear all results and reset inputs"):
        for k in ['maps_generated', 'final_risk_grid', 'location_query',
                  'scenario_a', 'scenario_b', 'last_location_point',
                  'last_accessible_roads', 'last_weights', 'last_recs',
                  'last_n_buildings', 'last_n_stations', 'last_n_water', 'last_n_hazards',
                  'mc_grid', 'moran_result', 'use_ahp', 'ahp_w']:
            st.session_state[k] = _defaults.get(k)
        st.rerun()

    st.markdown("---")
    st.caption("Created by Avishek Adhikari\navishek.jidpus@gmail.com")


# ══════════════════════════════════════════════════════════════════════════════
#  OSM FETCH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _target_epsg(location_point):
    gdf = gpd.GeoDataFrame(geometry=[Point(location_point[1], location_point[0])], crs="EPSG:4326")
    return gdf.estimate_utm_crs().to_epsg()

@st.cache_data(show_spinner=False)
def fetch_all_osm_parallel(location_point, distance, road_types_tuple, target_epsg):
    """Fetches road network, buildings, water, fire stations, and hazards all in parallel."""
    import osmnx as ox
    import geopandas as gpd
    ox.settings.requests_timeout = 180

    def _graph():
        try:
            g  = ox.graph_from_point(location_point, dist=distance, network_type='all')
            # IMPROVEMENT 3: add travel time attributes before projecting
            g = ox.add_edge_speeds(g)
            g = ox.add_edge_travel_times(g)
            gp = ox.project_graph(g, to_crs=target_epsg)
            edges = ox.graph_to_gdfs(gp, nodes=False)
            ar = edges[edges['highway'].isin(list(road_types_tuple))]
            return gp, ar
        except Exception as e:
            print(f"Graph error: {e}"); return None, None

    def _buildings():
        try:
            b = ox.features_from_point(location_point, {"building": True}, dist=distance)
            return b.to_crs(target_epsg)
        except: return gpd.GeoDataFrame(columns=['geometry'], crs=target_epsg)

    def _water():
        try:
            w = ox.features_from_point(location_point,
                {"natural": "water", "amenity": "fire_hydrant"}, dist=distance)
            return w.to_crs(target_epsg)
        except: return gpd.GeoDataFrame(columns=['geometry'], crs=target_epsg)

    def _stations():
        try:
            s = ox.features_from_point(location_point, {"amenity": "fire_station"}, dist=distance)
            return s.to_crs(target_epsg)
        except: return gpd.GeoDataFrame(columns=['geometry'], crs=target_epsg)

    def _hazards():
        try:
            h = ox.features_from_point(location_point,
                {"amenity": ["fuel", "hospital", "school", "marketplace"]}, dist=distance)
            return h.to_crs(target_epsg)
        except: return gpd.GeoDataFrame(columns=['geometry'], crs=target_epsg)

    with ThreadPoolExecutor(max_workers=5) as ex:
        fg = ex.submit(_graph)
        fb = ex.submit(_buildings)
        fw = ex.submit(_water)
        fs = ex.submit(_stations)
        fh = ex.submit(_hazards)
        graph_proj, accessible_roads = fg.result()
        buildings    = fb.result()
        water        = fw.result()
        stations     = fs.result()
        hazards      = fh.result()

    return graph_proj, accessible_roads, buildings, water, stations, hazards


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA — title
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔥 Fire Risk Analysis Tool")
st.caption("Geospatial fire-risk scoring for urban settlements using OpenStreetMap data.")

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS (triggered from sidebar button)
# ══════════════════════════════════════════════════════════════════════════════
if analyse_clicked:
    # IMPROVEMENT 2: use AHP weights if enabled
    if st.session_state.get('use_ahp') and st.session_state.get('ahp_w'):
        weights = st.session_state.ahp_w
    else:
        weights = {
            "density": density_weight / 100, "access": access_weight / 100,
            "water":   water_weight   / 100, "height": height_weight / 100,
            "hazard":  hazard_weight  / 100,
        }
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        else:
            st.error("All weights are zero — cannot run analysis."); st.stop()

    if not selected_road_types:
        st.error("Select at least one road type in Advanced Options."); st.stop()

    if input_method == 'Search by Name':
        if location_query:
            try:
                geo = Nominatim(user_agent="fire_risk_app", timeout=10).geocode(location_query)
                if geo:
                    location_point = (geo.latitude, geo.longitude)
                else:
                    st.error("Location not found. Try a more specific name."); st.stop()
            except Exception as e:
                st.error(f"Geocoding failed: {e}"); st.stop()
        else:
            st.warning("Enter a location name in the sidebar."); st.stop()

    if location_point:
        progress = st.progress(0, text="Starting analysis…")

        try:
            progress.progress(5,  text="Computing coordinate reference system…")
            tepsg = _target_epsg(location_point)

            progress.progress(10, text="Fetching OSM data in parallel (roads · buildings · water · stations · hazards)…")
            graph, accessible_roads, buildings, water_sources, fire_stations, hazards = fetch_all_osm_parallel(
                location_point, search_dist, tuple(selected_road_types), tepsg
            )

            if graph is None:
                st.error("Could not download road network. Check your internet or try a different location.")
                progress.empty(); st.stop()
            if buildings.empty:
                st.error("No buildings found. Try a larger radius.")
                progress.empty(); st.stop()

            n_buildings = len(buildings)
            n_stations  = len(fire_stations)  if not fire_stations.empty  else 0
            n_water     = len(water_sources)   if not water_sources.empty  else 0
            n_hazards   = len(hazards)         if not hazards.empty        else 0

            if n_stations == 0 and extra_station is None:
                st.warning("⚠ No fire stations found in OSM. Access risk defaults to maximum. Add a hypothetical station in Advanced Options.")
            if n_water   == 0:
                st.warning("⚠ No water sources found in OSM. Water risk defaults to maximum.")
            if n_hazards == 0:
                st.info("ℹ No hazard points found — hazard weight has no effect.")

            progress.progress(65, text="Calculating density grid…")
            bh  = calculate_height_risk(buildings)
            bho = calculate_occupancy_modifier(bh)
            # IMPROVEMENT 4: combustibility scoring
            bho = calculate_combustibility(bho)
            density_grid = calculate_density_grid(bho)

            # IMPROVEMENT 9: OSM data completeness flag
            density_grid = assess_data_completeness(bho, density_grid)

            progress.progress(69, text="Applying road-width modifier…")
            density_grid = calculate_road_width_modifier(density_grid, accessible_roads)

            # IMPROVEMENT 3: travel time
            progress.progress(75, text="Calculating travel time to fire stations…")
            bt = calculate_travel_risk(bho, fire_stations, graph, extra_station)

            progress.progress(80, text="Calculating water proximity risk…")
            bw = calculate_water_risk(bt, water_sources)

            progress.progress(83, text="Calculating hazard proximity risk…")
            ba = calculate_hazard_risk(bw, hazards)

            progress.progress(86, text="Computing composite risk scores…")
            # IMPROVEMENT 6: pass aggregation method
            final_risk_grid = calculate_composite_risk(density_grid, ba, weights, aggregation=aggregation_method)
            if wind_direction is not None:
                final_risk_grid = apply_wind_modifier(final_risk_grid, wind_direction)

            # IMPROVEMENT 7: Monte Carlo uncertainty
            if run_uncertainty:
                progress.progress(88, text="Running Monte Carlo uncertainty analysis (300 simulations)…")
                mc_grid = monte_carlo_uncertainty(density_grid, ba, weights)
                st.session_state.mc_grid = mc_grid
            else:
                st.session_state.mc_grid = None

            # IMPROVEMENT 8: Moran's I spatial autocorrelation
            progress.progress(92, text="Computing spatial autocorrelation (Moran's I)…")
            moran_result = calculate_spatial_autocorrelation(final_risk_grid)
            st.session_state.moran_result = moran_result

            progress.progress(93, text="Rendering static maps…")
            save_footprints_map(buildings, graph, 'building_footprints.png')
            save_roads_map(accessible_roads, graph, 'road_network.png')

            progress.progress(96, text="Generating risk heatmap…")
            generate_static_risk_map(final_risk_grid, graph)

            progress.progress(98, text="Generating interactive map…")
            folium_map = generate_interactive_risk_map(
                final_risk_grid, fire_stations, water_sources, extra_station, accessible_roads
            )
            st.session_state.folium_map = folium_map

            # Build recommendations
            bc   = final_risk_grid['risk_band'].value_counts()
            _avg = final_risk_grid['final_risk'].mean()
            _crit = int(bc.get('Critical', 0))
            _high = int(bc.get('High',     0))
            _max  = final_risk_grid['final_risk'].max()
            recs = []
            if _crit > 0:
                recs.append(f"⛔ **{_crit} Critical zone(s) detected.** Prioritise fire safety inspections and ensure emergency access routes are clear.")
            if _high > 0:
                recs.append(f"🔴 **{_high} High-risk zone(s) found.** Consider deploying additional fire hydrants or temporary water tanks.")
            if _avg > 0.5:
                recs.append("📍 **Overall average risk is high.** This area may benefit from a new fire station or major infrastructure review.")
            if _max >= 0.75:
                c1 = final_risk_grid.nlargest(1, 'final_risk').to_crs("EPSG:4326")
                recs.append(f"🗺 **Highest-risk cell at ({c1.centroid.y.values[0]:.4f}, {c1.centroid.x.values[0]:.4f}).** Field verification recommended.")
            if 'access_risk' in final_risk_grid.columns and final_risk_grid['access_risk'].mean() > 0.6:
                recs.append("🚒 **Fire station access risk is high.** Road improvements or an additional station would significantly reduce response times.")
            if 'water_risk' in final_risk_grid.columns and final_risk_grid['water_risk'].mean() > 0.6:
                recs.append("💧 **Water source proximity is low.** Installing more hydrants or ensuring water access could reduce suppression difficulty.")
            if not recs:
                recs.append("✅ Risk profile is within acceptable ranges. Continue routine monitoring.")

            # Save history
            os.makedirs("history", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sn = "".join(c if c.isalnum() else "_" for c in location_query)[:30]
            hp = f"history/{ts}_{sn}"
            final_risk_grid.to_crs("EPSG:4326").to_file(f"{hp}.geojson", driver="GeoJSON")
            with open(f"{hp}_meta.json", "w") as mf:
                json.dump({
                    "place": location_query, "timestamp": ts,
                    "radius_m": search_dist, "weights": {k: round(v, 3) for k, v in weights.items()},
                    "n_buildings": n_buildings, "n_stations": n_stations,
                    "n_water": n_water, "n_hazards": n_hazards,
                    "avg_risk": round(float(final_risk_grid['final_risk'].mean()), 4),
                    "max_risk": round(float(final_risk_grid['final_risk'].max()), 4),
                }, mf, indent=2)

            progress.progress(100, text="Analysis complete!")
            st.success(f"✅ Analysis complete for **{location_query}**!")
            st.toast("🔥 Fire risk analysis finished!", icon="✅")

            # Persist to session state
            st.session_state.maps_generated        = True
            st.session_state.final_risk_grid       = final_risk_grid
            st.session_state.location_query        = location_query
            st.session_state.last_location_point   = location_point
            st.session_state.last_accessible_roads = accessible_roads
            st.session_state.last_weights          = weights
            st.session_state.last_n_buildings      = n_buildings
            st.session_state.last_n_stations       = n_stations
            st.session_state.last_n_water          = n_water
            st.session_state.last_n_hazards        = n_hazards
            st.session_state.last_recs             = recs

            snap = {
                'label': f"{location_query} | r={search_dist}m | D:{weights['density']:.0%} A:{weights['access']:.0%} W:{weights['water']:.0%} H:{weights['height']:.0%} Z:{weights['hazard']:.0%}",
                'grid': final_risk_grid,
            }
            st.session_state.scenario_b = st.session_state.scenario_a
            st.session_state.scenario_a = snap

        except Exception as e:
            st.error(f"Analysis error: {e}"); st.exception(e); progress.empty()


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS (tab layout)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.maps_generated and st.session_state.final_risk_grid is not None:

    frg  = st.session_state.final_risk_grid
    lq   = st.session_state.location_query or "location"
    lpt  = st.session_state.last_location_point
    lar  = st.session_state.last_accessible_roads
    recs = st.session_state.last_recs or []
    bc   = frg['risk_band'].value_counts()

    tab_summary, tab_maps, tab_interactive, tab_hotspots, tab_export, tab_history, tab_batch, tab_compare = st.tabs([
        "📊 Summary",
        "🗺 Maps",
        "🌍 Interactive",
        "🔥 Hotspots",
        "📥 Export",
        "📂 History",
        "📋 Batch",
        "⚖ Compare",
    ])

    # ── TAB 1: Summary ─────────────────────────────────────────────────────────
    with tab_summary:
        st.subheader(f"Analysis Summary — {lq}")

        # Row 1: risk scores
        m1, m2, m3 = st.columns(3)
        with m1: st.metric("Average Risk Score", f"{frg['final_risk'].mean():.3f}")
        with m2: st.metric("Maximum Risk Score", f"{frg['final_risk'].max():.3f}")
        with m3:
            top1 = frg.nlargest(1, 'final_risk').to_crs("EPSG:4326")
            st.metric("Highest-Risk Cell",
                      f"{top1['risk_band'].values[0]}",
                      f"({top1.centroid.y.values[0]:.4f}, {top1.centroid.x.values[0]:.4f})")

        st.markdown("")

        # Row 2: band counts
        b1, b2, b3, b4 = st.columns(4)
        with b1: st.metric("⛔ Critical Zones", int(bc.get('Critical', 0)))
        with b2: st.metric("🔴 High Zones",     int(bc.get('High',     0)))
        with b3: st.metric("🟡 Medium Zones",   int(bc.get('Medium',   0)))
        with b4: st.metric("🟢 Low Zones",      int(bc.get('Low',      0)))

        st.markdown("")

        # Row 3: data quality
        st.markdown("**Data Quality**")
        d1, d2, d3, d4 = st.columns(4)
        with d1: st.metric("Buildings",     st.session_state.last_n_buildings)
        with d2: st.metric("Fire Stations", st.session_state.last_n_stations)
        with d3: st.metric("Water Sources", st.session_state.last_n_water)
        with d4: st.metric("Hazard Points", st.session_state.last_n_hazards)

        # IMPROVEMENT 5: Jenks break thresholds
        if 'jenks_breaks' in frg.columns:
            st.caption(f"Risk band thresholds (Jenks Natural Breaks): {frg['jenks_breaks'].iloc[0]}")
        else:
            st.caption("Risk band thresholds: quantile-based")

        # IMPROVEMENT 9: OSM data completeness warning
        low_conf_cells = int((frg['completeness_score'] < 0.5).sum()) if 'completeness_score' in frg.columns else 0
        if low_conf_cells > 0:
            st.warning(f"Data Completeness Warning: {low_conf_cells} grid cells have low OSM coverage confidence. Results in these cells should be interpreted cautiously. Consider field verification.")

        st.markdown("---")

        # IMPROVEMENT 7: Monte Carlo uncertainty metrics
        if st.session_state.get('mc_grid') is not None:
            mc = st.session_state.mc_grid
            st.markdown("**Monte Carlo Uncertainty Analysis (n=300)**")
            u1, u2, u3 = st.columns(3)
            with u1: st.metric("Mean Risk (MC)", f"{mc['risk_mean'].mean():.3f}")
            with u2: st.metric("Avg Uncertainty (σ)", f"{mc['risk_std'].mean():.3f}")
            with u3:
                stable_pct = (mc['risk_cv'] < 0.15).mean() * 100
                st.metric("Stably Classified Cells", f"{stable_pct:.0f}%")
            st.caption("Cells with coefficient of variation > 15% are sensitive to weight assumptions and should be interpreted cautiously.")
            st.markdown("")

        # IMPROVEMENT 8: Moran's I spatial autocorrelation
        if st.session_state.get('moran_result'):
            mr = st.session_state.moran_result
            st.markdown("**Spatial Autocorrelation (Moran's I)**")
            if 'error' not in mr:
                mc1, mc2, mc3 = st.columns(3)
                with mc1: st.metric("Moran's I", mr['moran_i'])
                with mc2: st.metric("p-value", mr['p_value'])
                with mc3: st.metric("Z-score", mr['z_score'])
                st.caption(f"{mr['significance']} — {mr['interpretation']}")
            else:
                st.caption(f"Spatial autocorrelation unavailable: {mr['error']}")
            st.markdown("")

        st.markdown("---")
        st.subheader("💡 Recommendations")
        for r in recs:
            st.markdown(f"- {r}")

    # ── TAB 2: Maps ────────────────────────────────────────────────────────────
    with tab_maps:
        st.subheader("Analysis Maps")
        mc1, mc2 = st.columns(2)
        with mc1:
            st.image('building_footprints.png', caption='Building Footprints',
                     use_container_width=True)
        with mc2:
            st.image('road_network.png', caption='Accessible Road Network',
                     use_container_width=True)
        st.image('final_risk_map.png', caption='Static Composite Risk Heatmap',
                 use_container_width=True)

    # ── TAB 3: Interactive map ─────────────────────────────────────────────────
    with tab_interactive:
        st.subheader("Interactive Fire Risk Map")
        st.caption("Click the **layers icon ⊞** (top-right of map) to toggle layers on/off.")

        if os.path.exists('interactive_risk_map.html'):
            # Load and display the saved HTML map — always renders correctly
            with open('interactive_risk_map.html', 'r', encoding='utf-8') as _f:
                _map_html = _f.read()
            components.html(_map_html, height=580, scrolling=True)

            # Download button
            with open('interactive_risk_map.html', 'rb') as _ff:
                st.download_button(
                    "🔲 Download Map for Fullscreen View",
                    data=_ff.read(),
                    file_name="fire_risk_interactive_map.html",
                    mime="text/html",
                )

            # Click-to-place: small dedicated st_folium map
            st.markdown("---")
            st.markdown("**📍 Click-to-Place — Get Coordinates for a Hypothetical Station**")
            st.caption("Click a spot below, then paste the coordinates into sidebar → Advanced Options.")
            import folium as _fl
            _gw = frg.to_crs("EPSG:4326")
            _ctr = [_gw.centroid.y.mean(), _gw.centroid.x.mean()]
            _click_map = _fl.Map(location=_ctr, zoom_start=15, tiles="CartoDB positron")
            _fl.Marker(_ctr, tooltip="Area centre").add_to(_click_map)
            _md = _st_folium(_click_map, height=300, use_container_width=True,
                             returned_objects=["last_clicked"], key="click_place_map")
            if _md and _md.get("last_clicked"):
                _cl = _md["last_clicked"]
                st.success(f"📌 lat={_cl['lat']:.5f}, lon={_cl['lng']:.5f}")
        else:
            st.info("Run an analysis to generate the interactive map.")

    # ── TAB 4: Hotspots ────────────────────────────────────────────────────────
    with tab_hotspots:
        st.subheader("Top Risk Hotspots")
        st.caption("🟢 Low (0–0.25) · 🟡 Medium (0.25–0.50) · 🔴 High (0.50–0.75) · ⛔ Critical (0.75–1.0)")

        top_n = st.slider("Hotspots to display", 5, 30, 10)
        hs = frg.nlargest(top_n, 'final_risk').to_crs("EPSG:4326").copy()
        hs['Latitude']        = hs.centroid.y.round(5)
        hs['Longitude']       = hs.centroid.x.round(5)
        hs['Risk Score']      = hs['final_risk'].round(4)
        hs['Band']            = hs['risk_band']
        hs['Buildings']       = hs['n_buildings'].astype(int)
        hs['Density Risk']    = hs['density_risk'].round(3)
        hs['Access Risk']     = hs['access_risk'].round(3)
        hs['Water Risk']      = hs['water_risk'].round(3)
        _hcols = ['Latitude','Longitude','Risk Score','Band','Buildings',
                  'Density Risk','Access Risk','Water Risk']
        # IMPROVEMENT 3: show travel time column if available
        if 'avg_travel_time' in hs.columns:
            hs['Travel Time (s)'] = hs['avg_travel_time'].round(1)
            _hcols.append('Travel Time (s)')
        if 'hazard_risk' in hs.columns:
            hs['Hazard Risk'] = hs['hazard_risk'].round(3)
            _hcols.append('Hazard Risk')
        st.dataframe(hs[_hcols].reset_index(drop=True), use_container_width=True)

    # ── TAB 5: Export ──────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Export & Share")
        _lqsafe = lq.replace(' ', '_').replace(',', '')

        # Shareable URL
        st.markdown("**🔗 Shareable URL**")
        if input_method == 'Enter Coordinates' and lpt:
            _url = f"?method=Enter+Coordinates&lat={lpt[0]}&lon={lpt[1]}&r={search_dist}"
        else:
            _url = f"?method=Search+by+Name&loc={lq.replace(' ', '+')}&r={search_dist}"
        st.code(f"http://localhost:8501/{_url}", language=None)
        st.caption("Copy this link to share the analysis setup.")

        st.markdown("---")
        st.markdown("**⬇ Download Files**")

        ex1, ex2 = st.columns(2)
        ex3, ex4 = st.columns(2)

        # GeoJSON
        with ex1:
            st.download_button("📦 GeoJSON",
                               data=frg.to_crs("EPSG:4326").to_json(),
                               file_name=f"risk_grid_{_lqsafe}.geojson",
                               mime="application/geo+json",
                               use_container_width=True)

        # CSV
        with ex2:
            _gw2 = frg.to_crs("EPSG:4326").copy()
            _gw2['lat'] = _gw2.centroid.y; _gw2['lon'] = _gw2.centroid.x
            _ccols = ['lat','lon','n_buildings','density_risk','access_risk',
                      'water_risk','final_risk','risk_band']
            if 'hazard_risk' in _gw2.columns: _ccols.append('hazard_risk')
            st.download_button("📊 CSV",
                               data=_gw2[_ccols].to_csv(index=False),
                               file_name=f"risk_data_{_lqsafe}.csv",
                               mime="text/csv",
                               use_container_width=True)

        # Shapefile (zipped)
        with ex3:
            try:
                _shp_buf = io.BytesIO()
                with zipfile.ZipFile(_shp_buf, 'w', zipfile.ZIP_DEFLATED) as _zf:
                    import tempfile, shutil
                    _td = tempfile.mkdtemp()
                    _sp = os.path.join(_td, "risk_grid")
                    frg.to_crs("EPSG:4326").to_file(_sp + ".shp")
                    for _ext in ['.shp','.shx','.dbf','.prj','.cpg']:
                        _fp = _sp + _ext
                        if os.path.exists(_fp):
                            _zf.write(_fp, f"risk_grid{_ext}")
                    shutil.rmtree(_td)
                st.download_button("🗂 Shapefile (.zip)",
                                   data=_shp_buf.getvalue(),
                                   file_name=f"risk_shapefile_{_lqsafe}.zip",
                                   mime="application/zip",
                                   use_container_width=True)
            except Exception as _se:
                st.caption(f"Shapefile error: {_se}")

        # KMZ
        with ex4:
            try:
                import simplekml
                _kml = simplekml.Kml()
                _gkml = frg.to_crs("EPSG:4326")
                for _, _row in _gkml.iterrows():
                    _pt = _row.geometry.centroid
                    _pk = _kml.newpoint(name=f"Risk {_row['final_risk']:.3f}",
                                        coords=[(_pt.x, _pt.y)])
                    _pk.description = f"Band: {_row['risk_band']} | Buildings: {int(_row['n_buildings'])}"
                _kmz_buf = io.BytesIO()
                with zipfile.ZipFile(_kmz_buf, 'w') as _zf:
                    _zf.writestr("doc.kml", _kml.kml())
                st.download_button("🌐 KMZ (Google Earth)",
                                   data=_kmz_buf.getvalue(),
                                   file_name=f"risk_map_{_lqsafe}.kmz",
                                   mime="application/vnd.google-earth.kmz",
                                   use_container_width=True)
            except ImportError:
                st.caption("KMZ: install simplekml (`pip install simplekml`)")
            except Exception as _ke:
                st.caption(f"KMZ error: {_ke}")

        # HTML report
        st.markdown("---")
        try:
            with open('interactive_risk_map.html', 'r', encoding='utf-8') as _rf:
                _me = _rf.read()
            _t5 = frg.nlargest(5,'final_risk').to_crs("EPSG:4326").copy()
            _t5['lat'] = _t5.centroid.y.round(5); _t5['lon'] = _t5.centroid.x.round(5)
            _rows = "".join(
                f"<tr><td>{i+1}</td><td>{r['lat']}</td><td>{r['lon']}</td>"
                f"<td>{r['final_risk']:.4f}</td><td>{r['risk_band']}</td><td>{int(r['n_buildings'])}</td></tr>"
                for i,(_, r) in enumerate(_t5.iterrows()))
            _hbc = frg['risk_band'].value_counts()
            _recs_html = "".join(f"<li>{r}</li>" for r in recs)
            _html_rep = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>Fire Risk Report – {lq}</title><style>
body{{font-family:Arial,sans-serif;max-width:1200px;margin:auto;padding:20px;background:#111;color:#eee}}
h1{{color:#ff4444}}h2{{color:#ffaa00;margin-top:30px}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #444;padding:8px;text-align:center}}
th{{background:#222}}.stat{{display:inline-block;background:#1e1e1e;border:1px solid #333;
border-radius:8px;padding:12px 20px;margin:6px;min-width:110px;text-align:center}}
.stat .val{{font-size:1.6em;font-weight:bold;color:#ff4444}}.stat .lbl{{font-size:.8em;color:#aaa}}
ul{{line-height:2}}iframe{{width:100%;height:520px;border:none;margin-top:10px}}</style></head><body>
<h1>🔥 Fire Risk Analysis Report</h1>
<p><b>Location:</b> {lq} &nbsp;|&nbsp; <b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>Summary</h2>
<div class='stat'><div class='val'>{frg['final_risk'].mean():.3f}</div><div class='lbl'>Avg Risk</div></div>
<div class='stat'><div class='val'>{frg['final_risk'].max():.3f}</div><div class='lbl'>Max Risk</div></div>
<div class='stat'><div class='val'>{int(_hbc.get('Critical',0))}</div><div class='lbl'>⛔ Critical</div></div>
<div class='stat'><div class='val'>{int(_hbc.get('High',0))}</div><div class='lbl'>🔴 High</div></div>
<div class='stat'><div class='val'>{int(_hbc.get('Medium',0))}</div><div class='lbl'>🟡 Medium</div></div>
<div class='stat'><div class='val'>{int(_hbc.get('Low',0))}</div><div class='lbl'>🟢 Low</div></div>
<h2>Recommendations</h2><ul>{_recs_html}</ul>
<h2>Top 5 Highest-Risk Zones</h2>
<table><tr><th>#</th><th>Latitude</th><th>Longitude</th><th>Risk Score</th><th>Band</th><th>Buildings</th></tr>
{_rows}</table>
<h2>Interactive Risk Map</h2>{_me}</body></html>"""
            st.download_button("📄 HTML Report (self-contained)",
                               data=_html_rep,
                               file_name=f"fire_risk_report_{_lqsafe}.html",
                               mime="text/html",
                               use_container_width=True)
        except Exception:
            pass

    # ── TAB 6: History ─────────────────────────────────────────────────────────
    with tab_history:
        st.subheader("Previous Analyses")
        _hdir = "history"
        if not os.path.exists(_hdir):
            st.info("No previous analyses yet.")
        else:
            _mfs = sorted([f for f in os.listdir(_hdir) if f.endswith("_meta.json")], reverse=True)
            if not _mfs:
                st.info("No previous analyses yet.")
            else:
                _hrs = []
                for _mf in _mfs[:20]:
                    try:
                        with open(os.path.join(_hdir, _mf)) as _f: _m = json.load(_f)
                        _m['_file'] = _mf.replace("_meta.json",""); _hrs.append(_m)
                    except: continue

                _hdf = pd.DataFrame([{
                    "Location":      r.get("place",""),
                    "Date/Time":     datetime.datetime.strptime(r["timestamp"],"%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M"),
                    "Radius (m)":    r.get("radius_m",""),
                    "Buildings":     r.get("n_buildings",""),
                    "Fire Stations": r.get("n_stations",""),
                    "Avg Risk":      r.get("avg_risk","—"),
                    "Max Risk":      r.get("max_risk","—"),
                } for r in _hrs])
                st.dataframe(_hdf, use_container_width=True)

                # History chart
                _chart_rows = [r for r in _hrs if r.get("avg_risk") is not None and r.get("max_risk") is not None]
                if len(_chart_rows) >= 2:
                    import matplotlib.pyplot as _mplt
                    _fig, _ax = _mplt.subplots(figsize=(10, 3))
                    _fig.patch.set_facecolor('#0e1117'); _ax.set_facecolor('#0e1117')
                    _dates = [datetime.datetime.strptime(r["timestamp"],"%Y%m%d_%H%M%S") for r in _chart_rows]
                    _avgs  = [r["avg_risk"] for r in _chart_rows]
                    _maxs  = [r["max_risk"] for r in _chart_rows]
                    _ax.plot(_dates, _avgs, 'o-', color='#ff9900', label='Avg Risk', linewidth=2)
                    _ax.plot(_dates, _maxs, 's--', color='#ff4444', label='Max Risk', linewidth=2)
                    _ax.set_ylabel("Risk Score", color='white'); _ax.tick_params(colors='white')
                    _ax.spines[:].set_color('#444')
                    _ax.legend(facecolor='#1e1e2e', labelcolor='white')
                    _ax.set_title("Risk Score Trend Across Analyses", color='white')
                    _mplt.xticks(rotation=30, color='white')
                    st.pyplot(_fig, use_container_width=True)
                    _mplt.close(_fig)

                # Download previous run
                st.markdown("**Download a previous run:**")
                _sel = st.selectbox("Select run:",
                    options=range(len(_hrs)),
                    format_func=lambda i: f"{_hrs[i].get('place','')} — {datetime.datetime.strptime(_hrs[i]['timestamp'],'%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M')}",
                    label_visibility="collapsed")
                _gp = os.path.join(_hdir, f"{_hrs[_sel]['_file']}.geojson")
                if os.path.exists(_gp):
                    with open(_gp) as _gf: _gc = _gf.read()
                    st.download_button("Download Selected Run (GeoJSON)", data=_gc,
                                       file_name=f"{_hrs[_sel]['_file']}.geojson",
                                       mime="application/geo+json")

    # ── TAB 7: Batch Analysis ──────────────────────────────────────────────────
    with tab_batch:
        st.subheader("Batch Analysis")
        st.caption("Analyse multiple locations at once. Results are summary stats only (no maps). Uses a fixed 1 km radius and equal weights.")

        _batch_input = st.text_area("Locations (one per line):",
                                    placeholder="Korail, Dhaka\nMirpur, Dhaka\nMohakhali, Dhaka",
                                    height=120)
        if st.button("▶ Run Batch Analysis", type="primary"):
            _locations = [l.strip() for l in _batch_input.strip().split('\n') if l.strip()]
            if not _locations:
                st.warning("Enter at least one location.")
            else:
                _batch_results = []
                _bp = st.progress(0, text="Running batch analysis…")
                _geo = Nominatim(user_agent="fire_risk_batch", timeout=10)
                for _i, _loc in enumerate(_locations):
                    try:
                        _bp.progress(int((_i / len(_locations)) * 100), text=f"Analysing {_loc}…")
                        _gl = _geo.geocode(_loc)
                        if not _gl:
                            _batch_results.append({"Location": _loc, "Status": "Not found"}); continue
                        _lpt2 = (_gl.latitude, _gl.longitude)
                        _bw = {"density":0.33,"access":0.33,"water":0.34,"height":0.0,"hazard":0.0}
                        _tepsg = _target_epsg(_lpt2)
                        _bg, _bar, _bb, _bws, _bfs, _bh = fetch_all_osm_parallel(
                            _lpt2, 1000, tuple(DEFAULT_ROAD_TYPES), _tepsg)
                        if _bg is None or _bb.empty:
                            _batch_results.append({"Location": _loc, "Status": "No data"}); continue
                        _bdg  = calculate_density_grid(_bb)
                        _bbh  = calculate_occupancy_modifier(calculate_height_risk(_bb))
                        _bdg  = calculate_road_width_modifier(_bdg, _bar)
                        _bbt  = calculate_travel_risk(_bbh, _bfs, _bg)
                        _bbw  = calculate_water_risk(_bbt, _bws)
                        _bba  = calculate_hazard_risk(_bbw, _bh)
                        _bfrg = calculate_composite_risk(_bdg, _bba, _bw)
                        _bbc  = _bfrg['risk_band'].value_counts()
                        _batch_results.append({
                            "Location": _loc, "Buildings": len(_bb),
                            "Avg Risk": round(float(_bfrg['final_risk'].mean()), 3),
                            "Max Risk": round(float(_bfrg['final_risk'].max()),  3),
                            "Critical": int(_bbc.get('Critical', 0)),
                            "High":     int(_bbc.get('High',     0)),
                            "Status":   "✅ OK",
                        })
                    except Exception as _be:
                        _batch_results.append({"Location": _loc, "Status": f"Error: {_be}"})
                _bp.progress(100, text="Batch complete!")
                _bdf = pd.DataFrame(_batch_results)
                st.dataframe(_bdf, use_container_width=True)
                st.download_button("Download Batch Results (CSV)",
                                   data=_bdf.to_csv(index=False),
                                   file_name="batch_results.csv",
                                   mime="text/csv")

    # ── TAB 8: Scenario Comparison ─────────────────────────────────────────────
    with tab_compare:
        st.subheader("Scenario Comparison")
        if st.session_state.scenario_a and st.session_state.scenario_b:
            st.caption("Your last two analyses side by side. Run another analysis to update Scenario B.")

            def _sc_sum(sc):
                g = sc['grid']; b = g['risk_band'].value_counts()
                return {
                    "Avg Risk":  f"{g['final_risk'].mean():.4f}",
                    "Max Risk":  f"{g['final_risk'].max():.4f}",
                    "Critical":  int(b.get('Critical', 0)),
                    "High":      int(b.get('High',     0)),
                    "Medium":    int(b.get('Medium',   0)),
                    "Low":       int(b.get('Low',      0)),
                }

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Scenario A (latest)**")
                st.caption(st.session_state.scenario_a['label'])
                st.table(_sc_sum(st.session_state.scenario_a))
            with c2:
                st.markdown(f"**Scenario B (previous)**")
                st.caption(st.session_state.scenario_b['label'])
                st.table(_sc_sum(st.session_state.scenario_b))
        else:
            st.info("Run **two** analyses to enable scenario comparison. Each new analysis automatically becomes Scenario A, pushing the previous one to Scenario B.")

else:
    # ── Welcome screen (shown before first analysis) ───────────────────────────
    st.markdown("""
<div class="welcome-card">
  <h2>Welcome to FRAT</h2>
  <p>Configure your location and settings in the <b>sidebar on the left</b>,<br>
  then click <b>🔍 Analyse Location</b> to generate your fire risk report.</p>
  <br>
  <p style="font-size:0.9rem; color:#888;">
    Results will appear here across 8 tabs:<br>
    📊 Summary &nbsp;·&nbsp; 🗺 Maps &nbsp;·&nbsp; 🌍 Interactive &nbsp;·&nbsp;
    🔥 Hotspots &nbsp;·&nbsp; 📥 Export &nbsp;·&nbsp; 📂 History &nbsp;·&nbsp;
    📋 Batch &nbsp;·&nbsp; ⚖ Compare
  </p>
</div>
""", unsafe_allow_html=True)
