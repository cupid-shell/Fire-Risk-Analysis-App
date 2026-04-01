import streamlit as st
from geopy.geocoders import Nominatim
import streamlit.components.v1 as components
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import json
import os
import datetime

from fire_risk_analyzer import (
    get_geospatial_data,
    calculate_density_grid,
    calculate_travel_risk,
    calculate_water_risk,
    calculate_composite_risk,
    save_footprints_map,
    save_roads_map,
    generate_static_risk_map,
    generate_interactive_risk_map,
    DEFAULT_ROAD_TYPES,
)

st.set_page_config(layout="wide")
st.title("Fire Risk Analysis Tool (FRAT)")
st.write("Enter the name of an urban settlement and select a radius to analyze its fire risk.")

# --- Session state ---
for _key, _default in [('maps_generated', False), ('final_risk_grid', None), ('location_query', '')]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# --- Input ---
input_method = st.radio("Choose input method:", ('Search by Name', 'Enter Coordinates'))

location_query = ""
location_point = None

if input_method == 'Search by Name':
    location_query = st.text_input("Enter Location Name (e.g., 'Korail, Dhaka')", "Korail, Dhaka")
else:
    lat_col, lon_col = st.columns(2)
    with lat_col:
        lat = st.number_input("Enter Latitude", value=23.774, min_value=-90.0, max_value=90.0, step=0.001, format="%.5f")
    with lon_col:
        lon = st.number_input("Enter Longitude", value=90.405, min_value=-180.0, max_value=180.0, step=0.001, format="%.5f")
    location_point = (lat, lon)
    location_query = f"{lat:.5f}, {lon:.5f}"

search_dist = st.slider("Select Search Radius (meters)", 500, 2500, 1000, 50)

# --- Risk weights ---
st.subheader("Adjust Risk Factor Weights")
col1_w, col2_w, col3_w = st.columns(3)
with col1_w:
    density_weight = st.slider("Density Importance", 0, 100, 33)
with col2_w:
    access_weight = st.slider("Access Importance", 0, 100, 33)
with col3_w:
    water_weight = st.slider("Water Importance", 0, 100, 34)

# --- Advanced options ---
ALL_ROAD_TYPES = [
    'primary', 'secondary', 'tertiary', 'residential',
    'service', 'unclassified', 'trunk', 'living_street', 'pedestrian', 'track'
]

with st.expander("Advanced Options"):
    st.markdown("**Road Type Filter**")
    st.caption("Select which road types count toward fire station accessibility scoring.")
    selected_road_types = st.multiselect(
        "Included road types:",
        options=ALL_ROAD_TYPES,
        default=DEFAULT_ROAD_TYPES,
    )

    st.markdown("---")
    st.markdown("**Simulate a Hypothetical Fire Station**")
    st.caption("Add a virtual fire station to see how it would reduce access risk in the area.")
    add_station = st.checkbox("Enable hypothetical fire station")
    extra_station = None
    if add_station:
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            s_lat = st.number_input("Station Latitude", value=23.774, min_value=-90.0, max_value=90.0, step=0.001, format="%.5f", key="s_lat")
        with s_col2:
            s_lon = st.number_input("Station Longitude", value=90.405, min_value=-180.0, max_value=180.0, step=0.001, format="%.5f", key="s_lon")
        extra_station = (s_lat, s_lon)
        st.info(f"Hypothetical station at ({s_lat:.5f}, {s_lon:.5f}) will be included in access risk calculations.")

# --- Cached data fetching (split into steps for progress feedback) ---
def _get_target_epsg(location_point):
    gdf = gpd.GeoDataFrame(geometry=[Point(location_point[1], location_point[0])], crs="EPSG:4326")
    return gdf.estimate_utm_crs().to_epsg()

@st.cache_data(show_spinner=False)
def fetch_road_network(location_point, distance, road_types_tuple, target_epsg):
    import osmnx as ox
    graph = ox.graph_from_point(location_point, dist=distance, network_type='all')
    graph_proj = ox.project_graph(graph, to_crs=target_epsg)
    edges = ox.graph_to_gdfs(graph_proj, nodes=False)
    accessible_roads = edges[edges['highway'].isin(list(road_types_tuple))]
    return graph_proj, accessible_roads

@st.cache_data(show_spinner=False)
def fetch_buildings(location_point, distance, target_epsg):
    import osmnx as ox
    import geopandas as gpd
    try:
        buildings = ox.features_from_point(location_point, {"building": True}, dist=distance)
        return buildings.to_crs(target_epsg)
    except Exception:
        return gpd.GeoDataFrame(columns=['geometry'], crs=target_epsg)

@st.cache_data(show_spinner=False)
def fetch_water_sources(location_point, distance, target_epsg):
    import osmnx as ox
    import geopandas as gpd
    try:
        water = ox.features_from_point(location_point, {"natural": "water", "amenity": "fire_hydrant"}, dist=distance)
        return water.to_crs(target_epsg)
    except Exception:
        return gpd.GeoDataFrame(columns=['geometry'], crs=target_epsg)

@st.cache_data(show_spinner=False)
def fetch_fire_stations(location_point, distance, target_epsg):
    import osmnx as ox
    import geopandas as gpd
    try:
        stations = ox.features_from_point(location_point, {"amenity": "fire_station"}, dist=distance)
        return stations.to_crs(target_epsg)
    except Exception:
        return gpd.GeoDataFrame(columns=['geometry'], crs=target_epsg)

# --- Analyze button ---
if st.button("Analyze Location"):
    weights = {"density": density_weight / 100, "access": access_weight / 100, "water": water_weight / 100}
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        st.error("Total weight cannot be zero."); st.stop()

    if not selected_road_types:
        st.error("Please select at least one road type in Advanced Options."); st.stop()

    if input_method == 'Search by Name':
        if location_query:
            try:
                geolocator = Nominatim(user_agent="fire_risk_app", timeout=10)
                location = geolocator.geocode(location_query)
                if location:
                    st.info(f"Found location: {location.address}")
                    location_point = (location.latitude, location.longitude)
                else:
                    st.error("Could not find the location."); location_point = None
            except Exception as e:
                st.error(f"Geocoding failed: {e}"); location_point = None
        else:
            st.warning("Please enter a location name."); location_point = None

    if location_point:
        st.info(f"Weights — Density: {weights['density']:.0%} | Access: {weights['access']:.0%} | Water: {weights['water']:.0%}")
        progress = st.progress(0, text="Starting analysis...")

        try:
            progress.progress(5, text="Computing coordinate system...")
            target_epsg = _get_target_epsg(location_point)

            progress.progress(10, text="Step 1/4 — Fetching road network from OSM (this is the slowest step, ~1-3 min on first run)...")
            graph, accessible_roads = fetch_road_network(
                location_point, search_dist, tuple(selected_road_types), target_epsg
            )

            if graph is None:
                st.error("Could not download road network. Check your internet connection or try a different location.")
                progress.empty()
                st.stop()

            progress.progress(30, text="Step 2/4 — Fetching building footprints...")
            buildings = fetch_buildings(location_point, search_dist, target_epsg)

            progress.progress(50, text="Step 3/4 — Fetching water sources...")
            water_sources = fetch_water_sources(location_point, search_dist, target_epsg)

            progress.progress(65, text="Step 4/4 — Fetching fire station locations...")
            fire_stations = fetch_fire_stations(location_point, search_dist, target_epsg)

            if buildings.empty:
                st.error("No buildings found for this location. Try a larger radius.")
                progress.empty()
                st.stop()

            # --- Data quality indicators ---
            n_buildings = len(buildings)
            n_stations = len(fire_stations) if not fire_stations.empty else 0
            n_water = len(water_sources) if not water_sources.empty else 0

            st.markdown("**Data Quality**")
            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Buildings Found", n_buildings)
            with q2:
                st.metric("Fire Stations (OSM)", n_stations)
            with q3:
                st.metric("Water Sources (OSM)", n_water)

            if n_stations == 0 and extra_station is None:
                st.warning("No fire stations found in OSM data for this area. Access risk defaults to maximum distance (5000m). Consider adding a hypothetical station in Advanced Options.")
            elif n_stations == 0 and extra_station is not None:
                st.info("No OSM fire stations found, but your hypothetical station will be used for access risk.")
            if n_water == 0:
                st.warning("No water sources found in OSM data for this area. Water proximity risk defaults to maximum distance (1000m).")

            progress.progress(72, text="Calculating building density grid...")
            density_grid = calculate_density_grid(buildings)

            progress.progress(78, text="Calculating travel distance to fire stations...")
            buildings_with_travel_risk = calculate_travel_risk(buildings, fire_stations, graph, extra_station)

            progress.progress(83, text="Calculating water source proximity risk...")
            buildings_with_all_risks = calculate_water_risk(buildings_with_travel_risk, water_sources)

            progress.progress(87, text="Computing composite risk scores...")
            final_risk_grid = calculate_composite_risk(density_grid, buildings_with_all_risks, weights)

            progress.progress(90, text="Rendering building footprints map...")
            save_footprints_map(buildings, graph, 'building_footprints.png')

            progress.progress(93, text="Rendering road network map...")
            save_roads_map(accessible_roads, graph, 'road_network.png')

            progress.progress(96, text="Generating static risk heatmap...")
            generate_static_risk_map(final_risk_grid, graph)

            progress.progress(98, text="Generating interactive map...")
            generate_interactive_risk_map(final_risk_grid)

            # --- Save to history ---
            os.makedirs("history", exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(c if c.isalnum() else "_" for c in location_query)[:30]
            history_prefix = f"history/{timestamp}_{safe_name}"
            final_risk_grid.to_crs("EPSG:4326").to_file(f"{history_prefix}.geojson", driver="GeoJSON")
            with open(f"{history_prefix}_meta.json", "w") as f:
                json.dump({
                    "place": location_query,
                    "timestamp": timestamp,
                    "radius_m": search_dist,
                    "weights": {k: round(v, 3) for k, v in weights.items()},
                    "n_buildings": n_buildings,
                    "n_stations": n_stations,
                    "n_water": n_water,
                    "road_types": selected_road_types,
                    "hypothetical_station": extra_station,
                }, f, indent=2)

            progress.progress(100, text="Analysis complete!")
            st.success("Analysis complete!")
            st.session_state.maps_generated = True
            st.session_state.final_risk_grid = final_risk_grid
            st.session_state.location_query = location_query

        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.exception(e)
            progress.empty()

# --- Results ---
if st.session_state.maps_generated and st.session_state.final_risk_grid is not None:
    final_risk_grid = st.session_state.final_risk_grid
    lq = st.session_state.location_query or "location"

    st.markdown("---")
    st.subheader("Analysis Summary Maps")
    map_col1, map_col2 = st.columns(2)
    with map_col1:
        st.image('building_footprints.png', caption='Building Footprints', use_container_width=True)
    with map_col2:
        st.image('road_network.png', caption='Accessible Road Network', use_container_width=True)
    st.image('final_risk_map.png', caption='Static Composite Risk Heatmap', use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Fire Risk Map")
    try:
        with open('interactive_risk_map.html', 'r', encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("Interactive map file not found. Please run the analysis again.")

    st.markdown("---")
    st.subheader("Top Risk Hotspots")
    top_n = st.slider("Number of hotspots to display", min_value=5, max_value=30, value=10)
    hotspots = final_risk_grid.nlargest(top_n, 'final_risk').to_crs("EPSG:4326").copy()
    hotspots['Latitude'] = hotspots.centroid.y.round(5)
    hotspots['Longitude'] = hotspots.centroid.x.round(5)
    hotspots['Risk Score'] = hotspots['final_risk'].round(4)
    hotspots['Buildings in Cell'] = hotspots['n_buildings'].astype(int)
    hotspots['Density Risk'] = hotspots['density_risk'].round(3)
    hotspots['Access Risk'] = hotspots['access_risk'].round(3)
    hotspots['Water Risk'] = hotspots['water_risk'].round(3)
    st.dataframe(
        hotspots[['Latitude', 'Longitude', 'Risk Score', 'Buildings in Cell', 'Density Risk', 'Access Risk', 'Water Risk']].reset_index(drop=True),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Export Data")
    st.caption("Click a button below to download the risk data for use in GIS tools like QGIS or ArcGIS.")
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        geojson_data = final_risk_grid.to_crs("EPSG:4326").to_json()
        st.download_button(
            label="Download Risk Grid (GeoJSON)",
            data=geojson_data,
            file_name=f"risk_grid_{lq.replace(' ', '_').replace(',', '')}.geojson",
            mime="application/geo+json",
        )

    with dl_col2:
        grid_wgs84 = final_risk_grid.to_crs("EPSG:4326").copy()
        grid_wgs84['lat'] = grid_wgs84.centroid.y
        grid_wgs84['lon'] = grid_wgs84.centroid.x
        csv_data = grid_wgs84[['lat', 'lon', 'n_buildings', 'density_risk', 'access_risk', 'water_risk', 'final_risk']].to_csv(index=False)
        st.download_button(
            label="Download Risk Data (CSV)",
            data=csv_data,
            file_name=f"risk_data_{lq.replace(' ', '_').replace(',', '')}.csv",
            mime="text/csv",
        )

# --- Previous Analyses ---
st.markdown("---")
st.subheader("Previous Analyses")

history_dir = "history"
if not os.path.exists(history_dir):
    st.info("No previous analyses yet. Run an analysis to start building history.")
else:
    meta_files = sorted(
        [f for f in os.listdir(history_dir) if f.endswith("_meta.json")],
        reverse=True
    )

    if not meta_files:
        st.info("No previous analyses yet. Run an analysis to start building history.")
    else:
        history_rows = []
        for mf in meta_files[:20]:
            try:
                with open(os.path.join(history_dir, mf)) as f:
                    meta = json.load(f)
                meta['_file'] = mf.replace("_meta.json", "")
                history_rows.append(meta)
            except Exception:
                continue

        display_df = pd.DataFrame([{
            "Location": r.get("place", ""),
            "Date/Time": datetime.datetime.strptime(r["timestamp"], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M"),
            "Radius (m)": r.get("radius_m", ""),
            "Buildings": r.get("n_buildings", ""),
            "Fire Stations": r.get("n_stations", ""),
            "Water Sources": r.get("n_water", ""),
        } for r in history_rows])

        st.dataframe(display_df, use_container_width=True)

        selected_label = st.selectbox(
            "Select a previous run to download:",
            options=range(len(history_rows)),
            format_func=lambda i: f"{history_rows[i].get('place','')} — {datetime.datetime.strptime(history_rows[i]['timestamp'], '%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M')}",
        )

        geojson_path = os.path.join(history_dir, f"{history_rows[selected_label]['_file']}.geojson")
        if os.path.exists(geojson_path):
            with open(geojson_path, "r") as f:
                past_geojson = f.read()
            st.download_button(
                label="Download Selected Run (GeoJSON)",
                data=past_geojson,
                file_name=f"{history_rows[selected_label]['_file']}.geojson",
                mime="application/geo+json",
            )
        else:
            st.warning("GeoJSON file for the selected run was not found.")

st.markdown("---")
st.caption("Created by Avishek Adhikari | avishek.jidpus@gmail.com")
