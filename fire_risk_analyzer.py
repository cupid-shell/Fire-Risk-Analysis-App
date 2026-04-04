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

def get_geospatial_data(location_point, distance, target_crs, road_types=None):
    """
    Downloads and projects all necessary data, handling cases where features are not found.
    """
    if road_types is None:
        road_types = DEFAULT_ROAD_TYPES

    print(f"Fetching data within {distance}m of {location_point}...")
    try:
        graph = ox.graph_from_point(location_point, dist=distance, network_type='all')
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
    buildings_centroids = buildings_proj.copy()
    buildings_centroids['geometry'] = buildings_centroids.geometry.centroid
    merged = gpd.sjoin(buildings_centroids, grid, how='left', predicate='within')
    merged['n_buildings'] = 1
    building_counts = merged.groupby('index_right')['n_buildings'].sum().reset_index()
    grid = grid.merge(building_counts, left_index=True, right_on='index_right', how='left')
    grid = grid.drop(columns=['index_right'])
    grid['n_buildings'] = grid['n_buildings'].fillna(0)
    print("Density calculation complete!")
    return grid

def calculate_travel_risk(buildings_proj, fire_stations_proj, graph_proj, extra_station=None):
    print("Calculating travel distance risk from fire stations...")
    stations = fire_stations_proj.copy()

    if extra_station is not None:
        target_crs = stations.crs if not stations.empty else buildings_proj.crs
        extra_pt = gpd.GeoDataFrame(
            geometry=[Point(extra_station[1], extra_station[0])],
            crs="EPSG:4326"
        ).to_crs(target_crs)
        stations = pd.concat([stations, extra_pt], ignore_index=True) if not stations.empty else extra_pt

    if stations.empty:
        print("No fire stations found. Assigning high-risk distance.")
        buildings_with_risk = buildings_proj.copy()
        buildings_with_risk['travel_distance'] = 5000
        return buildings_with_risk

    fire_station_points = stations.copy()
    fire_station_points['geometry'] = fire_station_points.geometry.centroid
    station_nodes = ox.nearest_nodes(graph_proj, fire_station_points.geometry.x, fire_station_points.geometry.y)
    building_nodes = ox.nearest_nodes(graph_proj, buildings_proj.geometry.centroid.x, buildings_proj.geometry.centroid.y)
    travel_distances = []
    for b_node in building_nodes:
        try:
            path_length = min([nx.shortest_path_length(graph_proj, b_node, s_node, weight='length') for s_node in np.atleast_1d(station_nodes)])
            travel_distances.append(path_length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            travel_distances.append(5000)
    buildings_with_risk = buildings_proj.copy()
    buildings_with_risk['travel_distance'] = travel_distances
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

def calculate_hazard_risk(buildings_proj, hazards_proj):
    """Proximity to hazard points (gas stations, hospitals, schools) — closer = higher risk."""
    b = buildings_proj.copy()
    if hazards_proj is None or hazards_proj.empty:
        b['distance_to_hazard'] = 2000
        return b
    combined = hazards_proj.union_all()
    b['distance_to_hazard'] = b.geometry.apply(lambda g: g.centroid.distance(combined))
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

def calculate_composite_risk(density_grid, buildings_with_all_risks, weights):
    print("Calculating composite risk score...")
    grid = density_grid.reset_index(drop=True)

    if buildings_with_all_risks.empty or 'travel_distance' not in buildings_with_all_risks.columns:
        grid['avg_travel_distance'] = 0
        grid['avg_distance_water'] = 0
    else:
        buildings_centroids = buildings_with_all_risks.copy()
        buildings_centroids['geometry'] = buildings_centroids.geometry.centroid
        merged = gpd.sjoin(buildings_centroids, grid, how='left', predicate='within')
        agg_cols = ['travel_distance', 'distance_to_water']
        for col in ['levels', 'distance_to_hazard', 'occupancy_proxy']:
            if col in buildings_centroids.columns:
                agg_cols.append(col)
        avg_risks_in_grid = merged.groupby('index_right')[agg_cols].mean()
        grid = grid.join(avg_risks_in_grid)

    grid = grid.rename(columns={'travel_distance': 'avg_travel_distance', 'distance_to_water': 'avg_distance_water'})
    for col in ['avg_travel_distance', 'avg_distance_water']:
        grid[col] = grid[col].fillna(0)

    def _norm(series):
        mn, mx = series.min(), series.max()
        return (series - mn) / (mx - mn) if mx > mn else pd.Series(0.0, index=series.index)

    grid['density_risk'] = _norm(grid['n_buildings'])

    # Occupancy modifier — boosts density risk by up to 50% for high-occupancy buildings
    if 'occupancy_proxy' in grid.columns:
        grid['occupancy_proxy'] = grid['occupancy_proxy'].fillna(1)
        occ_norm = _norm(grid['occupancy_proxy'])
        grid['density_risk'] = (grid['density_risk'] * (1 + occ_norm * 0.5)).clip(0, 1)

    grid['access_risk'] = _norm(grid['avg_travel_distance'])

    # Road width modifier — narrow roads boost access risk by up to 30%
    if 'avg_lanes' in grid.columns:
        grid['avg_lanes'] = grid['avg_lanes'].fillna(2)
        narrow_penalty = (1 / grid['avg_lanes'].clip(lower=0.5)).pipe(_norm)
        grid['access_risk'] = (grid['access_risk'] * (1 + narrow_penalty * 0.3)).clip(0, 1)

    grid['water_risk'] = _norm(grid['avg_distance_water'])

    if 'levels' in grid.columns and weights.get('height', 0) > 0:
        grid['levels'] = grid['levels'].fillna(1)
        grid['height_risk'] = _norm(grid['levels'])
    else:
        grid['height_risk'] = 0.0

    if 'distance_to_hazard' in grid.columns and weights.get('hazard', 0) > 0:
        grid['distance_to_hazard'] = grid['distance_to_hazard'].fillna(2000)
        grid['hazard_risk'] = _norm(-grid['distance_to_hazard'])  # closer = higher risk
    else:
        grid['hazard_risk'] = 0.0

    grid['final_risk'] = (
        grid['density_risk'] * weights.get('density', 0) +
        grid['access_risk']  * weights.get('access',  0) +
        grid['water_risk']   * weights.get('water',   0) +
        grid['height_risk']  * weights.get('height',  0) +
        grid['hazard_risk']  * weights.get('hazard',  0)
    ).clip(0, 1)

    def _band(s):
        if s >= 0.75: return 'Critical'
        if s >= 0.50: return 'High'
        if s >= 0.25: return 'Medium'
        return 'Low'

    grid['risk_band'] = grid['final_risk'].apply(_band)
    return grid

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
    m = folium.Map(location=map_center, zoom_start=15, tiles="CartoDB positron")

    # Satellite base layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite',
        overlay=False
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
        [row.centroid.y, row.centroid.x, row['final_risk']]
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
    folium.LayerControl(collapsed=False).add_to(m)
    m.save('interactive_risk_map.html')

def main(place_name, location_point, search_distance, weights, road_types=None, extra_station=None, wind_direction=None):
    """Orchestrates the entire analysis and map generation. Returns the final risk grid."""
    gdf_wgs84 = gpd.GeoDataFrame(geometry=[Point(location_point[1], location_point[0])], crs="EPSG:4326")
    target_crs = gdf_wgs84.estimate_utm_crs()
    graph, buildings, accessible_roads, water_sources, fire_stations = get_geospatial_data(location_point, search_distance, target_crs, road_types)
    if graph is None or buildings.empty:
        raise ValueError("No buildings or road network found. Cannot generate analysis.")
    density_grid = calculate_density_grid(buildings)
    buildings_with_heights = calculate_height_risk(buildings)
    buildings_with_occ = calculate_occupancy_modifier(buildings_with_heights)
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
