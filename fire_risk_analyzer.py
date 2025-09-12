import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import folium
import branca.colormap as cm
import networkx as nx
from fpdf import FPDF

def get_geospatial_data(location_point, distance, target_crs):
    """
    Downloads and projects all necessary data, handling cases where features are not found.
    """
    print(f"Fetching data within {distance}m of {location_point}...")
    try:
        graph = ox.graph_from_point(location_point, dist=distance, network_type='all')
        graph_proj = ox.project_graph(graph, to_crs=target_crs)
        edges = ox.graph_to_gdfs(graph_proj, nodes=False)
        road_types = ['primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified']
        accessible_roads = edges[edges['highway'].isin(road_types)]
    except Exception as e:
        print(f"Could not download road network. Error: {e}")
        return [None] * 5

    # --- THIS IS THE FIX: Reverting to a more general error handling ---
    try:
        tags = {"building": True}
        buildings = ox.features_from_point(location_point, tags, dist=distance)
        buildings_proj = buildings.to_crs(target_crs)
    except Exception: # Use a general Exception
        print("Warning: No buildings found in the area. Creating empty buildings dataset.")
        buildings_proj = gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)

    try:
        water_tags = {"natural": "water", "amenity": "fire_hydrant"}
        water_sources = ox.features_from_point(location_point, water_tags, dist=distance)
        water_sources_proj = water_sources.to_crs(target_crs)
    except Exception: # Use a general Exception
        print("Warning: No water sources found in the area. Creating empty water sources dataset.")
        water_sources_proj = gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)

    try:
        station_tags = {"amenity": "fire_station"}
        fire_stations = ox.features_from_point(location_point, station_tags, dist=distance)
        fire_stations_proj = fire_stations.to_crs(target_crs)
    except Exception: # Use a general Exception
        print("Warning: No fire stations found in the area. Creating empty fire stations dataset.")
        fire_stations_proj = gpd.GeoDataFrame(columns=['geometry'], crs=target_crs)

    print("Data fetching complete!")
    return graph_proj, buildings_proj, accessible_roads, water_sources_proj, fire_stations_proj

# (All other functions in the script remain the same)
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

def calculate_travel_risk(buildings_proj, fire_stations_proj, graph_proj):
    print("Calculating travel distance risk from fire stations...")
    if fire_stations_proj.empty:
        print("No fire stations found. Assigning high-risk distance.")
        buildings_with_risk = buildings_proj.copy()
        buildings_with_risk['travel_distance'] = 5000
        return buildings_with_risk
    fire_station_points = fire_stations_proj.copy()
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
        avg_risks_in_grid = merged.groupby('index_right')[['travel_distance', 'distance_to_water']].mean()
        grid = grid.join(avg_risks_in_grid)
    grid = grid.rename(columns={'travel_distance': 'avg_travel_distance', 'distance_to_water': 'avg_distance_water'})
    grid['avg_travel_distance'] = grid['avg_travel_distance'].fillna(0)
    grid['avg_distance_water'] = grid['avg_distance_water'].fillna(0)
    if not grid.empty and (grid['n_buildings'].max() - grid['n_buildings'].min()) > 0:
        grid['density_risk'] = (grid['n_buildings'] - grid['n_buildings'].min()) / (grid['n_buildings'].max() - grid['n_buildings'].min())
    else: grid['density_risk'] = 0
    if not grid.empty and (grid['avg_travel_distance'].max() - grid['avg_travel_distance'].min()) > 0:
        grid['access_risk'] = (grid['avg_travel_distance'] - grid['avg_travel_distance'].min()) / (grid['avg_travel_distance'].max() - grid['avg_travel_distance'].min())
    else: grid['access_risk'] = 0
    if not grid.empty and (grid['avg_distance_water'].max() - grid['avg_distance_water'].min()) > 0:
        grid['water_risk'] = (grid['avg_distance_water'] - grid['avg_distance_water'].min()) / (grid['avg_distance_water'].max() - grid['avg_distance_water'].min())
    else: grid['water_risk'] = 0
    grid['final_risk'] = (grid['density_risk'] * weights['density']) + (grid['access_risk'] * weights['access']) + (grid['water_risk'] * weights['water'])
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

def generate_interactive_risk_map(grid):
    print("Generating interactive risk map...")
    grid_wgs84 = grid.to_crs("EPSG:4326")
    map_center = [grid_wgs84.centroid.y.mean(), grid_wgs84.centroid.x.mean()]
    m = folium.Map(location=map_center, zoom_start=15, tiles="CartoDB positron")
    colormap = cm.LinearColormap(colors=['green', 'yellow', 'red'], vmin=grid['final_risk'].min(), vmax=grid['final_risk'].max())
    colormap.caption = 'Composite Fire Risk Score'
    style_function = lambda x: {'fillColor': colormap(x['properties']['final_risk']), 'color': 'black', 'weight': 0.5, 'fillOpacity': 0.7}
    folium.GeoJson(grid_wgs84, style_function=style_function, tooltip=folium.features.GeoJsonTooltip(fields=['n_buildings', 'final_risk'], aliases=['Buildings:', 'Risk Score:'])).add_to(m)
    m.add_child(colormap); folium.LayerControl().add_to(m)
    m.save('interactive_risk_map.html')

def generate_pdf_report(place_name, search_dist, weights, final_risk_grid):
    print("Generating PDF report...")
    if final_risk_grid.empty:
        print("Cannot generate report: risk grid is empty.")
        return
    top_5_hotspots = final_risk_grid.nlargest(5, 'final_risk')
    top_5_hotspots_wgs84 = top_5_hotspots.to_crs("EPSG:4326")
    top_5_hotspots_wgs84['lon'] = top_5_hotspots_wgs84.centroid.x
    top_5_hotspots_wgs84['lat'] = top_5_hotspots_wgs84.centroid.y
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, f'Fire Risk Analysis Report: {place_name}', 0, 1, 'C')
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Analysis Parameters', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 5, f"- Search Radius: {search_dist} meters", 0, 1)
    pdf.cell(0, 5, f"- Risk Weights: Density ({weights['density']:.0%}), Access ({weights['access']:.0%}), Water ({weights['water']:.0%})", 0, 1)
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Composite Fire Risk Map', 0, 1)
    pdf.image('final_risk_map.png', x=10, y=None, w=190)
    pdf.ln(5)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Top 5 Highest-Risk Zones', 0, 1)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(20, 10, 'Rank', 1)
    pdf.cell(50, 10, 'Latitude', 1)
    pdf.cell(50, 10, 'Longitude', 1)
    pdf.cell(40, 10, 'Risk Score', 1)
    pdf.ln()
    pdf.set_font('Arial', '', 10)
    rank = 1
    for index, row in top_5_hotspots_wgs84.iterrows():
        lat = f"{row['lat']:.5f}"
        lon = f"{row['lon']:.5f}"
        score = f"{row['final_risk']:.3f}"
        pdf.cell(20, 10, str(rank), 1)
        pdf.cell(50, 10, lat, 1)
        pdf.cell(50, 10, lon, 1)
        pdf.cell(40, 10, score, 1)
        pdf.ln()
        rank += 1
    pdf.output("Fire_Risk_Report.pdf")
    print("PDF report saved as 'Fire_Risk_Report.pdf'")

def main(place_name, location_point, search_distance, weights):
    """Orchestrates the entire analysis and map generation."""
    gdf_wgs84 = gpd.GeoDataFrame(geometry=[Point(location_point[1], location_point[0])], crs="EPSG:4326")
    target_crs = gdf_wgs84.estimate_utm_crs()
    graph, buildings, accessible_roads, water_sources, fire_stations = get_geospatial_data(location_point, search_distance, target_crs)
    if graph is None or buildings.empty:
        raise ValueError("No buildings or road network found. Cannot generate analysis.")
    density_grid = calculate_density_grid(buildings)
    buildings_with_travel_risk = calculate_travel_risk(buildings, fire_stations, graph)
    buildings_with_all_risks = calculate_water_risk(buildings_with_travel_risk, water_sources)
    final_risk_grid = calculate_composite_risk(density_grid, buildings_with_all_risks, weights)
    save_footprints_map(buildings, graph, 'building_footprints.png')
    save_roads_map(accessible_roads, graph, 'road_network.png')
    generate_static_risk_map(final_risk_grid, graph)
    generate_interactive_risk_map(final_risk_grid)
    generate_pdf_report(place_name, search_distance, weights, final_risk_grid)

if __name__ == "__main__":
    test_place = "Korail, Dhaka"
    test_point = (23.774, 90.405)
    search_dist = 1000
    test_weights = {"density": 0.33, "access": 0.33, "water": 0.34}
    main(test_place, test_point, search_dist, test_weights)