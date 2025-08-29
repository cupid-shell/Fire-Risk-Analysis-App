import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

def get_geospatial_data(location_point, distance, target_crs):
    """Downloads and projects road network and building data for a given point."""
    print("Fetching data...")
    G = ox.graph_from_point(location_point, dist=distance, network_type='all')
    G_proj = ox.project_graph(G, to_crs=target_crs)

    tags = {"building": True}
    gdf_buildings = ox.features_from_point(location_point, tags, dist=distance)
    buildings_proj = gdf_buildings.to_crs(target_crs)
    
    gdf_edges = ox.graph_to_gdfs(G_proj, nodes=False, edges=True)
    road_types = ['primary', 'secondary', 'tertiary', 'residential', 'service', 'unclassified']
    accessible_roads = gdf_edges[gdf_edges['highway'].isin(road_types)]

    print("Data fetching complete!")
    return G_proj, buildings_proj, accessible_roads

def calculate_density_grid(buildings_proj, cell_size=50):
    """Calculates building density on a grid."""
    print("Calculating density grid...")
    xmin, ymin, xmax, ymax = buildings_proj.total_bounds
    
    grid_cells = []
    for x0 in np.arange(xmin, xmax + cell_size, cell_size):
        for y0 in np.arange(ymin, ymax + cell_size, cell_size):
            x1 = x0 - cell_size
            y1 = y0 + cell_size
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

def calculate_access_risk(buildings_proj, accessible_roads):
    """Calculates the distance from each building to the nearest accessible road."""
    print("Calculating access risk...")
    accessible_network = accessible_roads.union_all()
    
    buildings_with_risk = buildings_proj.copy()
    buildings_with_risk['distance_to_access'] = buildings_with_risk.geometry.apply(lambda geom: geom.distance(accessible_network))
    
    print("Access risk calculation complete!")
    return buildings_with_risk

def generate_composite_risk_map(density_grid, buildings_with_access, graph):
    """Combines risks and generates the final map with a continuous color scale."""
    print("Generating final composite risk map...")

    buildings_centroids = buildings_with_access.copy()
    buildings_centroids['geometry'] = buildings_centroids.geometry.centroid
    
    merged = gpd.sjoin(buildings_centroids, density_grid, how='left', predicate='within')
    avg_distance_in_grid = merged.groupby('index_right')['distance_to_access'].mean().reset_index()
    
    grid = density_grid.merge(avg_distance_in_grid, left_index=True, right_on='index_right', how='left')
    grid = grid.rename(columns={'distance_to_access': 'avg_distance'})
    grid['avg_distance'] = grid['avg_distance'].fillna(0)

    grid['density_risk'] = (grid['n_buildings'] - grid['n_buildings'].min()) / (grid['n_buildings'].max() - grid['n_buildings'].min())
    grid['access_risk'] = (grid['avg_distance'] - grid['avg_distance'].min()) / (grid['avg_distance'].max() - grid['avg_distance'].min())
    grid['final_risk'] = (grid['density_risk'] * 0.5) + (grid['access_risk'] * 0.5)

    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=0, bgcolor='#060606', edge_color='w', edge_linewidth=0.1)
    grid.plot(column='final_risk', ax=ax, cmap='inferno', alpha=0.7, legend=True)
    ax.set_title('Composite Fire Risk Map\n(High-Density + Poor Access = High Risk)', color='white')

    fig.savefig('final_risk_map.png', dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='#060606')
    print("\nFinal risk map saved as 'final_risk_map.png'")
    
def save_footprints_map(buildings, graph, filepath):
    """Saves a map of the building footprints."""
    print(f"Saving building footprints map to {filepath}...")
    fig, ax = ox.plot_graph(graph, show=False, close=True, bgcolor='#060606', edge_color='grey', edge_linewidth=0.2, node_size=0)
    buildings.plot(ax=ax, color='cyan', alpha=0.7)
    ax.set_title('Building Footprints', color='white')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='#060606')
    plt.close(fig) 

def save_roads_map(accessible_roads, graph, filepath):
    """Saves a map of the road network."""
    print(f"Saving road network map to {filepath}...")
    fig, ax = ox.plot_graph(graph, show=False, close=True, bgcolor='#060606', edge_color='#333333', edge_linewidth=0.5, node_size=0)
    accessible_roads.plot(ax=ax, color='yellow', linewidth=1)
    ax.set_title('Accessible Road Network', color='white')
    fig.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='#060606')
    plt.close(fig) 

def main(location_point, search_distance, projected_crs):
    """
    The main function to orchestrate the entire fire risk analysis.
    """
    graph, buildings, accessible_roads = get_geospatial_data(location_point, search_distance, projected_crs)
    density_grid = calculate_density_grid(buildings)
    buildings_with_access_risk = calculate_access_risk(buildings, accessible_roads)
    save_footprints_map(buildings, graph, 'building_footprints.png')
    save_roads_map(accessible_roads, graph, 'road_network.png')
    generate_composite_risk_map(density_grid, buildings_with_access_risk, graph)

if __name__ == "__main__":
    korail_point = (23.774, 90.405)
    search_dist = 1000
    dhaka_crs = "EPSG:32646"
    main(korail_point, search_dist, dhaka_crs)