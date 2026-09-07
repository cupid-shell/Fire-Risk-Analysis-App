import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import Point, Polygon
from fire_risk_analyzer import (
    ahp_weights,
    calculate_network_vulnerability,
    run_analysis_pipeline,
    calculate_density_grid,
    calculate_composite_risk,
    calculate_combustibility,
    calculate_travel_risk,
    classify_risk_bands,
    monte_carlo_uncertainty,
)

def test_ahp_weights():
    """Verify that AHP weights are correctly calculated and consistency ratio works."""
    # Perfectly consistent 5x5 matrix
    perfect_matrix = [
        [1.0, 2.0, 4.0, 2.0, 4.0],
        [0.5, 1.0, 2.0, 1.0, 2.0],
        [0.25, 0.5, 1.0, 0.5, 1.0],
        [0.5, 1.0, 2.0, 1.0, 2.0],
        [0.25, 0.5, 1.0, 0.5, 1.0]
    ]
    weights, cr = ahp_weights(perfect_matrix)
    assert isinstance(weights, dict)
    assert len(weights) == 5
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert cr < 0.1  # Perfectly consistent matrix should have CR close to 0

    # Test names
    for key in ['density', 'access', 'water', 'height', 'hazard']:
        assert key in weights

def test_calculate_density_grid():
    """Verify that building density, GFA and counts are correctly aggregated into cells."""
    # Create 3 buildings in EPSG:32646 (UTM zone)
    b1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    b2 = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    b3 = Polygon([(40, 40), (50, 40), (50, 50), (40, 50)])
    
    buildings_gdf = gpd.GeoDataFrame(
        {'levels': [1, 2, 3], 'building': ['residential', 'commercial', 'yes']},
        geometry=[b1, b2, b3],
        crs="EPSG:32646"
    )
    
    # Calculate density grid (cell_size=50)
    grid = calculate_density_grid(buildings_gdf, cell_size=50)
    assert not grid.empty
    assert 'n_buildings' in grid.columns
    assert 'total_gfa' in grid.columns
    assert grid['n_buildings'].sum() == 3

def test_calculate_combustibility():
    """Verify building combustibility scores map correctly based on tags."""
    b1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    b2 = Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
    
    buildings_gdf = gpd.GeoDataFrame(
        {'building:material': ['wood', 'concrete'], 'building': ['industrial', 'government']},
        geometry=[b1, b2],
        crs="EPSG:32646"
    )
    
    res = calculate_combustibility(buildings_gdf)
    assert 'combustibility' in res.columns
    # Wood + Industrial should be much more combustible than Concrete + Government
    assert res.iloc[0]['combustibility'] > res.iloc[1]['combustibility']

def test_calculate_network_vulnerability():
    """Test road network dead-end calculation."""
    # Create a 2x2 grid cell setup
    cell1 = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    cell2 = Polygon([(50, 0), (100, 0), (100, 50), (50, 50)])
    grid = gpd.GeoDataFrame(geometry=[cell1, cell2], crs="EPSG:32646")

    # Create a mock projected street graph with two nodes: one is a dead end (degree 1)
    g = nx.MultiDiGraph(crs="EPSG:32646")
    g.add_node(1, x=25.0, y=25.0)
    g.add_node(2, x=75.0, y=25.0)
    # Undirected degree will be 1 for both if there's only 1 edge
    g.add_edge(1, 2, key=0, highway='residential', length=50.0)

    # Calculate
    vulnerable_grid = calculate_network_vulnerability(grid, g)
    assert 'network_vulnerability' in vulnerable_grid.columns
    assert vulnerable_grid.iloc[0]['network_vulnerability'] >= 0.0

def test_run_analysis_pipeline():
    """Verify run_analysis_pipeline executes successfully with mock datasets."""
    # Coordinates for central Dhaka (Korail)
    location_point = (23.774, 90.405)
    weights = {"density": 0.3, "access": 0.3, "water": 0.2, "height": 0.1, "hazard": 0.1}

    # Set up mock datasets in EPSG:32646 (projected UTM)
    b1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    buildings = gpd.GeoDataFrame(
        {'levels': [2], 'building': ['residential'], 'building:material': ['brick']},
        geometry=[b1],
        crs="EPSG:32646"
    )

    roads = gpd.GeoDataFrame(
        {'highway': ['residential'], 'lanes': ['2']},
        geometry=[Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)])],
        crs="EPSG:32646"
    )

    water = gpd.GeoDataFrame(
        {'natural': ['water']},
        geometry=[Point(20, 20)],
        crs="EPSG:32646"
    )

    stations = gpd.GeoDataFrame(
        {'amenity': ['fire_station']},
        geometry=[Point(-10, -10)],
        crs="EPSG:32646"
    )

    hazards = gpd.GeoDataFrame(
        {'amenity': ['fuel']},
        geometry=[Point(30, -30)],
        crs="EPSG:32646"
    )

    # Create mock MultiDiGraph
    g = nx.MultiDiGraph(crs="EPSG:32646")
    # ox.nearest_nodes searches nodes by x and y coordinates
    # We must add nodes with x, y attributes corresponding to the centroids of buildings & stations
    g.add_node(1, x=5.0, y=5.0)       # building centroid node
    g.add_node(2, x=-10.0, y=-10.0)   # fire station centroid node
    g.add_edge(1, 2, key=0, highway='residential', travel_time=10.0)

    # Run the pipeline with pre-fetched mock data
    res = run_analysis_pipeline(
        location_point=location_point,
        search_distance=1000,
        weights=weights,
        road_types=['residential'],
        generate_maps=False,
        graph=g,
        accessible_roads=roads,
        buildings=buildings,
        water_sources=water,
        fire_stations=stations,
        hazards=hazards
    )

    assert isinstance(res, dict)
    assert "final_risk_grid" in res
    assert res["n_buildings"] == 1
    assert res["n_stations"] == 1
    assert res["n_water"] == 1
    assert res["n_hazards"] == 1
    assert len(res["recs"]) > 0

    grid = res["final_risk_grid"]
    assert 'final_risk' in grid.columns
    assert 'risk_band' in grid.columns
    assert 'is_vacant' in grid.columns

def test_classify_risk_bands_low_variance():
    """Verify that classify_risk_bands handles degenerate / uniform scores without ValueError."""
    # Create a grid where all cells have the exact same score
    cell1 = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    cell2 = Polygon([(50, 0), (100, 0), (100, 50), (50, 50)])
    grid = gpd.GeoDataFrame({'final_risk': [0.20, 0.20]}, geometry=[cell1, cell2], crs="EPSG:32646")

    # This previously could fail with ValueError in jenkspy; now it falls back gracefully
    classified = classify_risk_bands(grid, n_classes=4)
    assert 'risk_band' in classified.columns
    assert (classified['risk_band'] == 'Low').all()

def test_calculate_travel_risk_multi_source():
    """Verify multi-source Dijkstra resolves shortest paths across multiple stations and disconnected nodes."""
    b1 = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    b2 = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
    buildings = gpd.GeoDataFrame(geometry=[b1, b2], crs="EPSG:32646")

    stations = gpd.GeoDataFrame(
        {'amenity': ['fire_station']},
        geometry=[Point(-10, -10)],
        crs="EPSG:32646"
    )

    g = nx.MultiDiGraph(crs="EPSG:32646")
    g.add_node(1, x=5.0, y=5.0)       # b1
    g.add_node(2, x=-10.0, y=-10.0)   # station
    g.add_node(3, x=105.0, y=105.0)   # b2 (disconnected)

    # Edge from station to b1
    g.add_edge(2, 1, key=0, highway='residential', travel_time=25.0)

    res = calculate_travel_risk(buildings, stations, g)
    assert 'travel_time' in res.columns
    # b1 is connected (travel time 25.0)
    assert res.iloc[0]['travel_time'] == 25.0
    # b2 is disconnected (should fallback to default max 480.0)
    assert res.iloc[1]['travel_time'] == 480.0

def test_empty_cell_vacant_handling():
    """Verify that cells with no buildings are flagged is_vacant=True and have zero built fire risk."""
    cell1 = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    cell2 = Polygon([(50, 0), (100, 0), (100, 50), (50, 50)])
    density_grid = gpd.GeoDataFrame(
        {'n_buildings': [1, 0]},
        geometry=[cell1, cell2],
        crs="EPSG:32646"
    )

    # Building in cell 1 only
    b1 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    buildings = gpd.GeoDataFrame(
        {'travel_time': [60.0], 'distance_to_water': [100.0]},
        geometry=[b1],
        crs="EPSG:32646"
    )

    weights = {"density": 0.3, "access": 0.3, "water": 0.4}
    res = calculate_composite_risk(density_grid, buildings, weights)

    assert 'is_vacant' in res.columns
    assert not res.iloc[0]['is_vacant']
    assert res.iloc[1]['is_vacant']
    # Occupied cell has computed risk
    assert res.iloc[0]['final_risk'] > 0
    # Vacant cell has 0 built risk
    assert res.iloc[1]['final_risk'] == 0.0

def test_monte_carlo_uncertainty():
    """Verify vectorized monte_carlo_uncertainty runs rapidly and outputs mean and std per cell."""
    cell1 = Polygon([(0, 0), (50, 0), (50, 50), (0, 50)])
    density_grid = gpd.GeoDataFrame(
        {'n_buildings': [2], 'total_gfa': [200.0], 'avg_lanes': [2.0]},
        geometry=[cell1],
        crs="EPSG:32646"
    )
    b1 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
    buildings = gpd.GeoDataFrame(
        {'travel_time': [100.0], 'distance_to_water': [150.0], 'levels': [2], 'combustibility': [0.5]},
        geometry=[b1],
        crs="EPSG:32646"
    )
    weights = {"density": 0.3, "access": 0.3, "water": 0.2, "height": 0.1, "hazard": 0.1}
    mc = monte_carlo_uncertainty(density_grid, buildings, weights, n_simulations=100)

    assert not mc.empty
    assert 'risk_mean' in mc.columns
    assert 'risk_std' in mc.columns
    assert 'risk_cv' in mc.columns
    assert mc.iloc[0]['risk_mean'] > 0.0
    assert 0.0 <= mc.iloc[0]['risk_cv'] <= 1.0


