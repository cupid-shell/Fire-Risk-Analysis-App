import streamlit as st
from geopy.geocoders import Nominatim
import streamlit.components.v1 as components


from fire_risk_analyzer import main as run_analysis

st.set_page_config(layout="wide")
st.title("🔥 Fire Risk Analysis Tool (FRAT)")
st.write("Enter the name of an urban settlement and select a radius to analyze its fire risk.")

if 'maps_generated' not in st.session_state:
    st.session_state.maps_generated = False

input_method = st.radio(
    "Choose input method:",
    ('Search by Name', 'Enter Coordinates')
)

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

st.subheader("Adjust Risk Factor Weights")
col1_weights, col2_weights, col3_weights = st.columns(3)
with col1_weights:
    density_weight = st.slider("Density Importance", 0, 100, 33)
with col2_weights:
    access_weight = st.slider("Access Importance", 0, 100, 33)
with col3_weights:
    water_weight = st.slider("Water Importance", 0, 100, 34)

if st.button("Analyze Location"):
    weights = {"density": (density_weight/100), "access": (access_weight/100), "water": (water_weight/100)}
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    else:
        st.error("Total weight cannot be zero."); st.stop()
    
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
        st.info(f"Running analysis with weights: Density ({weights['density']:.0%}), Access ({weights['access']:.0%}), Water ({weights['water']:.0%})")
        with st.spinner('Running analysis and generating report...'):
            try:
                run_analysis(location_query, location_point, search_dist, weights)
                st.success("Analysis Complete!")
                st.session_state.maps_generated = True
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}"); st.exception(e)

if st.session_state.maps_generated:
    st.markdown("---")
    st.subheader("Download Full Report")
    try:
        with open("Fire_Risk_Report.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button(label="Download Report (PDF)", data=PDFbyte, file_name=f"Fire_Risk_Report_{location_query.replace(' ', '_')}.pdf", mime='application/octet-stream')
    except FileNotFoundError:
        st.warning("PDF Report not found. Please run the analysis again.")

    st.subheader("Analysis Summary Maps")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image('building_footprints.png', caption='Building Footprints', use_container_width=True)
    with col2:
        st.image('road_network.png', caption='Accessible Road Network', use_container_width=True)
    with col3:
        st.image('final_risk_map.png', caption='Static Composite Risk', use_container_width=True)

    st.markdown("---")
    st.subheader("Interactive Fire Risk Map")
    try:
        with open('interactive_risk_map.html', 'r', encoding='utf-8') as f:
            map_html = f.read()
        components.html(map_html, height=600, scrolling=True)
    except FileNotFoundError:
        st.error("Interactive map file not found. Please run the analysis again.")
    
    st.markdown("---")
    st.caption("Created by Avishek Adhikari | avishek.jidpus@gmail.com")
