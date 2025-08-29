import streamlit as st
from geopy.geocoders import Nominatim

from fire_risk_analyzer import main as run_analysis

st.set_page_config(layout="wide")
st.title("🔥 Fire Risk Analysis Tool")
st.write("Enter the name of an urban settlement and select a radius to analyze its fire risk.")

location_query = st.text_input("Enter Location Name (e.g., 'Kibera, Nairobi')", "Korail, Dhaka")
search_dist = st.slider("Select Search Radius (meters)", 500, 2000, 1000)

if st.button("Analyze Location"):
    if location_query:
        try:
            geolocator = Nominatim(user_agent="fire_risk_app")
            location = geolocator.geocode(location_query)
            
            if location:
                st.info(f"Found location: {location.address}")
                location_point = (location.latitude, location.longitude)
                dhaka_crs = "EPSG:32646"

                with st.spinner('Running analysis... This may take a minute or two.'):
                    run_analysis(location_point, search_dist, dhaka_crs)
                
                st.success("Analysis Complete!")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Building Footprints")
                    st.image('building_footprints.png')

                with col2:
                    st.subheader("Road Network")
                    st.image('road_network.png')

                with col3:
                    st.subheader("Composite Fire Risk")
                    st.image('final_risk_map.png')

            else:
                st.error("Could not find the location. Please try a different query (e.g., 'Settlement, City').")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a location name.")