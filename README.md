# Fire Risk Analysis Tool (FRAT)

## Project Description
This project is a Python-based tool that analyzes fire risk in urban settlements. It uses geospatial data from OpenStreetMap to calculate a composite risk score based on **building density**, **network travel distance to the nearest fire station**, and **proximity to water sources**. The results are displayed in a powerful, interactive web application built with Streamlit.

The tool is highly interactive, allowing users to define an analysis area by name or coordinates, select a radius, and adjust the weights of each risk factor to perform dynamic scenario analysis.

## Key Features
* **Interactive UI:** Built with Streamlit for a user-friendly experience.
* **Flexible Input:** Analyze an area by searching for its name or by inputting precise Latitude/Longitude coordinates.
* **Risk Model:** Calculates a 3-factor risk score using a realistic network travel-distance model.
* **Analysis:** Features interactive sliders to adjust the weight/importance of density, access, and water risk factors.
* **Visualization:** Displays a dashboard with building footprints, road networks, a high-quality static risk map, and a fully interactive risk map with informational pop-ups.
* **Actionable Reporting:** Generates a downloadable PDF report summarizing the analysis and identifying the "Top 5 Highest-Risk Zones."

## Files Included
- `fire_risk_analyzer.py`: The core backend script that contains all the analysis functions.
- `app.py`: The Streamlit script that runs the user interface.
- `requirements.txt`: A list of all required Python libraries for setup.

---

## How to Set Up and Run the Project

### Prerequisites
You must have Anaconda or Miniconda installed.
- **Download Anaconda:** [https://www.anaconda.com/download](https://www.anaconda.com/download)

### Setup Instructions

1.  **Create and Activate a Conda Environment:**
    ```bash
    conda create --name fire_risk_env python=3.9
    conda activate fire_risk_env
    ```
2.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
The application will then open in your web browser.