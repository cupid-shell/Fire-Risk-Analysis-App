# Fire Risk Analysis Tool

## Project Description
This project is a Python-based tool that analyzes fire risk in urban informal settlements. It uses geospatial data from OpenStreetMap to calculate a composite risk score based on building density and emergency vehicle access. The results are displayed in an interactive web application built with Streamlit.

## Files Included
- `fire_risk_analyzer.py`: The core backend script that contains all the analysis functions.
- `app.py`: The Streamlit script that runs the user interface.
- `requirements.txt`: A list of all required Python libraries.

## How to Set Up and Run the Project

1.  **Create a Conda Environment:**
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