import streamlit as st
import os
import sys

# Add the parent directory to the path to allow imports from `app.pages`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


st.set_page_config(
    page_title="Deep Learning Age Prediction",
    page_icon="👴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Navigation")

PAGES = {
    "Home": "app.pages.home",
    "Live Prediction": "app.pages.live_prediction",
    "Evaluation Metrics": "app.pages.evaluation_metrics"
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Dynamically import the selected page
if selection in PAGES:
    page_module_name = PAGES[selection]
    module = __import__(page_module_name, fromlist=[""])
    if hasattr(module, "main"):
        module.main()
    else:
        st.error(f"Page '{selection}' does not have a 'main' function. Please ensure your page script defines a 'main' function.")
