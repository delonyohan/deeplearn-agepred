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

# Define pages as a dictionary
PAGES = {
    "Live Prediction": "app.pages.live_prediction",
    "Evaluation Metrics": "app.pages.evaluation_metrics"
}

selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Dynamically import the selected page
if selection in PAGES:
    page_module_name = PAGES[selection]
    # Use __import__ for dynamic import, and reload to ensure fresh import in case of changes
    module = __import__(page_module_name, fromlist=[""])
    # If the page module has a 'main' function, call it
    if hasattr(module, "main"):
        module.main()
    # Otherwise, assume the entire script is the Streamlit app
    else:
        # Re-run the script by setting the current file to the selected page file
        # This is a bit of a hack, but Streamlit's multi-page app feature is still evolving
        # A better approach would be to have each page implement a 'run' function
        # and call it here. For now, we'll just display a message if 'main' is not found.
        st.error(f"Page '{selection}' does not have a 'main' function. Please ensure your page script defines a 'main' function.")
