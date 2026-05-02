import streamlit as st

from utils.prediction_helpers import initialize_session_state
from utils.ui_helpers import load_css


st.set_page_config(
    page_title="Customer Retention Prediction System",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_css()
initialize_session_state()

dashboard_page = st.Page(
    "pages/dashboard.py",
    title="Dashboard",
    icon=":material/dashboard:",
    default=True,
)

descriptive_page = st.Page(
    "pages/descriptive_statistics.py",
    title="Descriptive Statistics",
    icon=":material/analytics:",
)

prediction_page = st.Page(
    "pages/prediction.py",
    title="Prediction",
    icon=":material/visibility:",
)

weighted_page = st.Page(
    "pages/weighted_final_prediction.py",
    title="Weighted Final Prediction",
    icon=":material/balance:",
)

about_page = st.Page(
    "pages/about_project.py",
    title="Project Overview",
    icon=":material/info:",
)

pg = st.navigation(
    {
        "Analytics": [dashboard_page, descriptive_page],
        "Prediction": [prediction_page, weighted_page],
        "Project": [about_page],
    },
    expanded=True,
)

pg.run()