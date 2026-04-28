import streamlit as st

from utils.app_config import CAT1_WEIGHT, CAT2_WEIGHT
from utils.prediction_helpers import initialize_session_state, probability_to_percentage
from utils.ui_helpers import load_css, page_header, info_card

load_css()
initialize_session_state()

page_header(
    "Dashboard",
    "Overview of the predictive analytics system."
)

st.markdown("### Project Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Category 1 Weight", f"{CAT1_WEIGHT * 100:.0f}%")

with col2:
    st.metric("Category 2 Weight", f"{CAT2_WEIGHT * 100:.0f}%")

with col3:
    st.metric("Prediction Type", "Binary")

with col4:
    st.metric("Final Threshold", "50%")

st.markdown("### Current Prediction Status")

status1 = "Completed" if st.session_state["cat1_prediction_done"] else "Pending"
status2 = "Completed" if st.session_state["cat2_prediction_done"] else "Pending"
status3 = "Completed" if st.session_state["weighted_prediction_done"] else "Pending"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Category 1", status1)

with col2:
    st.metric("Category 2", status2)

with col3:
    st.metric("Weighted Final", status3)

if st.session_state["cat1_probability"] is not None or st.session_state["cat2_probability"] is not None:
    st.markdown("### Latest Prediction Probabilities")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Category 1 Probability",
            probability_to_percentage(st.session_state["cat1_probability"])
        )

    with col2:
        st.metric(
            "Category 2 Probability",
            probability_to_percentage(st.session_state["cat2_probability"])
        )

    with col3:
        st.metric(
            "Final Probability",
            probability_to_percentage(st.session_state["final_probability"])
        )

st.markdown("### Category Overview")

col1, col2 = st.columns(2)

with col1:
    info_card(
        "Responsiveness & Support Speed",
        "Focuses on the operational service side: how fast and responsive the support interaction was."
    )

with col2:
    info_card(
        "Interaction Experience",
        "Focuses on the human service quality side: how polite, helpful, clear, and successful the interaction was."
    )