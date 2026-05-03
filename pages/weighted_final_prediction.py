import streamlit as st

from utils.app_config import CAT1_WEIGHT, CAT2_WEIGHT
from utils.prediction_helpers import (
    initialize_session_state,
    combine_weighted_predictions,
    probability_to_percentage,
)
from utils.ui_helpers import load_css, page_header, result_card, locked_card

load_css()
initialize_session_state()

page_header(
    "Weighted Final Prediction",
    "Combining Category 1 and Category 2 predictions using topic-based weights."
)

st.markdown("### Weighted Prediction Formula")

st.latex(
    r"""
    Final\ Probability =
    (Category\ 1\ Probability \times 0.60)
    +
    (Category\ 2\ Probability \times 0.40)
    """
)

st.info(
    "Category 1 receives 60% weight and Category 2 receives 40% weight. "
    "Category 2 receives higher weight because it directly represents the quality of the customer interaction."
)

if not st.session_state["cat1_prediction_done"] or not st.session_state["cat2_prediction_done"]:
    locked_card("Both category predictions are required before calculating the weighted final prediction.")
    st.markdown("---")
    st.warning("Go to the Prediction page and complete Category 1 and Category 2 first.")
    st.stop()

cat1_probability = st.session_state["cat1_probability"]
cat2_probability = st.session_state["cat2_probability"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Category 1 Probability", probability_to_percentage(cat1_probability))

with col2:
    st.metric("Category 2 Probability", probability_to_percentage(cat2_probability))

with col3:
    st.metric("Weighting Method", "60% + 40%")

weighted_result = combine_weighted_predictions(
    cat1_probability,
    cat2_probability,
    CAT1_WEIGHT,
    CAT2_WEIGHT,
)

st.session_state["weighted_result"] = weighted_result
st.session_state["weighted_prediction_done"] = True
st.session_state["final_probability"] = weighted_result["final_retention_probability"]
st.session_state["final_label"] = weighted_result["final_prediction_label"]

result_card(
    "Final Weighted Result",
    weighted_result["final_prediction_text"],
    probability_to_percentage(weighted_result["final_retention_probability"]),
    weighted_result["risk_level"],
)

st.progress(
    weighted_result["final_retention_probability"],
    text=f"Final Probability: {probability_to_percentage(weighted_result['final_retention_probability'])}"
)

st.markdown("### Contribution Breakdown")

contribution_df = {
    "Component": [
        "Category 1 Contribution",
        "Category 2 Contribution",
        "Final Retention Probability",
    ],
    "Value": [
        probability_to_percentage(weighted_result["category_1_contribution"]),
        probability_to_percentage(weighted_result["category_2_contribution"]),
        probability_to_percentage(weighted_result["final_retention_probability"]),
    ],
}

st.dataframe(contribution_df, use_container_width=True)

with st.expander("Simple Interpretation"):
    st.write(
        "The final prediction is created by combining both category probabilities. "
        "Category 1 measures responsiveness and support speed, while Category 2 measures interaction experience. "
        "Since interaction experience directly represents the customer's service enjoyment, it receives a higher weight."
    )