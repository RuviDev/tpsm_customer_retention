import streamlit as st

from utils.app_config import CAT1_MODEL_PATH, CAT2_MODEL_PATH
from utils.model_loader import load_model_bundle
from utils.prediction_helpers import (
    initialize_session_state,
    reset_predictions,
    predict_from_bundle,
    combine_weighted_predictions,
    probability_to_percentage,
    get_risk_level,
)
from utils.ui_helpers import load_css, page_header, result_card, locked_card


load_css()
initialize_session_state()

page_header(
    "Prediction",
    "Test Category 1 and Category 2 models, then generate the final weighted prediction."
)

try:
    cat1_bundle = load_model_bundle(CAT1_MODEL_PATH)
    cat2_bundle = load_model_bundle(CAT2_MODEL_PATH)
except Exception as error:
    st.error(str(error))
    st.stop()

cat1_metadata = cat1_bundle["metadata"]
cat2_metadata = cat2_bundle["metadata"]


def get_options(metadata, column_name, fallback=None):
    fallback = fallback or []
    options = metadata.get("categorical_options", {}).get(column_name, fallback)

    if not options:
        options = fallback

    return options


with st.sidebar:
    st.markdown("### Prediction Progress")

    st.write("✅ Category 1" if st.session_state["cat1_prediction_done"] else "⬜ Category 1")
    st.write("✅ Category 2" if st.session_state["cat2_prediction_done"] else "⬜ Category 2")
    st.write("✅ Weighted Final" if st.session_state["weighted_prediction_done"] else "⬜ Weighted Final")

    st.divider()

    if st.button("Reset All Predictions"):
        reset_predictions()
        st.rerun()


tab1, tab2 = st.tabs([
    "Category 1: Responsiveness & Support Speed",
    "Category 2: Interaction Experience"
])


with tab1:
    st.markdown("## Category 1: Responsiveness & Support Speed")
    st.caption("This model uses support speed and responsiveness-related service variables.")

    with st.form("cat1_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            channel_name = st.selectbox(
                "Channel Name",
                get_options(cat1_metadata, "channel_name")
            )

            category = st.selectbox(
                "Category",
                get_options(cat1_metadata, "category")
            )

            sub_category = st.selectbox(
                "Sub Category",
                get_options(cat1_metadata, "sub_category")
            )

            tenure_bucket = st.selectbox(
                "Tenure Bucket",
                get_options(cat1_metadata, "tenure_bucket")
            )

        with col2:
            agent_shift = st.selectbox(
                "Agent Shift",
                get_options(cat1_metadata, "agent_shift")
            )

            csat_score = st.slider(
                "CSAT Score",
                min_value=1,
                max_value=5,
                value=4,
                step=1
            )

            response_minutes = st.number_input(
                "Response Minutes",
                min_value=0.0,
                value=5.0,
                step=1.0
            )

        submitted_cat1 = st.form_submit_button("Predict Category 1")

    if submitted_cat1:
        cat1_input = {
            "channel_name": channel_name,
            "category": category,
            "sub_category": sub_category,
            "tenure_bucket": tenure_bucket,
            "agent_shift": agent_shift,
            "csat_score": csat_score,
            "response_minutes": response_minutes,
        }

        try:
            result_df = predict_from_bundle(
                cat1_bundle,
                cat1_input,
                normalize_text=False
            )

            probability = float(result_df["retained_probability"].iloc[0])
            label = int(result_df["predicted_retention_label"].iloc[0])
            prediction_text = result_df["prediction_text"].iloc[0]

            st.session_state["cat1_prediction_done"] = True
            st.session_state["cat1_probability"] = probability
            st.session_state["cat1_label"] = label
            st.session_state["cat1_result_df"] = result_df

            st.success("Category 1 prediction completed.")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if st.session_state["cat1_prediction_done"]:
        probability = st.session_state["cat1_probability"]
        prediction_text = st.session_state["cat1_result_df"]["prediction_text"].iloc[0]

        result_card(
            "Category 1 Result",
            prediction_text,
            probability_to_percentage(probability),
            get_risk_level(probability),
        )

        st.progress(probability, text=f"Retained Probability: {probability_to_percentage(probability)}")


with tab2:
    st.markdown("## Category 2: Interaction Experience")
    st.caption("This model uses customer sentiment, service quality, agent behavior, and outcome quality.")

    with st.form("cat2_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            issue_category = st.selectbox(
                "Issue Category",
                get_options(cat2_metadata, "issue_category")
            )

            customer_sentiment = st.selectbox(
                "Customer Sentiment",
                get_options(cat2_metadata, "customer_sentiment")
            )

            issue_complexity = st.selectbox(
                "Issue Complexity",
                get_options(cat2_metadata, "issue_complexity")
            )

            agent_experience_level = st.selectbox(
                "Agent Experience Level",
                get_options(cat2_metadata, "agent_experience_level")
            )

            agent_politeness = st.selectbox(
                "Agent Politeness",
                get_options(cat2_metadata, "agent_politeness")
            )

            agent_empathy = st.selectbox(
                "Agent Empathy",
                get_options(cat2_metadata, "agent_empathy")
            )

        with col2:
            agent_professionalism = st.selectbox(
                "Agent Professionalism",
                get_options(cat2_metadata, "agent_professionalism")
            )

            communication_clarity = st.selectbox(
                "Communication Clarity",
                get_options(cat2_metadata, "communication_clarity")
            )

            helpfulness = st.selectbox(
                "Helpfulness",
                get_options(cat2_metadata, "helpfulness")
            )

            customer_tone_observed = st.selectbox(
                "Customer Tone Observed",
                get_options(cat2_metadata, "customer_tone_observed")
            )

            outcome_quality = st.selectbox(
                "Outcome Quality",
                get_options(cat2_metadata, "outcome_quality")
            )

        st.markdown("### Interaction Behavior Indicators")

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)

        with bcol1:
            apology_present_text = st.radio(
                "Apology Present?",
                ["Yes", "No"],
                horizontal=True
            )

        with bcol2:
            patience_present_text = st.radio(
                "Patience Present?",
                ["Yes", "No"],
                horizontal=True
            )

        with bcol3:
            active_listening_present_text = st.radio(
                "Active Listening?",
                ["Yes", "No"],
                horizontal=True
            )

        with bcol4:
            deescalation_present_text = st.radio(
                "De-escalation?",
                ["Yes", "No"],
                horizontal=True
            )

        submitted_cat2 = st.form_submit_button("Predict Category 2")

    if submitted_cat2:
        cat2_input = {
            "issue_category": issue_category,
            "customer_sentiment": customer_sentiment,
            "issue_complexity": issue_complexity,
            "agent_experience_level": agent_experience_level,
            "agent_politeness": agent_politeness,
            "agent_empathy": agent_empathy,
            "agent_professionalism": agent_professionalism,
            "communication_clarity": communication_clarity,
            "apology_present": 1 if apology_present_text == "Yes" else 0,
            "patience_present": 1 if patience_present_text == "Yes" else 0,
            "active_listening_present": 1 if active_listening_present_text == "Yes" else 0,
            "deescalation_present": 1 if deescalation_present_text == "Yes" else 0,
            "helpfulness": helpfulness,
            "customer_tone_observed": customer_tone_observed,
            "outcome_quality": outcome_quality,
        }

        try:
            result_df = predict_from_bundle(
                cat2_bundle,
                cat2_input,
                normalize_text=True
            )

            probability = float(result_df["retained_probability"].iloc[0])
            label = int(result_df["predicted_retention_label"].iloc[0])
            prediction_text = result_df["prediction_text"].iloc[0]

            st.session_state["cat2_prediction_done"] = True
            st.session_state["cat2_probability"] = probability
            st.session_state["cat2_label"] = label
            st.session_state["cat2_result_df"] = result_df

            st.success("Category 2 prediction completed.")

        except Exception as error:
            st.error(f"Prediction failed: {error}")

    if st.session_state["cat2_prediction_done"]:
        probability = st.session_state["cat2_probability"]
        prediction_text = st.session_state["cat2_result_df"]["prediction_text"].iloc[0]

        result_card(
            "Category 2 Result",
            prediction_text,
            probability_to_percentage(probability),
            get_risk_level(probability),
        )

        st.progress(probability, text=f"Retained Probability: {probability_to_percentage(probability)}")


st.divider()

st.markdown("## Final Weighted Prediction")

if not st.session_state["cat1_prediction_done"] or not st.session_state["cat2_prediction_done"]:
    locked_card("Complete both Category 1 and Category 2 predictions to unlock the final weighted prediction.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.write("✅ Category 1 completed" if st.session_state["cat1_prediction_done"] else "⬜ Category 1 not completed")

    with col2:
        st.write("✅ Category 2 completed" if st.session_state["cat2_prediction_done"] else "⬜ Category 2 not completed")

else:
    col1, col2 = st.columns(2)

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

    if st.button("Generate Weighted Final Prediction", type="primary"):
        weighted_result = combine_weighted_predictions(
            st.session_state["cat1_probability"],
            st.session_state["cat2_probability"],
        )

        st.session_state["weighted_result"] = weighted_result
        st.session_state["weighted_prediction_done"] = True
        st.session_state["final_probability"] = weighted_result["final_retention_probability"]
        st.session_state["final_label"] = weighted_result["final_prediction_label"]

        st.rerun()

    if st.session_state["weighted_prediction_done"]:
        weighted_result = st.session_state["weighted_result"]

        result_card(
            "Final Weighted Prediction",
            weighted_result["final_prediction_text"],
            probability_to_percentage(weighted_result["final_retention_probability"]),
            weighted_result["risk_level"],
        )

        st.progress(
            weighted_result["final_retention_probability"],
            text=f"Final Probability: {probability_to_percentage(weighted_result['final_retention_probability'])}"
        )

        with st.expander("View Weighted Calculation"):
            st.write(f"Category 1 Weight: {weighted_result['category_1_weight'] * 100:.0f}%")
            st.write(f"Category 2 Weight: {weighted_result['category_2_weight'] * 100:.0f}%")
            st.write(f"Category 1 Contribution: {probability_to_percentage(weighted_result['category_1_contribution'])}")
            st.write(f"Category 2 Contribution: {probability_to_percentage(weighted_result['category_2_contribution'])}")