import pandas as pd
import streamlit as st

from utils.app_config import (
    CAT1_WEIGHT,
    CAT2_WEIGHT,
    DEFAULT_THRESHOLD,
    CLASS_LABELS,
)


def initialize_session_state():
    defaults = {
        "cat1_prediction_done": False,
        "cat2_prediction_done": False,
        "weighted_prediction_done": False,

        "cat1_probability": None,
        "cat2_probability": None,
        "final_probability": None,

        "cat1_label": None,
        "cat2_label": None,
        "final_label": None,

        "cat1_result_df": None,
        "cat2_result_df": None,
        "weighted_result": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_predictions():
    keys_to_reset = [
        "cat1_prediction_done",
        "cat2_prediction_done",
        "weighted_prediction_done",
        "cat1_probability",
        "cat2_probability",
        "final_probability",
        "cat1_label",
        "cat2_label",
        "final_label",
        "cat1_result_df",
        "cat2_result_df",
        "weighted_result",
    ]

    for key in keys_to_reset:
        if key in st.session_state:
            if key.endswith("_done"):
                st.session_state[key] = False
            else:
                st.session_state[key] = None


def get_retained_probability(model, input_df):
    probabilities = model.predict_proba(input_df)

    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps["model"], "classes_", None)

    if classes is None:
        retained_index = 1
    else:
        classes = list(classes)
        retained_index = classes.index(1)

    return probabilities[:, retained_index]


def predict_from_bundle(bundle, input_data, normalize_text=False):
    model = bundle["model"]
    metadata = bundle["metadata"]

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()

    expected_cols = metadata["feature_columns"]

    missing_cols = [col for col in expected_cols if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required input columns: {missing_cols}")

    input_df = input_df[expected_cols].copy()

    if normalize_text:
        for col in input_df.select_dtypes(include=["object"]).columns:
            input_df[col] = input_df[col].astype(str).str.strip().str.lower()

    prediction = model.predict(input_df)

    if hasattr(model, "predict_proba"):
        retained_probability = get_retained_probability(model, input_df)
    else:
        retained_probability = [None] * len(input_df)

    result_df = input_df.copy()
    result_df["predicted_retention_label"] = prediction
    result_df["prediction_text"] = result_df["predicted_retention_label"].map(CLASS_LABELS)
    result_df["retained_probability"] = retained_probability

    return result_df


def combine_weighted_predictions(
    cat1_probability,
    cat2_probability,
    cat1_weight=CAT1_WEIGHT,
    cat2_weight=CAT2_WEIGHT,
    threshold=DEFAULT_THRESHOLD,
):
    total_weight = cat1_weight + cat2_weight

    if total_weight == 0:
        raise ValueError("Total weight cannot be zero.")

    cat1_weight = cat1_weight / total_weight
    cat2_weight = cat2_weight / total_weight

    final_probability = (
        cat1_probability * cat1_weight
        + cat2_probability * cat2_weight
    )

    final_label = 1 if final_probability >= threshold else 0

    return {
        "category_1_probability": cat1_probability,
        "category_2_probability": cat2_probability,
        "category_1_weight": cat1_weight,
        "category_2_weight": cat2_weight,
        "category_1_contribution": cat1_probability * cat1_weight,
        "category_2_contribution": cat2_probability * cat2_weight,
        "final_retention_probability": final_probability,
        "final_prediction_label": final_label,
        "final_prediction_text": CLASS_LABELS[final_label],
        "threshold": threshold,
        "risk_level": get_risk_level(final_probability),
    }


def get_risk_level(probability):
    if probability >= 0.70:
        return "Low Risk - Likely to Return"

    if probability >= 0.40:
        return "Medium Risk - Needs Attention"

    return "High Risk - Not Likely to Return"


def probability_to_percentage(probability):
    if probability is None:
        return "N/A"

    return f"{probability * 100:.2f}%"