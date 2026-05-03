import pandas as pd
import streamlit as st

from utils.ui_helpers import (
    load_css,
    page_header,
    info_card,
    section_title,
    method_box,
)


load_css()

page_header(
    "About Project & Model Performance",
    "Academic explanation of the predictive analytics part and model output summary."
)

section_title("Analytical Statement")

st.info("Customers who enjoy service interactions are more likely to return.")

section_title("Why Two Categories Were Used")

col1, col2 = st.columns(2)

with col1:
    info_card(
        "Category 1: Responsiveness & Support Speed",
        "This category represents the operational side of service interaction. It checks whether faster response, better CSAT score, support channel, issue category, tenure bucket, and agent shift can predict customer retention."
    )

with col2:
    info_card(
        "Category 2: Interaction Experience",
        "This category represents the human interaction side. It checks whether customer sentiment, politeness, empathy, professionalism, helpfulness, communication clarity, tone, and outcome quality can predict customer retention."
    )

section_title("Predictive Analytics Methodology")

st.markdown(
    """
    The predictive analytics part was completed using supervised machine learning.
    The target variable was `retention_label`.

    For both categories:

    - `1` means the customer was retained / returned.
    - `0` means the customer was not retained / did not return.

    The full workflow included dataset loading, cleaning, preprocessing, train-test split,
    baseline comparison, machine learning model training, SMOTE comparison,
    hyperparameter tuning, model evaluation, feature importance analysis,
    model export, and Streamlit testing.
    """
)

section_title("Category 1: Responsiveness & Support Speed")

cat1_metrics = pd.DataFrame(
    [
        {
            "Best Model": "Logistic Regression",
            "Accuracy": "1.0000",
            "Balanced Accuracy": "1.0000",
            "Macro F1-score": "1.0000",
            "ROC-AUC": "1.0000",
            "Main Predictors": "csat_score, response_minutes",
        }
    ]
)

st.dataframe(cat1_metrics, use_container_width=True, hide_index=True)

method_box(
    """
    <b>Conclusion:</b><br>
    Category 1 showed that CSAT score and response minutes are very strong predictors of customer retention.
    Customers with higher CSAT scores and faster response times were more likely to be retained.
    The model achieved perfect performance because CSAT score strongly separated retained and not-retained customers.
    """
)

section_title("Category 2: Interaction Experience")

cat2_metrics = pd.DataFrame(
    [
        {
            "Best Model": "Tuned Random Forest",
            "Accuracy": "0.9939",
            "Balanced Accuracy": "0.9965",
            "Macro F1-score": "0.9871",
            "ROC-AUC": "0.9987",
            "Main Predictors": "outcome_quality, helpfulness, agent_politeness, customer_sentiment",
        }
    ]
)

st.dataframe(cat2_metrics, use_container_width=True, hide_index=True)

method_box(
    """
    <b>Conclusion:</b><br>
    Category 2 showed that outcome quality was the strongest predictor of retention.
    Customers were more likely to return when their issue was resolved and the interaction was helpful,
    polite, clear, and professional.
    """
)

section_title("Why Accuracy Alone Was Not Enough")

st.warning(
    "Both datasets were imbalanced. Most records belonged to the retained class. "
    "Therefore, accuracy alone could be misleading. Balanced accuracy, macro F1-score, confusion matrix, "
    "and ROC-AUC were used to evaluate the models more fairly."
)

section_title("Model Comparison Summary")

comparison_df = pd.DataFrame(
    [
        {
            "Model Type": "Baseline Model",
            "Purpose": "Used as a simple comparison point.",
            "Reason": "Shows whether ML models perform better than simple guessing.",
        },
        {
            "Model Type": "Logistic Regression",
            "Purpose": "Used for binary classification.",
            "Reason": "Simple, interpretable, and suitable for viva explanation.",
        },
        {
            "Model Type": "Random Forest",
            "Purpose": "Used as a stronger machine learning model.",
            "Reason": "Can capture complex patterns using many decision trees.",
        },
        {
            "Model Type": "Class Weighting",
            "Purpose": "Used to handle class imbalance.",
            "Reason": "Gives more importance to the minority class.",
        },
        {
            "Model Type": "SMOTE",
            "Purpose": "Used to create synthetic minority-class examples.",
            "Reason": "Helps the model learn not-retained customers better.",
        },
        {
            "Model Type": "GridSearchCV",
            "Purpose": "Used for hyperparameter tuning.",
            "Reason": "Finds better model settings using cross-validation.",
        },
    ]
)

st.dataframe(comparison_df, use_container_width=True, hide_index=True)

section_title("Weighted Final Prediction Method")

st.latex(
    r"""
    Final\ Probability =
    (Category\ 1\ Probability \times 0.60)
    +
    (Category\ 2\ Probability \times 0.40)
    """
)

st.markdown(
    """
    Option 3 was used for the final prediction. This is a topic-based weighted prediction method.

    - Category 1 receives **60% weight**.
    - Category 2 receives **40% weight**.

    Category 2 receives a higher weight because it directly measures customer interaction enjoyment,
    such as politeness, empathy, helpfulness, communication clarity, sentiment, tone, and outcome quality.
    Category 1 is also important, but it mainly represents response speed and support responsiveness.
    """
)

section_title("Model Export and Streamlit Testing")

st.markdown(
    """
    The final trained models were exported as `.joblib` files. The full pipeline was exported,
    not only the classifier. This is important because the pipeline already contains preprocessing steps,
    such as encoding, scaling, and missing value handling.

    In the Streamlit application, the exported models are loaded and used to predict retention from user inputs.
    The app first predicts Category 1 and Category 2 separately. After both predictions are completed,
    it combines both retained probabilities using the weighted prediction formula.
    """
)

section_title("Limitations")

st.warning(
    "Category 1 achieved perfect performance because CSAT score strongly separates retained and not-retained customers. "
    "If retention_label was created directly using CSAT score, the result should be interpreted carefully."
)

st.warning(
    "The weighted final prediction is a decision-level combination of two separate models. "
    "It is not a single model trained on one combined dataset."
)

section_title("Final Conclusion")

st.success(
    "Both predictive models support the analytical statement. "
    "The results show that customers with better service responsiveness and better interaction experience "
    "are more likely to return."
)