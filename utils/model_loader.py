import joblib
import streamlit as st
from pathlib import Path


@st.cache_resource
def load_model_bundle(model_path):
    """
    Loads a saved model bundle.

    Expected bundle format:
    {
        "model": trained_pipeline,
        "metadata": model_metadata
    }
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please place the exported .joblib model inside the models folder."
        )

    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("Loaded file is not a valid model bundle dictionary.")

    if "model" not in bundle or "metadata" not in bundle:
        raise ValueError("Model bundle must contain 'model' and 'metadata' keys.")

    return bundle