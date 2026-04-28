from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT_DIR / "models"

CAT1_MODEL_PATH = MODEL_DIR / "cat1_retention_prediction_model.joblib"
CAT2_MODEL_PATH = MODEL_DIR / "cat2_retention_prediction_model.joblib"

CAT1_WEIGHT = 0.40
CAT2_WEIGHT = 0.60
DEFAULT_THRESHOLD = 0.50

CLASS_LABELS = {
    0: "Not Returned / Not Retained",
    1: "Returned / Retained",
}

CAT1_NAME = "Responsiveness & Support Speed"
CAT2_NAME = "Interaction Experience"