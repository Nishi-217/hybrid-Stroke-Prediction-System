"""Shared utilities, constants, and validation helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "healthcare-dataset-stroke-data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
HISTORY_PATH = MODELS_DIR / "prediction_history.json"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoder.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"

TARGET_COL = "stroke"
ID_COL = "id"

NUMERIC_FEATURES = ["age", "avg_glucose_level", "bmi"]
BINARY_FEATURES = ["hypertension", "heart_disease"]
CATEGORICAL_FEATURES = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

GENDER_OPTIONS = ["Male", "Female"]
EVER_MARRIED_OPTIONS = ["Yes", "No"]
WORK_TYPE_OPTIONS = [
    "Private",
    "Self-employed",
    "Govt_job",
    "children",
    "Never_worked",
]
RESIDENCE_OPTIONS = ["Urban", "Rural"]
SMOKING_OPTIONS = [
    "never smoked",
    "formerly smoked",
    "smokes",
    "Unknown",
]

RISK_THRESHOLDS = {
    "low": 0.15,
    "moderate": 0.35,
    "high": 0.60,
}

RISK_RECOMMENDATIONS = {
    "Low Risk": (
        "Maintain a healthy lifestyle with regular exercise, balanced nutrition, "
        "and routine health check-ups."
    ),
    "Moderate Risk": (
        "Monitor blood pressure and glucose levels regularly. "
        "Consider lifestyle modifications and annual physician visits."
    ),
    "High Risk": (
        "Consult a physician for a comprehensive cardiovascular assessment. "
        "Implement medical guidance for blood pressure and metabolic control."
    ),
    "Critical Risk": (
        "Immediate medical evaluation recommended. "
        "Contact a healthcare provider urgently for stroke risk assessment."
    ),
}


def ensure_directories() -> None:
    """Create required project directories if missing."""
    for path in (MODELS_DIR, ASSETS_DIR, DATASET_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the stroke dataset from disk."""
    csv_path = path or DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    return pd.read_csv(csv_path)


def get_risk_category(probability: float) -> str:
    """Map stroke probability to a clinical risk category."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low Risk"
    if probability < RISK_THRESHOLDS["moderate"]:
        return "Moderate Risk"
    if probability < RISK_THRESHOLDS["high"]:
        return "High Risk"
    return "Critical Risk"


def get_risk_color(category: str) -> str:
    """Return hex color for risk category badges."""
    colors = {
        "Low Risk": "#22c55e",
        "Moderate Risk": "#eab308",
        "High Risk": "#f97316",
        "Critical Risk": "#ef4444",
    }
    return colors.get(category, "#64748b")


def get_risk_emoji(category: str) -> str:
    """Return emoji indicator for risk category."""
    mapping = {
        "Low Risk": "🟢",
        "Moderate Risk": "🟡",
        "High Risk": "🟠",
        "Critical Risk": "🔴",
    }
    return mapping.get(category, "⚪")


def validate_patient_input(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce single-patient input fields."""
    errors: list[str] = []

    gender = str(data.get("gender", "")).strip()
    if gender not in GENDER_OPTIONS:
        errors.append(f"gender must be one of {GENDER_OPTIONS}")

    try:
        age = float(data["age"])
        if not 0 < age <= 120:
            errors.append("age must be between 0 and 120")
    except (KeyError, TypeError, ValueError):
        errors.append("age must be a valid number")
        age = 0.0

    for field in ("hypertension", "heart_disease"):
        try:
            value = int(data[field])
            if value not in (0, 1):
                errors.append(f"{field} must be 0 or 1")
        except (KeyError, TypeError, ValueError):
            errors.append(f"{field} must be 0 or 1")

    ever_married = str(data.get("ever_married", "")).strip()
    if ever_married not in EVER_MARRIED_OPTIONS:
        errors.append(f"ever_married must be one of {EVER_MARRIED_OPTIONS}")

    work_type = str(data.get("work_type", "")).strip()
    if work_type not in WORK_TYPE_OPTIONS:
        errors.append(f"work_type must be one of {WORK_TYPE_OPTIONS}")

    residence = str(data.get("Residence_type", "")).strip()
    if residence not in RESIDENCE_OPTIONS:
        errors.append(f"Residence_type must be one of {RESIDENCE_OPTIONS}")

    try:
        glucose = float(data["avg_glucose_level"])
        if not 40 <= glucose <= 500:
            errors.append("avg_glucose_level must be between 40 and 500")
    except (KeyError, TypeError, ValueError):
        errors.append("avg_glucose_level must be a valid number")
        glucose = 0.0

    try:
        bmi = float(data["bmi"])
        if not 10 <= bmi <= 80:
            errors.append("bmi must be between 10 and 80")
    except (KeyError, TypeError, ValueError):
        errors.append("bmi must be a valid number")
        bmi = 0.0

    smoking = str(data.get("smoking_status", "")).strip()
    if smoking not in SMOKING_OPTIONS:
        errors.append(f"smoking_status must be one of {SMOKING_OPTIONS}")

    if errors:
        raise ValueError("; ".join(errors))

    return {
        "gender": gender,
        "age": age,
        "hypertension": int(data["hypertension"]),
        "heart_disease": int(data["heart_disease"]),
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking,
    }


def patient_dict_to_dataframe(patient: dict[str, Any]) -> pd.DataFrame:
    """Convert a validated patient dict to a single-row DataFrame."""
    return pd.DataFrame([patient])


def append_prediction_history(record: dict[str, Any]) -> None:
    """Persist a prediction record to local JSON history."""
    ensure_directories()
    history: list[dict[str, Any]] = []
    if HISTORY_PATH.exists():
        with open(HISTORY_PATH, encoding="utf-8") as fh:
            history = json.load(fh)
    record["timestamp"] = datetime.now().isoformat()
    history.append(record)
    with open(HISTORY_PATH, "w", encoding="utf-8") as fh:
        json.dump(history[-500:], fh, indent=2)


def load_prediction_history() -> pd.DataFrame:
    """Load prediction history as a DataFrame."""
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    with open(HISTORY_PATH, encoding="utf-8") as fh:
        return pd.DataFrame(json.load(fh))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to JSON."""
    ensure_directories()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON from disk."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
