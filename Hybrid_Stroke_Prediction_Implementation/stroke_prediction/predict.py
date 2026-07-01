"""Inference and SHAP-based explainability for stroke prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from preprocessing import prepare_inference_frame
from utils import (
    BEST_MODEL_PATH,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    get_risk_category,
    load_json,
    validate_patient_input,
)


@dataclass
class PredictionResult:
    """Structured prediction output."""

    probability: float
    prediction: int
    risk_category: str
    top_features: list[dict[str, float]]
    shap_values: np.ndarray | None = None
    transformed_features: np.ndarray | None = None
    feature_names: list[str] | None = None


class StrokePredictor:
    """Load trained pipeline and produce predictions with explanations."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or BEST_MODEL_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {self.model_path}. "
                "Run `python train.py` first."
            )
        self.pipeline = joblib.load(self.model_path)
        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]
        self.feature_names = self._load_feature_names()
        self.training_metrics = load_json(METRICS_PATH)

    @staticmethod
    def _load_feature_names() -> list[str]:
        if FEATURE_NAMES_PATH.exists():
            with open(FEATURE_NAMES_PATH, encoding="utf-8") as fh:
                return json.load(fh)
        return []

    def _transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(x)

    def predict_proba(self, patient_data: dict[str, Any]) -> PredictionResult:
        """Run prediction with SHAP explanation for a single patient."""
        validated = validate_patient_input(patient_data)
        frame = prepare_inference_frame(pd.DataFrame([validated]))
        transformed = self._transform(frame)
        probability = float(self.pipeline.predict_proba(frame)[0, 1])
        prediction = int(probability >= 0.5)
        risk_category = get_risk_category(probability)

        shap_values, top_features = self._explain(transformed)
        names = self._resolved_feature_names(transformed.shape[1])

        return PredictionResult(
            probability=probability,
            prediction=prediction,
            risk_category=risk_category,
            top_features=top_features,
            shap_values=shap_values,
            transformed_features=transformed,
            feature_names=names,
        )

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict stroke risk for multiple patient records."""
        frame = prepare_inference_frame(df)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        results = df.copy()
        results["stroke_probability"] = probabilities
        results["stroke_prediction"] = (probabilities >= 0.5).astype(int)
        results["risk_category"] = [
            get_risk_category(p) for p in probabilities
        ]
        return results

    def _resolved_feature_names(self, n_features: int) -> list[str]:
        if self.feature_names and len(self.feature_names) == n_features:
            return self.feature_names
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:
            return [f"feature_{i}" for i in range(n_features)]

    def _normalize_shap_values(self, shap_values: Any) -> np.ndarray:
        """Normalize SHAP output to (n_samples, n_features)."""
        values = np.array(shap_values)
        if values.ndim == 3:
            # Binary classifiers may return (samples, features, classes)
            values = values[:, :, 1]
        if isinstance(shap_values, list):
            values = np.array(shap_values[1])
        return values

    def _explain(
        self, transformed: np.ndarray
    ) -> tuple[np.ndarray | None, list[dict[str, float]]]:
        """Compute SHAP values and top contributing features."""
        try:
            if hasattr(self.classifier, "feature_importances_"):
                explainer = shap.TreeExplainer(self.classifier)
                raw_shap = explainer.shap_values(transformed)
                shap_values = self._normalize_shap_values(raw_shap)
            elif hasattr(self.classifier, "coef_"):
                explainer = shap.LinearExplainer(self.classifier, transformed)
                raw_shap = explainer.shap_values(transformed)
                shap_values = self._normalize_shap_values(raw_shap)
            else:
                explainer = shap.KernelExplainer(
                    self.pipeline.predict_proba,
                    transformed,
                    link="logit",
                )
                raw_shap = explainer.shap_values(transformed, nsamples=100)
                shap_values = self._normalize_shap_values(raw_shap)

            names = self._resolved_feature_names(transformed.shape[1])
            contributions = shap_values[0]
            ranked = sorted(
                zip(names, contributions),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[:8]
            top_features = [
                {"feature": name, "shap_value": float(val)}
                for name, val in ranked
            ]
            return shap_values, top_features
        except Exception:
            return None, []

    def get_global_shap_summary_data(
        self, background: pd.DataFrame, sample_size: int = 200
    ) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
        """Compute SHAP values for a sample used in summary plots."""
        try:
            frame = prepare_inference_frame(background.head(sample_size))
            transformed = self._transform(frame)
            if hasattr(self.classifier, "feature_importances_"):
                explainer = shap.TreeExplainer(self.classifier)
                raw_shap = explainer.shap_values(transformed)
                shap_values = self._normalize_shap_values(raw_shap)
            else:
                return None
            names = self._resolved_feature_names(transformed.shape[1])
            return transformed, shap_values, names
        except Exception:
            return None


@lru_cache(maxsize=1)
def load_predictor() -> StrokePredictor:
    """Cached predictor loader for Streamlit."""
    return StrokePredictor()
