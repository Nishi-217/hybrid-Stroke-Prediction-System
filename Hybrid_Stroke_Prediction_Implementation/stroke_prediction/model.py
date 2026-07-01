"""Model definitions, pipelines, and hyperparameter search."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from preprocessing import build_preprocessor

try:
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


RESAMPLERS: dict[str, Any] = {
    "smote": SMOTE(random_state=42),
    "borderline_smote": BorderlineSMOTE(random_state=42),
    "adasyn": ADASYN(random_state=42),
    "smoteenn": SMOTEENN(random_state=42),
}


class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible LSTM wrapper for tabular stroke prediction."""

    def __init__(
        self,
        units_1: int = 64,
        units_2: int = 32,
        dropout: float = 0.2,
        epochs: int = 25,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ) -> None:
        self.units_1 = units_1
        self.units_2 = units_2
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_: Any = None
        self.classes_: np.ndarray | None = None
        self.n_features_: int = 0

    def _build_model(self, n_features: int) -> Any:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMClassifier.")
        tf.random.set_seed(self.random_state)
        model = Sequential(
            [
                LSTM(
                    self.units_1,
                    input_shape=(1, n_features),
                    return_sequences=True,
                ),
                Dropout(self.dropout),
                LSTM(self.units_2),
                Dropout(self.dropout),
                Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LSTMClassifier":
        self.classes_ = np.unique(y)
        self.n_features_ = x.shape[1]
        x_seq = x.astype(np.float32).reshape((x.shape[0], 1, x.shape[1]))
        self.model_ = self._build_model(self.n_features_)
        self.model_.fit(
            x_seq,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            validation_split=0.1,
        )
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model has not been fitted.")
        x_seq = x.astype(np.float32).reshape((x.shape[0], 1, x.shape[1]))
        prob_pos = self.model_.predict(x_seq, verbose=0).ravel()
        return np.column_stack([1 - prob_pos, prob_pos])

    def predict(self, x: np.ndarray) -> np.ndarray:
        prob = self.predict_proba(x)[:, 1]
        return (prob >= 0.5).astype(int)


def get_base_models(random_state: int = 42) -> dict[str, Any]:
    """Return untuned baseline estimators."""
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            solver="liblinear",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=10,
        ),
        "catboost": CatBoostClassifier(
            verbose=0,
            random_state=random_state,
            auto_class_weights="Balanced",
        ),
        "lightgbm": LGBMClassifier(
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if TF_AVAILABLE:
        models["lstm"] = LSTMClassifier(random_state=random_state)
    return models


def create_model_pipeline(
    model: Any,
    resampler: Any | None = None,
    imputer: str = "iterative",
) -> ImbPipeline:
    """Create preprocessing + optional resampling + estimator pipeline."""
    preprocessor = build_preprocessor(imputer=imputer)
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if resampler is not None:
        steps.append(("resampler", resampler))
    steps.append(("classifier", model))
    return ImbPipeline(steps=steps)


def compare_resamplers(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    cv_folds: int = 5,
) -> dict[str, float]:
    """Compare resampling strategies using cross-validated PR-AUC."""
    base_model = get_base_models()[model_name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores: dict[str, float] = {}

    for name, sampler in RESAMPLERS.items():
        pipeline = create_model_pipeline(clone(base_model), sampler)
        cv_scores = cross_val_score(
            pipeline,
            x,
            y,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
        scores[name] = float(cv_scores.mean())
    return scores


def _suggest_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Suggest hyperparameters for Optuna trials (classifier param names)."""
    if model_name == "logistic_regression":
        return {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l2", "l1"]),
            "solver": "liblinear",
        }
    if model_name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
    if model_name == "catboost":
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        }
    if model_name == "extra_trees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
        }
    if model_name == "lstm":
        return {
            "units_1": trial.suggest_int("units_1", 32, 128),
            "units_2": trial.suggest_int("units_2", 16, 64),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }
    return {}


def _pipeline_params(params: dict[str, Any]) -> dict[str, Any]:
    """Prefix classifier hyperparameters for sklearn Pipeline.set_params."""
    return {f"classifier__{key}": value for key, value in params.items()}


def tune_model_with_optuna(
    x: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    resampler_name: str,
    n_trials: int = 20,
    cv_folds: int = 5,
) -> tuple[Any, float, dict[str, Any]]:
    """
    Tune a model pipeline with Optuna using stratified cross-validation.

    Returns fitted best pipeline, best score, and best params.
    """
    resampler = RESAMPLERS[resampler_name]
    base_model = get_base_models()[model_name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        pipeline = create_model_pipeline(clone(base_model), clone(resampler))
        params = _suggest_params(trial, model_name)
        pipeline.set_params(**_pipeline_params(params))
        scores = cross_val_score(
            pipeline,
            x,
            y,
            cv=cv,
            scoring="average_precision",
            n_jobs=-1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_pipeline = create_model_pipeline(
        clone(base_model), clone(resampler)
    )
    best_pipeline.set_params(**_pipeline_params(study.best_params))
    best_pipeline.fit(x, y)
    return best_pipeline, float(study.best_value), study.best_params


def evaluate_model_cv(
    pipeline: Any,
    x: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> dict[str, float]:
    """Evaluate a pipeline with stratified cross-validation."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    metric_map = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    results: dict[str, float] = {}
    for name, scoring in metric_map.items():
        scores = cross_val_score(
            pipeline, x, y, cv=cv, scoring=scoring, n_jobs=-1
        )
        results[name] = float(scores.mean())
    return results


def get_feature_importance(
    pipeline: Any,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Extract feature importances from tree-based models."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame()

    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        names = feature_names or [f"feature_{i}" for i in range(
            len(classifier.feature_importances_)
        )]

    return (
        pd.DataFrame({"feature": names, "importance": classifier.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
