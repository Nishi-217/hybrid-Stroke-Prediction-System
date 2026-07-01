"""Data cleaning, feature engineering, and sklearn preprocessing pipelines."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COL,
    NUMERIC_FEATURES,
    TARGET_COL,
)

ENGINEERED_FEATURES = [
    "age_group",
    "bmi_category",
    "glucose_risk",
    "cardio_risk_score",
    "lifestyle_risk_score",
]


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply baseline cleaning steps to raw stroke data."""
    data = df.copy()
    if ID_COL in data.columns:
        data = data.drop(columns=[ID_COL])

    # Remove invalid gender category and coerce BMI
    if "gender" in data.columns:
        data = data[data["gender"] != "Other"]

    if "bmi" in data.columns:
        data["bmi"] = pd.to_numeric(
            data["bmi"].replace("N/A", np.nan), errors="coerce"
        )

    return data.reset_index(drop=True)


def _iqr_outlier_mask(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """Return boolean mask for IQR outliers."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (series < lower) | (series > upper)


def treat_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    method: str = "cap",
) -> pd.DataFrame:
    """
    Detect and treat outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : iterable, optional
        Numeric columns to process. Defaults to NUMERIC_FEATURES.
    method : str
        'cap' clips to IQR bounds; 'median' replaces outliers with median.
    """
    data = df.copy()
    cols = list(columns or NUMERIC_FEATURES)

    for col in cols:
        if col not in data.columns:
            continue
        mask = _iqr_outlier_mask(data[col])
        if not mask.any():
            continue
        q1 = data.loc[~mask, col].quantile(0.25)
        q3 = data.loc[~mask, col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        if method == "median":
            data.loc[mask, col] = data.loc[~mask, col].median()
        else:
            data[col] = data[col].clip(lower=lower, upper=upper)
    return data


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain-informed engineered features."""
    data = df.copy()

    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 18, 35, 50, 65, 120],
        labels=["child", "young_adult", "middle_age", "senior", "elderly"],
        include_lowest=True,
    ).astype(str)

    data["bmi_category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"],
        include_lowest=True,
    ).astype(str)

    data["glucose_risk"] = pd.cut(
        data["avg_glucose_level"],
        bins=[0, 100, 126, 200, 500],
        labels=["normal", "prediabetic", "diabetic", "severe"],
        include_lowest=True,
    ).astype(str)

    data["cardio_risk_score"] = (
        data["hypertension"].astype(int)
        + data["heart_disease"].astype(int)
        + (data["age"] >= 55).astype(int)
        + (data["avg_glucose_level"] >= 140).astype(int)
    )

    smoking_map = {
        "never smoked": 0,
        "formerly smoked": 1,
        "smokes": 2,
        "Unknown": 1,
    }
    data["lifestyle_risk_score"] = data["smoking_status"].map(smoking_map).fillna(1)
    data["lifestyle_risk_score"] += (data["bmi"] >= 30).astype(int)

    return data


def get_feature_columns(include_engineered: bool = True) -> list[str]:
    """Return ordered feature column names used for modeling."""
    base = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
    if include_engineered:
        return base + ENGINEERED_FEATURES
    return base


def build_preprocessor(
    imputer: str = "iterative",
    include_engineered: bool = True,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numeric and categorical features.

    Parameters
    ----------
    imputer : str
        'iterative' or 'knn' for numeric missing value imputation.
    include_engineered : bool
        Whether engineered categorical features are included.
    """
    numeric_cols = list(NUMERIC_FEATURES)
    engineered_numeric = ["cardio_risk_score", "lifestyle_risk_score"]
    if include_engineered:
        numeric_cols += engineered_numeric

    cat_cols = list(CATEGORICAL_FEATURES)
    if include_engineered:
        cat_cols += ["age_group", "bmi_category", "glucose_risk"]

    if imputer == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    else:
        num_imputer = IterativeImputer(random_state=42, max_iter=10)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", num_imputer),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
    )


def prepare_training_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean, engineer, and split features from target."""
    data = engineer_features(treat_outliers(clean_raw_data(df)))
    feature_cols = get_feature_columns(include_engineered=True)
    x = data[feature_cols].copy()
    y = data[TARGET_COL].astype(int)
    return x, y


def prepare_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare raw patient records for inference."""
    data = clean_raw_data(df)
    data = engineer_features(treat_outliers(data))
    return data[get_feature_columns(include_engineered=True)]
