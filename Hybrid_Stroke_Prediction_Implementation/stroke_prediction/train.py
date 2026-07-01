"""Training orchestration for stroke prediction models."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation import compute_metrics, metrics_to_dataframe
from model import (
    RESAMPLERS,
    compare_resamplers,
    create_model_pipeline,
    evaluate_model_cv,
    get_base_models,
    get_feature_importance,
    tune_model_with_optuna,
)
from preprocessing import prepare_training_frame
from utils import (
    BEST_MODEL_PATH,
    COMPARISON_PATH,
    ENCODER_PATH,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MODELS_DIR,
    SCALER_PATH,
    ensure_directories,
    load_dataset,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_all_models(
    quick: bool = False,
    n_trials: int = 15,
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> dict:
    """
    Run the full training workflow:
    1. Load and preprocess data
    2. Compare resampling strategies
    3. Train and tune all models
    4. Select best model by PR-AUC
    5. Persist artifacts
    """
    ensure_directories()
    df = load_dataset()
    x, y = prepare_training_frame(df)

    logger.info("Dataset shape: %s | Stroke rate: %.2f%%", x.shape, 100 * y.mean())

    # Compare resampling techniques
    logger.info("Comparing resampling strategies...")
    resampler_scores = compare_resamplers(
        x, y, cv_folds=3 if quick else cv_folds
    )
    best_resampler = max(resampler_scores, key=resampler_scores.get)
    logger.info("Resampler scores: %s", resampler_scores)
    logger.info("Best resampler: %s", best_resampler)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=42,
    )

    model_names = list(get_base_models().keys())
    if quick:
        model_names = ["logistic_regression", "random_forest", "xgboost"]

    comparison: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}

    trials = 5 if quick else n_trials
    folds = 3 if quick else cv_folds

    for name in model_names:
        logger.info("Training model: %s", name)
        try:
            pipeline, cv_score, best_params = tune_model_with_optuna(
                x_train,
                y_train,
                model_name=name,
                resampler_name=best_resampler,
                n_trials=trials,
                cv_folds=folds,
            )
            y_prob = pipeline.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            test_metrics = compute_metrics(
                y_test.values, y_pred, y_prob
            )
            cv_metrics = evaluate_model_cv(pipeline, x_train, y_train, folds)

            comparison[name] = {
                **test_metrics,
                "cv_pr_auc": cv_score,
                "cv_recall": cv_metrics.get("recall", 0.0),
                "best_params": best_params,
            }
            fitted_models[name] = pipeline
            logger.info(
                "%s | test PR-AUC: %.3f | recall: %.3f",
                name,
                test_metrics.get("pr_auc", 0),
                test_metrics.get("recall", 0),
            )
        except Exception as exc:
            logger.warning("Failed to train %s: %s", name, exc)

    if not comparison:
        raise RuntimeError("No models were successfully trained.")

    comparison_df = metrics_to_dataframe(
        {
            k: {m: v[m] for m in v if m != "best_params"}
            for k, v in comparison.items()
            if "pr_auc" in v
        }
    )
    best_model_name = comparison_df.index[0]
    best_pipeline = fitted_models[best_model_name]

    # Evaluate best model on hold-out set
    y_prob = best_pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    final_metrics = compute_metrics(y_test.values, y_pred, y_prob)

    # Persist artifacts
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    joblib.dump(best_pipeline.named_steps["preprocessor"], SCALER_PATH)
    joblib.dump(
        best_pipeline.named_steps["preprocessor"].named_transformers_["cat"],
        ENCODER_PATH,
    )

    feature_names = list(
        best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(feature_names, fh, indent=2)

    comparison_df.to_csv(COMPARISON_PATH)

    training_report = {
        "best_model": best_model_name,
        "best_resampler": best_resampler,
        "resampler_scores": resampler_scores,
        "holdout_metrics": final_metrics,
        "model_comparison": comparison,
        "feature_importance": get_feature_importance(best_pipeline).head(20).to_dict(
            orient="records"
        ),
    }
    save_json(METRICS_PATH, training_report)

    logger.info("Best model: %s | PR-AUC: %.3f", best_model_name, final_metrics["pr_auc"])
    logger.info("Artifacts saved to %s", MODELS_DIR)
    return training_report


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train stroke prediction models")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast training with fewer models and trials",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Optuna trials per model",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds",
    )
    args = parser.parse_args()
    train_all_models(
        quick=args.quick,
        n_trials=args.trials,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
