"""Model evaluation metrics and visualization helpers."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute comprehensive classification metrics."""
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return metrics


def metrics_to_dataframe(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert nested metrics dict to a comparison DataFrame."""
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Stroke", "Stroke"],
        yticklabels=["No Stroke", "Stroke"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
) -> plt.Figure:
    """Plot ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
) -> plt.Figure:
    """Plot precision-recall curve with AP score."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {ap:.3f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Calibration Curve",
) -> plt.Figure:
    """Plot reliability diagram for predicted probabilities."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_learning_curve_figure(
    estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    title: str = "Learning Curve",
) -> plt.Figure:
    """Plot training and validation learning curves."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        x,
        y,
        cv=5,
        scoring="average_precision",
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5),
    )
    train_mean = train_scores.mean(axis=1)
    val_mean = val_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_sizes, train_mean, marker="o", label="Training")
    ax.plot(train_sizes, val_mean, marker="o", label="Validation")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("PR-AUC")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Plot correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        numeric.corr(),
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    return fig


def plot_class_distribution(y: pd.Series) -> plt.Figure:
    """Plot target class distribution."""
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.value_counts().sort_index()
    sns.barplot(
        x=counts.index.map({0: "No Stroke", 1: "Stroke"}),
        y=counts.values,
        palette=["#3b82f6", "#ef4444"],
        ax=ax,
    )
    ax.set_title("Stroke Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Return sklearn classification report as dictionary."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)
