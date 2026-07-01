"""
Stroke Prediction — Production Streamlit Healthcare Dashboard.

Run: python -m streamlit run app.py
"""

from __future__ import annotations

import io
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from fpdf import FPDF
from sklearn.model_selection import train_test_split

from evaluation import (
    compute_metrics,
    plot_calibration_curve,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_learning_curve_figure,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from model import get_feature_importance
from predict import load_predictor
from preprocessing import clean_raw_data, prepare_training_frame
from ui import (
    NAV_ITEMS,
    inject_global_css,
    render_feature_card,
    render_form_section,
    render_model_status,
    render_page_hero,
    render_recommendation,
    render_risk_result,
    render_section_title,
    render_sidebar_brand,
    render_stat_card,
    render_workflow_step,
)
from utils import (
    BEST_MODEL_PATH,
    COMPARISON_PATH,
    EVER_MARRIED_OPTIONS,
    GENDER_OPTIONS,
    METRICS_PATH,
    RESIDENCE_OPTIONS,
    RISK_RECOMMENDATIONS,
    SMOKING_OPTIONS,
    WORK_TYPE_OPTIONS,
    append_prediction_history,
    ensure_directories,
    get_risk_color,
    get_risk_emoji,
    load_dataset,
    load_json,
    load_prediction_history,
)

# ---------------------------------------------------------------------------
# Page config & session state
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stroke Risk AI | Healthcare Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "predictions" not in st.session_state:
    st.session_state.predictions = []


@st.cache_resource
def get_model():
    """Load trained model pipeline with caching."""
    if not BEST_MODEL_PATH.exists():
        return None
    return joblib.load(BEST_MODEL_PATH)


@st.cache_data
def load_analysis_data() -> pd.DataFrame:
    """Cached dataset loader for analysis pages."""
    return clean_raw_data(load_dataset())


def model_is_ready() -> bool:
    return BEST_MODEL_PATH.exists()


def _apply_plot_style() -> None:
    """Use a clean matplotlib style for dashboard charts."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
    })


# ---------------------------------------------------------------------------
# PDF Report
# ---------------------------------------------------------------------------
def generate_pdf_report(
    patient: dict,
    probability: float,
    risk_category: str,
    top_features: list[dict],
) -> bytes:
    """Generate a doctor-friendly PDF prediction report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Stroke Risk Assessment Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for key, val in patient.items():
        pdf.cell(0, 6, f"{key.replace('_', ' ').title()}: {val}", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Risk Assessment", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Stroke Probability: {probability * 100:.1f}%", ln=True)
    pdf.cell(0, 6, f"Risk Category: {risk_category}", ln=True)
    pdf.cell(0, 6, f"Recommendation: {RISK_RECOMMENDATIONS.get(risk_category, '')}", ln=True)

    if top_features:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top Contributing Factors (SHAP)", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for item in top_features[:6]:
            pdf.cell(0, 6, f"- {item['feature']}: {item['shap_value']:.4f}", ln=True)

    pdf.set_font("Helvetica", "I", 8)
    pdf.ln(10)
    pdf.multi_cell(
        0, 5,
        "Disclaimer: This AI tool supports clinical decision-making and does "
        "not replace professional medical diagnosis.",
    )
    return pdf.output()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home() -> None:
    render_page_hero(
        "Stroke Risk AI Platform",
        "Intelligent clinical decision support for early stroke detection "
        "using machine learning and explainable AI.",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat_card("Models", "7+", "Compared & tuned")
    with c2:
        render_stat_card("Metrics", "9", "Clinical evaluation")
    with c3:
        render_stat_card("Explainability", "SHAP", "Per prediction")
    with c4:
        render_stat_card("Dataset", "5,110", "Patient records")

    render_section_title("Overview")
    st.markdown(
        '<div class="info-card">'
        "This platform transforms stroke prediction research into a deployable "
        "healthcare tool. It combines advanced imbalanced-learning techniques, "
        "automated hyperparameter optimization, and SHAP-based explanations to "
        "help clinicians identify at-risk patients earlier."
        "</div>",
        unsafe_allow_html=True,
    )

    render_section_title("Key Capabilities")
    col_l, col_r = st.columns(2)
    features = [
        ("⚙️", "Modular ML pipeline with ColumnTransformer and imblearn Pipelines"),
        ("⚖️", "Resampling: SMOTE, Borderline SMOTE, ADASYN, SMOTEENN"),
        ("🎯", "Optuna tuning with Stratified K-Fold cross-validation"),
        ("🤖", "LR, Random Forest, XGBoost, CatBoost, LightGBM, Extra Trees, LSTM"),
        ("🔍", "SHAP waterfall and summary plots for every prediction"),
        ("📁", "Batch prediction, PDF reports, and history tracking"),
    ]
    for i, (icon, text) in enumerate(features):
        with col_l if i % 2 == 0 else col_r:
            render_feature_card(icon, text)

    render_section_title("System Workflow")
    steps = ["Data Ingestion", "Preprocessing", "Model Training", "Evaluation", "Prediction"]
    cols = st.columns(5)
    for col, (i, step) in zip(cols, enumerate(steps, 1)):
        with col:
            render_workflow_step(i, step)

    if COMPARISON_PATH.exists():
        render_section_title("Latest Model Comparison")
        comparison = pd.read_csv(COMPARISON_PATH, index_col=0)
        st.dataframe(
            comparison.style.background_gradient(
                subset=["pr_auc", "recall", "roc_auc"], cmap="Blues"
            ).format("{:.3f}"),
            use_container_width=True,
        )
    elif not model_is_ready():
        st.warning("No trained model found. Run `python train.py --quick` to get started.")


def page_predict() -> None:
    render_page_hero(
        "Stroke Risk Assessment",
        "Enter patient details below to generate an AI-powered stroke risk score "
        "with explainable feature contributions.",
    )

    if not model_is_ready():
        st.error("Model not trained. Please run `python train.py` first.")
        return

    tab1, tab2 = st.tabs(["👤 Single Patient", "📂 Batch Upload"])

    with tab1:
        with st.form("prediction_form", border=False):
            render_form_section("Demographics")
            c1, c2, c3 = st.columns(3)
            gender = c1.selectbox("Gender", GENDER_OPTIONS)
            age = c2.number_input("Age", min_value=1, max_value=120, value=55)
            ever_married = c3.selectbox("Ever Married", EVER_MARRIED_OPTIONS)

            render_form_section("Clinical Indicators")
            c4, c5, c6 = st.columns(3)
            hypertension = c4.selectbox(
                "Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
            )
            heart_disease = c5.selectbox(
                "Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"
            )
            glucose = c6.number_input(
                "Avg. Glucose Level (mg/dL)", min_value=40.0, max_value=500.0, value=120.0
            )

            render_form_section("Lifestyle & Environment")
            c7, c8, c9 = st.columns(3)
            work_type = c7.selectbox("Work Type", WORK_TYPE_OPTIONS)
            residence = c8.selectbox("Residence Type", RESIDENCE_OPTIONS)
            bmi = c9.number_input("BMI", min_value=10.0, max_value=80.0, value=28.0)
            smoking = st.selectbox("Smoking Status", SMOKING_OPTIONS)

            submitted = st.form_submit_button(
                "Analyze Stroke Risk", type="primary", use_container_width=True
            )

        if submitted:
            patient = {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence,
                "avg_glucose_level": glucose,
                "bmi": bmi,
                "smoking_status": smoking,
            }
            try:
                predictor = load_predictor()
                result = predictor.predict_proba(patient)
                color = get_risk_color(result.risk_category)
                emoji = get_risk_emoji(result.risk_category)
                pred_label = "Stroke Detected" if result.prediction else "No Stroke Detected"

                st.markdown("<br>", unsafe_allow_html=True)
                res_col, detail_col = st.columns([1, 1.2])

                with res_col:
                    render_risk_result(
                        result.probability,
                        result.risk_category,
                        color,
                        emoji,
                        pred_label,
                    )
                    render_recommendation(
                        RISK_RECOMMENDATIONS.get(result.risk_category, "")
                    )

                with detail_col:
                    if result.top_features:
                        render_section_title("Feature Contributions")
                        feat_df = pd.DataFrame(result.top_features)
                        feat_df["abs"] = feat_df["shap_value"].abs()
                        feat_df = feat_df.sort_values("abs", ascending=True)
                        fig, ax = plt.subplots(figsize=(8, 5))
                        colors = [
                            "#ef4444" if v > 0 else "#22c55e"
                            for v in feat_df["shap_value"]
                        ]
                        ax.barh(feat_df["feature"], feat_df["shap_value"], color=colors)
                        ax.axvline(0, color="#94a3b8", linewidth=0.8)
                        ax.set_xlabel("SHAP Impact")
                        ax.set_title("Top Risk Factors", fontweight="600")
                        fig.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                if result.shap_values is not None and result.transformed_features is not None:
                    render_section_title("SHAP Waterfall")
                    try:
                        names = result.feature_names or []
                        base_value = float(result.shap_values[0].mean()) if len(result.shap_values) else 0
                        exp = shap.Explanation(
                            values=result.shap_values[0],
                            base_values=base_value,
                            data=result.transformed_features[0],
                            feature_names=names,
                        )
                        fig, _ = plt.subplots(figsize=(10, 5))
                        shap.waterfall_plot(exp, max_display=10, show=False)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as exc:
                        st.warning(f"Waterfall plot unavailable: {exc}")

                record = {
                    **patient,
                    "stroke_probability": result.probability,
                    "risk_category": result.risk_category,
                    "prediction": result.prediction,
                }
                append_prediction_history(record)
                st.session_state.predictions.append(record)

                dl1, dl2 = st.columns(2)
                with dl1:
                    pdf_bytes = generate_pdf_report(
                        patient, result.probability, result.risk_category, result.top_features
                    )
                    st.download_button(
                        "📄 Download PDF Report",
                        data=bytes(pdf_bytes),
                        file_name="stroke_risk_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    with tab2:
        st.markdown(
            '<div class="info-card">Upload a CSV file containing patient records. '
            "The file should include columns matching the dataset schema "
            "(gender, age, hypertension, etc.).</div>",
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("Choose CSV file", type=["csv"], label_visibility="collapsed")
        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.dataframe(batch_df.head(10), use_container_width=True, height=220)
            if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                try:
                    predictor = load_predictor()
                    results = predictor.predict_batch(batch_df)
                    st.dataframe(results, use_container_width=True)
                    csv_buf = io.StringIO()
                    results.to_csv(csv_buf, index=False)
                    st.download_button(
                        "📥 Download Results CSV",
                        data=csv_buf.getvalue(),
                        file_name="batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception as exc:
                    st.error(f"Batch prediction failed: {exc}")


def page_dataset_analysis() -> None:
    render_page_hero(
        "Dataset Analysis",
        "Exploratory analysis of the Healthcare Stroke Prediction dataset.",
    )
    df = load_analysis_data()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_stat_card("Records", f"{len(df):,}", "Patient entries")
    with c2:
        render_stat_card("Features", str(df.shape[1] - 1), "Input variables")
    with c3:
        render_stat_card("Stroke Cases", f"{df['stroke'].sum()}", "Positive class")
    with c4:
        render_stat_card("Stroke Rate", f"{100 * df['stroke'].mean():.1f}%", "Class imbalance")

    render_section_title("Data Preview")
    st.dataframe(df.head(15), use_container_width=True, height=320)

    render_section_title("Summary Statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    render_section_title("Visual Analysis")
    _apply_plot_style()
    col_a, col_b = st.columns(2)
    with col_a:
        fig = plot_class_distribution(df["stroke"])
        st.pyplot(fig)
        plt.close(fig)
    with col_b:
        fig = plot_correlation_heatmap(df)
        st.pyplot(fig)
        plt.close(fig)


def page_model_performance() -> None:
    render_page_hero(
        "Model Performance",
        "Evaluation metrics, comparison tables, and diagnostic charts for the trained model.",
    )

    if not model_is_ready():
        st.error("Train a model first with `python train.py`.")
        return

    metrics = load_json(METRICS_PATH)
    if metrics:
        info1, info2 = st.columns(2)
        with info1:
            st.markdown(
                f'<div class="stat-card"><div class="label">Best Model</div>'
                f'<div class="value" style="font-size:1.3rem;">'
                f'{metrics.get("best_model", "N/A").replace("_", " ").title()}</div></div>',
                unsafe_allow_html=True,
            )
        with info2:
            st.markdown(
                f'<div class="stat-card"><div class="label">Resampler</div>'
                f'<div class="value" style="font-size:1.3rem;">'
                f'{metrics.get("best_resampler", "N/A").upper()}</div></div>',
                unsafe_allow_html=True,
            )

        holdout = metrics.get("holdout_metrics", {})
        if holdout:
            render_section_title("Hold-out Metrics")
            cols = st.columns(5)
            labels = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            for col, label in zip(cols, labels):
                col.metric(label.replace("_", " ").title(), f"{holdout.get(label, 0):.3f}")

    if COMPARISON_PATH.exists():
        render_section_title("Model Comparison")
        st.dataframe(
            pd.read_csv(COMPARISON_PATH, index_col=0).style.format("{:.3f}"),
            use_container_width=True,
        )

    pipeline = get_model()
    df = load_analysis_data()
    x, y = prepare_training_frame(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    render_section_title("Diagnostic Charts")
    _apply_plot_style()
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_roc_curve(y_test, y_prob)
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig = plot_precision_recall_curve(y_test, y_prob)
        st.pyplot(fig)
        plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        fig = plot_confusion_matrix(y_test, y_pred)
        st.pyplot(fig)
        plt.close(fig)
    with col4:
        fig = plot_calibration_curve(y_test, y_prob)
        st.pyplot(fig)
        plt.close(fig)

    importance = get_feature_importance(pipeline)
    if not importance.empty:
        render_section_title("Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 6))
        top = importance.head(15)
        ax.barh(top["feature"], top["importance"], color="#0369a1")
        ax.invert_yaxis()
        ax.set_title("Top 15 Feature Importances")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("📉 Generate Learning Curve"):
        if st.button("Compute Learning Curve", use_container_width=True):
            with st.spinner("Computing..."):
                fig = plot_learning_curve_figure(pipeline, x_train, y_train)
                st.pyplot(fig)
                plt.close(fig)


def page_explainable_ai() -> None:
    render_page_hero(
        "Explainable AI",
        "SHAP analysis reveals which features drive stroke risk predictions at "
        "both individual and population levels.",
    )

    if not model_is_ready():
        st.error("Train a model first.")
        return

    st.markdown(
        '<div class="info-card">'
        "<strong>SHAP</strong> (SHapley Additive exPlanations) assigns each feature "
        "a contribution value showing how it pushes the prediction toward or away "
        "from a stroke outcome."
        "</div>",
        unsafe_allow_html=True,
    )

    df = load_analysis_data()
    predictor = load_predictor()
    summary = predictor.get_global_shap_summary_data(df, sample_size=150)

    if summary:
        transformed, shap_values, names = summary
        render_section_title("Global Feature Impact")
        _apply_plot_style()
        col1, col2 = st.columns(2)
        with col1:
            fig, _ = plt.subplots(figsize=(9, 6))
            shap.summary_plot(shap_values, transformed, feature_names=names, show=False)
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            fig, _ = plt.subplots(figsize=(9, 6))
            shap.summary_plot(
                shap_values, transformed, feature_names=names,
                plot_type="bar", show=False,
            )
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Global SHAP plots are available for tree-based models.")

    history = load_prediction_history()
    if not history.empty:
        render_section_title("Prediction History")
        st.dataframe(history.tail(20), use_container_width=True, height=300)
        st.download_button(
            "📥 Download History CSV",
            data=history.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
        )


def page_about() -> None:
    render_page_hero(
        "About Stroke Risk AI",
        "A production-grade healthcare analytics platform for MSc research and clinical decision support.",
    )

    col1, col2 = st.columns(2)
    with col1:
        render_section_title("Architecture")
        modules = [
            ("preprocessing.py", "Cleaning, feature engineering, ColumnTransformer"),
            ("model.py", "Estimators, pipelines, Optuna tuning"),
            ("train.py", "Automated training and model selection"),
            ("predict.py", "Inference with SHAP explanations"),
            ("evaluation.py", "Metrics and visualizations"),
            ("app.py", "Streamlit dashboard"),
        ]
        for name, desc in modules:
            render_feature_card("📄", f"<strong>{name}</strong> — {desc}")

    with col2:
        render_section_title("Ethical Notice")
        st.markdown(
            '<div class="recommendation-box">'
            "This tool is intended for <strong>research and clinical decision support</strong> "
            "only. It must not replace qualified medical professionals. All predictions "
            "should be validated by a licensed healthcare provider."
            "</div>",
            unsafe_allow_html=True,
        )
        render_section_title("Institution")
        st.markdown(
            '<div class="info-card">'
            "<strong>MSc Healthcare AI Project</strong><br>"
            "University of Roehampton<br><br>"
            "Built with Python, scikit-learn, XGBoost, TensorFlow, SHAP, and Streamlit."
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ensure_directories()
    inject_global_css(st.session_state.theme)

    with st.sidebar:
        render_sidebar_brand()

        labels = [item[0] for item in NAV_ITEMS]
        page = st.radio("Navigation", labels, label_visibility="collapsed")

        st.divider()

        theme_label = "🌙 Dark Mode" if st.session_state.theme == "light" else "☀️ Light Mode"
        if st.toggle(theme_label, value=st.session_state.theme == "dark"):
            st.session_state.theme = "dark"
        else:
            st.session_state.theme = "light"

        st.divider()
        render_model_status(model_is_ready())

        st.markdown(
            '<p style="text-align:center;color:#94a3b8;font-size:0.72rem;'
            'margin-top:1.5rem;">v1.0 · Production Build</p>',
            unsafe_allow_html=True,
        )

    pages = {name: fn for name, fn in [
        ("Home", page_home),
        ("Predict Stroke", page_predict),
        ("Dataset Analysis", page_dataset_analysis),
        ("Model Performance", page_model_performance),
        ("Explainable AI", page_explainable_ai),
        ("About", page_about),
    ]}
    pages[page]()


if __name__ == "__main__":
    main()
