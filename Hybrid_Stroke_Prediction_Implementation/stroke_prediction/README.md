# Stroke Risk AI — Healthcare Prediction Platform

A production-ready machine learning application for **early stroke risk detection**, built as an MSc healthcare AI project. The system transforms academic notebook research into a modular Python application with automated model training, explainable AI (SHAP), and a professional Streamlit clinical dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Tech Stack](#tech-stack)
4. [Libraries & Why They Were Used](#libraries--why-they-were-used)
5. [Machine Learning Models & Why They Were Used](#machine-learning-models--why-they-were-used)
6. [Resampling Techniques](#resampling-techniques)
7. [Project Structure](#project-structure)
8. [System Architecture & Workflow](#system-architecture--workflow)
9. [Data Pipeline (Step by Step)](#data-pipeline-step-by-step)
10. [Training Pipeline (Step by Step)](#training-pipeline-step-by-step)
11. [Prediction & Inference Flow](#prediction--inference-flow)
12. [Streamlit Application — Pages & Functionality](#streamlit-application--pages--functionality)
13. [Dataset Description](#dataset-description)
14. [Evaluation Metrics](#evaluation-metrics)
15. [Risk Categories](#risk-categories)
16. [Complete Setup & Run Guide](#complete-setup--run-guide)
17. [Using Batch Upload](#using-batch-upload)
18. [Output Artifacts](#output-artifacts)
19. [CLI Reference](#cli-reference)
20. [Deployment (Streamlit Cloud)](#deployment-streamlit-cloud)
21. [Troubleshooting](#troubleshooting)
22. [Disclaimer](#disclaimer)

---

## Project Overview

Stroke is a leading cause of death and disability worldwide. This project builds a **hybrid healthcare prediction system** that:

- Trains and compares **7 machine learning / deep learning models** on imbalanced clinical data
- Automatically selects the best model using **PR-AUC** (critical for rare stroke cases)
- Provides **SHAP-based explanations** for every prediction
- Delivers results through a **professional Streamlit dashboard** suitable for portfolio, interviews, and deployment

The application evolved from an academic Jupyter notebook into a clean, modular codebase following PEP-8 standards and software engineering best practices.

---

## Problem Statement

The Healthcare Stroke Prediction dataset is **highly imbalanced** — only ~4.9% of patients had a stroke. In this setting:

- A naive model predicting "no stroke" for everyone achieves ~95% accuracy but **fails clinically**
- **Recall** (catching actual stroke cases) and **PR-AUC** matter more than raw accuracy
- Tree-based models achieve high accuracy but often miss minority-class patients
- Deep learning (LSTM) can improve recall by learning non-linear feature interactions

This project addresses class imbalance, compares multiple algorithms, tunes hyperparameters, and prioritizes **clinically meaningful metrics** over misleading accuracy scores.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.10+ |
| **Data processing** | pandas, NumPy |
| **Machine learning** | scikit-learn, imbalanced-learn |
| **Gradient boosting** | XGBoost, CatBoost, LightGBM |
| **Deep learning** | TensorFlow / Keras (LSTM) |
| **Hyperparameter tuning** | Optuna |
| **Explainability** | SHAP |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Model persistence** | joblib |
| **Reporting** | fpdf2 (PDF reports) |
| **Frontend / deployment** | Streamlit |
| **Version control** | Git |

---

## Libraries & Why They Were Used

| Library | Purpose | Why chosen |
|---------|---------|------------|
| **pandas** | Data loading, manipulation, batch CSV handling | Industry standard for tabular healthcare data |
| **NumPy** | Numerical operations, array math | Foundation for ML pipelines |
| **scikit-learn** | Preprocessing, pipelines, metrics, cross-validation | Mature, well-documented ML framework; `ColumnTransformer` and `Pipeline` ensure reproducible preprocessing |
| **imbalanced-learn** | SMOTE, Borderline SMOTE, ADASYN, SMOTEENN | Essential for handling severe class imbalance in stroke data |
| **XGBoost** | Gradient boosted trees | High performance on structured/tabular data; handles non-linear relationships |
| **CatBoost** | Gradient boosting with categorical support | Strong on mixed feature types; built-in class weighting |
| **LightGBM** | Fast gradient boosting | Efficient training on larger datasets; good recall/precision trade-offs |
| **TensorFlow / Keras** | LSTM neural network | Captures complex non-linear patterns; dissertation showed LSTM improves recall on imbalanced medical data |
| **Optuna** | Automated hyperparameter search | More efficient than grid search; integrates with cross-validation |
| **SHAP** | Model explainability | Provides per-prediction feature contributions clinicians can interpret |
| **Matplotlib / Seaborn** | Static charts (ROC, confusion matrix, heatmaps) | Reliable plotting for evaluation and EDA |
| **Plotly** | Interactive charts (optional) | Rich visualizations for dashboard |
| **joblib** | Save/load trained pipelines | Fast serialization of sklearn/imblearn pipelines |
| **fpdf2** | PDF prediction reports | Doctor-friendly downloadable reports |
| **Streamlit** | Web dashboard | Rapid deployment of ML apps without separate frontend; Streamlit Cloud ready |

---

## Machine Learning Models & Why They Were Used

| Model | Type | Why included |
|-------|------|--------------|
| **Logistic Regression** | Linear classifier | Interpretable baseline; fast; good with balanced class weights; establishes minimum performance bar |
| **Random Forest** | Bagging ensemble | Robust to outliers; provides feature importance; strong baseline on structured healthcare data |
| **XGBoost** | Gradient boosting | State-of-the-art on tabular data; handles non-linear interactions (age × glucose × hypertension) |
| **CatBoost** | Gradient boosting | Handles categorical features natively; less preprocessing sensitivity; balanced class weights |
| **LightGBM** | Gradient boosting | Fast training; efficient on medium datasets; competitive precision/recall |
| **Extra Trees** | Randomized bagging | More randomness than Random Forest; reduces overfitting on noisy clinical data |
| **LSTM** | Deep learning (RNN) | Learns complex non-linear patterns; dissertation research showed higher recall/AUC on stroke class vs tree models |

### Model selection criterion

The **best model is selected automatically by PR-AUC** (Precision-Recall AUC) on the hold-out test set, because PR-AUC is more informative than ROC-AUC when the positive class (stroke) is rare.

---

## Resampling Techniques

Before training, the pipeline compares four resampling strategies using cross-validated PR-AUC:

| Technique | Description | Why used |
|-----------|-------------|----------|
| **SMOTE** | Synthetic Minority Over-sampling | Creates synthetic stroke samples; standard approach for imbalance |
| **Borderline SMOTE** | Focuses on borderline minority samples | Better at hard-to-classify cases near decision boundary |
| **ADASYN** | Adaptive synthetic sampling | Generates more samples in regions where minority class is harder to learn |
| **SMOTEENN** | SMOTE + Edited Nearest Neighbours cleaning | Oversamples minority then removes noisy/overlapping samples |

The resampler with the **highest cross-validated PR-AUC** is used for all model training in that run.

---

## Project Structure

```
stroke_prediction/
│
├── app.py                    # Streamlit dashboard (main entry point)
├── ui.py                     # UI theme, CSS, reusable dashboard components
├── train.py                  # Full training orchestration CLI
├── predict.py                # Inference + SHAP explainability
├── preprocessing.py          # Cleaning, feature engineering, ColumnTransformer
├── model.py                  # Model definitions, pipelines, Optuna tuning
├── evaluation.py             # Metrics computation and chart helpers
├── utils.py                  # Constants, paths, validation, risk categories
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── .streamlit/
│   └── config.toml           # Streamlit theme and server config
│
├── models/                   # Generated after training
│   ├── best_model.pkl        # Full sklearn/imblearn pipeline (deploy this)
│   ├── scaler.pkl            # Preprocessor component
│   ├── encoder.pkl           # Categorical encoder component
│   ├── feature_names.json    # Transformed feature names for SHAP
│   ├── training_metrics.json # Best model, resampler, all metrics
│   ├── model_comparison.csv  # Side-by-side model comparison table
│   └── prediction_history.json  # Log of dashboard predictions
│
├── dataset/
│   └── healthcare-dataset-stroke-data.csv
│
└── assets/                   # Optional images, logos
```

---

## System Architecture & Workflow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Raw CSV    │───▶│ Preprocessing│───▶│  Training   │───▶│ best_model   │
│  (5,110     │    │ + Feature    │    │ 7 models +  │    │ .pkl saved   │
│  patients)  │    │ Engineering  │    │ Optuna tune │    │ via joblib   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘
                                                                   │
┌─────────────┐    ┌──────────────┐    ┌─────────────┐            │
│  Streamlit  │◀───│  SHAP + Risk │◀───│  predict.py │◀───────────┘
│  Dashboard  │    │  Categories  │    │  Inference  │
└─────────────┘    └──────────────┘    └─────────────┘
```

---

## Data Pipeline (Step by Step)

Executed in `preprocessing.py` for every training and inference run:

| Step | Action | Why |
|------|--------|-----|
| 1 | **Load CSV** | Read 5,110 patient records |
| 2 | **Drop `id`** | Unique identifier has no predictive value; causes overfitting |
| 3 | **Remove `gender = Other`** | Only 1 record; insufficient for learning |
| 4 | **Coerce BMI** | Convert `N/A` strings to NaN for imputation |
| 5 | **Outlier treatment** | IQR-based capping on age, glucose, BMI — reduces extreme value distortion |
| 6 | **Feature engineering** | Create `age_group`, `bmi_category`, `glucose_risk`, `cardio_risk_score`, `lifestyle_risk_score` |
| 7 | **ColumnTransformer** | Numeric: IterativeImputer → StandardScaler; Categorical: OneHotEncoder |
| 8 | **Resampling** (train only) | Apply best SMOTE variant inside imblearn Pipeline |
| 9 | **Model fit / predict** | Classifier trained or inference executed on transformed features |

### Input features (10 original + 5 engineered)

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Patient age |
| `avg_glucose_level` | Numeric | Average blood glucose (mg/dL) |
| `bmi` | Numeric | Body Mass Index |
| `hypertension` | Binary | 0 = No, 1 = Yes |
| `heart_disease` | Binary | 0 = No, 1 = Yes |
| `gender` | Categorical | Male, Female |
| `ever_married` | Categorical | Yes, No |
| `work_type` | Categorical | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Categorical | Urban, Rural |
| `smoking_status` | Categorical | never smoked, formerly smoked, smokes, Unknown |
| `age_group` | Engineered | child, young_adult, middle_age, senior, elderly |
| `bmi_category` | Engineered | underweight, normal, overweight, obese |
| `glucose_risk` | Engineered | normal, prediabetic, diabetic, severe |
| `cardio_risk_score` | Engineered | Composite of hypertension, heart disease, age, glucose |
| `lifestyle_risk_score` | Engineered | Composite of smoking status and obesity |

---

## Training Pipeline (Step by Step)

Run via `python train.py`. The pipeline executes:

| Step | What happens | Module |
|------|----------------|--------|
| 1 | Load dataset from `dataset/healthcare-dataset-stroke-data.csv` | `utils.py` |
| 2 | Clean, engineer features, split X and y | `preprocessing.py` |
| 3 | Compare 4 resamplers with Stratified K-Fold PR-AUC | `model.py` |
| 4 | Select best resampler (e.g. SMOTEENN) | `train.py` |
| 5 | Stratified 80/20 train/test split | `train.py` |
| 6 | For each model: Optuna hyperparameter tuning | `model.py` |
| 7 | Cross-validate with Stratified K-Fold (5 folds default) | `model.py` |
| 8 | Evaluate all models on hold-out test set | `evaluation.py` |
| 9 | Select best model by **PR-AUC** | `train.py` |
| 10 | Save pipeline, metrics, comparison CSV | `joblib` + `utils.py` |

### Training modes

```powershell
# Quick (~1–2 min): 3 models, 5 Optuna trials, 3 CV folds
python train.py --quick

# Full (~15–30+ min): all 7 models, 15+ trials, 5 CV folds
python train.py --trials 20 --cv-folds 5
```

---

## Prediction & Inference Flow

Executed in `predict.py` when a user submits a prediction:

| Step | Action |
|------|--------|
| 1 | Validate user input (age range, valid categories, etc.) |
| 2 | Convert input dict to DataFrame row |
| 3 | Run `prepare_inference_frame()` — same preprocessing as training |
| 4 | Load `best_model.pkl` pipeline (cached in Streamlit) |
| 5 | `predict_proba()` → stroke probability |
| 6 | Map probability to risk category (Low / Moderate / High / Critical) |
| 7 | Compute SHAP values → top contributing features |
| 8 | Return result + optional PDF report / history log |

---

## Streamlit Application — Pages & Functionality

Launch with `python -m streamlit run app.py`.

| Page | Functionality |
|------|---------------|
| **Home** | Project overview, key stats, feature list, 5-step workflow diagram, latest model comparison table |
| **Predict Stroke** | **Single patient:** form with demographics, clinical, lifestyle fields → probability gauge, risk badge, recommendation, SHAP bar chart, waterfall plot, PDF download. **Batch upload:** CSV upload → predictions for all rows → CSV download |
| **Dataset Analysis** | Record count, stroke rate, data preview, summary statistics, class distribution chart, correlation heatmap |
| **Model Performance** | Best model name, resampler, hold-out metrics, model comparison table, ROC curve, PR curve, confusion matrix, calibration curve, feature importance, learning curve |
| **Explainable AI** | Global SHAP summary and bar plots, prediction history table, history CSV download |
| **About** | Architecture modules, ethical disclaimer, institution info |

### Dashboard features

- Professional hospital-themed UI with Inter font
- Light / dark mode toggle
- Color-coded risk categories (green → red)
- Clinical recommendations per risk level
- Model readiness status in sidebar
- Prediction history persistence

---

## Dataset Description

**Source:** Public Healthcare Stroke Prediction Dataset  
**Records:** 5,110 patients  
**Target:** `stroke` (0 = no stroke, 1 = stroke)  
**Stroke rate:** ~4.9% (severely imbalanced)

The dataset includes demographic, lifestyle, and clinical variables commonly associated with stroke risk in real-world screening data.

---

## Evaluation Metrics

| Metric | Why it matters for stroke prediction |
|--------|--------------------------------------|
| **Accuracy** | Overall correctness — misleading when classes are imbalanced |
| **Balanced Accuracy** | Average recall per class — fairer on imbalanced data |
| **Precision** | Of predicted strokes, how many are correct — reduces false alarms |
| **Recall (Sensitivity)** | Of actual strokes, how many are detected — **most critical clinically** |
| **F1 Score** | Harmonic mean of precision and recall |
| **ROC-AUC** | Discrimination ability across all thresholds |
| **PR-AUC** | **Primary selection metric** — better for rare positive class |
| **MCC** | Matthews Correlation Coefficient — balanced measure even with imbalance |
| **Confusion Matrix** | True/false positives and negatives visualized |

---

## Risk Categories

| Probability | Category | Color | Recommendation |
|-------------|----------|-------|----------------|
| < 15% | Low Risk | Green | Maintain a healthy lifestyle |
| 15–35% | Moderate Risk | Yellow | Monitor blood pressure regularly |
| 35–60% | High Risk | Orange | Consult a physician |
| ≥ 60% | Critical Risk | Red | Immediate medical evaluation recommended |

---

## Complete Setup & Run Guide

### Prerequisites

- Python 3.10 or higher
- pip
- ~2 GB free disk space (TensorFlow is large)

```powershell
python --version
```

### Step 1 — Navigate to project folder

```powershell
cd "C:\Users\pvija\OneDrive\Documents\ML Projects\Hybrid_Stroke_Prediction_Implementation\stroke_prediction"
```

### Step 2 — Create virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4 — Train the model

```powershell
# First-time / quick test
python train.py --quick

# Production-quality training
python train.py --trials 20 --cv-folds 5
```

Verify `models/best_model.pkl` was created.

### Step 5 — Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### Step 6 — Make a prediction

1. Sidebar → **Predict Stroke**
2. Fill in patient details
3. Click **Analyze Stroke Risk**
4. Review probability, risk category, SHAP chart, and recommendation
5. Optionally download PDF report

### Running again later

```powershell
cd stroke_prediction
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

---

## Using Batch Upload

1. Go to **Predict Stroke** → **Batch Upload** tab
2. Upload a CSV with these columns:

```
gender, age, hypertension, heart_disease, ever_married, work_type,
Residence_type, avg_glucose_level, bmi, smoking_status
```

3. Optional columns: `id`, `stroke` (kept in output, not used for prediction)
4. Click **Run Batch Prediction**
5. Download results CSV with added columns:
   - `stroke_probability`
   - `stroke_prediction` (0 or 1)
   - `risk_category`

### Example batch CSV

```csv
gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status
Male,67,0,1,Yes,Private,Urban,228.69,36.6,formerly smoked
Female,45,1,0,Yes,Private,Urban,120.0,28.5,never smoked
```

You can also upload the full dataset at `dataset/healthcare-dataset-stroke-data.csv`.

### Valid value reference

| Field | Allowed values |
|-------|----------------|
| `gender` | Male, Female |
| `hypertension`, `heart_disease` | 0, 1 |
| `ever_married` | Yes, No |
| `work_type` | Private, Self-employed, Govt_job, children, Never_worked |
| `Residence_type` | Urban, Rural |
| `smoking_status` | never smoked, formerly smoked, smokes, Unknown |
| `bmi` | Number or N/A |

---

## Output Artifacts

After training, `models/` contains:

| File | Description |
|------|-------------|
| `best_model.pkl` | Complete pipeline: preprocessor → resampler → classifier |
| `scaler.pkl` | Fitted ColumnTransformer (numeric scaling + encoding) |
| `encoder.pkl` | Categorical encoder component |
| `feature_names.json` | Names of transformed features for SHAP plots |
| `training_metrics.json` | Best model, resampler scores, hold-out metrics, per-model comparison |
| `model_comparison.csv` | Sortable table of all model metrics |

---

## CLI Reference

```powershell
# ── Training ──
python train.py --quick                  # Fast: 3 models, 5 trials
python train.py --trials 20 --cv-folds 5 # Full training
python train.py --trials 30 --cv-folds 5  # Extended tuning

# ── Application ──
python -m streamlit run app.py           # Launch dashboard (port 8501)
python -m streamlit run app.py --server.port 8502  # Custom port
```

---

## Deployment (Streamlit Cloud)

1. Push the repository to **GitHub**
2. Train locally and include `models/best_model.pkl` in the repo (or use Git LFS for large files)
3. Go to [share.streamlit.io](https://share.streamlit.io)
4. Connect your GitHub repository
5. Set **Main file path:** `stroke_prediction/app.py`
6. Click **Deploy**

No API keys or secrets are required for the default setup.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `streamlit` not recognized | Use `python -m streamlit run app.py` |
| `Model not trained` in app | Run `python train.py --quick` first |
| Virtual env won't activate (Windows) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Port 8501 in use | `python -m streamlit run app.py --server.port 8502` |
| Training very slow | Use `--quick` for testing; full training takes 15–30+ minutes |
| Batch upload fails | Check column names and spelling match the dataset exactly |
| TensorFlow install fails | Ensure Python 3.10–3.12; TensorFlow may not support bleeding-edge Python versions |

---

## Disclaimer

This application is developed for **educational, research, and portfolio purposes only**. It is a clinical **decision-support tool**, not a diagnostic device. It does **not** provide medical advice and must **not** replace consultation with qualified healthcare professionals. All predictions should be validated by a licensed physician before any clinical action is taken.

---

## License

MIT License — see repository for details.

---

## Author

MSc Healthcare AI Project — University of Roehampton  
Built with Python, scikit-learn, XGBoost, TensorFlow, SHAP, and Streamlit.
