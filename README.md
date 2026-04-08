<div align="center">

# Customer Churn Prediction

**End-to-end ML pipeline and interactive dashboard for predicting customer attrition**

![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-189FDD)
![FLAML](https://img.shields.io/badge/FLAML-AutoML-0078D4?logo=microsoft&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-blue)

[Getting Started](#getting-started) · [Architecture](#architecture) · [Models](#models) · [Dashboard](#dashboard-features) · [Docker](#docker-deployment)

</div>

---

## Overview

This project delivers a production-ready customer churn prediction system built on the [BankChurners](BankChurners.csv) dataset. It combines a multi-model training pipeline with an interactive Streamlit dashboard for data exploration, prediction, and model evaluation.

**Pipeline**: Data Ingestion → Preprocessing & SMOTE Resampling → Multi-Model Training → Evaluation → Serialized Model Bundle → Interactive Dashboard

## Models

Five models are trained, evaluated, and served through the dashboard:

| Model | Type | Highlights |
|---|---|---|
| **Random Forest** | Ensemble (Bagging) | 100 estimators, balanced class weights, parallel training |
| **XGBoost** | Ensemble (Boosting) | 200 estimators, learning rate 0.1, depth 6, subsampling 0.9 |
| **SVM** | Kernel-based | RBF kernel, probability calibration, balanced class weights |
| **Decision Tree** | Single tree | Balanced class weights, interpretable baseline |
| **FLAML AutoML** | Automated ML | Microsoft [FLAML](https://microsoft.github.io/FLAML/) — searches across LightGBM, Random Forest, Extra Trees, and Logistic Regression within a 60-second time budget to find the optimal model and hyperparameters |

Additionally, **[LazyPredict](https://github.com/shankarpandala/lazypredict)** benchmarks ~26 classifiers with default hyperparameters during training to provide a comprehensive model landscape — results are displayed in a dedicated dashboard tab.

### Benchmark Results

| Model | Accuracy | F1 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|
| Random Forest | 0.9615 | 0.9770 | 0.9822 | 0.9718 |
| XGBoost | 0.9704 | 0.9823 | 0.9869 | 0.9777 |
| SVM | 0.9235 | 0.9548 | 0.9479 | 0.9618 |
| Decision Tree | 0.9373 | 0.9623 | 0.9724 | 0.9524 |
| **FLAML AutoML** | **0.9719** | **0.9832** | **0.9864** | **0.9800** |

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  train.py                                                       │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐              │
│  │ Load CSV │──▶│ Preprocess │──▶│ SMOTE + Scale│              │
│  └──────────┘   └────────────┘   └──────┬───────┘              │
│                                         │                       │
│           ┌─────────────────────────────┼──────────────────┐    │
│           ▼              ▼              ▼         ▼        ▼    │
│     Random Forest    XGBoost    SVM   Dec.Tree   FLAML    Lazy  │
│           │              │        │       │        │     Predict │
│           └──────────────┴────────┴───────┴────────┘        │   │
│                          │                                  │   │
│                  ┌───────▼──────────┐    ┌──────────────────▼┐  │
│                  │ Evaluate Models  │    │ Benchmark Results  │  │
│                  └───────┬──────────┘    └────────┬──────────┘  │
│                          │                        │             │
│                     ┌────▼────────────────────────▼───┐         │
│                     │      models_bundle.pkl          │         │
│                     └────────────────┬────────────────┘         │
└──────────────────────────────────────┼──────────────────────────┘
                                       │
┌──────────────────────────────────────┼──────────────────────────┐
│  app.py (Streamlit)                  ▼                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Sidebar: Model Selector │ FLAML Details Expander        │   │
│  ├──────────────┬──────────────────┬───────────────────────┤   │
│  │ Predictions  │ Model Evaluation │ LazyPredict Benchmark │   │
│  │  Tab         │  Tab             │  Tab                  │   │
│  └──────────────┴──────────────────┴───────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Dashboard Features

| Feature | Description |
|---|---|
| **Model Selector** | Multi-select any combination of the 5 trained models from the sidebar |
| **FLAML Details** | Sidebar expander showing FLAML's best estimator and hyperparameter configuration |
| **Exploratory Data Analysis** | Data preview, descriptive statistics, missing value detection, correlation heatmap, distribution plotter |
| **Predictions** | Churn/Retained labels per model, churn rate metric, pie chart (single model) or cross-model comparison bar chart (multi-model) with agreement count |
| **Model Evaluation** | Confusion matrix with TN/FP/FN/TP annotations and full classification report (requires labeled data) |
| **LazyPredict Benchmark** | Sortable table of ~26 classifiers with interactive bar chart — select from Accuracy, Balanced Accuracy, F1 Score, or ROC AUC |

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/pypi-ahmad/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt
```

### Train

```bash
python train.py
```

This runs the full pipeline: preprocesses the data, trains all 5 models, runs FLAML AutoML search (60s), benchmarks ~26 classifiers via LazyPredict, and serializes everything to `models_bundle.pkl`.

### Run

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Upload a CSV/XLSX file in BankChurners format to generate predictions.

## Docker Deployment

```bash
# Build
docker build -t churn-prediction .

# Run
docker run -p 8501:8501 churn-prediction
```

The container exposes port `8501` with large file upload support (1 GB).

## Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.13 |
| **ML / AutoML** | scikit-learn, XGBoost, FLAML, LazyPredict |
| **Data** | Pandas, NumPy, imbalanced-learn (SMOTE) |
| **Visualization** | Plotly, Streamlit |
| **Serialization** | Joblib |
| **Infrastructure** | Docker |

## Project Structure

```text
.
├── train.py                            # Training pipeline — models, FLAML, LazyPredict
├── app.py                              # Streamlit dashboard — EDA, predictions, evaluation
├── BankChurners.csv                    # Source dataset (10,127 records × 23 features)
├── models_bundle.pkl                   # Serialized models, scaler, metrics, LazyPredict results
├── Customer Churn Prediction.ipynb     # Exploratory notebook
├── requirements.txt                    # Pinned Python dependencies
├── Dockerfile                          # Container configuration
├── LICENSE                             # GPL-3.0
└── README.md
```

## Roadmap

- [ ] Cloud deployment (Azure / AWS / GCP) with CI/CD
- [ ] Unit and integration test suite
- [ ] Model monitoring and data drift detection
- [ ] SHAP / LIME explainability layer
- [ ] REST API endpoint (FastAPI) for batch and real-time inference

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.