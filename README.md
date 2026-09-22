<div align="center">

# Customer Churn Prediction

Machine-learning pipeline and interactive dashboard for customer churn prediction

![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.53-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-189FDD)
![FLAML](https://img.shields.io/badge/FLAML-AutoML-0078D4?logo=microsoft&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPL--3.0-blue)

[Getting started](#getting-started) | [Architecture](#architecture) | [Models](#models) | [Technical docs](docs/README.md) | [Dashboard](#dashboard-features) | [Docker](#docker-deployment)

</div>

## Overview

This project trains churn models on the [BankChurners](BankChurners.csv) dataset and serves them through a Streamlit dashboard for data exploration, prediction, and evaluation.

Data moves through a stratified split, model-specific preprocessing, training, evaluation, bundle serialization, and the Streamlit dashboard.

## Models

The dashboard serves seven trained and evaluated classifiers. Classical models use one-hot encoding, SMOTE, and scaling. Mitra-v2 and TabFM receive the same raw mixed-type training split without SMOTE.

| Model | Type | Highlights |
|---|---|---|
| Random Forest | Ensemble (Bagging) | 100 estimators, balanced class weights, parallel training |
| XGBoost | Ensemble (Boosting) | 200 estimators, learning rate 0.1, depth 6, subsampling 0.9 |
| SVM | Kernel-based | RBF kernel, probability calibration, balanced class weights |
| Decision Tree | Single tree | Balanced class weights, interpretable baseline |
| FLAML AutoML | Automated ML | Microsoft [FLAML](https://microsoft.github.io/FLAML/) searches LightGBM, Random Forest, Extra Trees, and Logistic Regression within a 60-second time budget to choose a model and its hyperparameters |
| Mitra-v2 | Tabular foundation model | Fine-tuned from [`autogluon/mitra-classifier-2`](https://huggingface.co/autogluon/mitra-classifier-2) for 50 steps on CUDA |
| TabFM | Tabular foundation model | Local research evaluation from [`google/tabfm-1.0.0-pytorch`](https://huggingface.co/google/tabfm-1.0.0-pytorch), one estimator and a 100-row context |

[LazyPredict](https://github.com/shankarpandala/lazypredict) also benchmarks about 26 classifiers with default hyperparameters during training. Its results appear in a separate dashboard tab.

### Benchmark results

The positive class is `Attrited Customer`. Results use one stratified 80/20 split with random seed 42.

| Model | Accuracy | ROC AUC | F1 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| Random Forest | 0.9492 | 0.9832 | 0.8408 | 0.8447 | 0.8369 |
| XGBoost | 0.9669 | 0.9919 | 0.8951 | 0.9108 | 0.8800 |
| SVM | 0.9161 | 0.9457 | 0.7176 | 0.7798 | 0.6646 |
| Decision Tree | 0.9225 | 0.8792 | 0.7715 | 0.7320 | 0.8154 |
| FLAML AutoML | 0.9640 | 0.9890 | 0.8847 | 0.9091 | 0.8615 |
| Mitra-v2 | 0.9753 | 0.9949 | 0.9206 | 0.9508 | 0.8923 |
| TabFM | 0.9176 | 0.9496 | 0.7258 | 0.7782 | 0.6800 |

## Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│  train.py                                                       │
│  ┌──────────┐   ┌────────────┐   ┌──────────────┐              │
│  │ Load CSV │──▶│ Preprocess │──▶│ SMOTE + Scale│              │
│  └──────────┘   └────────────┘   └──────┬───────┘              │
│                                         │                       │
│           ┌─────────────────────────────┼──────────────────┐    │
│           ▼                             ▼                  ▼    │
│     Classical + AutoML              Mitra-v2            TabFM  │
│     SMOTE + scaling               raw features       raw features│
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

## Dashboard features

| Feature | Description |
|---|---|
| Model selector | Multi-select any combination of the seven trained models from the sidebar |
| FLAML details | Sidebar expander showing FLAML's best estimator and hyperparameter configuration |
| Exploratory data analysis | Data preview, descriptive statistics, missing value detection, correlation heatmap, distribution plotter |
| Predictions | Churn/Retained labels per model, churn rate metric, pie chart (single model) or cross-model comparison bar chart (multi-model) with agreement count |
| Model evaluation | Confusion matrix with TN/FP/FN/TP annotations and full classification report (requires labeled data) |
| LazyPredict benchmark | Sortable table of about 26 classifiers with an interactive bar chart. Select Accuracy, Balanced Accuracy, F1 Score, or ROC AUC |

## Getting started

### Prerequisites

- Python 3.13
- uv

### Installation

```bash
# Clone the repository
git clone https://github.com/pypi-ahmad/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
uv sync
```

Install the optional GPU models when training or serving them:

```bash
uv sync --extra mitra --extra tabfm
```

### Train

```bash
uv run python train.py
```

The default trains the five classical/AutoML models. Run all seven models on a CUDA GPU with:

```bash
uv run python train.py --models classical,mitra-v2,tabfm --mitra-time-limit 3600 --tabfm-context-rows 100
```

Mitra and TabFM checkpoints are pinned to immutable revisions. Generated artifacts are stored under `artifacts/`, while model metadata and metrics are written to `models_bundle.pkl`.

If a completed Mitra artifact already exists, reuse it instead of fine-tuning again:

```bash
uv run python train.py --models classical,mitra-v2,tabfm --reuse-mitra --tabfm-context-rows 100
```

`--mitra-fit-seconds` can preserve the original measured training time in a regenerated bundle. It does not affect prediction.

### Reproducibility

| Component | Pinned revision |
|---|---|
| Mitra-v2 checkpoint | `edada0d20759c58ada8c8605c25f22f6e98ea5f0` |
| TabFM checkpoint | `77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83` |
| TabFM source | `d8678b6895f1428a468d4cc299c1ff4cf704e726` (`v1.0.1`) |

The benchmark uses a stratified 80/20 split with seed 42. Classical models train on the SMOTE-resampled training partition. Mitra-v2 and TabFM use the original training partition, and all models are evaluated on the same untouched test partition.

### Run

```bash
uv run streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Upload a CSV/XLSX file in BankChurners format to generate predictions.

## Docker deployment

Docker deployment requires an NVIDIA GPU, NVIDIA Container Toolkit, and a completed local `artifacts/mitra-v2` training artifact. The build copies that fine-tuned artifact into the image.

```bash
# Build
docker build -t churn-prediction .

# Run
docker run --gpus all -p 8501:8501 churn-prediction
```

The container exposes port `8501`, supports uploads up to 1 GB, and includes Mitra-v2. The image excludes TabFM because its weights are licensed for non-commercial research use.

## Tech stack

| Category | Tools |
|---|---|
| Language | Python 3.13 |
| ML / AutoML | scikit-learn, XGBoost, FLAML, LazyPredict, AutoGluon Mitra-v2, TabFM |
| Data | Pandas, NumPy, imbalanced-learn (SMOTE) |
| Visualization | Plotly, Streamlit |
| Serialization | Joblib |
| Infrastructure | Docker |

## Project structure

```text
.
├── train.py                            # Training and evaluation pipeline
├── foundation_models.py                # Mitra-v2 and TabFM integration
├── app.py                              # Streamlit dashboard — EDA, predictions, evaluation
├── tests/                               # Pipeline regression tests
├── docs/                                # Architecture, development, training, and operations guides
├── artifacts/                           # Generated foundation-model artifacts (ignored by Git)
├── BankChurners.csv                    # Source dataset (10,127 records × 23 features)
├── models_bundle.pkl                   # Models, preprocessing metadata, and measured metrics
├── Customer Churn Prediction.ipynb     # Exploratory notebook
├── pyproject.toml                      # Project metadata and dependencies
├── uv.lock                             # Locked dependency versions
├── Dockerfile                          # Container configuration
├── LICENSE                             # GPL-3.0
└── README.md
```

## Roadmap

- [ ] Cloud deployment (Azure / AWS / GCP) with CI/CD
- [x] Core pipeline regression tests
- [ ] Model monitoring and data drift detection
- [ ] SHAP / LIME explainability layer
- [ ] REST API endpoint (FastAPI) for batch and real-time inference

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.

Mitra-v2 weights are Apache-2.0. TabFM source code is Apache-2.0, but its pretrained weights use the [`tabfm-non-commercial-v1.0`](https://github.com/google-research/tabfm/blob/d8678b6895f1428a468d4cc299c1ff4cf704e726/LICENSE) license and are intended only for non-commercial research.

<p align="center">Made by Ahmad Mujtaba</p>
