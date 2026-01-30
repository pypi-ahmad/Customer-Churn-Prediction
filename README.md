# Customer Churn Prediction Dashboard 🔮

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red) ![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED) ![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange) ![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This repository delivers an end-to-end customer churn solution: **Data Ingestion (Excel/CSV)** → **Multi-Model Training** via [train.py](train.py) → **Interactive Dashboard** via [app.py](app.py). The goal is to predict customer attrition with machine learning and present actionable insights through a Streamlit web interface.

## Key Features

- **Multi-Model Factory**: Trains and compares Random Forest, XGBoost, SVM, and Decision Tree models in [train.py](train.py).
- **Advanced EDA**: Correlation heatmaps, distribution plotter, and data previews are built into [app.py](app.py).
- **Robust Evaluation**: Confusion Matrix and Classification Report are generated when labeled data is provided.
- **Deployment Ready**: Dockerized Streamlit app with large file support (1GB+) via .streamlit config.

## Tech Stack

- Python 3.13
- Streamlit
- Pandas
- Plotly
- OpenPyXL
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Joblib

## Getting Started (Local)

### Prerequisites

- Python 3.13+
- Pip

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the models

```bash
python train.py
```

### Step 3: Run the app

```bash
streamlit run app.py
```

## Getting Started (Docker)

### Step 1: Build the image

```bash
docker build -t churn-app .
```

### Step 2: Run the container

```bash
docker run -p 8501:8501 churn-app
```

## Project Structure

```text
.
├─ app.py
├─ train.py
├─ BankChurners.csv
├─ models_bundle.pkl
├─ Customer Churn Prediction.ipynb
├─ requirements.txt
├─ Dockerfile
├─ LICENSE
├─ README.md
└─ .streamlit
	└─ config.toml
```

## Future Roadmap

- Cloud deployment (Azure/AWS/GCP) with automated CI/CD
- Unit tests for preprocessing and inference
- Model monitoring and drift detection

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).