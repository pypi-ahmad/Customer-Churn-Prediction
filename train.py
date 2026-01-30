from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_data(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath.resolve()}")

    try:
        df = pd.read_csv(filepath)
    except pd.errors.ParserError as exc:
        raise RuntimeError(f"Failed to parse CSV: {exc}") from exc

    logging.info("Loaded dataset: %s rows × %s columns", df.shape[0], df.shape[1])
    return df


def preprocess_data(
    df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
    list[str],
]:
    try:
        y = pd.get_dummies(df["Attrition_Flag"], drop_first=True).squeeze()
        X = df.iloc[:, 2:21].copy()
    except KeyError as exc:
        raise RuntimeError(f"Missing expected column: {exc}") from exc

    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"],
    ).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=list(categorical_cols), drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    feature_columns = list(X_train.columns)

    smote = SMOTE(sampling_strategy="auto", random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Preprocessing complete. Train shape: %s", X_train_scaled.shape)
    return (
        X_train_scaled,
        X_test_scaled,
        np.asarray(y_train_resampled),
        np.asarray(y_test),
        scaler,
        feature_columns,
    )


def build_model_factory() -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
        "SVM": SVC(
            probability=True,
            class_weight="balanced",
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
    }


def evaluate_models(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1": float(f1_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
        }

    return metrics


def log_metrics_table(metrics: dict[str, dict[str, float]]) -> None:
    header = f"{'Model':<20} | {'Accuracy':>9} | {'F1':>6} | {'Precision':>9} | {'Recall':>6}"
    sep = "-" * len(header)
    logging.info("\n%s\n%s", header, sep)
    for name, vals in metrics.items():
        logging.info(
            "% -20s | %9.4f | %6.4f | %9.4f | %6.4f",
            name,
            vals["accuracy"],
            vals["f1"],
            vals["precision"],
            vals["recall"],
        )


def save_model_bundle(
    models: dict[str, Any],
    scaler: StandardScaler,
    feature_names: list[str],
    metrics: dict[str, dict[str, float]],
    filepath: Path,
) -> None:
    payload = {
        "models": models,
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": metrics,
    }
    joblib.dump(payload, filepath)
    logging.info("Saved models bundle to %s", filepath.resolve())


def main() -> None:
    configure_logging()

    data_path = Path("BankChurners.csv")
    bundle_path = Path("models_bundle.pkl")

    try:
        df = load_data(data_path)
        X_train, X_test, y_train, y_test, scaler, feature_columns = preprocess_data(df)
        models = build_model_factory()

        for name, model in models.items():
            model.fit(X_train, y_train)
            logging.info("Trained model: %s", name)

        metrics = evaluate_models(models, X_test, y_test)
        log_metrics_table(metrics)
        save_model_bundle(models, scaler, feature_columns, metrics, bundle_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("Pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
