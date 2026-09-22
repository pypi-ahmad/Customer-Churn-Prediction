from __future__ import annotations

import argparse
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from imblearn.over_sampling import SMOTE
from lazypredict.Supervised import LazyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from foundation_models import (
    MITRA_REPO,
    MITRA_REVISION,
    TABFM_REPO,
    TABFM_REVISION,
    fit_and_predict_tabfm,
    load_mitra,
    predict_mitra,
    release_cuda,
    save_tabfm_context,
    train_mitra,
)

LABEL_MAPPING = {"Existing Customer": 0, "Attrited Customer": 1}
SUPPORTED_GROUPS = {"classical", "mitra-v2", "tabfm"}
LOGGER = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    return pd.read_csv(path)


def encode_target(target: pd.Series) -> pd.Series:
    unknown = sorted(set(target.dropna().unique()) - set(LABEL_MAPPING))
    if unknown:
        raise ValueError(f"Unknown target labels: {unknown}")
    encoded = target.map(LABEL_MAPPING)
    if encoded.isna().any():
        raise ValueError("Target contains missing values.")
    return encoded.astype("int8")


def prepare_split(df: pd.DataFrame) -> dict[str, Any]:
    X_raw = df.iloc[:, 2:21].copy()
    y = encode_target(df["Attrition_Flag"])
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_encoded = pd.get_dummies(X_train_raw, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True).reindex(
        columns=X_train_encoded.columns,
        fill_value=0,
    )
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(
        X_train_encoded,
        y_train,
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_encoded)
    return {
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train_balanced": y_train_balanced,
        "scaler": scaler,
        "feature_names": list(X_train_encoded.columns),
        "raw_feature_names": list(X_raw.columns),
    }


def build_model_factory() -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
        ),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
        ),
    }


def train_flaml(X_train: np.ndarray, y_train: pd.Series) -> AutoML:
    automl = AutoML()
    automl.fit(
        X_train,
        y_train,
        task="classification",
        time_budget=60,
        metric="accuracy",
        estimator_list=["lgbm", "rf", "extra_tree", "lrl1"],
        seed=42,
        verbose=0,
        log_file_name="",
    )
    return automl


def run_lazypredict(split: dict[str, Any]) -> pd.DataFrame:
    classifier = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    results, _ = classifier.fit(
        split["X_train_scaled"],
        split["X_test_scaled"],
        split["y_train_balanced"],
        split["y_test"],
    )
    return results


def positive_probabilities(model: Any, X: Any) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X))
    return probabilities[:, -1] if probabilities.ndim == 2 else probabilities


def evaluate_predictions(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "f1": f1_score(y_true, predictions),
        "precision": precision_score(y_true, predictions),
        "recall": recall_score(y_true, predictions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn prediction models.")
    parser.add_argument("--data", type=Path, default=Path("BankChurners.csv"))
    parser.add_argument("--bundle", type=Path, default=Path("models_bundle.pkl"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--models",
        default="classical",
        help="Comma-separated groups: classical, mitra-v2, tabfm",
    )
    parser.add_argument("--mitra-time-limit", type=int, default=3600)
    parser.add_argument(
        "--reuse-mitra",
        action="store_true",
        help="Load an existing Mitra artifact instead of fine-tuning it again.",
    )
    parser.add_argument(
        "--mitra-fit-seconds",
        type=float,
        help="Recorded fine-tuning time when --reuse-mitra is used.",
    )
    parser.add_argument("--tabfm-context-rows", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    groups = {item.strip().lower() for item in args.models.split(",") if item.strip()}
    unknown = groups - SUPPORTED_GROUPS
    if unknown:
        raise ValueError(f"Unknown model groups: {sorted(unknown)}")

    split = prepare_split(load_data(args.data))
    models: dict[str, Any] = {}
    foundation_models: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, float]] = {}
    timings: dict[str, dict[str, float]] = {}
    lazy_results: pd.DataFrame | None = None

    if "classical" in groups:
        for name, model in build_model_factory().items():
            started = perf_counter()
            model.fit(split["X_train_scaled"], split["y_train_balanced"])
            fit_seconds = perf_counter() - started
            prediction_started = perf_counter()
            predictions = model.predict(split["X_test_scaled"])
            probabilities = positive_probabilities(model, split["X_test_scaled"])
            prediction_seconds = perf_counter() - prediction_started
            models[name] = model
            metrics[name] = evaluate_predictions(
                split["y_test"], predictions, probabilities
            )
            timings[name] = {
                "fit_seconds": fit_seconds,
                "predict_seconds": prediction_seconds,
            }
            LOGGER.info("Trained %s: %s", name, metrics[name])

        started = perf_counter()
        flaml_model = train_flaml(split["X_train_scaled"], split["y_train_balanced"])
        fit_seconds = perf_counter() - started
        started = perf_counter()
        predictions = flaml_model.predict(split["X_test_scaled"])
        probabilities = positive_probabilities(flaml_model, split["X_test_scaled"])
        name = "FLAML AutoML"
        models[name] = flaml_model
        metrics[name] = evaluate_predictions(
            split["y_test"], predictions, probabilities
        )
        timings[name] = {
            "fit_seconds": fit_seconds,
            "predict_seconds": perf_counter() - started,
        }
        try:
            lazy_results = run_lazypredict(split)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("LazyPredict benchmark failed: %s", exc)

    if "mitra-v2" in groups:
        artifact_path = args.artifacts_dir / "mitra-v2"
        if args.reuse_mitra:
            started = perf_counter()
            predictor = load_mitra(artifact_path)
            load_seconds = perf_counter() - started
            fit_seconds = args.mitra_fit_seconds or load_seconds
        else:
            predictor, fit_seconds = train_mitra(
                split["X_train_raw"],
                split["y_train"],
                artifact_path,
                args.mitra_time_limit,
            )
        started = perf_counter()
        predictions, probabilities = predict_mitra(predictor, split["X_test_raw"])
        predict_seconds = perf_counter() - started
        name = "Mitra-v2"
        metrics[name] = evaluate_predictions(
            split["y_test"], predictions, probabilities
        )
        timings[name] = {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
        foundation_models[name] = {
            "type": "mitra-v2",
            "artifact_path": str(artifact_path),
            "repository": MITRA_REPO,
            "revision": MITRA_REVISION,
            "license": "Apache-2.0",
        }
        LOGGER.info("Trained %s: %s", name, metrics[name])
        del predictor
        release_cuda()

    if "tabfm" in groups:
        (
            classifier,
            predictions,
            probabilities,
            settings,
            fit_seconds,
            predict_seconds,
        ) = fit_and_predict_tabfm(
            split["X_train_raw"],
            split["y_train"],
            split["X_test_raw"],
            args.tabfm_context_rows,
        )
        del classifier
        context_path = args.artifacts_dir / "tabfm-context.pkl"
        save_tabfm_context(context_path, split["X_train_raw"], split["y_train"])
        name = "TabFM"
        metrics[name] = evaluate_predictions(
            split["y_test"], predictions, probabilities
        )
        timings[name] = {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
        foundation_models[name] = {
            "type": "tabfm",
            "context_path": str(context_path),
            "repository": TABFM_REPO,
            "revision": TABFM_REVISION,
            "license": "tabfm-non-commercial-v1.0",
            **settings,
        }
        LOGGER.info("Evaluated %s: %s", name, metrics[name])

    payload = {
        "schema_version": 2,
        "models": models,
        "foundation_models": foundation_models,
        "scaler": split["scaler"],
        "feature_names": split["feature_names"],
        "raw_feature_names": split["raw_feature_names"],
        "label_mapping": LABEL_MAPPING,
        "metrics": metrics,
        "timings": timings,
        "lazypredict_results": lazy_results,
    }
    joblib.dump(payload, args.bundle)
    LOGGER.info("Saved %s", args.bundle)


if __name__ == "__main__":
    main()
