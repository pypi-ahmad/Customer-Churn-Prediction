from __future__ import annotations

import gc
import logging
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd

MITRA_REPO = "autogluon/mitra-classifier-2"
MITRA_REVISION = "edada0d20759c58ada8c8605c25f22f6e98ea5f0"
TABFM_REPO = "google/tabfm-1.0.0-pytorch"
TABFM_REVISION = "77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83"
TARGET_COLUMN = "__target__"
LOGGER = logging.getLogger(__name__)


def require_cuda() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Mitra-v2 and TabFM.")
    return torch


def release_cuda() -> None:
    gc.collect()
    torch = require_cuda()
    torch.cuda.empty_cache()


def train_mitra(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifact_path: Path,
    time_limit: int,
) -> tuple[Any, float]:
    require_cuda()
    from autogluon.tabular import TabularPredictor
    from huggingface_hub import snapshot_download

    checkpoint_path = snapshot_download(
        MITRA_REPO,
        revision=MITRA_REVISION,
        allow_patterns=["config.json", "model.safetensors"],
    )
    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train.to_numpy()
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(artifact_path),
        verbosity=2,
    )
    start = perf_counter()
    predictor.fit(
        train_data,
        time_limit=time_limit,
        hyperparameters={
            "MITRA": {
                "hf_model": checkpoint_path,
                "fine_tune": True,
                "fine_tune_steps": 50,
                "n_estimators": 1,
                "device": "cuda",
            }
        },
        fit_weighted_ensemble=False,
        dynamic_stacking=False,
        num_gpus=1,
        ag_args_fit={"ag.max_memory_usage_ratio": 2.2},
    )
    return predictor, perf_counter() - start


def load_mitra(artifact_path: str | Path) -> Any:
    from autogluon.tabular import TabularPredictor

    return TabularPredictor.load(str(artifact_path))


def predict_mitra(
    predictor: Any,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictor.predict(X), dtype=np.int8)
    probabilities = predictor.predict_proba(X)
    if isinstance(probabilities, pd.DataFrame):
        positive = probabilities[1] if 1 in probabilities else probabilities.iloc[:, -1]
    else:
        array = np.asarray(probabilities)
        positive = array[:, -1] if array.ndim == 2 else array
    return predictions, np.asarray(positive, dtype=float)


def download_tabfm_checkpoint() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        TABFM_REPO,
        revision=TABFM_REVISION,
        allow_patterns=["classification/**"],
    )


def build_tabfm_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_num_rows: int,
) -> Any:
    torch = require_cuda()
    from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch

    checkpoint_path = download_tabfm_checkpoint()
    model = tabfm_v1_0_0_pytorch.load(
        model_type="classification",
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype=torch.bfloat16,
        use_cache=True,
    )
    classifier = TabFMClassifier(
        model=model,
        n_estimators=1,
        max_num_features=500,
        max_num_rows=max_num_rows,
        batch_size=1,
        random_state=42,
        verbose=True,
    )
    classifier.fit(X_train, y_train)
    return classifier


def predict_tabfm(
    classifier: Any,
    X: pd.DataFrame,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for start in range(0, len(X), chunk_size):
        chunk = X.iloc[start : start + chunk_size]
        predictions.append(np.asarray(classifier.predict(chunk), dtype=np.int8))
        probabilities.append(np.asarray(classifier.predict_proba(chunk))[:, -1])
    return np.concatenate(predictions), np.concatenate(probabilities)


def fit_and_predict_tabfm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    requested_context_rows: int,
) -> tuple[Any, np.ndarray, np.ndarray, dict[str, int], float, float]:
    torch = require_cuda()
    context_options = [
        size
        for size in (requested_context_rows, 64, 32)
        if size <= requested_context_rows
    ]
    context_options = list(dict.fromkeys(context_options))
    last_error: Exception | None = None
    for context_rows in context_options:
        classifier = None
        for chunk_size in (64, 32, 16, 8):
            try:
                start = perf_counter()
                if classifier is None:
                    classifier = build_tabfm_classifier(X_train, y_train, context_rows)
                fit_seconds = perf_counter() - start
                predict_start = perf_counter()
                predictions, probabilities = predict_tabfm(
                    classifier,
                    X_test,
                    chunk_size,
                )
                predict_seconds = perf_counter() - predict_start
                settings = {
                    "max_num_rows": context_rows,
                    "prediction_chunk_rows": chunk_size,
                    "n_estimators": 1,
                }
                return (
                    classifier,
                    predictions,
                    probabilities,
                    settings,
                    fit_seconds,
                    predict_seconds,
                )
            except torch.cuda.OutOfMemoryError as exc:
                last_error = exc
                LOGGER.warning(
                    "TabFM CUDA OOM with context=%d, chunk=%d",
                    context_rows,
                    chunk_size,
                )
                gc.collect()
                torch.cuda.empty_cache()
    raise RuntimeError("TabFM could not fit in available CUDA memory.") from last_error


def save_tabfm_context(
    path: Path,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X_train": X_train, "y_train": y_train}, path)


def load_tabfm_from_context(path: str | Path, max_num_rows: int) -> Any:
    context = joblib.load(path)
    return build_tabfm_classifier(
        context["X_train"],
        context["y_train"],
        max_num_rows=max_num_rows,
    )
