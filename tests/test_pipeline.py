from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app import raw_features_for_inference
from train import LABEL_MAPPING, encode_target, evaluate_predictions, prepare_split


def test_target_encoding_marks_attrition_as_positive() -> None:
    target = pd.Series(["Existing Customer", "Attrited Customer"])

    assert encode_target(target).tolist() == [0, 1]
    assert LABEL_MAPPING["Attrited Customer"] == 1


def test_target_encoding_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unknown target labels"):
        encode_target(pd.Series(["Unknown Customer"]))


def test_prepare_split_excludes_target_and_identifiers() -> None:
    df = pd.read_csv("BankChurners.csv")

    split = prepare_split(df)

    assert split["raw_feature_names"] == list(df.columns[2:21])
    assert "Attrition_Flag" not in split["raw_feature_names"]
    assert "CLIENTNUM" not in split["raw_feature_names"]
    assert set(split["y_test"].unique()) == {0, 1}


def test_metrics_use_positive_churn_class() -> None:
    metrics = evaluate_predictions(
        pd.Series([0, 1, 1, 0]),
        np.array([0, 1, 0, 0]),
        np.array([0.1, 0.9, 0.4, 0.2]),
    )

    assert metrics["recall"] == 0.5
    assert metrics["precision"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_raw_feature_validation_reports_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        raw_features_for_inference(pd.DataFrame({"a": [1]}), ["a", "b"])
