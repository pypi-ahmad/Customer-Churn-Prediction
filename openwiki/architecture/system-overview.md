---
type: Architecture
title: System overview
description: End-to-end flow from BankChurners data preparation through model evaluation and persisted artifacts to Streamlit batch prediction.
tags: [architecture, training, evaluation, streamlit]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:05:53.012Z
sources:
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-115b2dad781e2a2c5b5a980d
    resource: repo://docs/architecture.md
  - id: openwiki-source-5df3e5ec1d3c549ddc9a97c9
    resource: repo://tests/test_pipeline.py
  - id: openwiki-source-a939fb45dd00758ce74ef972
    resource: repo://train.py
generated: { by: "codex", at: "2026-09-23T14:05:53.012Z" }
---

# System overview

This repository trains binary customer-churn classifiers from the BankChurners dataset and serves batch predictions through a Streamlit dashboard. The target mapping is `Existing Customer = 0` and `Attrited Customer = 1`.

## End-to-end flow

1. `train.py` loads the CSV, selects the 19 feature columns at positions 2–20, encodes the target, and creates one stratified 80/20 train/test split with seed 42.
2. The classical and FLAML branch one-hot encodes the raw features, applies SMOTE to the training partition, and fits a scaler on that resampled partition. Mitra-v2 and TabFM instead receive raw mixed-type training features.
3. Each enabled model group is evaluated against the shared, untouched test partition. The pipeline records metrics and fit/prediction timings.
4. Training writes `models_bundle.pkl`; foundation-model artifacts are stored separately and referenced from that bundle.
5. The Streamlit app loads the bundle, accepts CSV or XLSX uploads, and routes each selected model through its corresponding inference preprocessing path.

## Component boundaries

- `train.py` owns data preparation, the model groups, evaluation, and bundle serialization.
- `foundation_models.py` isolates the CUDA-backed Mitra-v2 and TabFM integration.
- `app.py` owns dashboard input, model selection, prediction, and result display.
- `tests/test_pipeline.py` pins selected target, feature, metric, and uploaded-schema behavior; see [pipeline test contracts](/openwiki/testing/pipeline-contracts.md).

The benchmark is a single holdout comparison, not cross-validation, and the project documentation cautions that it does not predict production performance. For implementation details, follow [training and evaluation](/openwiki/architecture/training-pipeline.md), [inference and foundation models](/openwiki/architecture/inference-and-foundation-models.md), and [model bundle and artifacts](/openwiki/architecture/model-bundle.md).
