---
type: Architecture
title: Inference and foundation models
description: How uploaded customer data is validated and routed through classical, Mitra-v2, or TabFM prediction paths.
tags: [inference, streamlit, mitra, tabfm, cuda]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:05:53.012Z
sources:
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-115b2dad781e2a2c5b5a980d
    resource: repo://docs/architecture.md
  - id: openwiki-source-7a7e355f860196e2022b2a4d
    resource: repo://foundation_models.py
generated: { by: "codex", at: "2026-09-23T14:05:53.012Z" }
---

# Inference and foundation models

The Streamlit dashboard accepts CSV and XLSX uploads and runs predictions for the models selected by the user. Classical estimators and the two optional foundation models use different input contracts; the foundation paths consume raw mixed-type feature columns, while the classical path applies the saved encoding and scaler.

## Upload and schema handling

`app.py` reads CSV with pandas and Excel with `openpyxl`, then converts object columns to strings. For classical estimators, it selects columns at positions 2 through 20, one-hot encodes categorical values, fills missing values with zero, aligns the resulting columns to the feature list stored in the bundle, and applies the saved scaler before prediction. This positional selection assumes the upload follows the training dataset's column layout.

For Mitra-v2 and TabFM, inference selects the bundle's raw feature names by column name. If any expected raw feature is absent, inference raises an error listing the missing columns. The dashboard catches read and prediction errors and presents them in the UI.

Each prediction is rendered as `Churned` when the model returns class `1`, otherwise `Retained`; the dashboard adds one `Prediction_<model>` column per selected model.

## Optional CUDA model paths

The application lists a foundation model only when its referenced artifact or context file exists. Mitra-v2 loads the AutoGluon predictor from its artifact path. TabFM loads its saved training context and builds a classifier for inference; prediction is processed in the chunk size saved in the bundle descriptor.

Both foundation-model operations require CUDA. The integration checks CUDA availability and raises a runtime error if it is unavailable. In the dashboard, a foundation model is released and CUDA cache cleanup is requested after that model's prediction finishes, so selected foundation models are handled one at a time rather than held together.

TabFM's pretrained weights are restricted to non-commercial research use. The dashboard displays that warning when TabFM is selected; the deployment guide describes the Docker image as Mitra-enabled and omits TabFM when its local context artifact is not present.

## Related pages

- [System overview](/openwiki/architecture/system-overview.md)
- [Model bundle and artifacts](/openwiki/architecture/model-bundle.md)
- [Local and container operations](/openwiki/operations/local-and-container.md)
