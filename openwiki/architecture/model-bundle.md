---
type: Architecture
title: Model bundle and artifacts
description: The schema-v2 Joblib payload and external model files produced by training and consumed by the Streamlit dashboard.
tags: [model-bundle, artifacts, persistence, joblib]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:05:53.012Z
sources:
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-af3e2b46f926bfe56e3c0b60
    resource: repo://docs/model-bundle-reference.md
  - id: openwiki-source-5f8c47704115ca750ab777d4
    resource: repo://docs/training-and-evaluation.md
  - id: openwiki-source-a939fb45dd00758ce74ef972
    resource: repo://train.py
generated: { by: "codex", at: "2026-09-23T14:05:53.012Z" }
---

# Model bundle and artifacts

`train.py` writes a Joblib dictionary to `models_bundle.pkl` after completing the requested model groups. The bundle stores classical estimators and the metadata needed to preprocess their inference inputs, plus descriptors for any foundation-model artifacts written separately.

## Bundle contents

Current payloads use `schema_version: 2`. The top-level fields include classical `models`, `foundation_models` descriptors, the classical scaler, encoded and raw feature-name lists, label mapping, held-out metrics, fit/prediction timings, and optional LazyPredict results. The descriptor for Mitra-v2 points to an AutoGluon artifact directory. The TabFM descriptor points to a Joblib file containing raw training features and labels, along with the successful context and prediction-chunk settings.

## Production and consumption

Training writes the configured bundle after assembling model results. Foundation model state is not embedded in that dictionary: Mitra-v2 is stored under the artifacts directory, and TabFM's training context is stored in its own file. These paths are recorded in each foundation-model descriptor.

The Streamlit app loads `models_bundle.pkl` as a local Joblib payload and expects a dictionary. It then offers a foundation model only when the descriptor's artifact or context path exists. Classical inference uses the saved feature names and scaler; foundation inference uses the raw feature names and follows the descriptor path. See [inference and foundation models](/openwiki/architecture/inference-and-foundation-models.md) for routing details.

Joblib uses Python pickle semantics, so only load a bundle created locally or obtained from a trusted artifact channel. The model-bundle reference also states that consumers should reject unknown schema versions rather than guessing their structure.

## Related pages

- [Training and evaluation pipeline](/openwiki/architecture/training-pipeline.md)
- [Inference and foundation models](/openwiki/architecture/inference-and-foundation-models.md)
- [System overview](/openwiki/architecture/system-overview.md)
