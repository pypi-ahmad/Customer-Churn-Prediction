---
type: Architecture
title: Training and evaluation pipeline
description: Data preparation, model-group execution, shared holdout evaluation, and training outputs in train.py.
tags: [training, evaluation, preprocessing, metrics]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:05:53.012Z
sources:
  - id: openwiki-source-115b2dad781e2a2c5b5a980d
    resource: repo://docs/architecture.md
  - id: openwiki-source-5f8c47704115ca750ab777d4
    resource: repo://docs/training-and-evaluation.md
  - id: openwiki-source-5df3e5ec1d3c549ddc9a97c9
    resource: repo://tests/test_pipeline.py
  - id: openwiki-source-a939fb45dd00758ce74ef972
    resource: repo://train.py
generated: { by: "codex", at: "2026-09-23T14:05:53.012Z" }
---

# Training and evaluation pipeline

`train.py` loads the BankChurners CSV and accepts comma-separated model groups: `classical`, `mitra-v2`, and `tabfm`. The default is `classical`. The classical group contains Random Forest, XGBoost, SVM, Decision Tree, and FLAML AutoML; the optional foundation-model groups require their respective `uv` extra and a CUDA GPU.

## Shared data preparation

The target is `Attrition_Flag`, mapped as `Existing Customer = 0` and `Attrited Customer = 1`; unknown or missing target values are rejected. Features are selected from source columns at positions 2–20, excluding the target and customer identifier under the current dataset layout. A stratified 80/20 split with seed 42 is shared by all enabled groups.

The classical path one-hot encodes the raw training and test partitions and aligns test columns to the training feature schema. SMOTE is fit only on the encoded training data. `StandardScaler` is fit on the resampled training partition and then applied to both model-training data and the untouched test partition.

Mitra-v2 and TabFM use the raw mixed-type training and test partitions rather than the SMOTE/scaling path. Mitra-v2 is fine-tuned with AutoGluon, with an option to reuse a completed Mitra artifact. TabFM fits against a bounded training context; the documented implementation reduces prediction chunks and then context sizes after CUDA out-of-memory errors.

## Evaluation and outputs

Each enabled model is evaluated against the same test partition. The pipeline records accuracy, ROC AUC, F1, precision, recall, fit time, and prediction time. LazyPredict is an additional comparison; if its run fails, the main training flow continues and leaves its result empty.

The output bundle contains the enabled estimators and foundation-model descriptors, preprocessing metadata, label mapping, metrics, timings, and optional LazyPredict results. Mitra-v2's predictor directory and TabFM's raw training context are separate files under the artifacts directory. See [model bundle and artifacts](/openwiki/architecture/model-bundle.md).

The checked-in tests cover target encoding and rejection, feature/identifier selection, positive-class metrics, and missing raw feature rejection. They do not constitute a full model-training benchmark; see [pipeline test contracts](/openwiki/testing/pipeline-contracts.md).

## Related pages

- [System overview](/openwiki/architecture/system-overview.md)
- [Model bundle and artifacts](/openwiki/architecture/model-bundle.md)
- [Pipeline test contracts](/openwiki/testing/pipeline-contracts.md)
