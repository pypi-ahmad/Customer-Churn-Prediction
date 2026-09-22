# Model bundle reference

`models_bundle.pkl` is a trusted local Joblib payload produced by `train.py`. Current bundles use schema version 2.

> Joblib uses Python pickle semantics. Never load a bundle from an untrusted source.

## Top-level schema

| Key | Type | Description |
|---|---|---|
| `schema_version` | `int` | Bundle contract version; currently `2` |
| `models` | `dict[str, estimator]` | Serialized classical and FLAML estimators |
| `foundation_models` | `dict[str, dict]` | External artifact descriptors for Mitra-v2 and TabFM |
| `scaler` | `StandardScaler` | Classical inference scaler fitted on resampled training data |
| `feature_names` | `list[str]` | Ordered one-hot encoded classical feature columns |
| `raw_feature_names` | `list[str]` | Ordered raw columns required by foundation models |
| `label_mapping` | `dict[str, int]` | `Existing Customer: 0`, `Attrited Customer: 1` |
| `metrics` | `dict[str, dict[str, float]]` | Held-out metrics by model |
| `timings` | `dict[str, dict[str, float]]` | Fit and test-prediction durations by model |
| `lazypredict_results` | `DataFrame | None` | LazyPredict comparison table when available |

## Foundation-model descriptors

### Mitra-v2

| Key | Meaning |
|---|---|
| `type` | `mitra-v2` dispatch identifier |
| `artifact_path` | AutoGluon predictor directory |
| `repository` | Hugging Face repository ID |
| `revision` | Immutable checkpoint commit |
| `license` | Weight license recorded by training |

### TabFM

| Key | Meaning |
|---|---|
| `type` | `tabfm` dispatch identifier |
| `context_path` | Joblib training-context path |
| `repository` | Hugging Face repository ID |
| `revision` | Immutable checkpoint commit |
| `license` | Pretrained-weight license identifier |
| `max_num_rows` | Context size that completed successfully |
| `prediction_chunk_rows` | Prediction chunk size that completed successfully |
| `n_estimators` | TabFM ensemble count; currently `1` |

## Reading metadata

```powershell
uv run python -c "import joblib; b=joblib.load('models_bundle.pkl'); print(b['schema_version']); print(b['metrics'])"
```

Only load a bundle created locally or obtained through a trusted artifact channel.

## Compatibility rules

- Consumers should reject unknown schema versions instead of guessing their structure.
- Classical inference must reindex encoded data to `feature_names` before scaling.
- Foundation inference must select `raw_feature_names` in the stored order.
- A foundation descriptor requires the file or directory at its `artifact_path` or `context_path`.
- Relative artifact paths are resolved from the process working directory. Start the app from the repository or image work directory unless paths were written as absolute paths.

## Metrics and timings

`metrics` contains `accuracy`, `roc_auc`, `f1`, `precision`, and `recall`. `timings` contains `fit_seconds` and `predict_seconds`.

Timings describe the machine and run that generated the bundle. Use them as provenance for that run; they do not guarantee performance on another machine.
