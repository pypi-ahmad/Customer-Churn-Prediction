# Training and evaluation

## Choose a model group

`--models` accepts a comma-separated combination of three groups:

| Group | Models | Required environment |
|---|---|---|
| `classical` | Random Forest, XGBoost, SVM, Decision Tree, FLAML AutoML | Base uv environment |
| `mitra-v2` | Fine-tuned Mitra-v2 classifier | `mitra` extra and CUDA GPU |
| `tabfm` | TabFM classifier | `tabfm` extra and CUDA GPU |

The command writes one bundle containing only the groups selected in that run. A subset run replaces the existing output file unless `--bundle` points elsewhere.

## Train classical models

```powershell
uv run python train.py
```

This is equivalent to `--models classical`. FLAML receives a 60-second search budget. LazyPredict runs after model fitting; if its benchmark fails, training continues and stores `null` for `lazypredict_results`.

## Train all seven models

Install the optional dependencies first:

```powershell
uv sync --extra mitra --extra tabfm
```

Then run:

```powershell
uv run python train.py `
  --models classical,mitra-v2,tabfm `
  --mitra-time-limit 3600 `
  --tabfm-context-rows 100
```

Mitra-v2 uses one estimator and up to 50 fine-tuning steps. The time limit is passed to AutoGluon; model finalization and validation can make wall-clock time longer than the configured training budget.

TabFM starts with the requested context size and 64-row prediction chunks. On CUDA out-of-memory errors, it tries smaller prediction chunks and then context sizes of 64 and 32 rows when those are below the requested size. The successful settings are stored in the bundle.

## Reuse a completed Mitra artifact

Use this after Mitra has finished successfully and `artifacts/mitra-v2` exists:

```powershell
uv run python train.py `
  --models classical,mitra-v2,tabfm `
  --reuse-mitra `
  --mitra-fit-seconds 3759.65 `
  --tabfm-context-rows 100
```

`--mitra-fit-seconds` records the original training duration in the new bundle. Omit it if preserving that timing is not important; the loader duration will be recorded instead.

Do not use `--reuse-mitra` with a missing or incomplete artifact. For a fresh Mitra run, use an empty artifact path or a new `--artifacts-dir`.

## Command reference

| Option | Default | Meaning |
|---|---|---|
| `--data` | `BankChurners.csv` | Input CSV path |
| `--bundle` | `models_bundle.pkl` | Output bundle path |
| `--artifacts-dir` | `artifacts` | Foundation-model artifact root |
| `--models` | `classical` | Comma-separated model groups |
| `--mitra-time-limit` | `3600` | AutoGluon training budget in seconds |
| `--reuse-mitra` | Off | Load an existing Mitra artifact |
| `--mitra-fit-seconds` | Not set | Original Mitra training duration to retain |
| `--tabfm-context-rows` | `100` | Initial TabFM context-row limit |

## Evaluation contract

All model groups share the same stratified 80/20 split and random seed 42. Metrics are computed with churn as class `1`:

- accuracy;
- ROC AUC from positive-class probabilities;
- F1;
- precision;
- recall.

The benchmark provides a reproducible project comparison. It does not estimate uncertainty, performance drift, calibration, or results on a different customer population.

## Generated artifacts

| Path | Contents |
|---|---|
| `models_bundle.pkl` | Classical estimators, preprocessing state, foundation-model metadata, metrics, timings, and optional LazyPredict results |
| `artifacts/mitra-v2/` | AutoGluon predictor and fine-tuned Mitra state |
| `artifacts/tabfm-context.pkl` | Raw TabFM training context and labels |

The `artifacts/` directory is ignored by Git. Back it up separately when a trained foundation model must be retained.
