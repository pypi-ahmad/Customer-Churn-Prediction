# Deployment runbook

Use these procedures to operate Streamlit locally or run the Mitra-enabled Docker image. TabFM runs locally because its pretrained weights are licensed for non-commercial research.

## Local startup

### 1. Verify required files

```powershell
Test-Path models_bundle.pkl
Test-Path artifacts\mitra-v2
Test-Path artifacts\tabfm-context.pkl
```

The bundle is always required. Each foundation model appears in the dashboard only when its referenced artifact exists.

### 2. Synchronize dependencies

Choose the extras that match the bundle:

```powershell
uv sync
uv sync --extra mitra
uv sync --extra mitra --extra tabfm
```

### 3. Start Streamlit

```powershell
uv run streamlit run app.py
```

Open `http://localhost:8501` and upload a BankChurners-format CSV or XLSX file.

### 4. Check application health

```powershell
(Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health).Content
```

Expected response:

```text
ok
```

Stop the foreground server with `Ctrl+C`.

## Docker startup

### Prerequisites

- Docker with Compose-independent `docker build` and `docker run` support
- NVIDIA driver and NVIDIA Container Toolkit
- A completed `artifacts/mitra-v2` directory
- A schema-v2 `models_bundle.pkl` containing the Mitra descriptor

Build and run:

```powershell
docker build -t churn-prediction .
docker run --gpus all --name churn-prediction -p 8501:8501 churn-prediction
```

Verify health:

```powershell
(Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health).Content
```

Stop and remove the container:

```powershell
docker stop churn-prediction
docker rm churn-prediction
```

The image installs the `mitra` optional dependency and copies the Mitra artifact. The dashboard filters out TabFM because its context file is absent.

## Troubleshooting

### `models_bundle.pkl not found`

Cause: the app started from the wrong working directory or training has not produced a bundle.

Actions:

1. Start Streamlit from the repository root.
2. Confirm `Test-Path models_bundle.pkl` returns `True`.
3. Run `uv run python train.py` if a classical-only bundle is acceptable.

### A foundation model is missing from the selector

Cause: its artifact path does not exist.

Actions:

1. Inspect `foundation_models` in the bundle.
2. Confirm the descriptor path exists relative to the current working directory.
3. Restore or regenerate the missing artifact.

### `CUDA GPU is required for Mitra-v2 and TabFM`

Cause: PyTorch cannot access CUDA.

Actions:

1. Run `uv run python -c "import torch; print(torch.cuda.is_available())"`.
2. Confirm the NVIDIA driver is available with `nvidia-smi`.
3. Re-run `uv sync` with the required optional extra.
4. In Docker, confirm NVIDIA Container Toolkit is installed and `--gpus all` is present.

### AutoGluon reports that the Mitra artifact path already exists

Cause: a fresh run targets an existing predictor directory.

Actions:

- Use `--reuse-mitra` for a complete artifact, or
- choose a new empty directory with `--artifacts-dir`.

Do not reuse an artifact left by a failed or interrupted fit.

### TabFM runs out of GPU memory

The training helper automatically tries prediction chunks of 64, 32, 16, and 8 rows. It then tries context limits of 64 and 32 rows when applicable.

If all attempts fail:

1. Close other GPU workloads.
2. Retry with `--tabfm-context-rows 64` or `32`.
3. Confirm Mitra is not loaded in another process.

### Predictions fail because columns are missing

Foundation models require every name in `raw_feature_names`. Classical uploads must contain at least the BankChurners columns used during positional feature selection.

Compare the upload with `BankChurners.csv`, restore missing columns, and keep original column names.

## Operational limits

- The app has no authentication or authorization layer.
- Models and uploads are processed in the Streamlit process; there is no task queue.
- Foundation-model inference runs sequentially to control VRAM use.
- The project does not provide drift monitoring, scheduled retraining, or production observability.
- The 1 GB upload limit is a server setting, not a recommended workload size.
