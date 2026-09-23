---
type: Operations Guide
title: Local and container operations
description: Supported uv setup, model-group training, Streamlit startup, and the Mitra-enabled Docker deployment.
tags: [operations, uv, streamlit, docker, cuda]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:05:53.012Z
sources:
  - id: openwiki-source-bb1ebe868e35e9e500714501
    resource: repo://Dockerfile
  - id: openwiki-source-acd7cd16e2b091dc5e913e54
    resource: repo://docs/deployment-runbook.md
  - id: openwiki-source-83e9451fd99a1038793b8456
    resource: repo://docs/developer-guide.md
  - id: openwiki-source-5f8c47704115ca750ab777d4
    resource: repo://docs/training-and-evaluation.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
generated: { by: "codex", at: "2026-09-23T14:05:53.012Z" }
---

# Local and container operations

## Local development

The project documents Python 3.13 and `uv` as its baseline. From the repository root, synchronize the base environment with:

```powershell
uv sync
```

The classical/AutoML training group is the default and does not require a GPU. To use Mitra-v2 or TabFM, install the corresponding optional dependencies; the documented all-model setup is:

```powershell
uv sync --extra mitra --extra tabfm
```

Train the default group and start the dashboard with:

```powershell
uv run python train.py
uv run streamlit run app.py
```

The dashboard listens at `http://localhost:8501` and accepts BankChurners-format CSV or XLSX input. Foundation-model selections are only available when their bundle-referenced artifact files are present. See [inference and foundation models](/openwiki/architecture/inference-and-foundation-models.md).

## Docker deployment

The checked-in Dockerfile uses Python 3.13, performs a locked uv sync with the `mitra` extra, copies the model bundle and Mitra artifact into `/app`, exposes port 8501, and launches Streamlit. The documented run path requires a completed local `artifacts/mitra-v2` artifact, an NVIDIA GPU with Container Toolkit, and the matching schema-v2 bundle:

```powershell
docker build -t churn-prediction .
docker run --gpus all -p 8501:8501 churn-prediction
```

This image is Mitra-enabled; it does not package TabFM. The deployment guide describes TabFM as a local, non-commercial research path. For local startup checks and troubleshooting, use the [deployment runbook](../../docs/deployment-runbook.md).

## Verification commands

The developer guide documents these repository checks:

```powershell
uv run pytest -q
uv lock --check
```

The tests cover deterministic data/metric contracts, not GPU model training. A successful unit-test run therefore does not establish that either foundation-model integration has completed an end-to-end run.
