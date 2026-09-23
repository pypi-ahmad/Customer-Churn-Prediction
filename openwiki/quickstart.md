---
type: Quickstart
title: Quickstart
description: Set up the uv environment, run the current pipeline tests, train the default model group, and launch the Streamlit dashboard.
tags: [quickstart, uv, training, streamlit]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:05:53.012Z
sources:
  - id: openwiki-source-17dded95897e01ee430228e3
    resource: repo://app.py
  - id: openwiki-source-83e9451fd99a1038793b8456
    resource: repo://docs/developer-guide.md
  - id: openwiki-source-5f8c47704115ca750ab777d4
    resource: repo://docs/training-and-evaluation.md
  - id: openwiki-source-7a7e355f860196e2022b2a4d
    resource: repo://foundation_models.py
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-a939fb45dd00758ce74ef972
    resource: repo://train.py
generated: { by: "codex", at: "2026-09-23T14:05:53.012Z" }
---

# Quickstart

Run these commands from the repository root with Python 3.13 and `uv` installed. The default dataset path is `BankChurners.csv`.

```powershell
uv sync
uv run pytest -q
uv run python train.py
uv run streamlit run app.py
```

The default training run uses the classical/AutoML group and writes `models_bundle.pkl`, which the dashboard loads. Open `http://localhost:8501` and upload a BankChurners-format CSV or XLSX file.

To include Mitra-v2 and TabFM, install their optional dependencies and run with CUDA available:

```powershell
uv sync --extra mitra --extra tabfm
uv run python train.py --models classical,mitra-v2,tabfm
```

TabFM's pretrained weights are for non-commercial research use. The GPU training and local/container setup details are in [local and container operations](/openwiki/operations/local-and-container.md). For a map of the components and data paths, see [system overview](/openwiki/architecture/system-overview.md).
