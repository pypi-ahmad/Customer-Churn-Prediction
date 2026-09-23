---
type: Test Reference
title: Pipeline test contracts
description: The deterministic behaviors currently asserted by tests/test_pipeline.py and how they relate to training and inference code.
tags: [tests, pipeline, contracts]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:05:53.012Z
sources:
  - id: openwiki-source-83e9451fd99a1038793b8456
    resource: repo://docs/developer-guide.md
  - id: openwiki-source-5df3e5ec1d3c549ddc9a97c9
    resource: repo://tests/test_pipeline.py
generated: { by: "codex", at: "2026-09-23T14:05:53.012Z" }
---

# Pipeline test contracts

Run the repository's documented test command from its root:

```powershell
uv run pytest -q
```

The current `tests/test_pipeline.py` suite covers these contracts:

- Target encoding maps `Existing Customer` to 0 and `Attrited Customer` to 1, and rejects an unknown target label.
- The split uses the source columns at positions 2–20 as raw model features and excludes `Attrition_Flag` and `CLIENTNUM`.
- `evaluate_predictions` treats class 1 as the positive churn class; the test pins recall, precision, and ROC AUC for a small example.
- Raw-feature inference reports an error when an expected feature column is missing.

These are focused data and metric tests. The file does not run full training, GPU inference, or the Streamlit browser UI. See [training and evaluation](/openwiki/architecture/training-pipeline.md) and [inference and foundation models](/openwiki/architecture/inference-and-foundation-models.md) for the surrounding implementation.
