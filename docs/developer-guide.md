# Developer guide

## Prerequisites

- Windows 11 or Linux
- Python 3.13, managed by uv
- Git
- An NVIDIA CUDA-capable GPU for Mitra-v2 or TabFM work

The classical pipeline, tests, and most dashboard development do not require a GPU.

## Set up the repository

```powershell
git clone https://github.com/pypi-ahmad/Customer-Churn-Prediction.git
Set-Location Customer-Churn-Prediction
uv sync
```

For foundation-model development:

```powershell
uv sync --extra mitra --extra tabfm
```

uv installs PyTorch from the configured CUDA 13.2 index. Confirm the environment before a long run:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The second line must be `True` for either foundation model.

## Common development commands

Run the regression suite:

```powershell
uv run pytest -q
```

Check and format Python files without adding Ruff to the project dependencies:

```powershell
uv run --with ruff ruff check train.py app.py foundation_models.py tests
uv run --with ruff ruff format --check train.py app.py foundation_models.py tests
```

Validate dependency resolution:

```powershell
uv lock --check
```

Start the dashboard:

```powershell
uv run streamlit run app.py
```

## Project conventions

- Keep churn as the positive class: `Attrited Customer = 1`.
- Preserve the shared raw train/test split when comparing model families.
- Apply SMOTE and scaling only to the classical training path.
- Pin external model and source revisions in `foundation_models.py` and `pyproject.toml`.
- Keep TabFM out of commercial deployment paths.
- Release CUDA memory before loading a second foundation model.
- Update the bundle schema and its reference document together when serialized keys change.

## Test coverage

[`tests/test_pipeline.py`](../tests/test_pipeline.py) covers deterministic pipeline behavior:

- churn label direction;
- rejection of unknown labels;
- exclusion of identifiers and target data from model features;
- positive-class metric semantics;
- validation of raw foundation-model columns.

The unit suite does not run GPU training. Verify model integrations with bounded end-to-end runs on supported hardware.
