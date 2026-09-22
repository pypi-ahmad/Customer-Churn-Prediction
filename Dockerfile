FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra mitra

COPY app.py foundation_models.py models_bundle.pkl .streamlit/config.toml ./
COPY artifacts/mitra-v2 ./artifacts/mitra-v2

EXPOSE 8501

ENTRYPOINT ["/app/.venv/bin/streamlit", "run", "app.py"]
