from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = Path("models_bundle.pkl")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


@st.cache_resource
def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "models_bundle.pkl not found. Run train.py to generate the model bundle."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle format. Expected a dict payload.")
    return bundle


@st.cache_data
def load_data(filename: str, bytes_data: bytes) -> pd.DataFrame:
    extension = Path(filename).suffix.lower()
    buffer = io.BytesIO(bytes_data)
    if extension == ".xlsx":
        df = pd.read_excel(buffer, engine="openpyxl")
    else:
        df = pd.read_csv(buffer)

    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str)
    return df


def ensure_streamlit_config(max_upload_size: int = 1024) -> None:
    config_dir = Path(".streamlit")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    config_contents = "[server]\nmaxUploadSize = 1024\n"

    if config_path.exists():
        existing_contents = config_path.read_text(encoding="utf-8")
        if "maxUploadSize" in existing_contents:
            return

    config_path.write_text(config_contents, encoding="utf-8")


def preprocess_for_inference(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if df.shape[1] < 21:
        raise ValueError(
            "Uploaded dataset must include at least 21 columns to match training data."
        )

    X = df.iloc[:, 2:21].copy()

    categorical_cols = X.select_dtypes(
        include=["object", "category", "string"],
    ).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=list(categorical_cols), drop_first=True)

    X = X.fillna(0)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X


def generate_predictions(
    df: pd.DataFrame,
    models: dict[str, Any],
    scaler: Any,
    feature_columns: list[str],
    selected_models: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    X = preprocess_for_inference(df, feature_columns)
    X_scaled = scaler.transform(X)

    result = df.copy()
    predictions_map: dict[str, pd.Series] = {}
    for name in selected_models:
        model = models[name]
        predictions = model.predict(X_scaled)
        series = pd.Series(
            np.where(predictions == 1, "Churned", "Retained"),
            index=df.index,
            name=f"Prediction_{name}",
        )
        result[series.name] = series
        predictions_map[name] = series

    return result, predictions_map


def render_eda_section(df: pd.DataFrame) -> None:
    st.subheader("Data Preview")
    st.dataframe(df.head(), width="stretch")
    st.dataframe(df.describe(include="all"), width="stretch")

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        st.subheader("Missing Values")
        missing_df = missing_counts.reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        missing_fig = px.bar(
            missing_df,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column",
        )
        st.plotly_chart(missing_fig, width='stretch')

    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Matrix")
        corr = numeric_df.corr()
        heatmap_fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )
        st.plotly_chart(heatmap_fig, width='stretch')

    st.subheader("Distribution Plotter")
    column_to_plot = st.selectbox("Select a column", options=df.columns)
    if pd.api.types.is_numeric_dtype(df[column_to_plot]):
        dist_fig = px.histogram(
            df,
            x=column_to_plot,
            title=f"Distribution of {column_to_plot}",
        )
    else:
        value_counts = df[column_to_plot].astype(str).value_counts().reset_index()
        value_counts.columns = [column_to_plot, "Count"]
        dist_fig = px.bar(
            value_counts,
            x=column_to_plot,
            y="Count",
            title=f"Distribution of {column_to_plot}",
        )
    st.plotly_chart(dist_fig, width='stretch')


def render_predictions_section(
    result: pd.DataFrame,
    predictions_map: dict[str, pd.Series],
    selected_models: list[str],
) -> None:
    st.subheader("Predictions")
    st.dataframe(result, width="stretch")

    if len(selected_models) == 1:
        prediction_column = f"Prediction_{selected_models[0]}"
        churn_rate = (result[prediction_column] == "Churned").mean()
        st.metric("Total Churn Rate", f"{churn_rate:.2%}")

        counts = (
            result[prediction_column]
            .value_counts()
            .reindex(["Churned", "Retained"], fill_value=0)
            .reset_index()
        )
        counts.columns = ["Prediction", "Count"]
        fig = px.pie(
            counts,
            values="Count",
            names="Prediction",
            title="Churned vs Retained",
            color="Prediction",
            color_discrete_map={"Churned": "#EF553B", "Retained": "#00CC96"},
        )
        st.plotly_chart(fig, width='stretch')
    else:
        prediction_columns = [f"Prediction_{name}" for name in predictions_map]
        agreement_count = result[prediction_columns].nunique(axis=1).eq(1).sum()
        st.metric("Model Agreement", f"{agreement_count}")

        churn_counts = pd.DataFrame(
            {
                "Model": list(predictions_map.keys()),
                "Churn Count": [
                    (predictions_map[name] == "Churned").sum()
                    for name in predictions_map
                ],
            }
        )
        fig = px.bar(
            churn_counts,
            x="Model",
            y="Churn Count",
            title="Churn Count by Model",
            text="Churn Count",
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, width='stretch')

        comparison_rows = []
        for name, preds in predictions_map.items():
            counts = preds.value_counts().reindex(["Churned", "Retained"], fill_value=0)
            comparison_rows.append(
                {
                    "Model": name,
                    "Churned": int(counts["Churned"]),
                    "Retained": int(counts["Retained"]),
                }
            )
        comparison_table = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_table, width="stretch")


def render_evaluation_section(
    df: pd.DataFrame,
    predictions_map: dict[str, pd.Series],
    selected_models: list[str],
) -> None:
    if "Attrition_Flag" not in df.columns:
        st.info("Upload a dataset with 'Attrition_Flag' to see evaluation metrics.")
        return

    actual = df["Attrition_Flag"].map({"Existing Customer": 0, "Attrited Customer": 1})
    actual = actual.dropna()
    if actual.empty:
        st.info("Upload a dataset with 'Attrition_Flag' to see evaluation metrics.")
        return

    for model_name in selected_models:
        st.subheader(f"{model_name} Confusion Matrix")
        predicted = predictions_map[model_name].map({"Retained": 0, "Churned": 1})
        predicted = predicted.loc[actual.index]

        cm = confusion_matrix(actual, predicted, labels=[0, 1])
        z_text = [
            [f"TN: {cm[0, 0]}", f"FP: {cm[0, 1]}"],
            [f"FN: {cm[1, 0]}", f"TP: {cm[1, 1]}"],
        ]
        labels = ["Existing Customer", "Attrited Customer"]
        cm_fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            annotation_text=z_text,
            colorscale="Blues",
        )
        cm_fig.update_layout(
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
        )

        report = classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=labels,
            zero_division=0,
            output_dict=True,
        )
        report_df = (
            pd.DataFrame(report).T
            .rename(columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-Score",
                "support": "Support",
            })
        )

        left, right = st.columns(2)
        with left:
            st.plotly_chart(cm_fig, width='stretch')
        with right:
            st.dataframe(report_df, width="stretch")


def main() -> None:
    configure_logging()

    ensure_streamlit_config()

    st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
    st.title("Customer Churn Prediction")
    st.write(
        "Upload a CSV or Excel file (BankChurners format) to explore data and predict churn."
    )

    try:
        bundle = load_model_bundle(MODEL_PATH)
        models = bundle["models"]
        scaler = bundle["scaler"]
        feature_columns = bundle["feature_names"]
    except KeyError as exc:
        st.error(f"Model bundle is missing a required key: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model bundle: {exc}")
        return

    model_names = list(models.keys())
    selected_models = st.sidebar.multiselect(
        "Select Models",
        options=model_names,
        default=["XGBoost"] if "XGBoost" in model_names else model_names[:1],
    )
    if not selected_models:
        st.warning("Please select at least one model to continue.")
        return

    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file to get started.")
        return

    try:
        df = load_data(uploaded_file.name, uploaded_file.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read file: {exc}")
        return

    with st.expander("Exploratory Data Analysis", expanded=True):
        render_eda_section(df)

    try:
        result, predictions_map = generate_predictions(
            df,
            models,
            scaler,
            feature_columns,
            selected_models,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Prediction failed: {exc}")
        return

    predictions_tab, evaluation_tab = st.tabs(
        ["Predictions", "Model Evaluation"]
    )

    with predictions_tab:
        render_predictions_section(result, predictions_map, selected_models)

    with evaluation_tab:
        render_evaluation_section(df, predictions_map, selected_models)


if __name__ == "__main__":
    main()
