"""Streamlit analytics dashboard for the saved UCI model artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark.errors import PySparkException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    DATASET_PATH,
    NUMERIC_COLUMNS,
    UCI_DATASET_PAGE,
)
from src.model_training import load_saved_model  # noqa: E402
from src.prediction import predict_session  # noqa: E402
from src.spark_session import create_spark_session  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"
PREDICTIONS_PATH = OUTPUT_DIR / "purchase_predictions.csv"
MODELS_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Customer Purchase Behaviour Analytics",
    layout="wide",
)


@st.cache_data
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not METRICS_PATH.is_file() or not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError("Model outputs are missing. Run `python main.py` first.")
    dataset = pd.read_csv(DATASET_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return dataset, predictions, metrics


@st.cache_resource
def load_model_resources(model_directory: str):
    spark = create_spark_session("CustomerPurchaseDashboard")
    model = load_saved_model(model_directory)
    return spark, model


def _session_duration(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["administrative_duration"]
        + frame["informational_duration"]
        + frame["product_related_duration"]
    )


def _add_behavior_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["session_duration"] = _session_duration(result)
    result["browsing_activity"] = pd.cut(
        result["product_related"],
        bins=[-float("inf"), 5, 20, float("inf")],
        labels=["Low (0-5)", "Medium (6-20)", "High (21+)"],
    ).astype(str)
    result["session_duration_category"] = pd.cut(
        result["session_duration"],
        bins=[-float("inf"), 300, 1800, float("inf")],
        labels=["Short (0-5 min)", "Medium (5-30 min)", "Long (>30 min)"],
    ).astype(str)
    result["purchase_status"] = result["purchase"].map(
        {0: "No purchase", 1: "Purchase", False: "No purchase", True: "Purchase"}
    )
    return result


try:
    customers, prediction_results, model_metrics = load_artifacts()
except (FileNotFoundError, pd.errors.ParserError, json.JSONDecodeError) as error:
    st.error(str(error))
    st.info("Generate the dataset, trained models, and outputs with `python main.py`.")
    st.stop()

customers = _add_behavior_groups(customers)
predictions = prediction_results.copy()
best_model_name = model_metrics["best_model"]
best_metrics = model_metrics["models"][best_model_name]

st.title("Customer Purchase Behaviour Analytics Dashboard")
st.caption(
    "Apache Spark MLlib classification using the UCI Online Shoppers Purchasing "
    "Intention Dataset. The source contains sessions, not customer age or prior "
    "purchase counts."
)
st.markdown(f"Dataset source: [{UCI_DATASET_PAGE}]({UCI_DATASET_PAGE})")

total_sessions = len(customers)
total_purchases = int((customers["purchase"] == 1).sum())
total_non_purchases = total_sessions - total_purchases
holdout_predicted_purchases = int((predictions["predicted_purchase"] == 1).sum())
conversion_rate = total_purchases / total_sessions if total_sessions else 0

kpi_columns = st.columns(6)
kpis = [
    ("Total sessions", f"{total_sessions:,}"),
    ("Actual purchases", f"{total_purchases:,}"),
    ("No purchases", f"{total_non_purchases:,}"),
    ("Actual conversion", f"{conversion_rate:.1%}"),
    ("Holdout predicted purchases", f"{holdout_predicted_purchases:,}"),
    ("Holdout model accuracy", f"{best_metrics['accuracy']:.1%}"),
]
for column, (label, value) in zip(kpi_columns, kpis):
    column.metric(label, value)

overview, segments, performance, predictions_tab, inference = st.tabs(
    ["Purchase analysis", "Customer segmentation", "Model performance", "Predictions", "Try a prediction"]
)

with overview:
    st.subheader("Observed purchase outcomes")
    left, right = st.columns(2)
    outcome_counts = customers["purchase_status"].value_counts().rename_axis("Outcome").reset_index(name="Sessions")
    left.plotly_chart(
        px.pie(outcome_counts, names="Outcome", values="Sessions", hole=0.45, title="Purchase vs no purchase"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(outcome_counts, x="Outcome", y="Sessions", color="Outcome", title="Session outcome counts"),
        use_container_width=True,
    )
    st.info(
        "Age-group and past-purchase charts are not available: those fields are not in "
        "the source. `VisitorType` is shown as a new/returning-visitor descriptor, not "
        "as a count of previous purchases."
    )

with segments:
    st.subheader("Conversion by observed session behaviour")
    selected_group = st.selectbox(
        "Segment sessions by",
        ["Browsing activity", "Session duration", "Visitor type (proxy only)", "Month"],
    )
    group_column = {
        "Browsing activity": "browsing_activity",
        "Session duration": "session_duration_category",
        "Visitor type (proxy only)": "visitor_type",
        "Month": "month",
    }[selected_group]
    grouped = (
        customers.groupby(group_column, observed=False)["purchase"]
        .agg(sessions="size", purchases="sum", conversion_rate="mean")
        .reset_index()
        .rename(columns={group_column: "Segment"})
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(grouped, x="Segment", y="sessions", color="Segment", title="Sessions per segment"),
        use_container_width=True,
    )
    right.plotly_chart(
        px.bar(
            grouped,
            x="Segment",
            y="conversion_rate",
            color="Segment",
            title="Observed purchase conversion rate",
            labels={"conversion_rate": "Conversion rate"},
        ),
        use_container_width=True,
    )
    st.caption(
        "Session duration is derived as the sum of the three page-category duration fields. "
        "Browsing activity uses ProductRelated page views. Visitor type is not purchase history."
    )

with performance:
    st.subheader(f"Holdout evaluation - {best_model_name}")
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_columns = st.columns(len(metric_names))
    for column, metric_name in zip(metric_columns, metric_names):
        column.metric(metric_name.replace("_", " ").title(), f"{best_metrics[metric_name]:.3f}")
    st.caption(model_metrics["selection_metric"])
    comparison = pd.DataFrame(model_metrics["models"]).T.reset_index(names="Model")
    st.plotly_chart(
        px.bar(
            comparison.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            title="Model comparison on the same holdout set",
            range_y=[0, 1],
        ),
        use_container_width=True,
    )
    figure_columns = st.columns(2)
    for column, filename, title in zip(
        figure_columns,
        ["confusion_matrix.png", "roc_curve.png"],
        ["Confusion matrix", "ROC curve"],
    ):
        image_path = OUTPUT_DIR / filename
        if image_path.is_file():
            column.image(str(image_path), caption=title, use_container_width=True)
    st.image(
        str(OUTPUT_DIR / "actual_vs_predicted.png"),
        caption="Actual versus predicted test-session counts",
        use_container_width=True,
    )

with predictions_tab:
    st.subheader("Holdout prediction results")
    correct_count = int(predictions["correct_prediction"].astype(bool).sum())
    incorrect_count = len(predictions) - correct_count
    st.write(f"Correct: **{correct_count:,}** | Incorrect: **{incorrect_count:,}**")
    st.plotly_chart(
        px.histogram(
            predictions,
            x="actual_purchase",
            color="predicted_purchase",
            barmode="group",
            title="Actual versus predicted outcomes",
            labels={"actual_purchase": "Actual purchase", "predicted_purchase": "Predicted purchase"},
        ),
        use_container_width=True,
    )
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button(
        "Download holdout predictions",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name="purchase_predictions.csv",
        mime="text/csv",
    )

with inference:
    st.subheader("Estimate purchase probability for one session")
    st.warning(
        "Enter fields that exist in the source dataset. Age and prior purchase count are "
        "not available and are intentionally not requested."
    )
    with st.form("session_prediction_form"):
        numeric_values = {}
        columns = st.columns(2)
        integer_columns = {
            "administrative",
            "informational",
            "product_related",
            "operating_systems",
            "browser",
            "region",
            "traffic_type",
        }
        fractional_columns = {"bounce_rates", "exit_rates", "special_day"}
        for index, feature in enumerate(NUMERIC_COLUMNS):
            minimum = float(customers[feature].min())
            maximum = float(customers[feature].max())
            default = float(customers[feature].median())
            step = 0.01 if feature in fractional_columns else (
                1.0 if feature in integer_columns else 10.0
            )
            numeric_values[feature] = columns[index % 2].number_input(
                feature.replace("_", " ").title(),
                min_value=minimum,
                max_value=maximum,
                value=min(max(default, minimum), maximum),
                step=step,
                key=f"input_{feature}",
            )
        categorical_values = {}
        for index, feature in enumerate(CATEGORICAL_COLUMNS):
            choices = sorted(customers[feature].dropna().astype(str).unique().tolist())
            if "Unknown" not in choices:
                choices.append("Unknown")
            categorical_values[feature] = columns[index % 2].selectbox(
                feature.replace("_", " ").title(),
                choices,
                key=f"input_{feature}",
            )
        submitted = st.form_submit_button("Predict purchase likelihood")

    if submitted:
        model_directory = MODELS_DIR / best_model_name.lower().replace(" ", "_").replace("-", "")
        try:
            spark, model = load_model_resources(str(model_directory))
            prediction = predict_session(
                model,
                spark,
                {**numeric_values, **categorical_values},
            )
        except (ValueError, OSError, PySparkException) as error:
            st.error(f"Prediction failed: {error}")
        else:
            outcome = "Likely purchase" if prediction["predicted_purchase"] else "Likely no purchase"
            st.success(f"{outcome} - purchase probability {prediction['purchase_probability']:.1%}")
