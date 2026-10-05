import argparse
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from src.database_service.database_service_agent import read_recent_data
from src.predictive_maintenance import (
    AXIS_COLUMNS,
    AnalysisResult,
    analyze_datasets,
    normalize_readings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data"
TRAINING_FILE = DATA_DIRECTORY / "RMBR4-2_export_test.csv"
SYNTHETIC_FILES = {
    "RMBR4-3 synthetic scenario": DATA_DIRECTORY / "RMBR4-3_export_spiky.csv",
    "RMBR4-4 synthetic scenario": DATA_DIRECTORY / "RMBR4-4_export_spiky.csv",
}


@st.cache_data(show_spinner=False)
def load_csv(file_path: str) -> pd.DataFrame:
    return normalize_readings(pd.read_csv(file_path))


def regression_figure(
    training_data: pd.DataFrame,
    result: AnalysisResult,
) -> go.Figure:
    training = normalize_readings(training_data)
    testing = result.test_data
    step = max(1, len(testing) // 4000)
    training_step = max(1, len(training) // 2000)
    figure = make_subplots(
        rows=4,
        cols=2,
        shared_xaxes=True,
        subplot_titles=[f"Axis #{index}" for index in range(1, 9)],
        vertical_spacing=0.06,
    )
    events_by_axis = {
        axis: result.events[result.events["axis"] == axis]
        for axis in AXIS_COLUMNS
    }
    event_legend_shown: set[str] = set()

    for index, axis in enumerate(AXIS_COLUMNS):
        row, column = divmod(index, 2)
        row += 1
        column += 1
        axis_events = events_by_axis[axis]
        if index == 0:
            show_legend = True
        else:
            show_legend = False

        figure.add_trace(
            go.Scattergl(
                x=training["time"].iloc[::training_step],
                y=training[axis].iloc[::training_step],
                mode="markers",
                name="Baseline readings",
                legendgroup="baseline",
                showlegend=show_legend,
                marker={"size": 3, "color": "#94a3b8", "opacity": 0.35},
                hovertemplate="%{x}<br>Baseline: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scattergl(
                x=testing["time"].iloc[::step],
                y=testing[axis].iloc[::step],
                mode="markers",
                name="Synthetic/live readings",
                legendgroup="readings",
                showlegend=show_legend,
                marker={"size": 4, "color": "#2563eb", "opacity": 0.7},
                hovertemplate="%{x}<br>Observed: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=testing["time"].iloc[::step],
                y=testing[f"{axis}_prediction"].iloc[::step],
                mode="lines",
                name="Linear regression",
                legendgroup="prediction",
                showlegend=show_legend,
                line={"color": "#111827", "width": 2},
                hovertemplate="%{x}<br>Predicted: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=column,
        )

        axis_threshold = result.axis_summary.loc[
            result.axis_summary["axis"] == axis
        ].iloc[0]
        for level, color, dash in (
            ("MinC", "#f59e0b", "dot"),
            ("MaxC", "#dc2626", "dash"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=testing["time"].iloc[::step],
                    y=testing[f"{axis}_prediction"].iloc[::step]
                    + axis_threshold[level],
                    mode="lines",
                    name=f"{level} sustained threshold",
                    legendgroup=level,
                    showlegend=show_legend,
                    line={"color": color, "width": 1, "dash": dash},
                    hovertemplate=f"%{{x}}<br>Predicted + {level}: %{{y:.4f}}"
                    "<extra></extra>",
                ),
                row=row,
                col=column,
            )

        for severity, color, symbol in (
            ("Alert", "#f59e0b", "diamond"),
            ("Error", "#dc2626", "x"),
        ):
            severity_events = axis_events[axis_events["severity"] == severity]
            if severity_events.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=severity_events["peak_time"],
                    y=severity_events["peak_current"],
                    mode="markers+text",
                    text=[
                        f"{severity} · {duration:g}s"
                        for duration in severity_events["duration_seconds"]
                    ],
                    textposition="top center",
                    name=f"{severity} event",
                    legendgroup=severity,
                    showlegend=severity not in event_legend_shown,
                    marker={"size": 10, "color": color, "symbol": symbol},
                    hovertemplate=(
                        f"{severity}<br>%{{x}}<br>Current: %{{y:.4f}}"
                        "<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )
            event_legend_shown.add(severity)

    figure.update_layout(
        title="Synthetic/live current readings vs. baseline linear regression",
        height=1150,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"t": 125},
    )
    figure.update_yaxes(title_text="Current (dataset units)")
    figure.update_xaxes(title_text="Time", row=4)
    return figure


def residual_figure(result: AnalysisResult) -> go.Figure:
    testing = result.test_data
    step = max(1, len(testing) // 4000)
    figure = make_subplots(
        rows=4,
        cols=2,
        shared_xaxes=True,
        subplot_titles=[f"Axis #{index}: observed − predicted" for index in range(1, 9)],
        vertical_spacing=0.06,
    )

    for index, axis in enumerate(AXIS_COLUMNS):
        row, column = divmod(index, 2)
        row += 1
        column += 1
        threshold = result.axis_summary.loc[
            result.axis_summary["axis"] == axis
        ].iloc[0]
        figure.add_trace(
            go.Scattergl(
                x=testing["time"].iloc[::step],
                y=testing[f"{axis}_residual"].iloc[::step],
                mode="markers",
                name=f"Axis #{index + 1} residual",
                showlegend=False,
                marker={"size": 3, "color": "#475569", "opacity": 0.55},
                hovertemplate="%{x}<br>Residual: %{y:.4f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        for value, color, dash, name in (
            (0, "#111827", "solid", "Zero deviation"),
            (threshold["MinC"], "#f59e0b", "dot", "MinC"),
            (threshold["MaxC"], "#dc2626", "dash", "MaxC"),
        ):
            figure.add_hline(
                y=value,
                line_color=color,
                line_dash=dash,
                line_width=1,
                annotation_text=name if index == 0 else None,
                row=row,
                col=column,
            )

    figure.update_layout(
        title="Residuals and data-derived sustained thresholds",
        height=1050,
        hovermode="x unified",
        margin={"t": 90},
    )
    figure.update_yaxes(title_text="Current deviation")
    figure.update_xaxes(title_text="Time", row=4)
    return figure


def start(db_url: str | None = None, requested_table: str | None = None) -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    st.set_page_config(page_title="Robot Predictive Maintenance", layout="wide")
    st.title("Robot Predictive Maintenance Dashboard")
    st.caption(
        "Per-axis scikit-learn regression, residual-calibrated thresholds, "
        "and sustained anomaly events."
    )

    if not TRAINING_FILE.is_file():
        st.error(f"Baseline training file not found: {TRAINING_FILE}")
        st.stop()
    training_data = load_csv(str(TRAINING_FILE))

    st.sidebar.header("Analysis settings")
    source = st.sidebar.radio(
        "Evaluation data",
        ("Synthetic CSV scenario", "Live Neon readings"),
    )
    if source == "Synthetic CSV scenario":
        scenario = st.sidebar.selectbox("Test scenario", tuple(SYNTHETIC_FILES))
        test_data = load_csv(str(SYNTHETIC_FILES[scenario]))
        st.subheader(scenario)
    else:
        st_autorefresh(interval=2000, key="robot-current-refresh")
        database_url = db_url or os.getenv("DATABASE_URL")
        if not database_url:
            st.error("DATABASE_URL is missing. Add it to the project-root .env file.")
            st.stop()
        table_name = requested_table or os.getenv(
            "ROBOT_DATA_TABLE", "rmbr4_export_data"
        )
        try:
            test_data = read_recent_data(database_url, table_name)
        except psycopg2.Error as error:
            st.error(f"Unable to read live robot readings from Neon: {error}")
            st.stop()
        if test_data.empty:
            st.info("No readings have arrived in the last 90 seconds.")
            st.stop()
        st.subheader(f"Live Neon readings — {table_name}")

    st.sidebar.subheader("Discover thresholds")
    min_percentile = st.sidebar.slider(
        "MinC calibration percentile",
        min_value=90.0,
        max_value=99.5,
        value=95.0,
        step=0.1,
    )
    max_percentile = st.sidebar.slider(
        "MaxC calibration percentile",
        min_value=max(95.0, min_percentile + 0.1),
        max_value=99.99,
        value=max(99.0, min_percentile + 0.1),
        step=0.01,
    )
    duration_seconds = st.sidebar.slider(
        "Minimum sustained duration T (seconds)",
        min_value=2,
        max_value=120,
        value=5,
        step=1,
    )

    try:
        result = analyze_datasets(
            training_data,
            test_data,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            duration_seconds=float(duration_seconds),
        )
    except ValueError as error:
        st.error(f"Unable to analyze these readings: {error}")
        st.stop()

    interval = test_data["time"].diff().dt.total_seconds()
    typical_interval = interval[interval > 0].median()
    intervals_per_threshold = (
        f"{duration_seconds / typical_interval:.1f}"
        if pd.notna(typical_interval) and typical_interval > 0
        else "unavailable"
    )
    st.markdown(
        f"**Baseline:** RMBR4-2, chronological 80/20 calibration split. "
        f"**Thresholds:** per-axis {min_percentile:g}th and "
        f"{max_percentile:g}th percentiles of positive holdout residuals. "
        f"**Persistence:** {duration_seconds:g} seconds "
        f"(about {intervals_per_threshold} sample intervals). "
        "Current deviations use the units in the CSV; the source does not specify "
        "kWh, so no energy-unit conversion is assumed."
    )

    summary = result.axis_summary
    alert_count = int((result.events["severity"] == "Alert").sum())
    error_count = int((result.events["severity"] == "Error").sum())
    metric_columns = st.columns(4)
    metric_columns[0].metric("Readings analyzed", f"{len(result.test_data):,}")
    metric_columns[1].metric("Axes modeled", len(summary))
    metric_columns[2].metric("Sustained alerts", alert_count)
    metric_columns[3].metric("Sustained errors", error_count)

    st.subheader("Model fit and discovered thresholds by axis")
    st.dataframe(
        summary[
            [
                "axis",
                "slope_per_second",
                "intercept_at_training_start",
                "holdout_rmse",
                "holdout_r2",
                "MinC_percentile",
                "MinC",
                "MaxC_percentile",
                "MaxC",
                "calibration_positive_residuals",
            ]
        ].rename(
            columns={
                "axis": "Axis",
                "slope_per_second": "Slope / second",
                "intercept_at_training_start": "Intercept",
                "holdout_rmse": "Holdout RMSE",
                "holdout_r2": "Holdout R²",
                "MinC_percentile": "MinC percentile",
                "MinC": "MinC (current units)",
                "MaxC_percentile": "MaxC percentile",
                "MaxC": "MaxC (current units)",
                "calibration_positive_residuals": "Positive calibration residuals",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    median_r2 = float(summary["holdout_r2"].median())
    if median_r2 < 0:
        st.info(
            f"The median chronological holdout R² is {median_r2:.3f}. A negative "
            "score means the time-only linear trend predicts worse than a constant "
            "mean baseline on the held-out readings; treat this as a teaching "
            "baseline, not a validated failure predictor."
        )

    st.plotly_chart(regression_figure(training_data, result), use_container_width=True)
    st.plotly_chart(residual_figure(result), use_container_width=True)

    st.subheader("Sustained events")
    if result.events.empty:
        st.info(
            "No positive residual remained above MinC for the selected duration. "
            "Try a synthetic scenario, a longer observation window, or a different "
            "calibration percentile."
        )
    else:
        st.dataframe(result.events, hide_index=True, use_container_width=True)
    st.download_button(
        "Download event log as CSV",
        data=result.events.to_csv(index=False),
        file_name="robot_predictive_maintenance_events.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url")
    parser.add_argument("--table-name")
    arguments, _ = parser.parse_known_args()
    start(arguments.db_url, arguments.table_name)
