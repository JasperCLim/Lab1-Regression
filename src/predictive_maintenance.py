from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


AXIS_COLUMNS = tuple(f"axis_{index}" for index in range(1, 9))
EVENT_COLUMNS = (
    "axis",
    "severity",
    "start_time",
    "end_time",
    "peak_time",
    "peak_current",
    "duration_seconds",
    "peak_deviation",
    "MinC",
    "MaxC",
    "sample_count",
)


@dataclass(frozen=True)
class AnalysisResult:
    test_data: pd.DataFrame
    axis_summary: pd.DataFrame
    events: pd.DataFrame


def normalize_readings(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize CSV and database columns to time and axis_1 through axis_8."""
    if data.empty:
        raise ValueError("No readings are available for analysis.")

    renamed: dict[object, str] = {}
    for column in data.columns:
        text = str(column).strip()
        if text.casefold() == "time":
            renamed[column] = "time"
            continue
        match = re.fullmatch(
            r"axis(?:\s*#\s*|\s*_\s*|\s*)(\d+)", text, flags=re.IGNORECASE
        )
        if match:
            renamed[column] = f"axis_{int(match.group(1))}"

    readings = data.rename(columns=renamed).copy()
    required = ("time", *AXIS_COLUMNS)
    missing = [column for column in required if column not in readings.columns]
    if missing:
        raise ValueError(f"Readings are missing required columns: {', '.join(missing)}")

    readings = readings.loc[:, required]
    readings["time"] = pd.to_datetime(readings["time"], utc=True, errors="coerce")
    for axis in AXIS_COLUMNS:
        readings[axis] = pd.to_numeric(readings[axis], errors="coerce")
    readings = readings.dropna(subset=["time"])
    readings = readings.dropna(subset=list(AXIS_COLUMNS), how="all")
    if readings.empty:
        raise ValueError("No rows contain a valid timestamp and axis reading.")
    return readings.sort_values("time", kind="stable").reset_index(drop=True)


def _consecutive_runs(
    mask: np.ndarray, timestamps: pd.Series, max_gap_seconds: float
) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []

    times_ns = timestamps.astype("int64").to_numpy()
    split_points: list[int] = []
    for position in range(1, indices.size):
        previous = indices[position - 1]
        current = indices[position]
        elapsed = (times_ns[current] - times_ns[previous]) / 1_000_000_000
        if current != previous + 1 or elapsed > max_gap_seconds:
            split_points.append(position)
    return list(np.split(indices, split_points))


def _observed_duration_seconds(
    run: np.ndarray, timestamps: pd.Series, sample_interval_seconds: float
) -> float:
    elapsed = (timestamps.iloc[run[-1]] - timestamps.iloc[run[0]]).total_seconds()
    return max(0.0, elapsed + sample_interval_seconds)


def _find_events(
    test_data: pd.DataFrame,
    thresholds: pd.DataFrame,
    duration_seconds: float,
) -> pd.DataFrame:
    if duration_seconds <= 0:
        raise ValueError("The sustained duration T must be greater than zero.")

    timestamps = test_data["time"]
    sample_intervals = timestamps.diff().dt.total_seconds()
    positive_intervals = sample_intervals[sample_intervals > 0]
    sample_interval = (
        float(positive_intervals.median()) if not positive_intervals.empty else 0.0
    )
    max_gap = sample_interval * 1.5
    events: list[dict[str, object]] = []

    for threshold in thresholds.itertuples(index=False):
        axis = threshold.axis
        residual = test_data[f"{axis}_residual"]
        min_runs = _consecutive_runs(
            residual.ge(threshold.MinC).fillna(False).to_numpy(),
            timestamps,
            max_gap,
        )
        max_mask = residual.ge(threshold.MaxC).fillna(False).to_numpy()
        for run in min_runs:
            event_duration = _observed_duration_seconds(
                run, timestamps, sample_interval
            )
            if event_duration < duration_seconds:
                continue

            high_runs = _consecutive_runs(
                max_mask[run], timestamps.iloc[run].reset_index(drop=True), max_gap
            )
            severity = (
                "Error"
                if any(
                    _observed_duration_seconds(
                        high_run,
                        timestamps.iloc[run].reset_index(drop=True),
                        sample_interval,
                    )
                    >= duration_seconds
                    for high_run in high_runs
                )
                else "Alert"
            )
            event_rows = test_data.iloc[run]
            peak_position = run[int(np.nanargmax(residual.iloc[run].to_numpy()))]
            events.append(
                {
                    "axis": axis,
                    "severity": severity,
                    "start_time": event_rows["time"].iloc[0],
                    "end_time": event_rows["time"].iloc[-1],
                    "peak_time": test_data["time"].iloc[peak_position],
                    "peak_current": float(test_data[axis].iloc[peak_position]),
                    "duration_seconds": round(event_duration, 2),
                    "peak_deviation": float(residual.iloc[run].max()),
                    "MinC": threshold.MinC,
                    "MaxC": threshold.MaxC,
                    "sample_count": len(run),
                }
            )

    return pd.DataFrame(events, columns=EVENT_COLUMNS).sort_values(
        ["start_time", "axis", "severity"], ignore_index=True
    )


def analyze_datasets(
    training_data: pd.DataFrame,
    test_data: pd.DataFrame,
    min_percentile: float = 99.0,
    max_percentile: float = 99.9,
    duration_seconds: float = 10.0,
) -> AnalysisResult:
    """Fit one model per axis and analyze synthetic/live readings against it.

    Thresholds are calibrated from positive residuals on the final 20% of
    chronological baseline readings; final models are refit on all baseline
    readings before predicting the test set.
    """
    if not 0 < min_percentile < max_percentile < 100:
        raise ValueError(
            "Threshold percentiles must satisfy 0 < MinC percentile "
            "< MaxC percentile < 100."
        )
    if duration_seconds <= 0:
        raise ValueError("The sustained duration T must be greater than zero.")

    training = normalize_readings(training_data)
    testing = normalize_readings(test_data)
    origin = training["time"].iloc[0]
    training_elapsed = (
        training["time"] - origin
    ).dt.total_seconds().to_numpy().reshape(-1, 1)
    test_elapsed = (
        testing["time"] - origin
    ).dt.total_seconds().to_numpy().reshape(-1, 1)

    summary_rows: list[dict[str, object]] = []
    for axis in AXIS_COLUMNS:
        axis_training = training.dropna(subset=[axis])
        split_at = int(len(axis_training) * 0.8)
        if split_at < 2 or len(axis_training) - split_at < 2:
            raise ValueError(
                f"Axis {axis} needs at least two training and two calibration readings."
            )

        positions = axis_training.index.to_numpy()
        axis_elapsed = training_elapsed[positions]
        axis_values = axis_training[axis].to_numpy(dtype=float)
        validation_model = LinearRegression().fit(
            axis_elapsed[:split_at], axis_values[:split_at]
        )
        validation_residuals = (
            axis_values[split_at:]
            - validation_model.predict(axis_elapsed[split_at:])
        )
        positive_residuals = validation_residuals[validation_residuals > 0]
        if positive_residuals.size == 0:
            raise ValueError(
                f"Axis {axis} has no positive calibration residuals; "
                "a positive-excess threshold cannot be estimated."
            )
        min_threshold, max_threshold = np.percentile(
            positive_residuals, [min_percentile, max_percentile]
        )
        if min_threshold >= max_threshold:
            raise ValueError(
                f"Axis {axis} does not have distinct thresholds at the selected "
                "percentiles. Choose more widely separated percentiles."
            )

        model = LinearRegression().fit(axis_elapsed, axis_values)
        predictions = model.predict(test_elapsed)
        testing[f"{axis}_prediction"] = predictions
        testing[f"{axis}_residual"] = testing[axis] - predictions
        summary_rows.append(
            {
                "axis": axis,
                "slope_per_second": float(model.coef_[0]),
                "intercept_at_training_start": float(model.intercept_),
                "holdout_rmse": float(
                    np.sqrt(
                        mean_squared_error(
                            axis_values[split_at:],
                            validation_model.predict(axis_elapsed[split_at:]),
                        )
                    )
                ),
                "holdout_r2": float(
                    r2_score(
                        axis_values[split_at:],
                        validation_model.predict(axis_elapsed[split_at:]),
                    )
                ),
                "MinC_percentile": min_percentile,
                "MaxC_percentile": max_percentile,
                "MinC": float(min_threshold),
                "MaxC": float(max_threshold),
                "calibration_positive_residuals": int(positive_residuals.size),
            }
        )

    axis_summary = pd.DataFrame(summary_rows)
    events = _find_events(
        testing, axis_summary[["axis", "MinC", "MaxC"]], duration_seconds
    )
    return AnalysisResult(testing, axis_summary, events)
