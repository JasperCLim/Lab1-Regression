import unittest

import numpy as np
import pandas as pd

from src.predictive_maintenance import analyze_datasets, normalize_readings
from src.database_service.database_service_agent import read_recent_data


def readings_frame(rows: int = 200) -> pd.DataFrame:
    random = np.random.default_rng(42)
    seconds = np.arange(rows) * 2
    frame = pd.DataFrame(
        {
            "Time": pd.Timestamp("2025-01-01", tz="UTC")
            + pd.to_timedelta(seconds, unit="s")
        }
    )
    for index in range(1, 9):
        frame[f"Axis #{index}"] = (
            index * 0.1
            + seconds * index * 0.0001
            + np.sin(seconds / (3 + index)) * 0.1
            + random.normal(0, 0.08, rows)
        )
    return frame


class PredictiveMaintenanceTests(unittest.TestCase):
    def test_normalizes_csv_and_database_column_names(self) -> None:
        csv_readings = readings_frame()
        normalized = normalize_readings(csv_readings)
        self.assertEqual(
            list(normalized.columns),
            ["time", *(f"axis_{index}" for index in range(1, 9))],
        )

        database_readings = normalized.rename(
            columns={f"axis_{index}": f"Axis_{index}" for index in range(1, 9)}
        )
        normalized_database = normalize_readings(database_readings)
        self.assertEqual(normalized_database.shape, normalized.shape)

    def test_fits_each_axis_and_reports_sustained_alerts_and_errors(self) -> None:
        training = readings_frame()
        baseline = analyze_datasets(
            training,
            training,
            min_percentile=95,
            max_percentile=99,
            duration_seconds=5,
        )
        self.assertEqual(len(baseline.axis_summary), 8)
        self.assertTrue(
            (baseline.axis_summary["MinC"] < baseline.axis_summary["MaxC"]).all()
        )

        test = training.copy()
        threshold = baseline.axis_summary.set_index("axis").loc["axis_1"]
        prediction = baseline.test_data["axis_1_prediction"]
        error_start, error_end = 30, 38
        alert_start, alert_end = 70, 78
        test.loc[error_start:error_end - 1, "Axis #1"] = (
            prediction.iloc[error_start:error_end] + threshold["MaxC"] + 1
        ).to_numpy()
        test.loc[alert_start:alert_end - 1, "Axis #1"] = (
            prediction.iloc[alert_start:alert_end]
            + (threshold["MinC"] + threshold["MaxC"]) / 2
        ).to_numpy()

        result = analyze_datasets(
            training,
            test,
            min_percentile=95,
            max_percentile=99,
            duration_seconds=5,
        )
        axis_events = result.events[result.events["axis"] == "axis_1"]
        self.assertTrue(
            (axis_events["severity"] == "Error").any(),
            "A sustained MaxC exceedance should be classified as Error.",
        )
        self.assertTrue(
            (axis_events["severity"] == "Alert").any(),
            "A sustained MinC-only exceedance should be classified as Alert.",
        )
        self.assertTrue((axis_events["duration_seconds"] >= 5).all())

    def test_short_exceedance_does_not_become_an_event(self) -> None:
        training = readings_frame()
        baseline = analyze_datasets(
            training,
            training,
            min_percentile=95,
            max_percentile=99,
            duration_seconds=5,
        )
        test = training.copy()
        threshold = baseline.axis_summary.set_index("axis").loc["axis_1"]
        prediction = baseline.test_data["axis_1_prediction"]
        test.loc[30:31, "Axis #1"] = (
            prediction.iloc[30:32] + threshold["MaxC"] + 1
        ).to_numpy()

        result = analyze_datasets(
            training,
            test,
            min_percentile=95,
            max_percentile=99,
            duration_seconds=5,
        )
        short_window = result.events[
            result.events["start_time"].between(
                result.test_data["time"].iloc[29],
                result.test_data["time"].iloc[32],
            )
        ]
        self.assertTrue(short_window.empty)

    def test_rejects_non_identifier_database_table_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "simple SQL identifier"):
            read_recent_data("unused", "robot_data; DROP TABLE robot_data")


if __name__ == "__main__":
    unittest.main()
