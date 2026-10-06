# Group 6
## Jasper Lim 9117038


## Project Structure

| Path | Purpose |
|------|---------|
| `lab.ipynb` | Main lab notebook. The original workshop cells cover the problem statement and loading the `RMBR4-2` CSV. The upload cell bulk-loads the whole CSV into Neon in one transaction (it truncates the table first). The final cell reads the full Neon table, launches the dashboard in parallel and replays one reading every 2 seconds into it. |
| `src/predictive_maintenance.py` | Core model: normalizes column names, fits one scikit-learn `LinearRegression` per axis (time → current) on the baseline, derives MinC/MaxC from holdout residuals, and finds Alert/Error events via `analyze_datasets`. |
| `src/web_ui/web_ui_interface.py` | Streamlit + Plotly dashboard (regression plots, residual plots, metrics, events table, CSV download). |
| `src/database_service/database_service_agent.py` | Neon/PostgreSQL helpers: `stream_DF_to_neon` writes a DataFrame, `read_recent_data` reads the last 90 s. |
| `src/data_collection/data_collection_agent.py` | `read_csv_to_dataframe` CSV loader. |
| `src/controller/orchestra.py` | `orchestra` class tying streaming and the dashboard launch together. |
| `tests/test_predictive_maintenance.py` | Unit tests for the model logic. |
| `data/` | `RMBR4-2` (baseline), `RMBR4-3` and `RMBR4-4` (synthetic spiky test sets); axes 1–8 carry data. |
| `.env` | Git-ignored; holds `DATABASE_URL` (and optional `ROBOT_DATA_TABLE`). |

## How to Run

All commands run from the repository root in PowerShell.

1. **Setup** (once):
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   Create `.env` with `DATABASE_URL=postgresql://...` (needed only for the live Neon view/streaming).

2. **Run the model** — in Python:
   ```python
   from src.predictive_maintenance import analyze_datasets
   import pandas as pd
   base = pd.read_csv("data/RMBR4-2_export_test.csv")
   test = pd.read_csv("data/RMBR4-3_export_spiky.csv")
   result = analyze_datasets(base, test, min_percentile=95, max_percentile=99, duration_seconds=5)
   print(result.axis_summary); print(result.events)
   ```

3. **Run the dashboard**:
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run src\web_ui\web_ui_interface.py
   ```
   Pick the evaluation source in the sidebar: a synthetic CSV scenario (no database needed), **Live Neon readings**, or **Notebook stream (from Neon)**.

   **Streaming from the notebook:** open `lab.ipynb` (kernel `.venv`), run the upload cell once to load the full CSV into Neon, then run the final cell. It starts the dashboard in parallel (http://localhost:8501), and you choose **Notebook stream (from Neon)** in the sidebar. Interrupt the cell to stop both the stream and the dashboard.

4. **Run the tests**:
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

## Why These Default Thresholds

MinC, MaxC and T are tuning choices, not values derived from the data. The defaults (MinC = 95th percentile, MaxC = 99th, T = 5 s) were picked to balance missed failures against false alarms, and I checked them against the three datasets (median sample interval 1.89 s). Treat them as a reasoned starting point, not validated failure limits; the dashboard sliders let you change them.

**Percentiles.** MinC and MaxC are percentiles of the *positive* residuals (observed − predicted) on the baseline's final 20% window, per axis.
- **MinC at 95%:** only the top 5% of normal overshoot counts as an Alert candidate, so ordinary noise rarely triggers it. At 90%, 5 events appeared even on the baseline's own final 20% (false alarms). At 95% there were none.
- **MaxC at 99%:** an Error needs a sustained deviation larger than 99% of normal overshoot, which is clearly beyond normal operation and well above MinC.
- **Why not 99 / 99.9:** these are too strict for this dataset. RMBR4-3 gave 1 Error and RMBR4-4 none at T = 10 s, so most of the injected spikes would be missed.

**Duration T = 5 s.** About 3 consecutive samples at 1.89 s. A single sample can be sensor noise, so persistence is required, but a long T hides short real faults.
- **T = 2 s (about one sample):** 20 false events on the baseline window and 205 / 252 events on RMBR4-3 / RMBR4-4, i.e. noise is flagged.
- **T = 10 s:** RMBR4-3 drops to 1 event and RMBR4-4 to 0, so most spikes are missed. The RMBR4-3 events detected at 5 s last 5.7–11.4 s, with a median of about 5.8 s.
- **T = 5 s:** 0 false events on the baseline window, 8 events on RMBR4-3 and 17 on RMBR4-4.

**Limits of this check.**
- The baseline false-alarm test is partly in-sample, because the final model and the thresholds both use the baseline's last 20%.
- The synthetic datasets contain spikes of unknown ground truth, so "detected" means flagged, not confirmed as failures.
- Holdout R² is negative, so the time-only regression is a weak predictor. The thresholds measure deviation from a trend line, not proven failure risk.
- Units are the dataset's current units, not kWh.

## Predictive Maintenance Dashboard

The interactive Streamlit dashboard uses RMBR4-2 as the baseline and RMBR4-3 or RMBR4-4 as synthetic test scenarios. It fits a separate scikit-learn linear regression from elapsed time to current for each of axes 1–8, reports chronological holdout metrics and coefficients, plots observed readings and regression predictions with Plotly, and shows residuals and sustained events.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\web_ui\web_ui_interface.py
```

The defaults are MinC at the 95th percentile and MaxC at the 99th percentile of positive residuals from the baseline's final 20% calibration window, with a five-second persistence requirement. All three settings can be adjusted in the dashboard. Thresholds are per-axis and expressed in the dataset's current units; the CSVs do not provide a kWh conversion. Detected events can be downloaded as a structured CSV.

Choose **Live Neon readings** to monitor the configured database table (`rmbr4_export_data` by default). Set `DATABASE_URL` in the ignored project-root `.env` file; optionally set `ROBOT_DATA_TABLE` to a simple table name. This view refreshes every two seconds and queries the last 90 seconds of readings.

Check the holdout R² before interpreting predictions; negative values indicate that the linear time trend performed worse than a constant-mean baseline for that held-out period.