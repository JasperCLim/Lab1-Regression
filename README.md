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

## Predictive Maintenance Dashboard

The interactive Streamlit dashboard uses RMBR4-2 as the baseline and RMBR4-3 or RMBR4-4 as synthetic test scenarios. It fits a separate scikit-learn linear regression from elapsed time to current for each of axes 1–8, reports chronological holdout metrics and coefficients, plots observed readings and regression predictions with Plotly, and shows residuals and sustained events.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\web_ui\web_ui_interface.py
```

The defaults are MinC at the 95th percentile and MaxC at the 99th percentile of positive residuals from the baseline's final 20% calibration window, with a five-second persistence requirement. All three settings can be adjusted in the dashboard. Thresholds are per-axis and expressed in the dataset's current units; the CSVs do not provide a kWh conversion. Detected events can be downloaded as a structured CSV.

Choose **Live Neon readings** to monitor the configured database table (`rmbr4_export_data` by default). Set `DATABASE_URL` in the ignored project-root `.env` file; optionally set `ROBOT_DATA_TABLE` to a simple table name. This view refreshes every two seconds and queries the last 90 seconds of readings.

Check the holdout R² before interpreting predictions; negative values indicate that the linear time trend performed worse than a constant-mean baseline for that held-out period.