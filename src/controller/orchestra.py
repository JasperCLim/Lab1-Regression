import subprocess
import sys
from pathlib import Path

from src.data_collection.data_collection_agent import read_csv_to_dataframe
from src.database_service.database_service_agent import stream_DF_to_neon


class orchestra:
    @staticmethod
    def telnet_simulation(csv_file_path, start_row=0):
        return read_csv_to_dataframe(csv_file_path, start_row)

    @staticmethod
    def web_database_connection(axis_data, table_name, delay_seconds, db_url):
        return stream_DF_to_neon(axis_data, table_name, delay_seconds, db_url)

    @staticmethod
    def activate_web_ui(db_url, table_name="rmbr4_export_data"):
        web_ui_path = Path(__file__).resolve().parents[1] / "web_ui" / "web_ui_interface.py"
        return subprocess.Popen([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(web_ui_path),
            "--",
            "--db-url",
            db_url,
            "--table-name",
            table_name,
        ], cwd=Path(__file__).resolve().parents[2])

    # Launch the Plotly predictive-maintenance dashboard.

    