import argparse

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from src.database_service.database_service_agent import read_recent_data


class web_ui_interface:
    def __init__(self, db_url: str, table_name: str = "rmbr4_export_data"):
        self.db_url = db_url
        self.table_name = table_name

    def start(self):
        st.set_page_config(page_title="Robot Current Dashboard", layout="wide")
        st.title("Robot Current Dashboard")
        st_autorefresh(interval=2000, key="robot-current-refresh")

        data = read_recent_data(self.db_url, self.table_name)
        if data.empty:
            st.info("No readings have arrived in the last 90 seconds.")
            return

        data["time"] = pd.to_datetime(data["time"])
        axes = [f"axis_{index}" for index in range(1, 9)]
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.stackplot(data["time"], *(data[name].fillna(0) for name in axes), labels=axes)
        axis.set_xlabel("Time")
        axis.set_ylabel("Current (A)")
        axis.legend(loc="upper left", ncol=2)
        axis.grid(alpha=0.25)
        figure.autofmt_xdate()
        st.pyplot(figure)
        plt.close(figure)

    def stop(self):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    parser.add_argument("--table-name", default="rmbr4_export_data")
    arguments = parser.parse_args()
    web_ui_interface(arguments.db_url, arguments.table_name).start()