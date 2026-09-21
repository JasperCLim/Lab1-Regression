import pandas as pd


def read_csv_to_dataframe(csv_file_path: str, start_row: int = 0) -> pd.DataFrame:
    return pd.read_csv(csv_file_path).iloc[start_row:]
