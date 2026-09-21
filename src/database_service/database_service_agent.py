# stream_data.py
import time
import pandas as pd
import psycopg2


def read_recent_data(db_url: str, table_name: str) -> pd.DataFrame:
    query = f"""SELECT time, axis_1, axis_2, axis_3, axis_4,
        axis_5, axis_6, axis_7, axis_8
        FROM {table_name}
        WHERE time >= (
            SELECT MAX(time) FROM {table_name}
        ) - INTERVAL '90 seconds'
        ORDER BY time"""
    with psycopg2.connect(db_url) as connection:
        return pd.read_sql_query(query, connection)

def stream_DF_to_neon(axis_data: pd.DataFrame, table_name: str, delay_seconds: float, db_url: str):
    """
    Streams dataframe rows one by one into a Neon database every `delay_seconds`.
    """
    # connect to the Neon postgreSQL database
    connection = psycopg2.connect(db_url)
    cursor = connection.cursor()
    # Create DB table if it doesn't exist
    create_table_query = f"""CREATE TABLE IF NOT EXISTS {table_name} (
    trait VARCHAR(255),
    axis_1 DOUBLE PRECISION,
    axis_2 DOUBLE PRECISION,
    axis_3 DOUBLE PRECISION,
    axis_4 DOUBLE PRECISION,
    axis_5 DOUBLE PRECISION,
    axis_6 DOUBLE PRECISION,
    axis_7 DOUBLE PRECISION,
    axis_8 DOUBLE PRECISION,
    axis_9 DOUBLE PRECISION,
    axis_10 DOUBLE PRECISION,
    axis_11 DOUBLE PRECISION,
    axis_12 DOUBLE PRECISION,
    axis_13 DOUBLE PRECISION,
    axis_14 DOUBLE PRECISION,
    time TIMESTAMP
)"""
    cursor.execute(create_table_query)
    connection.commit()


    print("data stream receiving every 0.1 seconds")
    insert_query = f"""INSERT INTO {table_name} (Trait, Axis_1, Axis_2, Axis_3, Axis_4, Axis_5, Axis_6, Axis_7, Axis_8, Axis_9, Axis_10, Axis_11, Axis_12, Axis_13, Axis_14, Time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s)"""
    try:
        for index, row in axis_data.iterrows():
            values = tuple(None if pd.isna(value) else value for value in row)
            cursor.execute(insert_query, values)
            connection.commit()
            print(f"[{time.strftime('%H:%M:%S')}] Transmitted Row {index + 1}/{len(axis_data)}")
            time.sleep(delay_seconds)

        print("Data uploaded successfully to Neon!")

    except KeyboardInterrupt:
        print("\nStreaming paused by user.")
    finally:
        cursor.close()
        connection.close()