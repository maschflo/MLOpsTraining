import argparse
import httpx
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/production_data.csv"
COLUMN_MAP = {"air_temp": "Air temperature [K]", "process_temp": "Process temperature [K]",
               "rot_speed": "Rotational speed [rpm]", "torque": "Torque [Nm]", "tool_wear": "Tool wear [min]",
               "product_type": "Type"}

def apply_shift(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df.copy()
    rng = np.random.default_rng(22)
    noise = rng.uniform(low=0.2, high=8.0, size=len(df))
    shifted["Process temperature [K]"] = shifted["Process temperature [K]"] + noise
    mult_noise = rng.uniform(low=1.02, high=1.15, size=len(df))
    shifted["Rotational speed [rpm]"] = (shifted["Rotational speed [rpm]"] * mult_noise).round()
    return shifted

def send_requests(df: pd.DataFrame) -> None:
    with httpx.Client() as client:
        for row in df.to_dict(orient="records"):
            pl = {k: row[v] for k, v in COLUMN_MAP.items()}
            client.post(API_URL, json=pl)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--shift", action="store_true", default=False)
    df = pd.read_csv(DATA_PATH)
    if parser.parse_args().shift:
        print("*****************BEFORE SHIFT ******************")
        print(df.describe())
        df = apply_shift(df)
        print("*****************AFTER SHIFT ******************")
        print(df.describe())
    send_requests(df)

if __name__ == "__main__":
    main()