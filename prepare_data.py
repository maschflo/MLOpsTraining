import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/ai4i2020.csv"
RANDOM_STATE = 22
PRODUCTION_FRACTION = 0.2

def main():
    original = pd.read_csv(RAW_PATH)
    reference_data, production = train_test_split(
        original, test_size=PRODUCTION_FRACTION,
        stratify=original["Machine failure"], random_state=RANDOM_STATE)
    production.to_csv("./data/production_data.csv", index=False)
    reference_data.to_csv("./data/reference_data.csv", index=False)

if __name__ == "__main__":
    main()