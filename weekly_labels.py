import os
from pathlib import Path
import pandas as pd

# ======================================================
# Project Directory
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

print("Working Directory:", BASE_DIR)


# ======================================================
# Function: Create Weekly Labels
# ======================================================
def create_weekly_labels(input_csv, output_csv):
    """
    Creates labels for weekly classification.

    Label:
        1 -> Next week's Close > Current week's Close
        0 -> Otherwise
    """

    input_path = BASE_DIR / input_csv
    output_path = BASE_DIR / output_csv

    # Check if input exists
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    # Read weekly features
    df = pd.read_csv(input_path)

    # --------------------------------------------------
    # Standardize Date column
    # --------------------------------------------------
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)

    # --------------------------------------------------
    # Create Labels
    # --------------------------------------------------
    df["future_close"] = df["Close"].shift(-1)

    df["label"] = (
        df["future_close"] > df["Close"]
    ).astype(int)

    # Remove last row (future_close = NaN)
    df.dropna(inplace=True)

    # Remove helper column
    df.drop(columns=["future_close"], inplace=True)

    # Save
    df.to_csv(output_path)

    print(f"✓ {output_csv} created")
    print(f"Rows: {len(df)}")
    print(f"Latest Date: {df.index.max().date()}")
    print(f"Saved to: {output_path}")
    print("-" * 60)


# ======================================================
# Generate labels for all indices
# ======================================================

files = {
    "nifty_weekly_features.csv": "nifty_weekly_labeled.csv",
    "banknifty_weekly_features.csv": "banknifty_weekly_labeled.csv",
    "sensex_weekly_features.csv": "sensex_weekly_labeled.csv",
}

for input_file, output_file in files.items():
    create_weekly_labels(input_file, output_file)

print("\n✅ All weekly labeled datasets created successfully.")