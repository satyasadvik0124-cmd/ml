import pandas as pd
from pathlib import Path


# -------------------------------------------------------
# Project Directory
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


# -------------------------------------------------------
# Function: Convert Daily Data to Weekly Data
# -------------------------------------------------------
def daily_to_weekly(input_csv, output_csv):
    """
    Converts daily OHLCV data to weekly OHLCV data.

    Parameters
    ----------
    input_csv : str or Path
        Path to daily CSV.

    output_csv : str or Path
        Path to save weekly CSV.
    """

    # Read CSV
    df = pd.read_csv(input_csv)

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort by Date
    df = df.sort_values("Date")

    # Set Date as index
    df.set_index("Date", inplace=True)

    # Weekly Resampling (Friday close)
    weekly_df = df.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    })

    # Remove incomplete weeks
    weekly_df.dropna(inplace=True)

    # Save
    weekly_df.to_csv(output_csv)

    print(f"✅ Saved: {output_csv}")
    print(f"Total Weekly Records: {len(weekly_df)}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
# -------------------------------------------------------
# Main
# -------------------------------------------------------
if __name__ == "__main__":

    files = {
        "nifty_daily.csv": "nifty_weekly.csv",
        "banknifty_daily.csv": "banknifty_weekly.csv",
        "sensex_daily.csv": "sensex_weekly.csv"
    }

    for daily_name, weekly_name in files.items():

        daily_file = BASE_DIR / daily_name
        weekly_file = BASE_DIR / weekly_name

        if not daily_file.exists():
            print(f"❌ File not found: {daily_file}")
            continue

        print(f"\nProcessing {daily_name}...")

        daily_to_weekly(daily_file, weekly_file)

    print("\n✅ All weekly files generated successfully.")