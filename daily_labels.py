# =============================================================================
# daily_labels.py — Create next-day direction labels for NIFTY, SENSEX, BANKNIFTY
# Saves labeled CSVs to /content/drive/MyDrive/Colab Notebooks/market_ml_project/
# =============================================================================

import os
import sys
import pandas as pd

# --- Setup paths --------------------------------------------------------------
ROOT = "/content/drive/MyDrive/Colab Notebooks/market_ml_project"
sys.path.insert(0, ROOT)

SAVE_DIR   = f"{ROOT}/data/raw"
OUTPUT_DIR = ROOT  # saves labeled CSVs directly into old project folder
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Imports ------------------------------------------------------------------
from config.config import START_DATE, END_DATE
from data.download_price import get_price_data

# --- Tickers to label ---------------------------------------------------------
LABEL_TICKERS = {
    "^NSEI"   : "nifty_daily_labeled.csv",
    "^BSESN"  : "sensex_daily_labeled.csv",
    "^NSEBANK": "banknifty_daily_labeled.csv",
}

# =============================================================================
# CORE FUNCTION
# =============================================================================

def create_labels(df, ticker):
    """
    Adds next-day direction label to a price DataFrame.

    label = 1 → next day close > today close (bullish)
    label = 0 → next day close <= today close (bearish)

    Also adds useful derived columns:
      - pct_change   : today's % change
      - future_return: tomorrow's % return
    """
    df = df.copy()

    # Next day close
    df["future_close"]  = df["Close"].shift(-1)

    # Direction label
    df["label"]         = (df["future_close"] > df["Close"]).astype(int)

    # Extra useful columns
    df["pct_change"]    = df["Close"].pct_change() * 100
    df["future_return"] = (df["future_close"] - df["Close"]) / df["Close"] * 100

    # Drop last row (no future close available)
    df.dropna(inplace=True)

    # Drop helper column
    df.drop(columns=["future_close"], inplace=True)

    return df

# =============================================================================
# DOWNLOAD DAILY DATA (separate from weekly — daily needed for next-day labels)
# =============================================================================

print("Downloading daily data for indices...")
print(f"Period: {START_DATE} → {END_DATE}\n")

daily_data = get_price_data(
    tickers=list(LABEL_TICKERS.keys()),
    start=START_DATE,
    end=END_DATE,
    timeframe="daily",           # daily — not weekly
    save_dir=SAVE_DIR,
    chunk_size=3,
)

# =============================================================================
# CREATE AND SAVE LABELED CSVs
# =============================================================================

print("\nCreating labeled CSVs...\n")

for ticker, filename in LABEL_TICKERS.items():
    if ticker not in daily_data:
        print(f"[SKIP] {ticker} — not in downloaded data")
        continue

    df = daily_data[ticker]

    if df.empty:
        print(f"[SKIP] {ticker} — empty DataFrame")
        continue

    labeled_df = create_labels(df, ticker)
    out_path   = os.path.join(OUTPUT_DIR, filename)
    labeled_df.to_csv(out_path)

    # Stats
    total   = len(labeled_df)
    bullish = labeled_df["label"].sum()
    bearish = total - bullish
    avg_ret = labeled_df["future_return"].mean()

    print(f"[OK] {ticker}")
    print(f"     Saved to  : {out_path}")
    print(f"     Rows      : {total}")
    print(f"     Bullish   : {bullish} ({bullish/total*100:.1f}%)")
    print(f"     Bearish   : {bearish} ({bearish/total*100:.1f}%)")
    print(f"     Avg return: {avg_ret:.3f}%")
    print(f"     Date range: {labeled_df.index[0].date()} → {labeled_df.index[-1].date()}")
    print()

print("="*50)
print("All labeled CSVs saved successfully!")
print(f"Location: {OUTPUT_DIR}")