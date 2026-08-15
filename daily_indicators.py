import os
import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# ======================================================
# Automatically use the folder where this script exists
# ======================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

print("Working Directory:", os.getcwd())


def build_daily_features(input_csv, output_csv):
    # Read raw CSV
    df = pd.read_csv(input_csv)

    # --------------------------------------------------
    # Remove Yahoo Finance "Ticker" row if present
    # --------------------------------------------------
    if str(df.iloc[0, 0]) == "Ticker":
        df = df.iloc[1:].reset_index(drop=True)

    # --------------------------------------------------
    # Standardize Date column
    # --------------------------------------------------
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)

    # --------------------------------------------------
    # Convert price columns to numeric
    # --------------------------------------------------
    price_cols = ["Open", "High", "Low", "Close", "Volume"]

    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------
    # EMA
    # --------------------------------------------------
    df["ema_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    df["ema_100"] = EMAIndicator(df["Close"], window=100).ema_indicator()

    # --------------------------------------------------
    # RSI
    # --------------------------------------------------
    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
    df["rsi_change"] = df["rsi_14"].diff()

    # --------------------------------------------------
    # MACD
    # --------------------------------------------------
    macd = MACD(df["Close"])

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # --------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------
    bb = BollingerBands(df["Close"], window=20, window_dev=2)

    df["bb_percent"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    # --------------------------------------------------
    # OBV
    # --------------------------------------------------
    obv = OnBalanceVolumeIndicator(df["Close"], df["Volume"])

    df["obv"] = obv.on_balance_volume()
    df["obv_slope"] = df["obv"].diff()

    # --------------------------------------------------
    # ADX
    # --------------------------------------------------
    adx = ADXIndicator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()

    # --------------------------------------------------
    # Remove NaN rows
    # --------------------------------------------------
    df.dropna(inplace=True)

    # --------------------------------------------------
    # Save inside project folder
    # --------------------------------------------------
    output_path = os.path.join(PROJECT_DIR, output_csv)

    df.to_csv(output_path)

    print(f"✓ {output_csv} created")
    print(f"Rows: {len(df)}")
    print(f"Latest Date: {df.index.max().date()}")
    print(f"Saved to: {output_path}")
    print("-" * 60)


# ======================================================
# Generate features for all indices
# ======================================================

build_daily_features(
    "nifty_daily.csv",
    "nifty_daily_features.csv"
)

build_daily_features(
    "banknifty_daily.csv",
    "banknifty_daily_features.csv"
)

build_daily_features(
    "sensex_daily.csv",
    "sensex_daily_features.csv"
)

print("\nAll daily feature files generated successfully.")