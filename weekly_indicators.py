import pandas as pd
import numpy as np
import os
from pathlib import Path

# Project directory (folder where this script is stored)
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

print("Working Directory:", BASE_DIR)
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def build_weekly_features(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Handle Yahoo junk row
    if df.iloc[0, 0] == "Ticker":
        df = df.iloc[1:].reset_index(drop=True)

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df.set_index("Date", inplace=True)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =====================
    # EMA
    # =====================
    df["ema_20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(df["Close"], window=50).ema_indicator()
    df["ema_100"] = EMAIndicator(df["Close"], window=100).ema_indicator()

    # =====================
    # RSI
    # =====================
    df["rsi_14"] = RSIIndicator(df["Close"], window=14).rsi()
    df["rsi_change"] = df["rsi_14"].diff()

    # =====================
    # MACD
    # =====================
    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    # =====================
    # Bollinger Bands
    # =====================
    bb = BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_percent"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    # =====================
    # OBV (LOG SAFE)
    # =====================
    obv = OnBalanceVolumeIndicator(df["Close"], df["Volume"])
    df["obv_raw"] = obv.on_balance_volume()
    df["obv_slope_raw"] = df["obv_raw"].diff()

    df["obv"] = np.sign(df["obv_raw"]) * np.log1p(np.abs(df["obv_raw"]))
    df["obv_slope"] = np.sign(df["obv_slope_raw"]) * np.log1p(np.abs(df["obv_slope_raw"]))

    df.drop(columns=["obv_raw", "obv_slope_raw"], inplace=True)

    # =====================
    # ADX
    # =====================
    adx = ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()

    # =====================
    # CRITICAL CLEAN
    # =====================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # HARD CLIP (WEEKLY SAFE)
    df = df.clip(lower=-1e6, upper=1e6)

    df.to_csv(output_csv)
    print(output_csv, "created")


build_weekly_features(
    BASE_DIR / "nifty_weekly.csv",
    BASE_DIR / "nifty_weekly_features.csv"
)

build_weekly_features(
    BASE_DIR / "banknifty_weekly.csv",
    BASE_DIR / "banknifty_weekly_features.csv"
)

build_weekly_features(
    BASE_DIR / "sensex_weekly.csv",
    BASE_DIR / "sensex_weekly_features.csv"
)
