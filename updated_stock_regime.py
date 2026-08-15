# ==========================================================
# UPDATED UNIVERSAL NSE/BSE RISK REGIME TESTER (FULL METRICS)
# ==========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


ticker = "RELIANCE.NS"   # Change stock here


def run_risk_regime(ticker):

    print("\nRunning Updated Risk Regime Model for:", ticker)

    df = yf.download(ticker, start="2010-01-01", interval="1wk", auto_adjust=True)
    df = df.dropna().reset_index()

    if len(df) < 200:
        print("Not enough historical data.")
        return

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    df["ema_20"] = df["Close"].ewm(span=20).mean()
    df["ema_50"] = df["Close"].ewm(span=50).mean()
    df["ema_100"] = df["Close"].ewm(span=100).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_change"] = df["rsi_14"].diff()

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()
    df["bb_percent"] = (df["Close"] - rolling_mean) / (2 * rolling_std)
    df["bb_width"] = (2 * rolling_std) / rolling_mean

    df["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["obv_slope"] = df["obv"].diff()

    df["tr1"] = df["High"] - df["Low"]
    df["tr2"] = abs(df["High"] - df["Close"].shift())
    df["tr3"] = abs(df["Low"] - df["Close"].shift())
    df["tr"] = df[["tr1","tr2","tr3"]].max(axis=1)
    df["atr"] = df["tr"].rolling(14).mean()

    df["up_move"] = df["High"].diff()
    df["down_move"] = -df["Low"].diff()

    df["plus_dm"] = np.where(
        (df["up_move"] > df["down_move"]) & (df["up_move"] > 0),
        df["up_move"], 0
    )

    df["minus_dm"] = np.where(
        (df["down_move"] > df["up_move"]) & (df["down_move"] > 0),
        df["down_move"], 0
    )

    df["plus_di"] = 100 * (pd.Series(df["plus_dm"]).rolling(14).sum() / df["atr"])
    df["minus_di"] = 100 * (pd.Series(df["minus_dm"]).rolling(14).sum() / df["atr"])

    df["dx"] = 100 * abs(df["plus_di"] - df["minus_di"]) / (df["plus_di"] + df["minus_di"])
    df["adx"] = df["dx"].rolling(14).mean()

    df = df.dropna()

    # -----------------------------
    # Target
    # -----------------------------

    df["future_4w_return"] = df["Close"].shift(-4) / df["Close"] - 1
    df = df.dropna()

    features = [
        "ema_20","ema_50","ema_100",
        "rsi_14","rsi_change",
        "macd","macd_signal","macd_hist",
        "bb_percent","bb_width",
        "obv","obv_slope",
        "adx","plus_di","minus_di"
    ]

    X = df[features]
    y = df["future_4w_return"]

    split = int(len(df)*0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    test_df = df.iloc[split:].copy()

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\nRegression R²:", round(r2_score(y_test, pred),4))

    test_df["pred"] = pred
    test_df["buy_hold_return"] = y_test.values

    # -----------------------------
    # Regimes
    # -----------------------------

    perc_threshold = np.quantile(pred, 0.3)
    abs_threshold = -0.03

    test_df["perc_flag"] = pred < perc_threshold
    test_df["abs_flag"] = pred < abs_threshold
    test_df["hybrid_flag"] = (pred < abs_threshold) & (test_df["adx"] < 20)

    test_df["perc_return"] = np.where(test_df["perc_flag"], 0, y_test.values)
    test_df["abs_return"] = np.where(test_df["abs_flag"], 0, y_test.values)
    test_df["hybrid_return"] = np.where(test_df["hybrid_flag"], 0, y_test.values)

    # -----------------------------
    # Performance Function
    # -----------------------------

    def performance(name, returns, flag):
        equity = (1 + returns).cumprod()
        peak = equity.cummax()
        max_dd = ((equity - peak) / peak).min()
        sharpe = np.sqrt(52) * returns.mean() / returns.std()
        exposure = 100 * (1 - flag.mean())
        print(f"\n{name}")
        print("Final Return:", round(equity.iloc[-1],2))
        print("Max Drawdown:", round(max_dd,3))
        print("Sharpe:", round(sharpe,3))
        print("Exposure %:", round(exposure,2))

    # -----------------------------
    # Print Results
    # -----------------------------

    performance("Buy & Hold", test_df["buy_hold_return"], pd.Series([0]*len(test_df)))
    performance("Percentile Regime", test_df["perc_return"], test_df["perc_flag"])
    performance("Absolute Regime", test_df["abs_return"], test_df["abs_flag"])
    performance("Hybrid Regime", test_df["hybrid_return"], test_df["hybrid_flag"])

    # -----------------------------
    # Plot
    # -----------------------------

    plt.figure(figsize=(12,6))
    plt.plot(test_df["Date"], (1+test_df["buy_hold_return"]).cumprod())
    plt.plot(test_df["Date"], (1+test_df["perc_return"]).cumprod())
    plt.plot(test_df["Date"], (1+test_df["abs_return"]).cumprod())
    plt.plot(test_df["Date"], (1+test_df["hybrid_return"]).cumprod())

    plt.title(f"{ticker} - Regime Comparison")
    plt.legend(["Buy & Hold","Percentile","Absolute","Hybrid"])
    plt.grid(True)

    filename = f"{ticker.replace('.NS','').replace('.BO','')}_updated_regime.png"
    plt.savefig(filename)
    plt.close()

    print("\nGraph saved as:", filename)


run_risk_regime(ticker)
