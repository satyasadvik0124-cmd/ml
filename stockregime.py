# ==========================================================
# UNIVERSAL NSE/BSE RISK REGIME TESTER (PROBABILITY STYLE)
# ==========================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# ==========================================================
# 🔹 CHANGE STOCK HERE
# ==========================================================

ticker = "INFY.NS"

INITIAL_CAPITAL = 1000000   # ₹10 Lakhs


# ==========================================================
# FUNCTION
# ==========================================================

def run_risk_regime(ticker):

    print("\nRunning Risk Regime Model for:", ticker)

    # ------------------------------------------------------
    # 1. Download Data
    # ------------------------------------------------------

    df = yf.download(ticker, start="2010-01-01", interval="1wk", auto_adjust=True)
    df = df.dropna().reset_index()

    if len(df) < 200:
        print("Not enough historical data.")
        return

    # ------------------------------------------------------
    # 2. Feature Engineering
    # ------------------------------------------------------

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

    df = df.dropna()

    # ------------------------------------------------------
    # 3. Target
    # ------------------------------------------------------

    df["future_4w_return"] = df["Close"].shift(-4) / df["Close"] - 1
    df = df.dropna()

    features = [
        "ema_20","ema_50","ema_100",
        "rsi_14","rsi_change",
        "macd","macd_signal","macd_hist",
        "bb_percent","bb_width",
        "obv","obv_slope"
    ]

    X = df[features]
    y = df["future_4w_return"]

    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    test_df = df.iloc[split:].copy()

    # ------------------------------------------------------
    # 4. Train Model
    # ------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("R²:", round(r2_score(y_test, pred), 4))

    # ------------------------------------------------------
    # 5. Probability-Based Risk Regime (UNCHANGED LOGIC)
    # ------------------------------------------------------

    threshold = np.quantile(pred, 0.3)  # bottom 30% filter

    test_df["predicted_return"] = pred
    test_df["risk_flag"] = test_df["predicted_return"] < threshold

    test_df["buy_hold_return"] = y_test.values
    test_df["strategy_return"] = np.where(
        test_df["risk_flag"],
        0,
        test_df["buy_hold_return"]
    )

    exposure = 100 * (1 - test_df["risk_flag"].mean())

    # ------------------------------------------------------
    # 6. Capital Growth
    # ------------------------------------------------------

    test_df["buy_hold_value"] = INITIAL_CAPITAL * (1 + test_df["buy_hold_return"]).cumprod()
    test_df["strategy_value"] = INITIAL_CAPITAL * (1 + test_df["strategy_return"]).cumprod()

    # ------------------------------------------------------
    # 7. Performance
    # ------------------------------------------------------

    years = (test_df["Date"].iloc[-1] - test_df["Date"].iloc[0]).days / 365.25

    final_bh = test_df["buy_hold_value"].iloc[-1]
    final_strat = test_df["strategy_value"].iloc[-1]

    cagr_bh = (final_bh / INITIAL_CAPITAL) ** (1 / years) - 1
    cagr_strat = (final_strat / INITIAL_CAPITAL) ** (1 / years) - 1

    def max_dd(series):
        peak = series.cummax()
        return ((series - peak) / peak).min()

    mdd_bh = max_dd(test_df["buy_hold_value"])
    mdd_strat = max_dd(test_df["strategy_value"])

    sharpe_bh = (test_df["buy_hold_return"].mean() /
                 test_df["buy_hold_return"].std()) * np.sqrt(52)

    sharpe_strat = (test_df["strategy_return"].mean() /
                    test_df["strategy_return"].std()) * np.sqrt(52)

    # ------------------------------------------------------
    # 8. Print Results
    # ------------------------------------------------------

    print("\n===== PERFORMANCE =====")
    print(f"Initial Capital: ₹{INITIAL_CAPITAL:,.0f}")
    print(f"Bottom 30% Threshold: {threshold:.4f}")
    print(f"Exposure %: {exposure:.2f}%\n")

    print("BUY & HOLD")
    print(f"Final Value: ₹{final_bh:,.0f}")
    print(f"CAGR: {cagr_bh*100:.2f}%")
    print(f"MaxDD: {mdd_bh*100:.2f}%")
    print(f"Sharpe: {sharpe_bh:.2f}\n")

    print("RISK REGIME")
    print(f"Final Value: ₹{final_strat:,.0f}")
    print(f"CAGR: {cagr_strat*100:.2f}%")
    print(f"MaxDD: {mdd_strat*100:.2f}%")
    print(f"Sharpe: {sharpe_strat:.2f}")

    # ------------------------------------------------------
    # 9. Plot
    # ------------------------------------------------------

    plt.figure(figsize=(12,6))

    plt.plot(
        test_df["Date"],
        test_df["buy_hold_value"],
        label=f"Buy & Hold (₹{final_bh:,.0f})"
    )

    plt.plot(
        test_df["Date"],
        test_df["strategy_value"],
        label=f"Risk Regime (₹{final_strat:,.0f})"
    )

    plt.title(f"{ticker} - Capital Growth Comparison")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (₹)")
    plt.legend()
    plt.grid(True)

    filename = f"{ticker.replace('.NS','').replace('.BO','')}_risk_regime.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print("\nGraph saved as:", filename)


# ==========================================================
# RUN
# ==========================================================

run_risk_regime(ticker)
