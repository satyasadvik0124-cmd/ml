# ==========================================================
# CONDITIONAL LEVERAGE MODEL (FINAL STABLE VERSION)
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

ticker = "TRENT.NS"
INITIAL_CAPITAL = 1000000


def run_model(ticker):

    print("\nRunning Conditional Leverage Model for:", ticker)

    # ------------------------------------------------------
    # 1. Download Data
    # ------------------------------------------------------

    df = yf.download(ticker, start="2010-01-01", interval="1wk", auto_adjust=True)

    # 🔥 FIX MultiIndex issue from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().reset_index()

    if len(df) < 200:
        print("Not enough historical data.")
        return

    # ------------------------------------------------------
    # 2. Indicators
    # ------------------------------------------------------

    df["ema_50"] = df["Close"].ewm(span=50).mean()
    df["ema_100"] = df["Close"].ewm(span=100).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26

    # ----- ADX -----
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

    df = df.dropna().reset_index(drop=True)

    # ------------------------------------------------------
    # 3. Target (4W prediction)
    # ------------------------------------------------------

    df["future_4w_return"] = df["Close"].shift(-4) / df["Close"] - 1
    df = df.dropna().reset_index(drop=True)

    features = ["ema_50","ema_100","rsi_14","macd","adx"]

    X = df[features]
    y = df["future_4w_return"]

    split = int(len(df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    test_df = df.iloc[split:].copy().reset_index(drop=True)

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

    print("R²:", round(r2_score(y_test, pred),4))

    test_df["predicted_return"] = pred

    # ------------------------------------------------------
    # 5. Exposure Logic
    # ------------------------------------------------------

    crash_condition = test_df["predicted_return"] < -0.05

    bull_condition = (
        (test_df["adx"] > 25) &
        (test_df["Close"] > test_df["ema_50"]) &
        (test_df["ema_50"] > test_df["ema_100"])
    )

    test_df["exposure"] = 1.0
    test_df.loc[crash_condition, "exposure"] = 0.0
    test_df.loc[bull_condition & (~crash_condition), "exposure"] = 1.5

    # ------------------------------------------------------
    # 6. Weekly Compounding (Aligned)
    # ------------------------------------------------------

    test_df["actual_1w_return"] = test_df["Close"].shift(-1) / test_df["Close"] - 1

    test_df["strategy_return"] = (
        test_df["exposure"].shift(1) * test_df["actual_1w_return"]
    )

    test_df["buy_hold_return"] = test_df["actual_1w_return"]

    test_df = test_df.dropna().reset_index(drop=True)

    # ------------------------------------------------------
    # 7. Capital Growth
    # ------------------------------------------------------

    test_df["buy_hold_value"] = INITIAL_CAPITAL * (1 + test_df["buy_hold_return"]).cumprod()
    test_df["strategy_value"] = INITIAL_CAPITAL * (1 + test_df["strategy_return"]).cumprod()

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
    print(f"BUY & HOLD CAGR: {round(cagr_bh*100,2)}%")
    print(f"STRATEGY CAGR: {round(cagr_strat*100,2)}%")
    print(f"BUY & HOLD MaxDD: {round(mdd_bh*100,2)}%")
    print(f"STRATEGY MaxDD: {round(mdd_strat*100,2)}%")
    print(f"BUY & HOLD Sharpe: {round(sharpe_bh,2)}")
    print(f"STRATEGY Sharpe: {round(sharpe_strat,2)}")

    # ------------------------------------------------------
    # 9. Plot
  # ------------------------------------------------------
# 9. Plot (Professional Capital Style)
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
    label=f"Conditional Leverage (₹{final_strat:,.0f})"
)

plt.title(f"{ticker} - Capital Growth Comparison (Leverage Model)")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (₹)")
plt.legend()
plt.grid(True)

filename = f"{ticker.replace('.NS','').replace('.BO','')}_leverage_model.png"
plt.savefig(filename, dpi=300)
plt.show()

print("\nGraph saved as:", filename)

run_model(ticker)