# ==============================================
# COMPARE BUY & HOLD VS RISK REGIME (COLAB)
# ==============================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor


# ----------------------------------------------
# 1. SELECT INDEX
# ----------------------------------------------

INDEX_NAME = "banknifty"   # change to banknifty / sensex

BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/market_ml_project/"
DATA_PATH = os.path.join(BASE_PATH, f"{INDEX_NAME}_weekly_features.csv")

print("Loading:", DATA_PATH)


# ----------------------------------------------
# 2. LOAD DATA
# ----------------------------------------------

df = pd.read_csv(DATA_PATH)

# Ensure correct date column name
if "Date" not in df.columns:
    df.rename(columns={"date": "Date"}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


# ----------------------------------------------
# 3. CREATE TARGET (4 WEEK RETURN)
# ----------------------------------------------

df["future_4w_return"] = df["Close"].shift(-4) / df["Close"] - 1
df = df.dropna(subset=["future_4w_return"]).copy()


# ----------------------------------------------
# 4. FEATURES
# ----------------------------------------------

feature_cols = [
    "ema_20",
    "ema_50",
    "ema_100",
    "rsi_14",
    "rsi_change",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_percent",
    "bb_width",
    "obv",
    "obv_slope",
    "adx",
    "plus_di",
    "minus_di"
]

df = df.dropna(subset=feature_cols)


# ----------------------------------------------
# 5. TRAIN / TEST SPLIT
# ----------------------------------------------

split = int(len(df) * 0.8)

X_train = df[feature_cols].iloc[:split]
y_train = df["future_4w_return"].iloc[:split]

X_test = df[feature_cols].iloc[split:]
y_test = df["future_4w_return"].iloc[split:]

test_df = df.iloc[split:].copy()


# ----------------------------------------------
# 6. TRAIN MODEL
# ----------------------------------------------

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=6,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# ----------------------------------------------
# 7. RISK FILTER (ABSOLUTE -5%)
# ----------------------------------------------

risk_threshold = -0.05  # -5%

test_df["predicted_return"] = y_pred
test_df["risk_flag"] = test_df["predicted_return"] < risk_threshold

test_df["buy_hold_return"] = y_test.values
test_df["strategy_return"] = np.where(
    test_df["risk_flag"],
    0,
    y_test.values
)


# ----------------------------------------------
# 8. EQUITY CURVES
# ----------------------------------------------

test_df["buy_hold_equity"] = (1 + test_df["buy_hold_return"]).cumprod()
test_df["strategy_equity"] = (1 + test_df["strategy_return"]).cumprod()


# ----------------------------------------------
# 9. PERFORMANCE METRICS
# ----------------------------------------------

def max_drawdown(series):
    peak = series.cummax()
    drawdown = (series - peak) / peak
    return drawdown.min()

def sharpe_ratio(returns):
    return np.sqrt(52) * returns.mean() / returns.std()

print("\n=== PERFORMANCE ===")

print("Buy & Hold Final:", round(test_df["buy_hold_equity"].iloc[-1],2))
print("Strategy Final:", round(test_df["strategy_equity"].iloc[-1],2))

print("Buy & Hold MaxDD:", round(max_drawdown(test_df["buy_hold_equity"]),3))
print("Strategy MaxDD:", round(max_drawdown(test_df["strategy_equity"]),3))

print("Buy & Hold Sharpe:", round(sharpe_ratio(test_df["buy_hold_return"]),3))
print("Strategy Sharpe:", round(sharpe_ratio(test_df["strategy_return"]),3))

print("Exposure %:", round(100 * (1 - test_df["risk_flag"].mean()),2))


# ----------------------------------------------
# 10. PLOT GRAPH (INLINE)
# ----------------------------------------------
# ----------------------------------------------
# 10. PLOT GRAPH (SCRIPT SAFE VERSION)
# ----------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(test_df["Date"], test_df["buy_hold_equity"])
plt.plot(test_df["Date"], test_df["strategy_equity"])

plt.title(f"{INDEX_NAME.upper()} - Buy & Hold vs Risk Regime (-5%)")
plt.xlabel("Date")
plt.ylabel("Cumulative Equity")
plt.legend(["Buy & Hold", "Risk Regime"])
plt.grid(True)

plt.tight_layout()

plt.savefig("equity_curve.png")   # save image file
plt.show()
