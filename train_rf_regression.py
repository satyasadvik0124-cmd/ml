# ==========================================================
# WEEKLY RISK-FILTER CLASSIFIER (v3)
# Switches from regression -> classification of "risk week",
# adds a volatility/regime feature, prunes weak features,
# evaluates across multiple walk-forward folds (not one split),
# and statistically tests whether results beat random chance.
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)


# ==========================================================
# 1. SELECT INDEX
# ==========================================================

INDEX_NAME = "banknifty"

BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/market_ml_project/"
DATA_PATH = os.path.join(BASE_PATH, f"{INDEX_NAME}_weekly_features.csv")

print(f"\nRunning Risk-Filter Classifier for: {INDEX_NAME}")
print("Loading:", DATA_PATH)


# ==========================================================
# 2. LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)


# ==========================================================
# 3. TARGET: was the NEXT 4 weeks a "risk" period?
#    (future return below threshold -> 1, else 0)
# ==========================================================

RISK_THRESHOLD = -0.05  # -1%

df["future_4w_return"] = df["Close"].shift(-4) / df["Close"] - 1
df["is_risk"] = (df["future_4w_return"] < RISK_THRESHOLD).astype(int)
df = df.dropna(subset=["future_4w_return"]).copy()

print("Risk-week base rate:", round(df["is_risk"].mean(), 4), "(class balance)")


# ==========================================================
# 4. FEATURE ENGINEERING
#    Pruned to the features that actually carried importance
#    last run, PLUS a new realized-volatility regime feature
#    (volatility clustering is usually the strongest predictor
#    of near-term risk, and was missing before).
# ==========================================================

df["close_vs_ema50"]  = df["Close"] / df["ema_50"] - 1
df["close_vs_ema100"] = df["Close"] / df["ema_100"] - 1
df["ema20_vs_ema50"]  = df["ema_20"] / df["ema_50"] - 1
df["ema50_vs_ema100"] = df["ema_50"] / df["ema_100"] - 1

# NEW: realized volatility regime (rolling std of weekly returns)
df["weekly_return"] = df["Close"].pct_change()
df["realized_vol_8w"]  = df["weekly_return"].rolling(8).std()
df["realized_vol_20w"] = df["weekly_return"].rolling(20).std()
df["vol_ratio"] = df["realized_vol_8w"] / df["realized_vol_20w"]  # short vs long vol regime

feature_cols = [
    "ema50_vs_ema100", "macd_signal", "ema20_vs_ema50",
    "close_vs_ema100", "obv_slope", "macd", "macd_hist",
    "close_vs_ema50", "bb_width", "obv_slope",
    "realized_vol_8w", "realized_vol_20w", "vol_ratio",
]
feature_cols = list(dict.fromkeys(feature_cols))  # dedupe, preserve order

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=feature_cols + ["is_risk"]).reset_index(drop=True)

print("Feature count:", len(feature_cols))
print("Usable rows:", len(df))


# ==========================================================
# 5. WALK-FORWARD EVALUATION (multiple folds, not one split)
#    Single 80/20 splits on ~800 rows are too noisy to trust.
#    This trains on an expanding window and tests on the next
#    chunk, repeated 5x, then aggregates results.
# ==========================================================

X = df[feature_cols]
y = df["is_risk"]

tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(df) * 0.1))

param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "min_samples_leaf": [5, 10, 15, 20],
    "max_features": ["sqrt", "log2", 0.6],
    "class_weight": ["balanced"],
}

fold_results = []
all_test_idx, all_preds, all_probs, all_actual = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    inner_cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=1),
        param_distributions=param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=inner_cv,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan

    fold_results.append({"fold": fold, "accuracy": acc, "auc": auc, "n_test": len(y_test)})
    all_test_idx.extend(test_idx)
    all_preds.extend(pred)
    all_probs.extend(proba)
    all_actual.extend(y_test)

    print(f"Fold {fold}: accuracy={acc:.3f}, AUC={auc:.3f}, n={len(y_test)}")

results_df = pd.DataFrame(fold_results)
print("\n=== WALK-FORWARD SUMMARY ===")
print(results_df)
print("Mean accuracy:", round(results_df["accuracy"].mean(), 4))
print("Mean AUC     :", round(results_df["auc"].mean(), 4))


# ==========================================================
# 6. STATISTICAL SIGNIFICANCE CHECK
#    Is accuracy actually beating the "always predict majority
#    class" baseline, and is it beyond what chance alone
#    could produce at this sample size?
# ==========================================================

all_actual = np.array(all_actual)
all_preds = np.array(all_preds)

overall_acc = accuracy_score(all_actual, all_preds)
majority_baseline = max(all_actual.mean(), 1 - all_actual.mean())

# Binomial test: is overall_acc significantly above the majority-class baseline?
n_correct = int((all_actual == all_preds).sum())
n_total = len(all_actual)
binom_result = stats.binomtest(n_correct, n_total, p=majority_baseline, alternative="greater")

print("\n=== SIGNIFICANCE CHECK ===")
print("Overall accuracy      :", round(overall_acc, 4))
print("Majority-class baseline:", round(majority_baseline, 4))
print("p-value (vs baseline)  :", round(binom_result.pvalue, 4))
if binom_result.pvalue < 0.05:
    print(">> Statistically significant edge over baseline (p < 0.05)")
else:
    print(">> NOT statistically distinguishable from the majority-class baseline.")
    print("   Treat any backtest 'improvement' from this model with caution.")

print("\nConfusion matrix (rows=actual, cols=predicted):")
print(confusion_matrix(all_actual, all_preds))
print("Precision (risk class):", round(precision_score(all_actual, all_preds, zero_division=0), 4))
print("Recall (risk class)   :", round(recall_score(all_actual, all_preds, zero_division=0), 4))


# ==========================================================
# 7. STRATEGY SIMULATION using probability, not hard class
#    Only sit out when the model is CONFIDENT (proba high),
#    rather than at the arbitrary 0.5 boundary. Threshold is
#    reported for a few options so you can compare.
# ==========================================================

sim_df = df.iloc[all_test_idx].copy().reset_index(drop=True)
sim_df["risk_proba"] = all_probs
sim_df["buy_hold_return"] = sim_df["future_4w_return"]

def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    return ((equity_curve - peak) / peak).min()

def sharpe_ratio(returns, freq=52):
    if returns.std() == 0:
        return 0
    return np.sqrt(freq) * returns.mean() / returns.std()

print("\n=== STRATEGY COMPARISON ACROSS CONFIDENCE THRESHOLDS ===")
bh_equity = (1 + sim_df["buy_hold_return"]).cumprod()
print("Buy & Hold  | Final:", round(bh_equity.iloc[-1], 3),
      "| MaxDD:", round(max_drawdown(bh_equity), 3),
      "| Sharpe:", round(sharpe_ratio(sim_df["buy_hold_return"]), 3))

for thresh in [0.5, 0.6, 0.7]:
    sim_df["strategy_return"] = np.where(
        sim_df["risk_proba"] >= thresh, 0, sim_df["future_4w_return"]
    )
    strat_equity = (1 + sim_df["strategy_return"]).cumprod()
    print(f"Threshold {thresh} | Final:", round(strat_equity.iloc[-1], 3),
          "| MaxDD:", round(max_drawdown(strat_equity), 3),
          "| Sharpe:", round(sharpe_ratio(sim_df["strategy_return"]), 3),
          "| Weeks filtered:", int((sim_df["risk_proba"] >= thresh).sum()), "/", len(sim_df))


# ==========================================================
# 8. SAVE RESULTS
# ==========================================================

output_path = os.path.join(BASE_PATH, f"{INDEX_NAME}_classifier_results.csv")
sim_df.to_csv(output_path, index=False)
print("\nResults saved to:", output_path)