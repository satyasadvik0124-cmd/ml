# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

# ============================================================
# CONFIG
# ============================================================
PROJECT_PATH   = "/content/drive/MyDrive/Colab Notebooks/market_ml_project"
INDEX_NAME     = "BANKNIFTY"     # NIFTY / BANKNIFTY / SENSEX
SPLIT_DATE     = "2021-01-01"
PROB_THRESHOLD = 0.60

DATA_FILES = {
    "NIFTY"    : "nifty_weekly_features.csv",
    "BANKNIFTY": "banknifty_weekly_features.csv",
    "SENSEX"   : "sensex_weekly_features.csv"
}

DATA_FILE = f"{PROJECT_PATH}/{DATA_FILES[INDEX_NAME]}"

# ============================================================
# LOAD WEEKLY TECHNICAL FEATURES
# ============================================================
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)
print(f"Loaded {INDEX_NAME} technical data: {df.shape}")

# ============================================================
# CREATE WEEK COLUMN (FOR SENTIMENT MERGE)
# ============================================================
df["week"] = df.index.to_period("W-FRI").astype(str)

# ============================================================
# LOAD GDELT WEEKLY SENTIMENT
# ============================================================
sentiment = pd.read_pickle(f"{PROJECT_PATH}/gdelt_weekly_sentiment.pkl")
print(f"Loaded GDELT sentiment: {sentiment.shape}")
print(f"Sentiment columns: {list(sentiment.columns)}")

# ============================================================
# MERGE TECHNICAL + SENTIMENT
# ============================================================
df = pd.merge(
    df.reset_index(),
    sentiment,
    on="week",
    how="left"
).set_index("Date")

print(f"After sentiment merge: {df.shape}")

# ============================================================
# HANDLE MISSING SENTIMENT (FILL WITH NEUTRAL = 0)
# ============================================================
sent_cols = ["mean_sentiment", "median_sentiment", "sentiment_std", "news_count"]
for col in sent_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ============================================================
# IMPROVED SENTIMENT FEATURES
# ============================================================

# --- Original interaction features ---
df["sent_x_adx"]    = df["mean_sentiment"] * df["adx"]
df["sent_x_rsi"]    = df["mean_sentiment"] * df["rsi_14"]
df["sent_x_bb"]     = df["mean_sentiment"] * df["bb_width"]
df["sent_weighted"] = df["mean_sentiment"] * np.log1p(df["news_count"])

# --- Sentiment lags (last 1, 2, 3 weeks) ---
df["sent_lag1"] = df["mean_sentiment"].shift(1)
df["sent_lag2"] = df["mean_sentiment"].shift(2)
df["sent_lag3"] = df["mean_sentiment"].shift(3)   # NEW: 3rd lag

# --- Sentiment momentum (is sentiment improving or worsening?) ---
df["sent_momentum"]  = df["mean_sentiment"] - df["mean_sentiment"].shift(2)   # change over 2 weeks
df["sent_accel"]     = df["sent_momentum"] - df["sent_momentum"].shift(1)     # acceleration

# --- Sentiment regime (strongly positive / neutral / strongly negative) ---
df["sent_positive"]  = (df["mean_sentiment"] > 0.1).astype(int)
df["sent_negative"]  = (df["mean_sentiment"] < -0.1).astype(int)
df["sent_extreme"]   = (df["mean_sentiment"].abs() > 0.3).astype(int)        # extreme sentiment

# --- Sentiment divergence (price up but sentiment down = warning) ---
df["price_up"]       = (df["Close"].pct_change() > 0).astype(int)
df["sent_divergence"]= ((df["price_up"] == 1) & (df["mean_sentiment"] < 0)).astype(int)

# --- Rolling sentiment (smoothed) ---
df["sent_4w_avg"]    = df["mean_sentiment"].rolling(4).mean()
df["sent_8w_avg"]    = df["mean_sentiment"].rolling(8).mean()
df["sent_vs_avg"]    = df["mean_sentiment"] - df["sent_4w_avg"]              # current vs recent avg

# --- News volume features ---
df["news_spike"]     = (df["news_count"] > df["news_count"].rolling(8).mean() * 1.5).astype(int)
df["news_drought"]   = (df["news_count"] < df["news_count"].rolling(8).mean() * 0.5).astype(int)

# ============================================================
# TECHNICAL FEATURE ENGINEERING (same as optimized RF weekly)
# ============================================================

# --- Price action ---
df["price_range"]     = (df["High"] - df["Low"]) / df["Close"]
df["close_position"]  = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)
df["gap"]             = df["Open"] / df["Close"].shift(1) - 1

# --- EMA crossovers ---
df["ema_20_50_cross"]  = (df["ema_20"] > df["ema_50"]).astype(int)
df["ema_50_100_cross"] = (df["ema_50"] > df["ema_100"]).astype(int)
df["ema_20_slope"]     = df["ema_20"].pct_change(2)
df["ema_50_slope"]     = df["ema_50"].pct_change(4)

# --- RSI zones ---
df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)

# --- MACD cross ---
df["macd_cross"]      = (df["macd"] > df["macd_signal"]).astype(int)
df["macd_cross_chg"]  = df["macd_cross"].diff()

# --- ADX trend ---
df["trend_strong"]    = (df["adx"] > 25).astype(int)
df["di_cross"]        = (df["plus_di"] > df["minus_di"]).astype(int)

# --- Momentum returns ---
df["return_1w"]  = df["Close"].pct_change(1)
df["return_2w"]  = df["Close"].pct_change(2)
df["return_4w"]  = df["Close"].pct_change(4)
df["return_8w"]  = df["Close"].pct_change(8)
df["return_12w"] = df["Close"].pct_change(12)
df["mom_accel"]  = df["return_1w"] - df["return_2w"]

# --- Volatility ---
df["vol_20w"]    = df["return_1w"].rolling(20).std()
df["vol_ratio"]  = df["return_1w"].rolling(4).std() / (df["vol_20w"] + 1e-9)

# --- OBV momentum ---
df["obv_momentum"] = df["obv"].pct_change(4)

# ============================================================
# CREATE LABEL (LEAKAGE SAFE — next week's return)
# ============================================================
df = df.sort_index()
df["future_return"] = df["Close"].pct_change().shift(-1)
df["label"]         = (df["future_return"] > 0).astype(int)
df.dropna(inplace=True)

print(f"\nFinal dataset: {df.shape[0]} rows | Label balance: {df['label'].value_counts().to_dict()}")

# ============================================================
# FEATURES & LABEL
# ============================================================
DROP_COLS = ["label", "future_return", "week", "Open", "High", "Low", "Close", "Volume"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X = df[feature_cols]
y = df["label"]

# Numerical safety
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype("float64")
X = X.clip(lower=-1e6, upper=1e6)

print(f"Total features: {X.shape[1]}")

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
X_train = X[X.index < SPLIT_DATE]
X_test  = X[X.index >= SPLIT_DATE]
y_train = y[y.index < SPLIT_DATE]
y_test  = y[y.index >= SPLIT_DATE]

print(f"\nTrain: {X_train.shape[0]} weeks ({X_train.index[0].date()} → {X_train.index[-1].date()})")
print(f"Test : {X_test.shape[0]} weeks  ({X_test.index[0].date()} → {X_test.index[-1].date()})")

# ============================================================
# MODEL 1 — Optimized Random Forest
# ============================================================
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features="sqrt",
    class_weight="balanced",
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print(f"\nRF OOB Score: {rf.oob_score_:.4f}")

# ============================================================
# MODEL 2 — Gradient Boosting
# ============================================================
gbc = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42
)
gbc.fit(X_train, y_train)

# ============================================================
# MODEL 3 — Ensemble
# ============================================================
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gbc", gbc)],
    voting="soft",
    weights=[0.5, 0.5]
)
ensemble.fit(X_train, y_train)

# ============================================================
# EVALUATE ALL 3
# ============================================================
print("\n" + "="*60)
print("  MODEL COMPARISON")
print("="*60)

for name, model in [("Random Forest", rf), ("Gradient Boosting", gbc), ("Ensemble", ensemble)]:
    y_pred = model.predict(X_test)
    print(f"\n{name}:")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ============================================================
# DETAILED REPORT — ENSEMBLE
# ============================================================
print("\n" + "="*60)
print("  DETAILED REPORT — ENSEMBLE")
print("="*60)

y_pred  = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Bearish", "Bullish"]))

# ============================================================
# HIGH CONFIDENCE FILTER
# ============================================================
df_results = X_test.copy()
df_results["actual"]    = y_test.values
df_results["prob_up"]   = y_proba
df_results["predicted"] = y_pred

print("\nSample predictions:")
print(df_results[["prob_up", "predicted", "actual"]].head(10))

print("\nConfidence threshold analysis:")
for thresh in [0.55, 0.60, 0.65, 0.70]:
    hc = df_results[df_results["prob_up"] >= thresh]
    if len(hc) > 0:
        acc = accuracy_score(hc["actual"], hc["predicted"])
        print(f"  Threshold {thresh}: {len(hc):3d} weeks | accuracy {acc:.4f}")

# ============================================================
# SENTIMENT IMPACT ANALYSIS
# ============================================================
print("\n" + "="*60)
print("  SENTIMENT IMPACT ANALYSIS")
print("="*60)

# Compare accuracy: weeks with strong sentiment vs neutral
test_df_full = df[df.index >= SPLIT_DATE].copy()
test_df_full["predicted"] = y_pred
test_df_full["actual"]    = y_test.values

pos_weeks  = test_df_full[test_df_full["mean_sentiment"] > 0.1]
neg_weeks  = test_df_full[test_df_full["mean_sentiment"] < -0.1]
neut_weeks = test_df_full[test_df_full["mean_sentiment"].abs() <= 0.1]

for label, subset in [("Positive sentiment weeks", pos_weeks),
                       ("Negative sentiment weeks", neg_weeks),
                       ("Neutral sentiment weeks",  neut_weeks)]:
    if len(subset) > 0:
        acc = accuracy_score(subset["actual"], subset["predicted"])
        print(f"  {label}: {len(subset):3d} weeks | accuracy {acc:.4f}")

# ============================================================
# FEATURE IMPORTANCE — Top 25
# ============================================================
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 25 Features (RF):")
print(importance.head(25).round(4))

# Sentiment features specifically
sent_features = [f for f in importance.index if "sent" in f or "news" in f]
print(f"\nSentiment feature importances:")
print(importance[sent_features].round(4))

# ============================================================
# WALK-FORWARD VALIDATION (within training period only)
# ============================================================
print("\n" + "="*60)
print("  WALK-FORWARD VALIDATION (training period only)")
print("="*60)

X_wf = X[X.index < SPLIT_DATE]
y_wf = y[y.index < SPLIT_DATE]

tscv   = TimeSeriesSplit(n_splits=5)
wf_acc = []

for fold, (tr_idx, te_idx) in enumerate(tscv.split(X_wf)):
    Xtr, Xte = X_wf.iloc[tr_idx], X_wf.iloc[te_idx]
    ytr, yte = y_wf.iloc[tr_idx], y_wf.iloc[te_idx]

    fold_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    fold_rf.fit(Xtr, ytr)
    acc = accuracy_score(yte, fold_rf.predict(Xte))
    wf_acc.append(acc)
    print(f"  Fold {fold+1}: train={len(Xtr)} | test={len(Xte)} | accuracy {acc:.4f}")

print(f"\n  Mean  : {np.mean(wf_acc):.4f}")
print(f"  Std   : {np.std(wf_acc):.4f}")
print(f"  Range : {min(wf_acc):.4f} – {max(wf_acc):.4f}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print(f"  FINAL SUMMARY — {INDEX_NAME} with GDELT Sentiment")
print("="*60)
print(f"  Total features          : {X.shape[1]}")
print(f"  Sentiment features      : {len(sent_features)}")
print(f"  Technical features      : {X.shape[1] - len(sent_features)}")
print(f"  Train weeks             : {X_train.shape[0]}")
print(f"  Test weeks              : {X_test.shape[0]}")
print(f"  RF Accuracy             : {accuracy_score(y_test, rf.predict(X_test)):.4f}")
print(f"  GBC Accuracy            : {accuracy_score(y_test, gbc.predict(X_test)):.4f}")
print(f"  Ensemble Accuracy       : {accuracy_score(y_test, ensemble.predict(X_test)):.4f}")
print(f"  RF OOB Score            : {rf.oob_score_:.4f}")
print(f"  Walk-forward mean       : {np.mean(wf_acc):.4f} ± {np.std(wf_acc):.4f}")
print("="*60)