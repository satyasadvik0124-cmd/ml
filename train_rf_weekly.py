import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_PATH  = "/content/drive/MyDrive/Colab Notebooks/market_ml_project"
INDEX_NAME    = "BANKNIFTY"   # NIFTY / BANKNIFTY / SENSEX
SPLIT_DATE    = "2024-01-01"
PROB_THRESHOLD = 0.60

FILES = {
    "NIFTY"    : "nifty_weekly_labeled.csv",
    "BANKNIFTY": "banknifty_weekly_labeled.csv",
    "SENSEX"   : "sensex_weekly_labeled.csv"
}

DATA_FILE = f"{PROJECT_PATH}/{FILES[INDEX_NAME]}"

# =============================================================================
# LOAD
# =============================================================================
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print(f"Loaded {INDEX_NAME}: {df.shape[0]} rows | {df.shape[1]-1} features + label")
print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
print(f"Label balance: {df['label'].value_counts().to_dict()}")

# =============================================================================
# FEATURE ENGINEERING — add more signals on top of existing ones
# =============================================================================

# --- Price-derived features ---
df["price_range"]     = (df["High"] - df["Low"]) / df["Close"]         # weekly volatility
df["close_position"]  = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)  # where close sits in range
df["gap"]             = df["Open"] / df["Close"].shift(1) - 1           # gap from prev close

# --- EMA slopes (rate of change of EMA) ---
df["ema_20_slope"]    = df["ema_20"].pct_change(2)
df["ema_50_slope"]    = df["ema_50"].pct_change(4)

# --- EMA crossover signals ---
df["ema_20_50_cross"] = (df["ema_20"] > df["ema_50"]).astype(int)
df["ema_50_100_cross"]= (df["ema_50"] > df["ema_100"]).astype(int)

# --- RSI zones ---
df["rsi_overbought"]  = (df["rsi_14"] > 70).astype(int)
df["rsi_oversold"]    = (df["rsi_14"] < 30).astype(int)
df["rsi_mid"]         = ((df["rsi_14"] >= 45) & (df["rsi_14"] <= 55)).astype(int)

# --- MACD cross ---
df["macd_cross"]      = (df["macd"] > df["macd_signal"]).astype(int)
df["macd_cross_change"]= df["macd_cross"].diff()                        # 1=just crossed up, -1=just crossed down

# --- BB squeeze (low width = squeeze before breakout) ---
df["bb_squeeze"]      = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.2)).astype(int)

# --- ADX trend strength ---
df["trend_strong"]    = (df["adx"] > 25).astype(int)
df["di_cross"]        = (df["plus_di"] > df["minus_di"]).astype(int)   # bullish DI cross

# --- Momentum: multi-period returns ---
df["return_1w"]       = df["Close"].pct_change(1)
df["return_2w"]       = df["Close"].pct_change(2)
df["return_4w"]       = df["Close"].pct_change(4)
df["return_8w"]       = df["Close"].pct_change(8)
df["return_12w"]      = df["Close"].pct_change(12)

# --- Momentum acceleration ---
df["mom_accel"]       = df["return_1w"] - df["return_2w"]

# --- Lagged labels (recent market direction memory) ---
df["label_lag1"]      = df["label"].shift(1)
df["label_lag2"]      = df["label"].shift(2)
df["label_lag3"]      = df["label"].shift(3)

# --- Consecutive up/down weeks ---
df["consec_up"]       = df["label"].groupby(
    (df["label"] != df["label"].shift()).cumsum()
).cumcount()
df["consec_up"]       = df["consec_up"] * df["label"]                  # only count when label=1

# --- OBV momentum ---
df["obv_momentum"]    = df["obv"].pct_change(4)

# --- Volatility regime ---
df["vol_20w"]         = df["return_1w"].rolling(20).std()
df["vol_ratio"]       = df["return_1w"].rolling(4).std() / (df["vol_20w"] + 1e-9)  # short vs long vol

# --- RSI divergence proxy ---
df["price_up_rsi_down"] = (
    (df["Close"] > df["Close"].shift(1)) &
    (df["rsi_14"] < df["rsi_14"].shift(1))
).astype(int)  # bearish divergence signal

# Drop NaN rows from feature engineering
df.dropna(inplace=True)

# =============================================================================
# FEATURE SELECTION
# =============================================================================
DROP_COLS = ["label", "Open", "High", "Low", "Close", "Volume"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]

X = df[feature_cols]
y = df["label"]

# Numerical safety
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)
X = X.astype("float64")
X = X.clip(lower=-1e6, upper=1e6)

print(f"\nTotal features: {X.shape[1]}")

# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
X_train = X[X.index < SPLIT_DATE]
X_test  = X[X.index >= SPLIT_DATE]
y_train = y[y.index < SPLIT_DATE]
y_test  = y[y.index >= SPLIT_DATE]

print(f"\nTrain: {X_train.shape[0]} weeks ({X_train.index[0].date()} → {X_train.index[-1].date()})")
print(f"Test : {X_test.shape[0]} weeks  ({X_test.index[0].date()} → {X_test.index[-1].date()})")
print(f"Train label balance: {y_train.value_counts().to_dict()}")
print(f"Test  label balance: {y_test.value_counts().to_dict()}")

# =============================================================================
# MODEL 1 — Optimized Random Forest
# =============================================================================
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=10,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    oob_score=True,
)
rf.fit(X_train, y_train)
print(f"\nRF OOB Score: {rf.oob_score_:.4f}")

# =============================================================================
# MODEL 2 — Gradient Boosting Classifier
# =============================================================================
gbc = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=10,
    random_state=42,
)
gbc.fit(X_train, y_train)

# =============================================================================
# MODEL 3 — Ensemble (Soft Voting)
# =============================================================================
ensemble = VotingClassifier(
    estimators=[
        ("rf",  rf),
        ("gbc", gbc),
    ],
    voting="soft",
    weights=[0.5, 0.5],
)
ensemble.fit(X_train, y_train)

# =============================================================================
# EVALUATE ALL 3 MODELS
# =============================================================================
print("\n" + "="*60)
print("  MODEL COMPARISON")
print("="*60)

for name, model in [("Random Forest", rf), ("Gradient Boosting", gbc), ("Ensemble", ensemble)]:
    y_pred  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# =============================================================================
# BEST MODEL — detailed report (Ensemble)
# =============================================================================
print("\n" + "="*60)
print("  DETAILED REPORT — ENSEMBLE MODEL")
print("="*60)

y_pred  = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Bearish", "Bullish"]))

# =============================================================================
# HIGH CONFIDENCE FILTER
# =============================================================================
df_results = X_test.copy()
df_results["actual"]   = y_test.values
df_results["prob_up"]  = y_proba
df_results["predicted"]= y_pred

print(f"\nSample probabilities:")
print(df_results[["prob_up", "predicted", "actual"]].head(10))

hc = df_results[df_results["prob_up"] >= PROB_THRESHOLD]
if len(hc) > 0:
    hc_acc = accuracy_score(hc["actual"], hc["predicted"])
    print(f"\nHigh-confidence weeks (prob >= {PROB_THRESHOLD}): {len(hc)}")
    print(f"High-confidence accuracy: {hc_acc:.4f}")
    print(f"High-confidence label balance: {hc['actual'].value_counts().to_dict()}")
else:
    print(f"\nNo high-confidence weeks at threshold {PROB_THRESHOLD}.")

# Lower threshold check
for thresh in [0.55, 0.60, 0.65, 0.70]:
    hc_t = df_results[df_results["prob_up"] >= thresh]
    if len(hc_t) > 0:
        acc_t = accuracy_score(hc_t["actual"], hc_t["predicted"])
        print(f"  Threshold {thresh}: {len(hc_t)} weeks | accuracy {acc_t:.4f}")

# =============================================================================
# FEATURE IMPORTANCE — Top 20
# =============================================================================
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 20 Features (RF):")
print(importance.head(20).round(4))

# =============================================================================
# WALK-FORWARD VALIDATION (time series cross-validation)
# =============================================================================
print("\n" + "="*60)
print("  WALK-FORWARD VALIDATION (5 folds)")
print("="*60)

tscv   = TimeSeriesSplit(n_splits=5)
wf_acc = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

    fold_rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    fold_rf.fit(Xtr, ytr)
    fold_acc = accuracy_score(yte, fold_rf.predict(Xte))
    wf_acc.append(fold_acc)
    print(f"  Fold {fold+1}: {len(Xtr)} train | {len(Xte)} test | accuracy {fold_acc:.4f}")

print(f"\n  Mean accuracy : {np.mean(wf_acc):.4f}")
print(f"  Std           : {np.std(wf_acc):.4f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print(f"  FINAL SUMMARY — {INDEX_NAME}")
print("="*60)
print(f"  Features used        : {X.shape[1]}")
print(f"  Train size           : {X_train.shape[0]} weeks")
print(f"  Test size            : {X_test.shape[0]} weeks")
print(f"  RF Accuracy          : {accuracy_score(y_test, rf.predict(X_test)):.4f}")
print(f"  GBC Accuracy         : {accuracy_score(y_test, gbc.predict(X_test)):.4f}")
print(f"  Ensemble Accuracy    : {accuracy_score(y_test, ensemble.predict(X_test)):.4f}")
print(f"  Walk-forward mean    : {np.mean(wf_acc):.4f} ± {np.std(wf_acc):.4f}")
print(f"  OOB Score (RF)       : {rf.oob_score_:.4f}")
print("="*60)