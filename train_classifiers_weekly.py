import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit

# Optional libraries. Install if unavailable: pip install xgboost lightgbm
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# ============================================================
# CONFIG
# ============================================================
PROJECT_PATH = "/content/drive/MyDrive/Colab Notebooks/market_ml_project"
INDEX_NAME = "BANKNIFTY"       # NIFTY / BANKNIFTY / SENSEX
SPLIT_DATE = "2024-01-01"

FEATURE_FILES = {
    "NIFTY": "nifty_weekly_features.csv",
    "BANKNIFTY": "banknifty_weekly_features.csv",
    "SENSEX": "sensex_weekly_features.csv"
}

DATA_FILE = f"{PROJECT_PATH}/{FEATURE_FILES[INDEX_NAME]}"

# ============================================================
# LOAD EXISTING TECHNICAL DATA
# ============================================================
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

print(f"Loaded {INDEX_NAME}: {df.shape}")
print(f"Date range: {df.index[0].date()} -> {df.index[-1].date()}")

# ============================================================
# MERGE EXISTING GDELT WEEKLY SENTIMENT
# ============================================================
df["week"] = df.index.to_period("W-FRI").astype(str)

sentiment = pd.read_pickle(f"{PROJECT_PATH}/gdelt_weekly_sentiment.pkl")

df = pd.merge(
    df.reset_index(), sentiment, on="week", how="left"
).set_index("Date")
df.sort_index(inplace=True)

sent_cols = ["mean_sentiment", "median_sentiment", "sentiment_std", "news_count"]
for col in sent_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ============================================================
# FEATURE ENGINEERING
# NOTE: NO CURRENT/FUTURE LABEL IS USED AS A FEATURE.
# ============================================================
df["price_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)
df["gap"] = df["Open"] / (df["Close"].shift(1) + 1e-9) - 1

df["ema_20_slope"] = df["ema_20"].pct_change(2)
df["ema_50_slope"] = df["ema_50"].pct_change(4)
df["ema_20_50_cross"] = (df["ema_20"] > df["ema_50"]).astype(int)
df["ema_50_100_cross"] = (df["ema_50"] > df["ema_100"]).astype(int)

df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
df["rsi_mid"] = ((df["rsi_14"] >= 45) & (df["rsi_14"] <= 55)).astype(int)

df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)
df["macd_cross_change"] = df["macd_cross"].diff()
df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).quantile(0.2)).astype(int)
df["trend_strong"] = (df["adx"] > 25).astype(int)
df["di_cross"] = (df["plus_di"] > df["minus_di"]).astype(int)

df["return_1w"] = df["Close"].pct_change(1)
df["return_2w"] = df["Close"].pct_change(2)
df["return_4w"] = df["Close"].pct_change(4)
df["return_8w"] = df["Close"].pct_change(8)
df["return_12w"] = df["Close"].pct_change(12)
df["mom_accel"] = df["return_1w"] - df["return_2w"]

df["obv_momentum"] = df["obv"].pct_change(4)
df["vol_20w"] = df["return_1w"].rolling(20).std()
df["vol_ratio"] = df["return_1w"].rolling(4).std() / (df["vol_20w"] + 1e-9)

# Sentiment features
df["sent_x_adx"] = df["mean_sentiment"] * df["adx"]
df["sent_x_rsi"] = df["mean_sentiment"] * df["rsi_14"]
df["sent_x_bb"] = df["mean_sentiment"] * df["bb_width"]
df["sent_weighted"] = df["mean_sentiment"] * np.log1p(df["news_count"])
df["sent_lag1"] = df["mean_sentiment"].shift(1)
df["sent_lag2"] = df["mean_sentiment"].shift(2)
df["sent_lag3"] = df["mean_sentiment"].shift(3)
df["sent_momentum"] = df["mean_sentiment"] - df["mean_sentiment"].shift(2)
df["sent_accel"] = df["sent_momentum"] - df["sent_momentum"].shift(1)
df["sent_positive"] = (df["mean_sentiment"] > 0.1).astype(int)
df["sent_negative"] = (df["mean_sentiment"] < -0.1).astype(int)
df["sent_extreme"] = (df["mean_sentiment"].abs() > 0.3).astype(int)
df["sent_4w_avg"] = df["mean_sentiment"].rolling(4).mean()
df["sent_8w_avg"] = df["mean_sentiment"].rolling(8).mean()
df["sent_vs_avg"] = df["mean_sentiment"] - df["sent_4w_avg"]
df["news_spike"] = (df["news_count"] > df["news_count"].rolling(8).mean() * 1.5).astype(int)
df["news_drought"] = (df["news_count"] < df["news_count"].rolling(8).mean() * 0.5).astype(int)

# ============================================================
# LEAKAGE-SAFE NEXT-WEEK TARGET
# ============================================================
df["future_return"] = df["Close"].pct_change().shift(-1)
df["label"] = (df["future_return"] > 0).astype(int)
df.dropna(inplace=True)

DROP_COLS = ["label", "future_return", "week", "Open", "High", "Low", "Close", "Volume"]
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].replace([np.inf, -np.inf], np.nan).clip(-1e6, 1e6)
y = df["label"]

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples : {len(X)}")
print(f"Labels  : {y.value_counts().to_dict()}")

# ============================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================
X_train = X[X.index < SPLIT_DATE]
X_test = X[X.index >= SPLIT_DATE]
y_train = y[y.index < SPLIT_DATE]
y_test = y[y.index >= SPLIT_DATE]

print(f"\nTrain: {len(X_train)}")
print(f"Test : {len(X_test)}")

# ============================================================
# CLASSIFIERS
# Scaling is applied only to models that need it.
# ============================================================
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=42))
    ]),
    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42))
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=15, weights="distance"))
    ]),
    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42))
    ]),
    "Extra Trees": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(n_estimators=600, max_depth=10, min_samples_leaf=5, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=800, max_depth=8, min_samples_leaf=5, min_samples_split=10, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1))
    ]),
    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=42))
    ]),
    "MLP": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.001, learning_rate_init=0.001, max_iter=1000, early_stopping=True, random_state=42))
    ])
}

if XGBClassifier is not None:
    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss", random_state=42, n_jobs=-1))
    ])
else:
    print("XGBoost not installed - skipping XGBoost.")

if LGBMClassifier is not None:
    models["LightGBM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LGBMClassifier(n_estimators=500, num_leaves=31, learning_rate=0.03, max_depth=-1, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, verbosity=-1, random_state=42, n_jobs=-1))
    ])
else:
    print("LightGBM not installed - skipping LightGBM.")

# ============================================================
# TRAIN + EVALUATE
# ============================================================
results = []
predictions = {}
probabilities = {}

print("\n" + "=" * 80)
print("CLASSIFICATION MODEL COMPARISON")
print("=" * 80)

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, proba) if proba is not None else np.nan

    results.append({"Model": name, "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1, "ROC_AUC": auc})
    predictions[name] = pred
    probabilities[name] = proba

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n" + "=" * 80)
print("FINAL RANKING")
print("=" * 80)
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# SAVE RESULTS
# ============================================================
results_df.to_csv(f"{PROJECT_PATH}/{INDEX_NAME.lower()}_all_classifier_results.csv", index=False)

pred_df = pd.DataFrame({"Date": X_test.index, "actual": y_test.values})
for name, pred in predictions.items():
    safe = name.lower().replace(" ", "_")
    pred_df[f"{safe}_pred"] = pred
    if probabilities[name] is not None:
        pred_df[f"{safe}_prob_up"] = probabilities[name]
pred_df.to_csv(f"{PROJECT_PATH}/{INDEX_NAME.lower()}_all_classifier_predictions.csv", index=False)

# ============================================================
# WALK-FORWARD VALIDATION OF TOP 3 MODELS
# ============================================================
top_models = results_df.head(3)["Model"].tolist()
print("\n" + "=" * 80)
print("WALK-FORWARD VALIDATION - TOP 3")
print("=" * 80)

tscv = TimeSeriesSplit(n_splits=5)
for name in top_models:
    fold_scores = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        model = models[name]
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        score = accuracy_score(y.iloc[te_idx], model.predict(X.iloc[te_idx]))
        fold_scores.append(score)
        print(f"{name:20s} Fold {fold}: {score:.4f}")
    print(f"{name:20s} Mean: {np.mean(fold_scores):.4f} | Std: {np.std(fold_scores):.4f}\n")

print("Done.")
