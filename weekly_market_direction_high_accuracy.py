"""
High-accuracy weekly market direction model.

This script follows the result style of train_rf_weekly.py:
- Uses weekly technical indicators + GDELT sentiment
- Adds momentum, lag-label, streak, volatility, and sentiment features
- Trains Random Forest, Gradient Boosting, and Ensemble models
- Prints model comparison, high-confidence accuracy, top features, walk-forward validation
- Saves predictions and summary CSVs

Run from the project folder:
    python weekly_market_direction_high_accuracy.py --index BANKNIFTY --split-date 2024-01-01
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

FEATURE_FILES = {
    "NIFTY": "nifty_weekly_features.csv",
    "BANKNIFTY": "banknifty_weekly_features.csv",
    "SENSEX": "sensex_weekly_features.csv",
}

SENTIMENT_COLUMNS = ["mean_sentiment", "median_sentiment", "sentiment_std", "news_count"]


def parse_args():
    parser = argparse.ArgumentParser(description="High-accuracy weekly market direction model")
    parser.add_argument("--index", default="BANKNIFTY", choices=FEATURE_FILES.keys())
    parser.add_argument("--project-path", default=".")
    parser.add_argument("--split-date", default="2024-01-01")
    parser.add_argument("--high-confidence", type=float, default=0.60)
    parser.add_argument("--thresholds", default="0.55,0.60,0.65,0.70")
    args, _unknown = parser.parse_known_args()
    return args


def find_file(project_path, filename):
    project_path = Path(project_path)
    direct = project_path / filename
    if direct.exists():
        return direct

    matches = list(project_path.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Missing required file: {filename} under {project_path}")


def load_data(project_path, index_name):
    feature_path = find_file(project_path, FEATURE_FILES[index_name])
    sentiment_path = find_file(project_path, "gdelt_weekly_sentiment.pkl")

    df = pd.read_csv(feature_path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    sentiment = pd.read_pickle(sentiment_path)
    df["week"] = df.index.to_period("W-FRI").astype(str)
    df = pd.merge(df.reset_index(), sentiment, on="week", how="left").set_index("Date")
    df = df.sort_index()

    for col in SENTIMENT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def add_features(df):
    df = df.copy()

    df["future_return"] = df["Close"].pct_change().shift(-1)
    df["label"] = (df["future_return"] > 0).astype(int)

    df["price_range"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-9)
    df["close_position"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"] + 1e-9)
    df["gap"] = df["Open"] / (df["Close"].shift(1) + 1e-9) - 1

    df["close_vs_ema20"] = df["Close"] / (df["ema_20"] + 1e-9) - 1
    df["close_vs_ema50"] = df["Close"] / (df["ema_50"] + 1e-9) - 1
    df["close_vs_ema100"] = df["Close"] / (df["ema_100"] + 1e-9) - 1
    df["ema20_vs_ema50"] = df["ema_20"] / (df["ema_50"] + 1e-9) - 1
    df["ema50_vs_ema100"] = df["ema_50"] / (df["ema_100"] + 1e-9) - 1
    df["ema20_slope"] = df["ema_20"].pct_change(2)
    df["ema50_slope"] = df["ema_50"].pct_change(4)

    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
    df["rsi_mid"] = ((df["rsi_14"] >= 45) & (df["rsi_14"] <= 55)).astype(int)

    df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int)
    df["macd_cross_change"] = df["macd_cross"].diff()
    df["trend_strong"] = (df["adx"] > 25).astype(int)
    df["di_cross"] = (df["plus_di"] > df["minus_di"]).astype(int)

    df["return_1w"] = df["Close"].pct_change(1)
    df["return_2w"] = df["Close"].pct_change(2)
    df["return_4w"] = df["Close"].pct_change(4)
    df["return_8w"] = df["Close"].pct_change(8)
    df["return_12w"] = df["Close"].pct_change(12)
    df["mom_accel"] = df["return_1w"] - df["return_2w"]

    df["vol_4w"] = df["return_1w"].rolling(4).std()
    df["vol_12w"] = df["return_1w"].rolling(12).std()
    df["vol_20w"] = df["return_1w"].rolling(20).std()
    df["vol_ratio"] = df["vol_4w"] / (df["vol_20w"] + 1e-9)
    df["obv_momentum"] = df["obv"].pct_change(4)

    df["label_lag1"] = df["label"].shift(1)
    df["label_lag2"] = df["label"].shift(2)
    df["label_lag3"] = df["label"].shift(3)

    is_up = (df["Close"].diff() > 0).astype(int)
    groups = (is_up != is_up.shift()).cumsum()
    streak = is_up.groupby(groups).cumcount() + 1
    df["consec_up"] = np.where(is_up == 1, streak, 0)
    df["consec_down"] = np.where(is_up == 0, streak, 0)

    if "mean_sentiment" in df.columns:
        df["sent_x_adx"] = df["mean_sentiment"] * df["adx"]
        df["sent_x_rsi"] = df["mean_sentiment"] * df["rsi_14"]
        df["sent_x_bb"] = df["mean_sentiment"] * df["bb_width"]
        df["sent_weighted"] = df["mean_sentiment"] * np.log1p(df.get("news_count", 0))
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
        df["news_spike"] = (df.get("news_count", 0) > df.get("news_count", 0).rolling(8).mean() * 1.5).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["future_return", "label"])
    return df


def feature_columns(df):
    drop_cols = {
        "week",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "future_return",
        "label",
    }
    cols = [c for c in df.columns if c not in drop_cols]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def make_models():
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=8,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    gbc = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gbc", gbc)],
        voting="soft",
        weights=[0.5, 0.5],
    )

    return rf, gbc, ensemble


def print_model_result(name, model, X_test, y_test):
    pred = model.predict(X_test)
    print(f"\n{name}")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))


def main():
    args = parse_args()
    project_path = Path(args.project_path)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]

    df = load_data(project_path, args.index)
    df = add_features(df)
    cols = feature_columns(df)

    df = df.dropna(subset=cols + ["label"])
    X = df[cols].clip(-1e6, 1e6)
    y = df["label"].astype(int)

    train_mask = X.index < pd.Timestamp(args.split_date)
    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]

    print(f"Loaded {args.index}: {len(df)} rows | {len(cols)} features + label")
    print(f"Date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"Label balance: {y.value_counts().to_dict()}")
    print(f"\nTotal features: {X.shape[1]}")
    print(f"\nTrain: {len(X_train)} weeks ({X_train.index.min().date()} -> {X_train.index.max().date()})")
    print(f"Test : {len(X_test)} weeks  ({X_test.index.min().date()} -> {X_test.index.max().date()})")
    print(f"Train label balance: {y_train.value_counts().to_dict()}")
    print(f"Test  label balance: {y_test.value_counts().to_dict()}")

    rf, gbc, ensemble = make_models()
    rf.fit(X_train, y_train)
    gbc.fit(X_train, y_train)
    ensemble.fit(X_train, y_train)

    print(f"\nRF OOB Score: {rf.oob_score_:.4f}")
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print_model_result("Random Forest", rf, X_test, y_test)
    print_model_result("Gradient Boosting", gbc, X_test, y_test)
    print_model_result("Ensemble", ensemble, X_test, y_test)

    y_pred = ensemble.predict(X_test)
    proba_up = ensemble.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 60)
    print("DETAILED REPORT - ENSEMBLE MODEL")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Bearish", "Bullish"]))

    sample = pd.DataFrame({"prob_up": proba_up, "predicted": y_pred, "actual": y_test.values}, index=X_test.index)
    print("Sample probabilities:")
    print(sample.head(10))

    high_conf = sample[sample["prob_up"] >= args.high_confidence]
    print(f"\nHigh-confidence weeks (prob >= {args.high_confidence}): {len(high_conf)}")
    if len(high_conf) > 0:
        print(f"High-confidence accuracy: {accuracy_score(high_conf['actual'], high_conf['predicted']):.4f}")
        print(f"High-confidence label balance: {high_conf['actual'].value_counts().to_dict()}")

    threshold_rows = []
    for threshold in thresholds:
        rows = sample[sample["prob_up"] >= threshold]
        if len(rows) == 0:
            continue
        acc = accuracy_score(rows["actual"], rows["predicted"])
        threshold_rows.append({"threshold": threshold, "weeks": len(rows), "accuracy": acc})
        print(f"Threshold {threshold}: {len(rows)} weeks | accuracy {acc:.4f}")

    importance = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nTop 20 Features (RF):")
    print(importance.head(20).round(4))

    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION (5 folds)")
    print("=" * 60)

    wf_scores = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(X), start=1):
        fold_rf, _fold_gbc, _fold_ensemble = make_models()
        fold_rf.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = fold_rf.predict(X.iloc[test_idx])
        acc = accuracy_score(y.iloc[test_idx], fold_pred)
        wf_scores.append(acc)
        print(f"Fold {fold}: {len(train_idx)} train | {len(test_idx)} test | accuracy {acc:.4f}")

    print(f"\nMean accuracy : {np.mean(wf_scores):.4f}")
    print(f"Std           : {np.std(wf_scores):.4f}")

    output_prefix = args.index.lower()
    model_summary = pd.DataFrame([
        {"model": "Random Forest", "accuracy": accuracy_score(y_test, rf.predict(X_test))},
        {"model": "Gradient Boosting", "accuracy": accuracy_score(y_test, gbc.predict(X_test))},
        {"model": "Ensemble", "accuracy": accuracy_score(y_test, y_pred)},
    ])
    threshold_summary = pd.DataFrame(threshold_rows)
    predictions = sample.reset_index().rename(columns={"index": "Date"})

    model_summary.to_csv(project_path / f"{output_prefix}_high_accuracy_model_summary.csv", index=False)
    threshold_summary.to_csv(project_path / f"{output_prefix}_high_accuracy_thresholds.csv", index=False)
    predictions.to_csv(project_path / f"{output_prefix}_high_accuracy_predictions.csv", index=False)
    importance.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(
        project_path / f"{output_prefix}_high_accuracy_feature_importance.csv", index=False
    )

    print("\n" + "=" * 60)
    print(f"FINAL SUMMARY - {args.index}")
    print("=" * 60)
    print(f"Features used        : {X.shape[1]}")
    print(f"Train size           : {len(X_train)} weeks")
    print(f"Test size            : {len(X_test)} weeks")
    print(f"RF Accuracy          : {accuracy_score(y_test, rf.predict(X_test)):.4f}")
    print(f"GBC Accuracy         : {accuracy_score(y_test, gbc.predict(X_test)):.4f}")
    print(f"Ensemble Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Walk-forward mean    : {np.mean(wf_scores):.4f} +/- {np.std(wf_scores):.4f}")
    print(f"OOB Score (RF)       : {rf.oob_score_:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
