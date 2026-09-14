"""
Weekly market direction + bearish loss pipeline.

Goal:
1. Predict weekly market direction using technical indicators + GDELT sentiment.
2. Improve reported accuracy using confidence thresholds.
3. If the model predicts bearish, estimate expected percentage loss with regression.
4. Save clean comparison files for review.

Run this from the project folder that contains:
- nifty_weekly_features.csv / banknifty_weekly_features.csv / sensex_weekly_features.csv
- gdelt_weekly_sentiment.pkl

Example:
    python weekly_market_direction_pipeline.py --index BANKNIFTY --split-date 2024-01-01
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # optional dependency
    XGBClassifier = None
    XGBRegressor = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # optional dependency
    LGBMClassifier = None
    LGBMRegressor = None


FEATURE_FILES = {
    "NIFTY": "nifty_weekly_features.csv",
    "BANKNIFTY": "banknifty_weekly_features.csv",
    "SENSEX": "sensex_weekly_features.csv",
}

SENTIMENT_COLUMNS = ["mean_sentiment", "median_sentiment", "sentiment_std", "news_count"]


def parse_args():
    parser = argparse.ArgumentParser(description="Weekly direction + bearish loss model")
    parser.add_argument("--index", default="BANKNIFTY", choices=FEATURE_FILES.keys())
    parser.add_argument("--project-path", default=".", help="Folder containing weekly feature CSVs and GDELT pickle")
    parser.add_argument("--split-date", default="2024-01-01")
    parser.add_argument("--flat-threshold", type=float, default=0.005, help="Ignore weeks between +/- this return")
    parser.add_argument("--bearish-loss-threshold", type=float, default=-0.005, help="Regression trains on returns below this")
    parser.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80")
    return parser.parse_args()


def load_dataset(project_path, index_name):
    project_path = Path(project_path)
    feature_path = project_path / FEATURE_FILES[index_name]
    sentiment_path = project_path / "gdelt_weekly_sentiment.pkl"

    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature file: {feature_path}")
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Missing sentiment file: {sentiment_path}")

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


def add_features_and_targets(df, flat_threshold):
    df = df.copy()

    df["future_return_1w"] = df["Close"].pct_change().shift(-1)
    df["direction_label"] = np.where(df["future_return_1w"] > 0, 1, 0)
    df["strong_direction"] = np.where(
        df["future_return_1w"] > flat_threshold,
        1,
        np.where(df["future_return_1w"] < -flat_threshold, 0, np.nan),
    )

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
    df["momentum_accel"] = df["return_1w"] - df["return_2w"]
    df["vol_4w"] = df["return_1w"].rolling(4).std()
    df["vol_12w"] = df["return_1w"].rolling(12).std()
    df["vol_20w"] = df["return_1w"].rolling(20).std()
    df["vol_ratio"] = df["vol_4w"] / (df["vol_20w"] + 1e-9)
    df["obv_momentum"] = df["obv"].pct_change(4)

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

    return df.replace([np.inf, -np.inf], np.nan)


def build_feature_matrix(df):
    drop_cols = {
        "week",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "future_return_1w",
        "direction_label",
        "strong_direction",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    X = X.select_dtypes(include=[np.number]).clip(-1e6, 1e6)
    return X, list(X.columns)


def classifier_models():
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)),
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.5, kernel="rbf", gamma="scale", probability=True, class_weight="balanced", random_state=42)),
        ]),
        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=800, max_depth=10, min_samples_leaf=4, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=1000, max_depth=8, min_samples_leaf=5, min_samples_split=10, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.04, subsample=0.8, min_samples_leaf=8, random_state=42)),
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=1000, early_stopping=True, random_state=42)),
        ]),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss", random_state=42, n_jobs=-1)),
        ])

    if LGBMClassifier is not None:
        models["LightGBM"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)),
        ])

    return models


def evaluate_classifiers(X_train, y_train, X_test, y_test):
    rows = []
    fitted = {}

    for name, model in classifier_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision_bullish": precision_score(y_test, pred, pos_label=1, zero_division=0),
            "recall_bullish": recall_score(y_test, pred, pos_label=1, zero_division=0),
            "precision_bearish": precision_score(y_test, pred, pos_label=0, zero_division=0),
            "recall_bearish": recall_score(y_test, pred, pos_label=0, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() > 1 else np.nan,
        })
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values(["accuracy", "roc_auc"], ascending=False)
    return results, fitted


def threshold_scoreboard(model, X_test, y_test, thresholds):
    proba_up = model.predict_proba(X_test)[:, 1]
    rows = []

    for threshold in thresholds:
        confident = (proba_up >= threshold) | (proba_up <= 1 - threshold)
        if confident.sum() == 0:
            continue

        pred = (proba_up[confident] >= threshold).astype(int)
        actual = y_test[confident]
        rows.append({
            "threshold": threshold,
            "predictions_made": int(confident.sum()),
            "coverage": confident.mean(),
            "accuracy": accuracy_score(actual, pred),
            "precision_bullish": precision_score(actual, pred, pos_label=1, zero_division=0),
            "precision_bearish": precision_score(actual, pred, pos_label=0, zero_division=0),
            "recall_bearish": recall_score(actual, pred, pos_label=0, zero_division=0),
        })

    return pd.DataFrame(rows).sort_values(["accuracy", "coverage"], ascending=False)


def regression_models():
    models = {
        "Ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "Random Forest Regressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=800, max_depth=8, min_samples_leaf=4, random_state=42, n_jobs=-1)),
        ]),
    }

    if XGBRegressor is not None:
        models["XGBoost Regressor"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=42, n_jobs=-1)),
        ])

    if LGBMRegressor is not None:
        models["LightGBM Regressor"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=31, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1)),
        ])

    return models


def evaluate_bearish_regression(X_train, future_train, X_test, future_test, bearish_loss_threshold):
    train_mask = future_train < bearish_loss_threshold
    test_mask = future_test < bearish_loss_threshold

    rows = []
    predictions = pd.DataFrame(index=X_test.index)
    predictions["actual_return"] = future_test
    predictions["actual_loss_pct"] = np.where(future_test < 0, -future_test * 100, 0)
    predictions["is_actual_bearish_loss"] = test_mask

    if train_mask.sum() < 20 or test_mask.sum() < 5:
        return pd.DataFrame([{
            "model": "not_enough_bearish_samples",
            "train_bearish_rows": int(train_mask.sum()),
            "test_bearish_rows": int(test_mask.sum()),
        }]), predictions

    y_train_loss = (-future_train[train_mask] * 100).clip(lower=0)
    y_test_loss = (-future_test[test_mask] * 100).clip(lower=0)

    for name, model in regression_models().items():
        model.fit(X_train[train_mask], y_train_loss)
        pred_loss = pd.Series(model.predict(X_test), index=X_test.index).clip(lower=0)
        predictions[f"{name.lower().replace(' ', '_')}_predicted_loss_pct"] = pred_loss

        rows.append({
            "model": name,
            "train_bearish_rows": int(train_mask.sum()),
            "test_bearish_rows": int(test_mask.sum()),
            "mae_loss_pct_on_actual_bearish": mean_absolute_error(y_test_loss, pred_loss[test_mask]),
            "r2_on_actual_bearish": r2_score(y_test_loss, pred_loss[test_mask]) if test_mask.sum() > 1 else np.nan,
        })

    return pd.DataFrame(rows).sort_values("mae_loss_pct_on_actual_bearish"), predictions


def walk_forward_best_model(model_name, X, y):
    scores = []
    for fold, (train_idx, test_idx) in enumerate(TimeSeriesSplit(n_splits=5).split(X), start=1):
        model = classifier_models()[model_name]
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        scores.append({
            "fold": fold,
            "accuracy": accuracy_score(y.iloc[test_idx], pred),
            "test_rows": len(test_idx),
        })
    return pd.DataFrame(scores)


def main():
    args = parse_args()
    project_path = Path(args.project_path)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]

    raw_df = load_dataset(project_path, args.index)
    df = add_features_and_targets(raw_df, args.flat_threshold)
    X, feature_cols = build_feature_matrix(df)

    modeling_df = df.loc[X.index].dropna(subset=["strong_direction", "future_return_1w"])
    X = X.loc[modeling_df.index]
    y = modeling_df["strong_direction"].astype(int)
    future_returns = modeling_df["future_return_1w"]

    usable = X.notna().any(axis=1)
    X = X.loc[usable]
    y = y.loc[usable]
    future_returns = future_returns.loc[usable]

    train_mask = X.index < pd.Timestamp(args.split_date)
    X_train, X_test = X.loc[train_mask], X.loc[~train_mask]
    y_train, y_test = y.loc[train_mask], y.loc[~train_mask]
    future_train, future_test = future_returns.loc[train_mask], future_returns.loc[~train_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Train/test split produced empty data. Adjust --split-date.")

    comparison, fitted = evaluate_classifiers(X_train, y_train, X_test, y_test)
    best_model_name = comparison.iloc[0]["model"]
    best_model = fitted[best_model_name]

    threshold_results = threshold_scoreboard(best_model, X_test, y_test, thresholds)
    regression_results, regression_predictions = evaluate_bearish_regression(
        X_train, future_train, X_test, future_test, args.bearish_loss_threshold
    )
    wf_results = walk_forward_best_model(best_model_name, X, y)

    proba_up = pd.Series(best_model.predict_proba(X_test)[:, 1], index=X_test.index)
    predictions = pd.DataFrame({
        "Date": X_test.index,
        "actual_direction": y_test.values,
        "actual_return_1w": future_test.values,
        "best_model": best_model_name,
        "prob_bullish": proba_up.values,
        "prob_bearish": 1 - proba_up.values,
        "predicted_direction": (proba_up >= 0.5).astype(int).values,
    })
    predictions["predicted_label"] = np.where(predictions["predicted_direction"] == 1, "bullish", "bearish")

    output_prefix = args.index.lower()
    comparison.to_csv(project_path / f"{output_prefix}_weekly_model_comparison.csv", index=False)
    threshold_results.to_csv(project_path / f"{output_prefix}_weekly_threshold_results.csv", index=False)
    predictions.to_csv(project_path / f"{output_prefix}_weekly_predictions.csv", index=False)
    regression_results.to_csv(project_path / f"{output_prefix}_bearish_loss_regression_results.csv", index=False)
    regression_predictions.to_csv(project_path / f"{output_prefix}_bearish_loss_predictions.csv")
    wf_results.to_csv(project_path / f"{output_prefix}_walk_forward_results.csv", index=False)

    print("\n=== BEST CLASSIFIER ===")
    print(comparison.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== CONFIDENCE THRESHOLD RESULTS ===")
    print(threshold_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== BEARISH LOSS REGRESSION ===")
    print(regression_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== WALK-FORWARD RESULTS FOR BEST MODEL ===")
    print(wf_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSaved files:")
    for name in [
        f"{output_prefix}_weekly_model_comparison.csv",
        f"{output_prefix}_weekly_threshold_results.csv",
        f"{output_prefix}_weekly_predictions.csv",
        f"{output_prefix}_bearish_loss_regression_results.csv",
        f"{output_prefix}_bearish_loss_predictions.csv",
        f"{output_prefix}_walk_forward_results.csv",
    ]:
        print("-", project_path / name)

    print("\nClassification report for best model:")
    print(classification_report(y_test, best_model.predict(X_test), target_names=["bearish", "bullish"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, best_model.predict(X_test)))


if __name__ == "__main__":
    main()
