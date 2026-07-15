"""Train and evaluate the churn classifier.

Input: the feature table from src/processing/build_dataset.py.
Output: a fitted model in models/ plus churn scores for every customer
in data/processed/churn_scores.parquet (the dashboard reads those).

The split is 60/20/20 train/validation/test, stratified so every part
keeps the same 85/15 class ratio:
  train      -> fitting the models
  validation -> comparing models and picking one
  test       -> touched exactly once at the end

With 85% churned, accuracy is useless (always predicting churn scores
85%), so models get class_weight="balanced" and are compared on
ROC-AUC and PR-AUC.

Run:
    python -m src.ml.train
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = "data/processed/churn_dataset"
SCORES_PATH = "data/processed/churn_scores.parquet"
MODEL_DIR = Path("models")

CATEGORICAL = ["gender", "state", "traffic_source"]
TARGET = "churned"
NOT_FEATURES = ["user_id", TARGET]

RANDOM_STATE = 42  # fixed seed so runs are reproducible


def load_data() -> pd.DataFrame:
    return pd.read_parquet(DATASET_PATH)


def split(df: pd.DataFrame):
    """60/20/20 stratified split. test_size=0.5 of the 40% remainder."""
    train, rest = train_test_split(
        df, test_size=0.4, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest[TARGET], random_state=RANDOM_STATE
    )
    return train, val, test


def make_preprocessor(numeric_cols):
    # categoricals: one hot, unknown categories at predict time become
    # all zeros instead of crashing.
    # numerics: median imputation for the few nulls, scaling so that
    # logistic regression treats all features on the same footing
    # (tree models don't care about scale, it does them no harm).
    return ColumnTransformer([
        # sparse_output=False because HistGradientBoosting needs dense
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
    ])


def candidate_models():
    return {
        "always_churn_baseline": DummyClassifier(strategy="constant", constant=1),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


def evaluate(name, pipeline, X, y) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = pipeline.predict(X)
    return {
        "model": name,
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    df = load_data()
    train, val, test = split(df)
    feature_cols = [c for c in df.columns if c not in NOT_FEATURES]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL]

    X_train, y_train = train[feature_cols], train[TARGET]
    X_val, y_val = val[feature_cols], val[TARGET]
    X_test, y_test = test[feature_cols], test[TARGET]

    results = []
    fitted = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([
            ("prep", make_preprocessor(numeric_cols)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        res = evaluate(name, pipe, X_val, y_val)
        results.append(res)
        print(f"{name:>24}  roc_auc={res['roc_auc']}  pr_auc={res['pr_auc']}")

    # pick the winner on validation ROC-AUC (skip the dummy baseline)
    real = [r for r in results if r["model"] != "always_churn_baseline"]
    best_name = max(real, key=lambda r: r["roc_auc"])["model"]
    best = fitted[best_name]
    print(f"\nselected: {best_name}")

    # the one and only look at the test set
    test_result = evaluate(best_name, best, X_test, y_test)
    print(f"test roc_auc={test_result['roc_auc']}  pr_auc={test_result['pr_auc']}")
    print("test confusion matrix [tn fp / fn tp]:")
    for row in test_result["confusion_matrix"]:
        print(f"  {row}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_DIR / "churn_model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({"validation": results, "test": test_result}, f, indent=2)

    # score every customer for the dashboard
    scores = df[["user_id", "state", TARGET]].copy()
    scores["churn_probability"] = best.predict_proba(df[feature_cols])[:, 1]
    scores.to_parquet(SCORES_PATH, index=False)
    print(f"\nwrote {len(scores):,} customer scores to {SCORES_PATH}")


if __name__ == "__main__":
    main()
