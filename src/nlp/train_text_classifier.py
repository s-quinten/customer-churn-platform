"""Train a complaint topic classifier on the narrative text.

Predicts which product a complaint is about from the text alone, the
product column gives the labels for free. TF-IDF vectorizer plus
logistic regression: the usual combo for sparse text features, fast to
train and the per word coefficients stay interpretable.

Run (locally is fine, the sample fits in memory):
    python -m src.nlp.train_text_classifier
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

SAMPLE_PATH = "data/processed/complaints_sample"
MODEL_DIR = Path("models")

RANDOM_STATE = 42


def load_sample() -> pd.DataFrame:
    df = pd.read_parquet(SAMPLE_PATH)
    # collapse product renames CFPB did over the years, otherwise the
    # same thing appears as two labels
    df["product"] = df["product"].replace({
        "Credit reporting, credit repair services, or other personal consumer reports":
            "Credit reporting or other personal consumer reports",
        "Payday loan, title loan, or personal loan":
            "Payday loan, title loan, personal loan, or advance loan",
        "Money transfer, virtual currency, or money service":
            "Money transfers",
    })
    return df


def main() -> None:
    df = load_sample()
    print(f"training sample: {len(df):,} narratives, {df['product'].nunique()} products")

    train, test = train_test_split(
        df, test_size=0.2, stratify=df["product"], random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),   # single words plus two-word phrases
            min_df=5,             # ignore words seen in fewer than 5 docs
            stop_words="english",
        )),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(train["narrative"], train["product"])

    pred = pipe.predict(test["narrative"])
    macro_f1 = f1_score(test["product"], pred, average="macro")
    print(f"macro F1 on held out narratives: {macro_f1:.3f}")
    print(classification_report(test["product"], pred, zero_division=0))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "complaint_topic_model.joblib")
    report = classification_report(
        test["product"], pred, output_dict=True, zero_division=0
    )
    with open(MODEL_DIR / "nlp_metrics.json", "w") as f:
        json.dump({"macro_f1": macro_f1, "report": report}, f, indent=2)
    print("model saved to models/complaint_topic_model.joblib")


if __name__ == "__main__":
    main()
