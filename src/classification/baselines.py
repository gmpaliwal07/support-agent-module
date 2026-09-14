from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from sklearn.dummy import DummyClassifier

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"

TRAIN_PATH = DATA_DIR / "training_set.csv"
GOLDEN_PATH = EVAL_DIR / "golden_set.csv"

def load_data():
    train = pd.read_csv(TRAIN_PATH)
    golden = pd.read_csv(GOLDEN_PATH)
    return train, golden

def run_majority_baseline(train, golden):
    print("\n-- Baseline 1: Majority Class --")
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(train[["clean_text"]], train["intent"])
    preds = clf.predict(golden[["clean_text"]])
 
    acc = accuracy_score(golden["intent"], preds)
    macro_f1 = f1_score(golden["intent"], preds, average="macro", zero_division=0)
    print(f"accuracy: {acc:.4f}")
    print(f"macro F1: {macro_f1:.4f}")
    print(classification_report(golden["intent"], preds, zero_division=0))
    return {"name": "majority_class", "accuracy": acc, "macro_f1": macro_f1}

def run_tfidf_lr_baseline(train, golden):
    print("\n-- Baseline 2: TF-IDF + Logistic Regression --")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    X_train = vectorizer.fit_transform(train["clean_text"].astype(str))
    X_golden = vectorizer.transform(golden["clean_text"].astype(str))
 
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # accounts for the intent imbalance we found
    )
    clf.fit(X_train, train["intent"])
    preds = clf.predict(X_golden)
 
    acc = accuracy_score(golden["intent"], preds)
    macro_f1 = f1_score(golden["intent"], preds, average="macro", zero_division=0)
    print(f"accuracy: {acc:.4f}")
    print(f"macro F1: {macro_f1:.4f}")
    print(classification_report(golden["intent"], preds, zero_division=0))
    return {"name": "tfidf_logreg", "accuracy": acc, "macro_f1": macro_f1}
 

def main():
    train, golden = load_data()
    print(f"training pool: {len(train)} rows")
    print(f"golden eval set: {len(golden)} rows")
 
    results = []
    results.append(run_majority_baseline(train, golden))
    results.append(run_tfidf_lr_baseline(train, golden))
 
    print("\n Summary ")
    for r in results:
        print(f"{r['name']:<20} accuracy={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}")
 
if __name__ == "__main__":
    main()