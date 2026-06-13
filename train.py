"""
Task-07: Train a Spam Email Classifier (Naive Bayes or SVM)
Prodigy Infotech Internship
"""

import argparse
import pickle
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train(data_path, model_type, output_path):
    print(f"📂 Loading dataset: {data_path}")
    df = pd.read_csv(data_path)

    # Support both 'label'/'text' and 'v1'/'v2' column names (UCI format)
    if "v1" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    df = df[["label", "text"]].dropna()
    df["text"] = df["text"].apply(preprocess)
    df["label_num"] = (df["label"].str.lower() == "spam").astype(int)

    print(f"   Total samples : {len(df)}")
    print(f"   Spam          : {df['label_num'].sum()}")
    print(f"   Ham           : {(df['label_num'] == 0).sum()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
    )

    classifier = MultinomialNB() if model_type == "nb" else LinearSVC(max_iter=1000)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True)),
        ("clf",   classifier),
    ])

    print(f"🚀 Training {'Naive Bayes' if model_type == 'nb' else 'SVM (LinearSVC)'}...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\n✅ Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

    with open(output_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"💾 Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spam email classifier")
    parser.add_argument("--data",   type=str, default="data/sample_emails.csv", help="Path to CSV dataset")
    parser.add_argument("--model",  type=str, default="nb", choices=["nb", "svm"], help="Classifier: nb or svm")
    parser.add_argument("--output", type=str, default="spam_model.pkl",           help="Output model path")
    args = parser.parse_args()
    train(args.data, args.model, args.output)
