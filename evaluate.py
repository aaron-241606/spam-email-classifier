"""
Task-07: Evaluate spam classifier with full metrics
Prodigy Infotech Internship
"""

import argparse
import pickle
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate(model_path, data_path):
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    df = pd.read_csv(data_path)
    if "v1" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    df = df[["label", "text"]].dropna()
    df["text"]      = df["text"].apply(preprocess)
    df["label_num"] = (df["label"].str.lower() == "spam").astype(int)

    _, X_test, _, y_test = train_test_split(
        df["text"], df["label_num"], test_size=0.2, random_state=42, stratify=df["label_num"]
    )

    preds = pipeline.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    cm   = confusion_matrix(y_test, preds)

    print("\n" + "="*45)
    print("       SPAM CLASSIFIER EVALUATION")
    print("="*45)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print("="*45)
    print("\nConfusion Matrix:")
    print(f"               Predicted Ham   Predicted Spam")
    print(f"  Actual Ham   {cm[0][0]:>12}   {cm[0][1]:>14}")
    print(f"  Actual Spam  {cm[1][0]:>12}   {cm[1][1]:>14}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate spam classifier")
    parser.add_argument("--model", type=str, default="spam_model.pkl",          help="Path to trained model")
    parser.add_argument("--data",  type=str, default="data/sample_emails.csv",  help="Path to CSV dataset")
    args = parser.parse_args()
    evaluate(args.model, args.data)
