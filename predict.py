"""
Task-07: Predict whether an email is spam or ham
Prodigy Infotech Internship
"""

import argparse
import pickle
import re


def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(model_path, text):
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    cleaned = preprocess(text)
    pred    = pipeline.predict([cleaned])[0]
    label   = "🚨 SPAM" if pred == 1 else "✅ HAM (Not Spam)"

    # Probability (only for Naive Bayes pipeline)
    try:
        proba = pipeline.predict_proba([cleaned])[0]
        confidence = max(proba) * 100
        print(f"\n📧 Email Text  : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"   Prediction  : {label}")
        print(f"   Confidence  : {confidence:.1f}%")
        print(f"   Ham prob    : {proba[0]*100:.1f}%")
        print(f"   Spam prob   : {proba[1]*100:.1f}%\n")
    except AttributeError:
        print(f"\n📧 Email Text  : {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"   Prediction  : {label}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an email as spam or ham")
    parser.add_argument("--model", type=str, default="spam_model.pkl", help="Path to trained model")
    parser.add_argument("--text",  type=str, required=True,            help="Email text to classify")
    args = parser.parse_args()
    predict(args.model, args.text)
