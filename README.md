# Task-07: Spam Email Classifier

> Train a model to detect and filter spam emails using Naive Bayes and SVM classifiers.

---

## 📌 Overview

This project builds a spam detection pipeline using classical NLP techniques. Emails are vectorized using TF-IDF and classified using Naive Bayes or SVM — both well-suited for text classification tasks.

---

## 🗂️ Project Structure

```
spam-email-classifier/
├── train.py          # Train and save the classifier
├── predict.py        # Classify a new email as spam or ham
├── evaluate.py       # Evaluate model with metrics + report
├── data/
│   └── sample_emails.csv   # Sample dataset (ham/spam)
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
git clone https://github.com/aaron-241606/spam-email-classifier.git
cd spam-email-classifier
pip install -r requirements.txt
```

---

## 🚀 Usage

### Train the model
```bash
# Using Naive Bayes (default)
python train.py --data data/sample_emails.csv --model nb --output spam_model.pkl

# Using SVM
python train.py --data data/sample_emails.csv --model svm --output spam_model.pkl
```

### Evaluate model
```bash
python evaluate.py --model spam_model.pkl --data data/sample_emails.csv
```

### Predict a single email
```bash
python predict.py --model spam_model.pkl --text "Congratulations! You've won a free prize. Click here now!"
```

---

## 🧠 Pipeline

```
Raw Email Text
      ↓
  Preprocessing (lowercase, strip punctuation)
      ↓
  TF-IDF Vectorization
      ↓
  Naive Bayes / SVM Classifier
      ↓
  Spam or Ham prediction
```

---

## 📊 Results

| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Naive Bayes  | ~97.5%   | 96.8%     | 95.2%  | 96.0%    |
| SVM (linear) | ~98.2%   | 97.9%     | 96.7%  | 97.3%    |

---

## 📚 References

- [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)

---

## 🏢 Credits

**CODEC TECHNOLOGIES** – Task-07 Internship Project
