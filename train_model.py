"""Train & evaluate phishing models.
Usage: python train_model.py data.csv
CSV needs columns: text, label  (label: 1/phishing/spam = phishing, 0/ham/legitimate = safe).
Tip: merge an English dataset with a Vietnamese one so the model supports both languages.
"""
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data.csv").dropna(subset=["text", "label"])
pos = {"1", "phishing", "spam", "true", "phishing email"}
df["y"] = df["label"].astype(str).str.lower().isin(pos).astype(int)

X_tr, X_te, y_tr, y_te = train_test_split(df["text"], df["y"], test_size=0.2, stratify=df["y"], random_state=42)
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True, lowercase=True)
A_tr, A_te = vec.fit_transform(X_tr), vec.transform(X_te)

candidates = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "NaiveBayes": MultinomialNB(),
    "LinearSVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced")),
}
best, best_score = None, -1
for name, clf in candidates.items():
    clf.fit(A_tr, y_tr)
    pred = clf.predict(A_te)
    print(f"\n=== {name} ===")
    print(classification_report(y_te, pred, target_names=["legitimate", "phishing"]))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    score = 0.5 * recall_score(y_te, pred) + 0.5 * f1_score(y_te, pred)  # favour recall
    if score > best_score:
        best, best_score, best_name = clf, score, name

print(f"\nBest model: {best_name}")
joblib.dump(best, "phishing_model.pkl")
joblib.dump(vec, "tfidf_vectorizer.pkl")
print("Saved phishing_model.pkl and tfidf_vectorizer.pkl")
