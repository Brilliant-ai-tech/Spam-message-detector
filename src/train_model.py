from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA = Path("data/spam.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA).dropna()
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"],
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced")
}

results = []
fitted = {}

for name, classifier in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98
        )),
        ("classifier", classifier)
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, pos_label="spam"),
        "recall": recall_score(y_test, pred, pos_label="spam"),
        "f1": f1_score(y_test, pred, pos_label="spam")
    }
    results.append(row)
    fitted[name] = (pipe, pred)

    report = classification_report(y_test, pred, target_names=["ham", "spam"])
    (REPORT_DIR / f"{name.lower().replace(' ', '_')}_report.txt").write_text(report)

results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

best_name = results_df.iloc[0]["model"]
best_model, best_pred = fitted[best_name]
joblib.dump(best_model, MODEL_DIR / "spam_detector.pkl")

cm = confusion_matrix(y_test, best_pred, labels=["ham", "spam"])
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm)
ax.set_xticks([0, 1], ["Ham", "Spam"])
ax.set_yticks([0, 1], ["Ham", "Spam"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
plt.close(fig)

metrics = {
    "dataset_rows": int(len(df)),
    "ham_count": int((df.label == "ham").sum()),
    "spam_count": int((df.label == "spam").sum()),
    "test_size": 0.20,
    "random_state": 42,
    "best_model": best_name,
    "best_metrics": results_df.iloc[0].to_dict(),
    "models": results
}
(REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
print("Saved models/spam_detector.pkl")
