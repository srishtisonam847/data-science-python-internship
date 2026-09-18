"""
Week 4 Task: Supervised Learning Model Implementation
Dataset: UCI Wine Recognition Dataset
Task: Multiclass classification of wine cultivar from chemical measurements.

Run:
    pip install numpy pandas matplotlib scikit-learn
    python week4_wine_supervised_learning.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ------------------------------------------------------------
# 1. Load the public dataset
# ------------------------------------------------------------
wine = load_wine(as_frame=True)
df = wine.frame.copy()
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

feature_cols = list(df.columns[:13])
df["target_name"] = df["target"].map(dict(enumerate(wine.target_names)))

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["target_name"].value_counts())

print("\nMissing values:")
print(df[feature_cols].isna().sum())

# ------------------------------------------------------------
# 2. Basic transformation / feature engineering
# ------------------------------------------------------------
X = df[feature_cols].copy()
y = df["target"].copy()

# Four positively skewed features receive log1p transforms.
# The original columns are retained so the model can learn from both
# the original scale and the transformed representation.
X_eng = X.copy()

for c in ["magnesium", "malic_acid", "color_intensity", "proline"]:
    X_eng[c + "_log1p"] = np.log1p(X_eng[c])

# Interpretable feature engineering:
# ratio = flavonoids relative to total phenols
# index = combined phenolic signal
X_eng["phenol_flav_ratio"] = X_eng["flavanoids"] / (
    X_eng["total_phenols"] + 1e-6
)
X_eng["phenol_index"] = (
    X_eng["total_phenols"] + X_eng["flavanoids"]
)

# ------------------------------------------------------------
# 3. Train/test split
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_eng,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# ------------------------------------------------------------
# 4. Pipeline: scaling + Logistic Regression
# ------------------------------------------------------------
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# ------------------------------------------------------------
# 5. Test-set evaluation
# ------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
roc_auc = roc_auc_score(
    y_test, y_proba, multi_class="ovr", average="macro"
)

print("\nTest-set metrics")
print(f"Accuracy:           {accuracy:.4f}")
print(f"Macro Precision:    {precision:.4f}")
print(f"Macro Recall:       {recall:.4f}")
print(f"Macro F1:            {f1:.4f}")
print(f"Macro ROC-AUC (OvR): {roc_auc:.4f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------------------------------------
# 6. Five-fold stratified cross-validation
# ------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "roc_auc": "roc_auc_ovr"
}

cv_results = cross_validate(
    model,
    X_eng,
    y,
    cv=cv,
    scoring=scoring
)

print("\n5-fold cross-validation:")
for metric in scoring:
    scores = cv_results["test_" + metric]
    print(
        f"{metric:10s}: mean={scores.mean():.4f}, "
        f"std={scores.std():.4f}"
    )

# ------------------------------------------------------------
# 7. Aggregation examples
# ------------------------------------------------------------
class_summary = (
    df.groupby(["target", "target_name"])[feature_cols]
      .mean()
      .round(3)
)

print("\nMean feature values by class:")
print(class_summary)

df["alcohol_band"] = pd.cut(
    df["alcohol"],
    bins=[10.9, 12.0, 13.0, 14.0, 15.0],
    labels=["11–12", "12–13", "13–14", "14–15"],
    include_lowest=True
)

alcohol_summary = (
    df.groupby("alcohol_band", observed=False)
      .agg(
          samples=("target", "size"),
          mean_proline=("proline", "mean"),
          class_0_share=("target", lambda s: (s == 0).mean()),
          class_1_share=("target", lambda s: (s == 1).mean()),
          class_2_share=("target", lambda s: (s == 2).mean()),
      )
)

print("\nAlcohol-band aggregation:")
print(alcohol_summary.round(3))

# ------------------------------------------------------------
# 8. Fully annotated visualizations
# ------------------------------------------------------------

# Class distribution
counts = df["target_name"].value_counts().reindex(wine.target_names)
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(counts.index, counts.values)
ax.set_title("Wine Cultivar Class Distribution")
ax.set_xlabel("Cultivar class")
ax.set_ylabel("Number of samples")
ax.grid(axis="y", alpha=0.25)
for b, v in zip(bars, counts.values):
    ax.annotate(
        f"{v} samples ({v/len(df):.1%})",
        xy=(b.get_x() + b.get_width()/2, v),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center"
    )
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm)
ax.set_title("Confusion Matrix — Test Set")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_xticks(range(3), wine.target_names)
ax.set_yticks(range(3), wine.target_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 9. Model interpretation
# ------------------------------------------------------------
coef = model.named_steps["classifier"].coef_
importance = np.mean(np.abs(coef), axis=0)

importance_df = pd.DataFrame({
    "feature": X_eng.columns,
    "mean_abs_coefficient": importance
}).sort_values("mean_abs_coefficient", ascending=False)

print("\nTop 10 features by mean absolute coefficient:")
print(importance_df.head(10).to_string(index=False))
