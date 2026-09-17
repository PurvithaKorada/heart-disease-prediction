"""
Heart Disease Prediction Using Machine Learning
=================================================
Internship capstone project.

Dataset: Heart Disease UCI (combined Cleveland, Hungary, VA Long Beach,
Switzerland sources) — 920 rows, 16 columns.

This script:
1. Loads and inspects the raw dataset
2. Cleans it (missing values, duplicates, type fixes)
3. Performs EDA and saves plots
4. Trains Logistic Regression and Decision Tree classifiers
5. Evaluates both models on a held-out test set
6. Saves the better model + a comparison table

Run with:  python src/heart_disease_prediction.py
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save-to-file only, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
sns.set_style("whitegrid")

# Path setup (works whether run from repo root or src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_disease.csv")
VIZ_DIR = os.path.join(BASE_DIR, "visualizations")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(VIZ_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Loaded dataset with shape: {df.shape}")


# ---------------------------------------------------------------------------
# 2. INSPECT DATA
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: INSPECTING DATA")
print("=" * 70)

print(f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn names: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

missing = df.isnull().sum()
print(f"\nMissing values per column:\n{missing[missing > 0]}")

dup_count = df.drop(columns=["id"]).duplicated().sum()
print(f"\nDuplicate rows (excluding id column): {dup_count}")

print(f"\nTarget column: 'num' (0 = no disease, 1-4 = increasing severity)")
print(f"Raw target class distribution:\n{df['num'].value_counts().sort_index()}")


# ---------------------------------------------------------------------------
# 3. CLEAN DATA
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 3: CLEANING DATA")
print("=" * 70)

df_clean = df.copy()

# Drop the 'id' column - it's just a row identifier, not a predictive feature
df_clean = df_clean.drop(columns=["id"])

# Drop exact duplicate rows (ignoring id, which we've already removed)
before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates()
print(f"Dropped {before - df_clean.shape[0]} duplicate row(s). New shape: {df_clean.shape}")

# Convert TRUE/FALSE string columns to proper boolean/int
for col in ["fbs", "exang"]:
    df_clean[col] = df_clean[col].map({"TRUE": 1, "FALSE": 0, True: 1, False: 0})

# Binarize the target: 0 = no heart disease, 1 = heart disease present
# (standard convention for this dataset - collapses severity levels 1-4 into "present")
df_clean["target"] = (df_clean["num"] > 0).astype(int)
df_clean = df_clean.drop(columns=["num"])

print(f"\nBinarized target class distribution:\n{df_clean['target'].value_counts()}")
print(f"Class balance: {df_clean['target'].value_counts(normalize=True).round(3).to_dict()}")

# ---- Handle missing values ----
numeric_cols = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
categorical_cols = ["sex", "dataset", "cp", "fbs", "restecg", "exang", "slope", "thal"]

print("\nMissing values BEFORE imputation:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Numeric columns: impute with median (robust to outliers)
for col in numeric_cols:
    if df_clean[col].isnull().sum() > 0:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

# Categorical columns: impute with mode (most frequent value)
for col in categorical_cols:
    if df_clean[col].isnull().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)

print("\nMissing values AFTER imputation:", df_clean.isnull().sum().sum())


# ---------------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# --- Target distribution ---
plt.figure(figsize=(6, 5))
ax = sns.countplot(x="target", data=df_clean, hue="target", palette=["#4C72B0", "#DD8452"], legend=False)
plt.title("Target Class Distribution\n(0 = No Disease, 1 = Disease Present)")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "01_target_distribution.png"), dpi=150)
plt.close()
print("Saved: 01_target_distribution.png")

# --- Key numeric feature distributions ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(data=df_clean, x=col, hue="target", kde=True, ax=ax,
                 palette=["#4C72B0", "#DD8452"], element="step")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "02_feature_distributions.png"), dpi=150)
plt.close()
print("Saved: 02_feature_distributions.png")

# --- Correlation heatmap (numeric features + target) ---
corr_df = df_clean[numeric_cols + ["target"]].copy()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "03_correlation_heatmap.png"), dpi=150)
plt.close()
print("Saved: 03_correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 5. FEATURE / TARGET SPLIT + ENCODING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: PREPARING FEATURES (ENCODING + TRAIN/TEST SPLIT)")
print("=" * 70)

X = df_clean.drop(columns=["target"])
y = df_clean["target"]

# Label-encode categorical (object/string) columns so models can use them
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"Features used ({X.shape[1]}): {list(X.columns)}")

# ---------------------------------------------------------------------------
# 6. TRAIN/TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")


# ---------------------------------------------------------------------------
# 7. FEATURE SCALING
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 7: FEATURE SCALING")
print("=" * 70)

# Scaling matters for Logistic Regression (distance/gradient based).
# Decision Trees don't need it, but using the same scaled data for both
# keeps the pipeline simple and doesn't hurt tree performance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Applied StandardScaler (fit on training data only, to avoid data leakage).")


# ---------------------------------------------------------------------------
# 8. TRAIN MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 8: TRAINING MODELS")
print("=" * 70)

log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
print("Logistic Regression trained.")

dt_clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)
dt_clf.fit(X_train_scaled, y_train)
print("Decision Tree Classifier trained (max_depth=5 to reduce overfitting).")


# ---------------------------------------------------------------------------
# 9 & 10. PREDICTIONS + EVALUATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 9: EVALUATING MODELS")
print("=" * 70)

results = {}

def evaluate_model(name, model, X_test_data):
    y_pred = model.predict(X_test_data)
    y_proba = model.predict_proba(X_test_data)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auc": auc, "confusion_matrix": cm.tolist(),
        "y_pred": y_pred, "y_proba": y_proba
    }
    return y_pred, y_proba, cm

lr_pred, lr_proba, lr_cm = evaluate_model("Logistic Regression", log_reg, X_test_scaled)
dt_pred, dt_proba, dt_cm = evaluate_model("Decision Tree", dt_clf, X_test_scaled)


# ---------------------------------------------------------------------------
# VISUALIZATIONS: Confusion matrices
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
axes[0].set_title("Confusion Matrix - Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Greens", ax=axes[1],
            xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
axes[1].set_title("Confusion Matrix - Decision Tree")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "04_confusion_matrices.png"), dpi=150)
plt.close()
print("\nSaved: 04_confusion_matrices.png")


# ---------------------------------------------------------------------------
# VISUALIZATIONS: ROC curves
# ---------------------------------------------------------------------------
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {results['Logistic Regression']['auc']:.3f})", linewidth=2)
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {results['Decision Tree']['auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Model Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "05_roc_curves.png"), dpi=150)
plt.close()
print("Saved: 05_roc_curves.png")


# ---------------------------------------------------------------------------
# STEP 13: MODEL COMPARISON TABLE
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 13: MODEL COMPARISON")
print("=" * 70)

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree"],
    "Accuracy": [results["Logistic Regression"]["accuracy"], results["Decision Tree"]["accuracy"]],
    "Precision": [results["Logistic Regression"]["precision"], results["Decision Tree"]["precision"]],
    "Recall": [results["Logistic Regression"]["recall"], results["Decision Tree"]["recall"]],
    "F1-Score": [results["Logistic Regression"]["f1"], results["Decision Tree"]["f1"]],
    "ROC AUC": [results["Logistic Regression"]["auc"], results["Decision Tree"]["auc"]],
}).round(4)

print(comparison_df.to_string(index=False))
comparison_df.to_csv(os.path.join(BASE_DIR, "models", "model_comparison.csv"), index=False)

# Comparison bar chart
plt.figure(figsize=(9, 6))
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
x = np.arange(len(metrics_to_plot))
width = 0.35
lr_vals = comparison_df.loc[comparison_df["Model"] == "Logistic Regression", metrics_to_plot].values.flatten()
dt_vals = comparison_df.loc[comparison_df["Model"] == "Decision Tree", metrics_to_plot].values.flatten()
plt.bar(x - width/2, lr_vals, width, label="Logistic Regression", color="#4C72B0")
plt.bar(x + width/2, dt_vals, width, label="Decision Tree", color="#55A868")
plt.xticks(x, metrics_to_plot)
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()
for i, v in enumerate(lr_vals):
    plt.text(i - width/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
for i, v in enumerate(dt_vals):
    plt.text(i + width/2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, "06_model_comparison.png"), dpi=150)
plt.close()
print("\nSaved: 06_model_comparison.png")


# ---------------------------------------------------------------------------
# STEP 14: SELECT AND SAVE THE BETTER MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 14: MODEL SELECTION")
print("=" * 70)

# Select based on ROC AUC (a robust metric for binary classification that
# accounts for the class imbalance in this dataset), with F1 as tiebreaker.
lr_auc = results["Logistic Regression"]["auc"]
dt_auc = results["Decision Tree"]["auc"]

if lr_auc > dt_auc:
    best_name, best_model = "Logistic Regression", log_reg
elif dt_auc > lr_auc:
    best_name, best_model = "Decision Tree", dt_clf
else:
    # tie on AUC -> use F1 as tiebreaker
    if results["Logistic Regression"]["f1"] >= results["Decision Tree"]["f1"]:
        best_name, best_model = "Logistic Regression", log_reg
    else:
        best_name, best_model = "Decision Tree", dt_clf

print(f"Selected model: {best_name}")
print(f"Reason: highest ROC AUC on the held-out test set "
      f"(Logistic Regression = {lr_auc:.4f}, Decision Tree = {dt_auc:.4f}).")

model_bundle = {
    "model": best_model,
    "scaler": scaler,
    "label_encoders": label_encoders,
    "feature_columns": list(X.columns),
    "model_name": best_name,
}
model_path = os.path.join(MODEL_DIR, "heart_disease_model.joblib")
joblib.dump(model_bundle, model_path)
print(f"Saved model bundle to: {model_path}")


# ---------------------------------------------------------------------------
# STEP 15: SAMPLE PREDICTION
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 15: SAMPLE PREDICTION DEMONSTRATION")
print("=" * 70)

sample_idx = X_test.index[0]
sample_X = X_test.loc[[sample_idx]]
sample_X_scaled = scaler.transform(sample_X)
sample_true = y_test.loc[sample_idx]
sample_pred = best_model.predict(sample_X_scaled)[0]
sample_proba = best_model.predict_proba(sample_X_scaled)[0][1]

print(f"Sample test row (index {sample_idx}):")
print(sample_X.to_string(index=False))
print(f"\nActual label:    {'Disease' if sample_true == 1 else 'No Disease'} ({sample_true})")
print(f"Predicted label: {'Disease' if sample_pred == 1 else 'No Disease'} ({sample_pred})")
print(f"Predicted probability of disease: {sample_proba:.4f}")

# Save a small results summary as JSON for reference
summary = {
    "dataset_shape_raw": list(df.shape),
    "dataset_shape_after_cleaning": list(df_clean.shape),
    "duplicates_removed": int(before - df_clean.drop(columns=[]).shape[0]) if False else int(dup_count),
    "class_distribution": df_clean["target"].value_counts().to_dict(),
    "comparison_table": comparison_df.to_dict(orient="records"),
    "best_model": best_name,
    "sample_prediction": {
        "index": int(sample_idx),
        "actual": int(sample_true),
        "predicted": int(sample_pred),
        "predicted_probability": float(sample_proba),
    }
}
with open(os.path.join(MODEL_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
