# Heart Disease Prediction Using Machine Learning

Internship capstone project: predicting the presence of heart disease from
patient clinical data using classical machine learning classifiers.

## Project Objective

Build and evaluate a machine learning pipeline that predicts whether a
patient has heart disease based on clinical measurements (age, cholesterol,
blood pressure, chest pain type, ECG results, etc.), comparing two
classification approaches — Logistic Regression and a Decision Tree — and
selecting the better-performing model.

## Dataset

**Heart Disease UCI dataset** (combined multi-center version) — 920 patient
records across four sources: Cleveland, Hungary, VA Long Beach, and
Switzerland, with 15 clinical feature columns plus an `id` column and a
`num` target column.

Key facts found during inspection of the actual file:

| Property | Value |
|---|---|
| Rows (raw) | 920 |
| Columns (raw) | 16 |
| Duplicate rows (excluding `id`) | 2 |
| Target column | `num` (0 = no disease, 1–4 = increasing severity) |
| Class distribution (raw) | 0: 411, 1: 265, 2: 109, 3: 107, 4: 28 |

**Missing values** (notably high for several columns):

| Column | Missing |
|---|---|
| `ca` | 611 (66%) |
| `thal` | 486 (53%) |
| `slope` | 309 (34%) |
| `oldpeak` | 62 |
| `trestbps` | 59 |
| `thalch` | 55 |
| `exang` | 55 |
| `fbs` | 90 |
| `chol` | 30 |
| `restecg` | 2 |

### Preprocessing decisions (documented, not silent)

- **Target binarization**: `num` (0–4) was converted to a binary `target`
  column — 0 = no disease, 1 = disease present (`num > 0`). This is the
  standard convention for this dataset. After cleaning, the binary
  distribution is **410 "no disease" vs 508 "disease"** (~55/45 split).
- **Duplicates**: the 2 duplicate rows (excluding `id`) were dropped.
- **Missing values**: rather than dropping rows (which would have removed
  the majority of the dataset given `ca`/`thal` missingness), missing
  values were imputed — **median** for numeric columns, **mode** for
  categorical columns. This is a reasonable default but is a real
  limitation worth noting: `ca` and `thal` in particular have over half
  their values imputed, which likely reduces their true predictive signal.
- `fbs` and `exang` (stored as `TRUE`/`FALSE` strings) were converted to 0/1.
- The `id` column was dropped (not predictive).

## Technologies Used

- Python 3
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — modeling and evaluation
- joblib — model persistence

## ML Workflow

1. Load dataset
2. Inspect (shape, dtypes, missing values, duplicates, target distribution)
3. Clean (drop id/duplicates, fix types, binarize target)
4. Handle missing values (median/mode imputation)
5. Exploratory data analysis (distributions, correlation heatmap)
6. Separate features (X) and target (y); label-encode categoricals
7. Train/test split (80/20, stratified, `random_state=42`)
8. Feature scaling (`StandardScaler`, fit on training data only)
9. Train Logistic Regression and Decision Tree classifiers
10. Generate predictions on the test set
11. Evaluate: accuracy, precision, recall, F1, confusion matrix, ROC/AUC
12. Compare models in a results table
13. Select and save the better model
14. Demonstrate a sample prediction

## Models Used

- **Logistic Regression** (`max_iter=1000`, `random_state=42`)
- **Decision Tree Classifier** (`max_depth=5` to limit overfitting, `random_state=42`)

## Evaluation Metrics (actual results, test set of 184 rows)

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8152 | 0.8148 | 0.8627 | 0.8381 | **0.9073** |
| Decision Tree | 0.8152 | 0.8542 | 0.8039 | 0.8283 | 0.8684 |

**Confusion matrices** (rows = actual, columns = predicted; order: No Disease, Disease):

- Logistic Regression:
  ```
  [[62 20]
   [14 88]]
  ```
- Decision Tree:
  ```
  [[68 14]
   [20 82]]
  ```

### Model selected: **Logistic Regression**

Both models reached identical accuracy (0.8152), so accuracy alone doesn't
separate them. Logistic Regression has a clearly higher **ROC AUC (0.9073
vs 0.8684)**, meaning it ranks disease-positive patients above
disease-negative patients more reliably across all classification
thresholds — a more robust measure than a single-threshold metric like
accuracy. It also has higher recall (0.8627 vs 0.8039), which matters for
a health-screening use case where missing an actual disease case (a false
negative) is costlier than a false alarm. The Decision Tree has slightly
better precision, but the gap in AUC and recall favors Logistic Regression
here.

### Sample prediction (from the notebook/script output)

For one held-out test patient: **actual label = Disease**, **model
predicted = No Disease**, with a predicted probability of disease of
**0.343**. This is shown deliberately as an actual (not cherry-picked)
example — it illustrates that the model, like any real classifier at
~82% accuracy, does make mistakes near the decision boundary.

## Visualizations (in `visualizations/`)

1. `01_target_distribution.png` — class balance (No Disease vs Disease)
2. `02_feature_distributions.png` — distributions of key numeric features, split by class
3. `03_correlation_heatmap.png` — correlation among numeric features
4. `04_confusion_matrices.png` — confusion matrices for both models
5. `05_roc_curves.png` — ROC curves with AUC for both models
6. `06_model_comparison.png` — bar chart comparing all metrics

## Project Structure

```
heart-disease-prediction/
├── data/
│   └── heart_disease.csv
├── notebooks/
│   └── heart_disease_prediction.ipynb
├── src/
│   └── heart_disease_prediction.py
├── visualizations/
│   ├── 01_target_distribution.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_confusion_matrices.png
│   ├── 05_roc_curves.png
│   └── 06_model_comparison.png
├── models/
│   ├── heart_disease_model.joblib
│   ├── model_comparison.csv
│   └── results_summary.json
├── requirements.txt
└── README.md
```

## How to Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run the Project

**Script:**
```bash
python src/heart_disease_prediction.py
```

**Notebook:**
```bash
jupyter notebook notebooks/heart_disease_prediction.ipynb
```

Both produce the same results, since they run the identical pipeline.

## Results Summary

- Final dataset used for modeling: 918 rows × 14 features after cleaning
- Best model: **Logistic Regression**, ROC AUC = **0.9073**, Accuracy = **0.8152**
- Model and preprocessing objects (scaler, label encoders) saved together
  in `models/heart_disease_model.joblib` for reuse on new patient data

## Limitations

- `ca` and `thal` — two clinically important features — have >50% missing
  values in the raw data, handled here via mode imputation. A more
  advanced approach (e.g. multiple imputation, or a missingness indicator
  feature) could improve on this.
- The dataset combines four different medical centers with different data
  collection practices, which is a likely source of some of the
  missingness pattern.

## Dataset Citation

The Heart Disease dataset is from the **UCI Machine Learning Repository**.

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease*. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C52P4X

Source: https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset is licensed under **CC BY 4.0**. Please retain attribution when redistributing the dataset.
