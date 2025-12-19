"""
Hepatitis Disease Prediction
Leakage-free, medically sound ML pipeline
Dataset: UCI Hepatitis (ID: 46)
"""

# =========================
# Imports
# =========================
import os
import joblib
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# =========================
# Config
# =========================
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================
# Load Dataset
# =========================
hepatitis = fetch_ucirepo(id=46)

X = hepatitis.data.features.copy()
y = hepatitis.data.targets.iloc[:, 0].copy()

# Normalize column names (CRITICAL)
X.columns = (
    X.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_") 
)

# Target: 1 = Dies, 2 = Lives → 0/1
y = y.map({1: 0, 2: 1})

assert X.shape[1] == 19, "Expected 19 features"

# =========================
# Feature Typing
# =========================
NUMERIC_FEATURES = [
    "age",
    "bilirubin",
    "alk_phosphate",
    "sgot",
    "albumin",
    "protime"
]

BINARY_FEATURES = [
    "sex",
    "steroid",
    "antivirals",
    "fatigue",
    "malaise",
    "anorexia",
    "liver_big",
    "liver_firm",
    "spleen_palpable",
    "spiders",
    "ascites",
    "varices",
    "histology"
]

# Defensive check (prevents your error forever)
expected_cols = set(NUMERIC_FEATURES + BINARY_FEATURES)
actual_cols = set(X.columns)

assert expected_cols == actual_cols, (
    f"\nColumn mismatch!\nExpected: {expected_cols}\nGot: {actual_cols}"
)

# =========================
# Preprocessing Pipelines
# =========================
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("bin", binary_pipeline, BINARY_FEATURES)
    ],
    remainder="drop"
)

# =========================
# Models (Constrained)
# =========================
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Linear SVM": SVC(
        kernel="linear",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance"
    ),

    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.5,
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE
    )
}

# =========================
# Cross-Validation
# =========================
cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

# =========================
# Train & Evaluate
# =========================
results = []
best_model = None
best_auc = 0.0

for name, model in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    mean_auc = cv_results["test_roc_auc"].mean()

    results.append({
        "Model": name,
        "Accuracy": cv_results["test_accuracy"].mean(),
        "Precision": cv_results["test_precision"].mean(),
        "Recall": cv_results["test_recall"].mean(),
        "F1": cv_results["test_f1"].mean(),
        "ROC-AUC": mean_auc
    })

    if mean_auc > best_auc:
        best_auc = mean_auc
        best_model = pipeline

# =========================
# Fit Best Model on Full Data
# =========================
best_model.fit(X, y)

# =========================
# Save Artifacts
# =========================
joblib.dump(best_model, f"{ARTIFACT_DIR}/best_model.pkl")
joblib.dump(preprocessor, f"{ARTIFACT_DIR}/preprocessor.pkl")

# =========================
# Results
# =========================
results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

print("\n================ FINAL RESULTS ================\n")
print(results_df.to_string(index=False))

print(f"\nBest Model: {results_df.iloc[0]['Model']}")
print(f"Best ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.3f}")
