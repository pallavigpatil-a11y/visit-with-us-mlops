
import pandas as pd
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

train_df = pd.read_csv(f"{ARTIFACT_DIR}/train.csv")
test_df = pd.read_csv(f"{ARTIFACT_DIR}/test.csv")

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]

X_test = test_df.drop(columns=["ProdTaken"])
y_test = test_df["ProdTaken"]

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    ))
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

print("\nStarting hyperparameter tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

y_pred = best_model.predict(X_test)

print("\nModel Evaluation")
print("----------------")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))

if hasattr(best_model, "predict_proba"):
    y_probability = best_model.predict_proba(X_test)[:, 1]
    print("ROC-AUC  :", roc_auc_score(y_test, y_probability))

model_path = f"{MODEL_DIR}/best_model.pkl"

joblib.dump(best_model, model_path)

print(f"\nBest model saved to: {model_path}")
