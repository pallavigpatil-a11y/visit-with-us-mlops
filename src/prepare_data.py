
import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_PATH = "data/tourism.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=["number"]).columns
categorical_cols = df.select_dtypes(
    include=["object", "string", "category"]
).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    if df[col].isna().any():
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value[0])

print("Cleaned shape:", df.shape)

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train_df = X_train.copy()
train_df["ProdTaken"] = y_train.values

test_df = X_test.copy()
test_df["ProdTaken"] = y_test.values

train_df.to_csv(
    f"{ARTIFACT_DIR}/train.csv",
    index=False
)

test_df.to_csv(
    f"{ARTIFACT_DIR}/test.csv",
    index=False
)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)
print("Train/test data saved successfully.")
