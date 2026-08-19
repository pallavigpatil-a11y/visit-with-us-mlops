
import pandas as pd
import os
from sklearn.model_selection import train_test_split

DATA_PATH = "data/tourism.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# Remove unnecessary identifier
if "CustomerID" in df.columns:
    df = df.drop(columns=["CustomerID"])

# Remove duplicate rows
df = df.drop_duplicates()

# Basic missing-value treatment
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

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

print("\nTrain/test data saved successfully.")
