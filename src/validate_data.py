
import pandas as pd
import os

DATA_PATH = "data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

print("Loading dataset:", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

missing_columns = [
    col for col in EXPECTED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    print("\nMissing expected columns:")
    print(missing_columns)
    raise ValueError("Dataset validation failed.")

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["ProdTaken"].value_counts())

print("\nDataset validation successful.")
