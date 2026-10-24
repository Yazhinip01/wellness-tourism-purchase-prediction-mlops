import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = ["Unnamed: 0", "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "Occupation", "Gender",
                    "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
                    "Passport", "OwnCar", "NumberOfChildrenVisiting", "Designation",
                    "MonthlyIncome", "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups", "DurationOfPitch"]

missing = [column for column in expected_columns if column not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

# Display registration summary
print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))

# Display missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Display target distribution
print("\nPurchase distribution:")
print(df["ProdTaken"].value_counts().sort_index())
