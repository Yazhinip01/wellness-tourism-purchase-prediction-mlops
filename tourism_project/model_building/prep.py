import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")
df.drop(columns=["CustomerID", "Unnamed: 0"], inplace=True)

# Keep categorical columns as raw strings.
# The model's preprocessing pipeline will one-hot encode them during training.
# Streamlit will supply the same raw categorical values during prediction.

df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')


X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Split the data into training and testing sets.
# Stratification preserves the purchase/non-purchase ratio.
# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Gender values kept as:", sorted(X["Gender"].unique()))
print("MaritalStatus values kept as:", sorted(X["MaritalStatus"].unique()))
