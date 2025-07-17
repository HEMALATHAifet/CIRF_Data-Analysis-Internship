import pandas as pd

# Load the dataset
df = pd.read_csv("tested.csv")

# Display first 5 rows
print("🔹 First 5 Rows of Dataset:")
print(df.head())

# Check for missing values
print("\n🔹 Missing Values in Each Column:")
print(df.isnull().sum())

# Drop rows with missing Age or Fare
df.dropna(subset=["Age", "Fare"], inplace=True)

# Survival count by Gender
print("\n🔹 Survival Count by Gender:")
print(df.groupby("Sex")["Survived"].value_counts().unstack())

# Survival count by Passenger Class
print("\n🔹 Survival Count by Pclass:")
print(df.groupby("Pclass")["Survived"].value_counts().unstack())

# Average Age and Fare by Survival
print("\n🔹 Average Age and Fare by Survival Status:")
print(df.groupby("Survived")[["Age", "Fare"]].mean())

# Correlation Matrix
print("\n🔹 Correlation Matrix:")
print(df.corr(numeric_only=True))
