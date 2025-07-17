import numpy as np
import pandas as pd

# Load the dataset using pandas
df = pd.read_csv("tested.csv")

# Drop rows with missing Age or Fare
df_clean = df.dropna(subset=["Age", "Fare"])

# Extract "Age" and "Fare" columns as NumPy arrays
ages = df_clean["Age"].to_numpy()
fares = df_clean["Fare"].to_numpy()

# 1. Mean Age and Fare
mean_age = np.mean(ages)
mean_fare = np.mean(fares)

print(f"Mean Age: {mean_age:.2f}")
print(f"Mean Fare: {mean_fare:.2f}")

# 2. Standard Deviation
std_age = np.std(ages)
std_fare = np.std(fares)

print(f"Standard Deviation of Age: {std_age:.2f}")
print(f"Standard Deviation of Fare: {std_fare:.2f}")

# 3. Maximum and Minimum Age
max_age = np.max(ages)
min_age = np.min(ages)

print(f"Max Age: {max_age}")
print(f"Min Age: {min_age}")

# 4. Count number of passengers under 18
under_18_count = np.sum(ages < 18)
print(f"Number of passengers under 18: {under_18_count}")

# 5. Normalize Fare (0 to 1)
fare_normalized = (fares - np.min(fares)) / (np.max(fares) - np.min(fares))
print(f"First 5 normalized fares: {fare_normalized[:5]}")
