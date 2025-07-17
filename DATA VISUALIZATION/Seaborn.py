import pandas as pd

# Load the dataset
df = pd.read_csv("tested.csv")

# Display first few rows
df.head()
df.info()
df.describe()
df.columns

pip install pandas seaborn matplotlib


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("tested.csv")

# Drop rows with missing Age or Fare
df.dropna(subset=["Age", "Fare"], inplace=True)

# 1. Survival Count by Gender
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival Count by Gender")
plt.show()

# 2. Survival by Passenger Class
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.show()

# 3. Age distribution of Survivors vs Non-survivors
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", kde=True)
plt.title("Age Distribution by Survival")
plt.show()

# 4. Fare vs Age by Survival
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived")
plt.title("Fare vs Age by Survival")
plt.show()

# 5. Heatmap of Correlations
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
