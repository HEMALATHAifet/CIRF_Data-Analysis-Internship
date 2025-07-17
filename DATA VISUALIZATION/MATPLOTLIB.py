pip install matplotlib pandas

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("tested.csv")

# Replace missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Set figure size
plt.figure(figsize=(10, 6))

# Plot 1: Bar chart of survivors vs non-survivors
survivors = df['Survived'].value_counts()
plt.subplot(2, 2, 1)
plt.bar(['Not Survived', 'Survived'], survivors, color=['red', 'green'])
plt.title('Survivor Count')
plt.ylabel('Number of Passengers')

# Plot 2: Bar chart of survival by gender
gender_survival = df.groupby('Sex')['Survived'].sum()
plt.subplot(2, 2, 2)
plt.bar(gender_survival.index, gender_survival.values, color=['blue', 'pink'])
plt.title('Survivors by Gender')

# Plot 3: Histogram of Age
plt.subplot(2, 2, 3)
plt.hist(df['Age'], bins=20, color='orange', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')

# Plot 4: Boxplot of Fare
plt.subplot(2, 2, 4)
plt.boxplot(df['Fare'].dropna())
plt.title('Fare Distribution')
plt.ylabel('Fare')

plt.tight_layout()
plt.show()
