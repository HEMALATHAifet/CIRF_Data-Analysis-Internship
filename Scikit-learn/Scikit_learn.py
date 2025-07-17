import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
df = pd.read_csv("tested.csv")

# Step 2: Drop rows with missing values in 'Age', 'Fare', or 'Embarked'
df.dropna(subset=["Age", "Fare", "Embarked"], inplace=True)

# Step 3: Convert categorical variables to numeric
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Step 4: Define features and target
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]

# Step 5: Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
