import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Load dataset
df = pd.read_csv("tested.csv")

# 2. Drop rows with missing values in selected columns
df.dropna(subset=["Age", "Fare", "Embarked"], inplace=True)

# 3. Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 4. Feature matrix and target variable
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]

# 5. Scale features for better training
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 7. Define Keras model
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))   # input layer + hidden layer 1
model.add(Dense(8, activation='relu'))                 # hidden layer 2
model.add(Dense(1, activation='sigmoid'))              # output layer (binary classification)

# 8. Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 9. Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# 10. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
