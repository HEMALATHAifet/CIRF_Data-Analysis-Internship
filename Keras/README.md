
## 🧠 Titanic Survival Prediction using Keras

This project uses **Keras**, a high-level deep learning API running on top of TensorFlow, to build a simple neural network that predicts whether a passenger survived the Titanic disaster based on features like age, fare, sex, and more.

---

### 📂 Dataset Used

File: `tested.csv` (uploaded by the user)

This dataset contains information such as:

* `Pclass` (Passenger class)
* `Sex`
* `Age`
* `Fare`
* `Embarked` (Port of Embarkation)
* `Survived` (Target: 0 = No, 1 = Yes)

---

### 📚 Python Libraries Used

| Library            | Purpose                                                     |
| ------------------ | ----------------------------------------------------------- |
| `pandas`           | Data loading and manipulation                               |
| `numpy`            | Numerical computations                                      |
| `scikit-learn`     | Preprocessing (standard scaling), splitting data            |
| `tensorflow.keras` | Defining, training, and evaluating the neural network model |

---

### 🤖 What is Keras?

Keras is an **open-source deep learning framework** built on top of TensorFlow. It allows quick prototyping of deep learning models with simple Python code. It is beginner-friendly and very useful for building **neural networks** for classification, regression, and other ML tasks.

---

### 🧪 Use of This Project

This model is used to **predict passenger survival** based on several features. It applies a **binary classification** deep learning approach using a **feedforward neural network** built with Keras.

---

### 🧾 Line-by-Line Code Explanation

```python
# Load data using pandas
df = pd.read_csv("tested.csv")
```

📌 Loads the dataset into a pandas DataFrame.

```python
df.dropna(subset=["Age", "Fare", "Embarked"], inplace=True)
```

📌 Removes rows that have missing values in key columns to ensure clean data.

```python
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
```

📌 Converts categorical data to numeric format (label encoding).

```python
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]
```

📌 Defines features (`X`) and target (`y`).

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

📌 Standardizes the feature values to improve training performance.

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

📌 Splits the data into training and testing sets (80-20 split).

```python
model = Sequential()
model.add(Dense(16, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

📌 Defines the neural network:

* 2 hidden layers (`relu` activation)
* 1 output layer with `sigmoid` for binary classification.

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

📌 Compiles the model using:

* `binary_crossentropy` (loss for binary classification)
* `adam` optimizer (fast & adaptive)
* accuracy metric

```python
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
```

📌 Trains the model on the training data for 50 epochs with mini-batches of 10 samples.

```python
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
```

📌 Evaluates the model on unseen test data and prints the accuracy.

---

### 📈 Expected Output

```
Epoch 1/50
...
Epoch 50/50
...
Test Accuracy: 0.83
```

---

### ✅ How to Run

1. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn tensorflow
   ```

2. Place `tested.csv` in the same directory as your `.py` file.

3. Run the script.

---
