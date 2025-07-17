# 🚢 Titanic Survival Prediction using TensorFlow

This is a beginner-friendly machine learning project that uses **TensorFlow** to predict whether a passenger survived the Titanic shipwreck based on their information such as class, age, sex, and fare.

---

## 📌 What is TensorFlow?

**TensorFlow** is an open-source machine learning library developed by Google. It allows you to build and train deep learning models easily using high-level APIs.

We use TensorFlow in this project to:
- Build a neural network (a model inspired by how the human brain works)
- Train it on historical Titanic passenger data
- Predict whether a new passenger survived or not

---

## 📊 Dataset

We use the `tested.csv` dataset from the Titanic dataset. Each row represents a passenger, and columns include:
- `Pclass`: Passenger class (1 = First, 2 = Second, 3 = Third)
- `Sex`: Gender
- `Age`: Age of the passenger
- `Fare`: Ticket fare
- `Survived`: Whether the passenger survived (1) or not (0)

---

## 📦 Python Libraries Used

| Library       | Purpose                                                                 |
|---------------|-------------------------------------------------------------------------|
| `pandas`      | For reading and manipulating the dataset                               |
| `tensorflow`  | To build, compile, and train the neural network model                  |
| `sklearn`     | For preprocessing (encoding, scaling) and train-test split             |

Install them using:

```bash
pip install pandas tensorflow scikit-learn
````

---

## 📁 Files

* `tested.csv`: Titanic dataset
* `tensorflow_model.py`: Python script that contains the full TensorFlow model code

---

## 🧠 How the Code Works (Explained Step-by-Step)

### 1. **Import Libraries**

```python
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
```

We import libraries for data handling, preprocessing, and building the model.

---

### 2. **Read Dataset**

```python
df = pd.read_csv("tested.csv")
```

We read the CSV file containing Titanic passenger information.

---

### 3. **Clean and Prepare the Data**

```python
df.dropna(subset=['Age', 'Fare', 'Sex'], inplace=True)
df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
```

* We remove rows with missing `Age`, `Fare`, or `Sex`.
* Convert `Sex` (male/female) into numbers (0/1) using `LabelEncoder`.

---

### 4. **Select Features and Labels**

```python
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']
```

We choose the input features (X) and output (y).

---

### 5. **Normalize the Features**

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

We scale all numeric features to a similar range to improve model performance.

---

### 6. **Split the Data**

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

We split the data into training (80%) and testing (20%).

---

### 7. **Build the Neural Network Model**

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

* The model has 3 layers.

  * First layer: 16 neurons, ReLU activation
  * Second layer: 8 neurons, ReLU activation
  * Output layer: 1 neuron, sigmoid activation (for binary classification)

---

### 8. **Compile the Model**

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

We define how the model learns:

* `adam` optimizer adjusts weights
* `binary_crossentropy` is the loss function for classification
* `accuracy` helps track performance

---

### 9. **Train the Model**

```python
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)
```

The model is trained on the training set for 50 iterations (epochs).

---

### 10. **Evaluate the Model**

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

We check how well the model performs on unseen data (testing set).

---

## ✅ Output

At the end, you'll see something like:

```
Test Accuracy: 0.81
```

This means the model correctly predicted survival 81% of the time on the test data.

---

## 🔁 What's Next?

* Add more features like `Embarked`, `SibSp`, etc.
* Try different neural network architectures
* Compare results with traditional models like Logistic Regression or Decision Trees

---

## 🙋 Why Use TensorFlow?

TensorFlow allows you to:

* Build complex models with ease
* Run on CPUs, GPUs, or TPUs
* Scale models for real-world applications

---

## 📬 Feedback

Feel free to contribute, suggest improvements, or ask questions!

---
