
# Titanic Survival Prediction using Scikit-learn

This is a simple machine learning project that uses the Titanic dataset (`tested.csv`) to predict passenger survival using Logistic Regression in Scikit-learn.

---

## 📊 Dataset

The dataset `tested.csv` contains passenger information such as:

- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`: Gender (male, female)
- `Age`: Age of the passenger
- `Fare`: Ticket fare
- `Embarked`: Port of embarkation (S = Southampton, C = Cherbourg, Q = Queenstown)
- `Survived`: Whether the passenger survived (1) or not (0)

---

## 🧠 ML Model Used

- **Logistic Regression** (a classification algorithm suitable for binary outcomes)

---

## 📦 Requirements

- `pandas`
- `scikit-learn`

Install them via:

```bash
pip install pandas scikit-learn
````

---

## 🧾 Code Explanation

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

📌 *Import required libraries*

---

```python
df = pd.read_csv("tested.csv")
```

📌 *Load the dataset*

---

```python
df.dropna(subset=["Age", "Fare", "Embarked"], inplace=True)
```

📌 *Remove rows with missing values in important columns*

---

```python
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
```

📌 *Convert categorical columns into numeric values for model compatibility*

---

```python
X = df[["Pclass", "Sex", "Age", "Fare", "Embarked"]]
y = df["Survived"]
```

📌 *Define feature matrix `X` and target variable `y`*

---

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

📌 *Split the dataset into training and testing sets (80-20 split)*

---

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

📌 *Initialize and train the Logistic Regression model*

---

```python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

📌 *Predict survival for the test set and calculate accuracy*

---


| Line                           | Explanation                                               |
| ------------------------------ | --------------------------------------------------------- |
| `import pandas...`             | Import necessary libraries                                |
| `df = pd.read_csv(...)`        | Load Titanic dataset                                      |
| `df.dropna(...)`               | Remove rows with missing Age, Fare, or Embarked           |
| `df["Sex"] = ...`              | Convert 'Sex' column to numeric: male → 0, female → 1     |
| `df["Embarked"] = ...`         | Convert 'Embarked' column to numeric: S → 0, C → 1, Q → 2 |
| `X = df[...]`                  | Select feature columns                                    |
| `y = df["Survived"]`           | Target column (Survived: 1 = Yes, 0 = No)                 |
| `train_test_split(...)`        | Split dataset into training and testing sets              |
| `model = LogisticRegression()` | Create logistic regression model                          |
| `model.fit(...)`               | Train model on training data                              |
| `y_pred = model.predict(...)`  | Predict survival on test data                             |
| `accuracy_score(...)`          | Calculate accuracy of predictions                         |

---

## ✅ Output

The program prints the accuracy of the trained model on the test dataset, for example:

```
Accuracy: 0.81
```

---

## 📌 Note

* This project assumes cleaned and preprocessed input data.
* For better performance, consider additional preprocessing like scaling or using advanced models like Random Forest.

---

## 📚 References

* Titanic dataset from Kaggle or similar educational sources.
* [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)

```

---

