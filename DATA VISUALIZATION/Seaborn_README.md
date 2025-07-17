# 🚢 Titanic Survival Visualization Project

This project uses **Seaborn** and **Matplotlib** to explore the Titanic dataset (`tested.csv`) visually, analyzing how variables such as gender, class, age, and fare influence survival outcomes.

---

## 📁 Dataset

- **File Used**: `tested.csv`
- **Source**: Based on the Titanic dataset from [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic)
- **Important Columns**: `Survived`, `Sex`, `Pclass`, `Age`, `Fare`

---

## 📊 Visualizations

1. **Survival Count by Gender**
2. **Survival by Passenger Class**
3. **Age Distribution of Survivors vs Non-survivors**
4. **Fare vs Age by Survival**
5. **Correlation Heatmap**

> *(You can add your plots in an `/images` folder and display them here)*

---

## 🧪 How to Run

### 🔧 Requirements
```bash
pip install pandas seaborn matplotlib
````

### ▶️ Run the Script

```bash
python titanic_visualization.py
```

Or open in Jupyter:

```bash
jupyter notebook titanic_visualization.ipynb
```

---

## 📌 Summary Table

| Plot Type   | What It Shows                                  |
| ----------- | ---------------------------------------------- |
| Countplot   | Survival by gender and class                   |
| Histplot    | Age distribution of survivors vs non-survivors |
| Scatterplot | Relationship between age and fare by survival  |
| Heatmap     | Correlations between numeric features          |

---

## 🧠 Code Explanation (Line-by-Line)

### ✅ Setup and Imports

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

* `pandas` (pd): For loading and manipulating CSV data.
* `seaborn` (sns): For plotting statistical graphs easily.
* `matplotlib.pyplot` (plt): For showing the plots created with Seaborn.

---

### 📥 Load Dataset

```python
df = pd.read_csv("tested.csv")
```

* Loads the dataset into a DataFrame named `df`.

---

### 🧹 Clean the Data

```python
df.dropna(subset=["Age", "Fare"], inplace=True)
```

* Removes rows with missing `Age` or `Fare` values.

---

### 📊 1. Survival Count by Gender

```python
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival Count by Gender")
plt.show()
```

* Visualizes survival by gender.
* `hue="Survived"` adds color for survived vs. not survived.

---

### 📊 2. Survival by Passenger Class

```python
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.show()
```

* Shows the effect of passenger class on survival.

---

### 📊 3. Age Distribution of Survivors vs Non-survivors

```python
sns.histplot(data=df, x="Age", hue="Survived", multiple="stack", kde=True)
plt.title("Age Distribution by Survival")
plt.show()
```

* Displays how survival varies by age.
* `kde=True` overlays a density curve.

---

### 📊 4. Fare vs Age by Survival

```python
sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived")
plt.title("Fare vs Age by Survival")
plt.show()
```

* Plots each passenger by age and fare, colored by survival.

---

### 📊 5. Heatmap of Correlations

```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```

* Calculates and visualizes correlations between numerical columns.
* `annot=True` displays correlation values inside the heatmap.

---

## 🙌 Acknowledgments

* [Seaborn Documentation](https://seaborn.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)

---

## 👩‍💻 Author

**A. Hemalatha**
*Data Science and AI Enthusiast*
[LinkedIn](https://www.linkedin.com/in/hemalatha-a-developer/) • [GitHub](https://github.com/HEMALATHAifet/)

---


