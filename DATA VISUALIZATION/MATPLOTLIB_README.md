# 🚢 Titanic Data Visualization using Matplotlib

This project uses the Titanic dataset (`tested.csv`) to explore survival patterns among passengers using **Matplotlib** for data visualization.

---

## 📦 Dependencies

```bash
pip install pandas matplotlib
````

---

## 📥 Dataset

Make sure the file `tested.csv` is in the same directory. This dataset should contain columns like:

* `Sex`
* `Age`
* `Fare`
* `Pclass`
* `Survived`

---

## 🧪 Python Program with Matplotlib

```python
import pandas as pd
import matplotlib.pyplot as plt
```

✅ **Explanation**:

* `pandas`: For reading and managing the dataset.
* `matplotlib.pyplot`: For creating and displaying plots.

---

```python
df = pd.read_csv("tested.csv")
```

✅ Reads the Titanic data into a DataFrame named `df`.

---

```python
df.dropna(subset=["Age", "Fare"], inplace=True)
```

✅ Removes rows with missing values in the `Age` or `Fare` columns to avoid plotting issues.

---

### 1️⃣ Survival Count by Gender

```python
survived_gender = df[df['Survived'] == 1]['Sex'].value_counts()
not_survived_gender = df[df['Survived'] == 0]['Sex'].value_counts()

labels = survived_gender.index
x = range(len(labels))

plt.bar(x, survived_gender, width=0.4, label='Survived', align='center')
plt.bar([p + 0.4 for p in x], not_survived_gender, width=0.4, label='Not Survived', align='center')
plt.xticks([p + 0.2 for p in x], labels)
plt.title("Survival Count by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend()
plt.show()
```

✅ Explanation:

* Filters and counts survivors/non-survivors by gender.
* Uses `plt.bar()` to plot grouped bar charts.
* Adjusts bar positions to show two bars side-by-side per gender.
* Adds labels, title, and legend.

---

### 2️⃣ Survival by Passenger Class

```python
survived_class = df[df['Survived'] == 1]['Pclass'].value_counts().sort_index()
not_survived_class = df[df['Survived'] == 0]['Pclass'].value_counts().sort_index()

labels = survived_class.index
x = range(len(labels))

plt.bar(x, survived_class, width=0.4, label='Survived', align='center')
plt.bar([p + 0.4 for p in x], not_survived_class, width=0.4, label='Not Survived', align='center')
plt.xticks([p + 0.2 for p in x], labels)
plt.title("Survival by Passenger Class")
plt.xlabel("Class")
plt.ylabel("Count")
plt.legend()
plt.show()
```

✅ Explanation:

* Compares survival across Pclass (1st, 2nd, 3rd).
* Uses the same grouped bar approach.

---

### 3️⃣ Age Distribution by Survival

```python
plt.hist(df[df['Survived'] == 1]['Age'], bins=20, alpha=0.5, label='Survived')
plt.hist(df[df['Survived'] == 0]['Age'], bins=20, alpha=0.5, label='Not Survived')
plt.title("Age Distribution by Survival")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.legend()
plt.show()
```

✅ Explanation:

* `plt.hist()` plots two overlaid histograms.
* `alpha=0.5` makes them semi-transparent.
* Shows how age distribution varies with survival.

---

### 4️⃣ Fare vs. Age by Survival

```python
colors = {0: 'red', 1: 'green'}
plt.scatter(df['Age'], df['Fare'], c=df['Survived'].map(colors))
plt.title("Fare vs Age by Survival")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
```

✅ Explanation:

* `plt.scatter()` plots fare vs age.
* Uses color mapping: red for not survived, green for survived.

---

### 5️⃣ Correlation Heatmap (Optional)

```python
import numpy as np

corr = df[["Age", "Fare", "Pclass", "Survived"]].corr()
fig, ax = plt.subplots()
cax = ax.matshow(corr, cmap="coolwarm")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar(cax)
plt.title("Correlation Heatmap", pad=20)
plt.show()
```

✅ Explanation:

* Uses `matshow()` to simulate a heatmap.
* Adds ticks, color bar, and title.

---

## 🔍 Seaborn vs. Matplotlib

| Feature        | Seaborn                            | Matplotlib                    |
| -------------- | ---------------------------------- | ----------------------------- |
| Style & Theme  | Built-in themes, cleaner visuals   | Requires manual styling       |
| Ease of Use    | High-level interface               | Low-level, more customization |
| Built-in Plots | `countplot`, `histplot`, `heatmap` | Must be manually coded        |
| Integration    | Built on top of Matplotlib         | Base library                  |
| Ideal For      | Statistical data visualization     | Custom plots and control      |

---

## ✅ Summary

This project provides beginner-friendly **Matplotlib visualizations** to explore Titanic survival data, comparing it with the capabilities of Seaborn.

---

## 📁 File Structure

```
📦 titanic-matplotlib/
 ┣ 🗃 tested.csv
 ┣ 📄 titanic_matplotlib.py
 ┣ 📄 README.md
```

---

## 🚀 Author

**Hemalatha A**
GitHub: \[https://github.com/HEMALATHAifet/]
LinkedIn: \[https://www.linkedin.com/in/hemalatha-a-developer/]

---

