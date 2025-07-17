# 🚢 Titanic Data Analysis using NumPy

This is a simple data analysis project that uses **NumPy** and **Pandas** to analyze the Titanic dataset (`tested.csv`). It demonstrates how to clean data, extract relevant information, and perform basic statistical operations using NumPy.

---

## 📂 Dataset

The dataset is a modified version of the Titanic passenger data and includes columns such as:

- `Age`
- `Fare`
- `Survived`
- `Pclass`, etc.

Make sure the file is named `tested.csv` and is in the same directory as your Python script.

---

## 🧠 What This Project Does

This script performs:

- Mean and standard deviation calculation
- Finding max and min values
- Counting specific conditions (e.g., passengers under 18)
- Min-Max normalization

---

## 🧾 Requirements

```bash
pip install numpy pandas
````

---

## 🚀 How to Run

```bash
python numpy_analysis.py
```

Make sure the script and the `tested.csv` file are in the same folder.

---

## 📌 Code Explanation (Line-by-Line)

```python
import numpy as np
import pandas as pd
```

🔹 Import NumPy and Pandas libraries.

```python
df = pd.read_csv("tested.csv")
```

🔹 Read the Titanic dataset into a Pandas DataFrame.

```python
df_clean = df.dropna(subset=["Age", "Fare"])
```

🔹 Drop rows that have missing Age or Fare values to avoid errors.

```python
ages = df_clean["Age"].to_numpy()
fares = df_clean["Fare"].to_numpy()
```

🔹 Convert the "Age" and "Fare" columns into NumPy arrays for numeric operations.

```python
mean_age = np.mean(ages)
mean_fare = np.mean(fares)
```

🔹 Compute the **mean** (average) of ages and fares.

```python
std_age = np.std(ages)
std_fare = np.std(fares)
```

🔹 Compute the **standard deviation** of ages and fares.

```python
max_age = np.max(ages)
min_age = np.min(ages)
```

🔹 Get the **maximum** and **minimum** values of the Age column.

```python
under_18_count = np.sum(ages < 18)
```

🔹 Count how many passengers are under 18 using Boolean masking.

```python
fare_normalized = (fares - np.min(fares)) / (np.max(fares) - np.min(fares))
```

🔹 Normalize the fare column using **Min-Max normalization**.

```python
print(...)  # multiple print statements
```

🔹 Output the results of each computation.

---


| Line                                  | Explanation                                     |
| ------------------------------------- | ----------------------------------------------- |
| `import numpy as np`                  | Imports NumPy library for numerical operations. |
| `import pandas as pd`                 | Imports Pandas to load and clean the CSV data.  |
| `df = pd.read_csv(...)`               | Reads the Titanic dataset.                      |
| `df_clean = df.dropna(...)`           | Removes rows where Age or Fare is missing.      |
| `ages = df_clean["Age"].to_numpy()`   | Converts the Age column to a NumPy array.       |
| `fares = df_clean["Fare"].to_numpy()` | Converts the Fare column to a NumPy array.      |
| `np.mean(...)`                        | Calculates the mean of Age and Fare.            |
| `np.std(...)`                         | Computes standard deviation.                    |
| `np.max(...)`, `np.min(...)`          | Gets maximum and minimum values.                |
| `np.sum(ages < 18)`                   | Counts how many are below 18 years.             |
| `(fares - min) / (max - min)`         | Performs min-max normalization.                 |

---

## 📊 Seaborn vs. Matplotlib vs. NumPy

| Feature       | NumPy                  | Matplotlib                     | Seaborn                    |
| ------------- | ---------------------- | ------------------------------ | -------------------------- |
| Purpose       | Numerical computation  | Plotting (low-level)           | Plotting (high-level)      |
| Visualization | ❌ No plotting          | ✅ Yes (basic plots)            | ✅ Yes (aesthetic plots)    |
| Abstraction   | ✅ Fast and simple      | ⚠️ Requires manual styling     | ✅ Prettier default styles  |
| Usage         | Data stats & math only | Line, bar, scatter, pie charts | Heatmaps, histograms, etc. |

---

## 📁 Output Sample

```txt
Mean Age: 29.70
Mean Fare: 32.20
Standard Deviation of Age: 14.42
Standard Deviation of Fare: 49.69
Max Age: 80.0
Min Age: 0.42
Number of passengers under 18: 113
First 5 normalized fares: [0.014 0.103 0.056 ...]
```

---

## 🙌 Acknowledgements

* Titanic dataset from Kaggle
* Libraries used: NumPy, Pandas

---



