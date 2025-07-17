# 🚢 Titanic Dataset Analysis with Pandas

This project analyzes the Titanic dataset (`tested.csv`) using Pandas.

## 📂 Files

- `titanic_pandas_analysis.py`: Python script with Pandas-based data analysis.
- `tested.csv`: Dataset file (make sure this is in the same folder).

## ✅ What It Does

1. Loads the dataset
2. Displays initial records
3. Handles missing data (Age and Fare)
4. Analyzes survival count by gender and class
5. Calculates average age/fare by survival
6. Shows correlation between numeric features

### 📘 Line-by-Line Explanation

| Line                                             | Description                                                       |
| ------------------------------------------------ | ----------------------------------------------------------------- |
| `import pandas as pd`                            | Imports the Pandas library for data manipulation.                 |
| `df = pd.read_csv("tested.csv")`                 | Reads the Titanic CSV dataset into a DataFrame.                   |
| `df.head()`                                      | Displays the first 5 rows to understand the structure.            |
| `df.isnull().sum()`                              | Counts missing (NaN) values in each column.                       |
| `df.dropna(...)`                                 | Drops rows where `Age` or `Fare` are missing.                     |
| `df.groupby("Sex")...`                           | Groups by gender and counts survival status (0/1).                |
| `df.groupby("Pclass")...`                        | Shows how survival varied across passenger classes.               |
| `df.groupby("Survived")[["Age", "Fare"]].mean()` | Shows the average age and fare for survivors and non-survivors.   |
| `df.corr()`                                      | Calculates correlation between numeric columns (e.g., Age, Fare). |

---

## 📊 Pandas vs. Seaborn vs. Matplotlib

| Tool        | Purpose                               | Strength                     |
|-------------|----------------------------------------|------------------------------|
| **Pandas**  | Data cleaning, filtering, aggregation | Easy for tabular insights    |
| **Seaborn** | Statistical plotting                   | Beautiful and high-level API |
| **Matplotlib** | Base plotting library              | Fine control over plots      |

Use **Pandas** for quick insights, **Seaborn** for attractive visualizations, and **Matplotlib** when you want custom plots.

---

## 🧪 Sample Output

Run this using:
```bash
python titanic_pandas_analysis.py
````

```
🔹 First 5 Rows of Dataset:
  PassengerId  Survived  Pclass     Name   Sex   Age  Fare  ...
0           1         0       3   ...    male  22.0  7.25
1           2         1       1   ...  female  38.0  71.2833
...

🔹 Survival Count by Gender:
Survived    0    1
Sex                
female     81  233
male      468  109

...
```

---

