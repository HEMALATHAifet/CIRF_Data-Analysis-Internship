## 🚢 Titanic Survival Analysis using SciPy

This project demonstrates how to perform a **Chi-Square Test of Independence** using the **SciPy** library on the Titanic dataset. The test helps us determine whether there is a **statistically significant relationship** between a passenger’s **gender** and their **chance of survival**.

---

### 📂 Dataset

We use the Titanic dataset: `tested.csv`, which contains columns like:

* `Sex`: Passenger gender (`male`, `female`)
* `Survived`: Survival status (`0 = No`, `1 = Yes`)

---

### 📊 Objective

To test the hypothesis:

> "Is a passenger’s **gender** associated with their **chance of survival**?"

We use **Chi-Square Test of Independence** to answer this.

---

### 🧪 Libraries Used

| Library       | Purpose                                      |
| ------------- | -------------------------------------------- |
| `pandas`      | To load and manipulate CSV data              |
| `scipy.stats` | To perform statistical tests like Chi-Square |

---

### 🧠 What is SciPy?

> **SciPy** (Scientific Python) is a Python-based ecosystem of open-source software for mathematics, science, and engineering. It builds on NumPy and provides many user-friendly and efficient numerical routines.

We use `scipy.stats.chi2_contingency` to test **categorical associations**.

---

### 📌 Chi-Square Test Explanation

The Chi-Square Test checks if two categorical variables are **independent**.

* **Null Hypothesis (H₀):** Survival is independent of gender.
* **Alternative Hypothesis (H₁):** Survival is dependent on gender.

If p-value < 0.05, we **reject H₀** and conclude there **is an association**.

---
### 🔍 Explanation:

| Line  | Code                                             | Purpose                                      |
| ----- | ------------------------------------------------ | -------------------------------------------- |
| 1     | `import pandas as pd`                            | For data loading and manipulation            |
| 2     | `from scipy.stats import chi2_contingency`       | Import Chi-Square test function              |
| 5     | `df = pd.read_csv("tested.csv")`                 | Load the Titanic dataset                     |
| 8     | `pd.crosstab(df['Sex'], df['Survived'])`         | Create a contingency table (Sex vs Survived) |
| 11    | `chi2, p, dof, expected = chi2_contingency(...)` | Perform the Chi-Square test                  |
| 13–17 | `print(...)`                                     | Display test results                         |
| 20–23 | `if p < 0.05: ...`                               | Interpret statistical significance           |

---



### 📈 Sample Output

```
Contingency Table:
 Survived    0    1
Sex               
female      81  233
male       468  109

Expected Frequencies:
[[192.84 121.15]
 [356.15 220.85]]

Chi-Square Statistic: 265.73
Degrees of Freedom: 1
P-value: 1.1e-59

Conclusion: Survival and Gender are dependent.
```

---

### ✅ Conclusion

Using SciPy’s `chi2_contingency`, we found a **significant association between gender and survival**, meaning **female passengers had a better chance of survival** on the Titanic.

---

