

# 🧠 CIRF Data Analysis Internship – Python Library Exploration

**Internship Organization:** Computational Intelligence Research Foundation (CIRF)  
**Domain:** Data Analysis  
**Task:** Explore multiple Python libraries using the same dataset (`tested.csv`) on Google Colab  

---

## 🗂 Project Overview

During my internship at **CIRF**, I explored **eight key Python libraries** for data analysis and machine learning, all focused on the same dataset:

1. **Seaborn** – Statistical plotting  
2. **Matplotlib** – Core plotting  
3. **Pandas** – Data manipulation  
4. **Scikit-learn** – Traditional machine learning  
5. **Keras** – Neural networks (high-level)  
6. **NumPy** – Numerical computing  
7. **SciPy** – Scientific/statistics  
8. **TensorFlow** – Deep learning framework  

Each folder in this repo contains a Jupyter/Colab notebook with:
- Code
- Line-by-line explanations
- README for that library

---

## 📘 Libraries at a Glance

### 1. **Seaborn**
- **Description:** Statistical data visualization built on Matplotlib  
- **Syntax Example:** `sns.countplot(data=df, x='Sex', hue='Survived')`  
- **Use When:** Quick, attractive statistical plots (count, histogram, heatmap)  
- **Purpose:** Visual exploratory data analysis  
- **Applications:** EDA, data storytelling, correlation insights  

---
### 2. **Matplotlib**
- **Description:** Low-level foundational plotting library  
- **Syntax Example:**  
  ```python
  plt.subplot(2,2,1)
  plt.bar(['No', 'Yes'], df['Survived'].value_counts())
  ```

---

### 3. **Pandas**

* **Description:** Data loading and manipulation with DataFrames
* **Syntax Example:** `df.groupby('Sex')['Survived'].value_counts().unstack()`
* **Use When:** Cleaning, filtering, aggregating data
* **Purpose:** Prepping data for analysis or model input
* **Applications:** Data wrangling, quick statistical summaries

---

### 4. **Scikit-learn**

* **Description:** Traditional ML library (classic algorithms)
* **Syntax Example:**

  ```python
  model = LogisticRegression()
  model.fit(X_train, y_train)
  ```
* **Use When:** Quick model building with preprocessing
* **Purpose:** Train/test split, algorithms, evaluation
* **Applications:** Classification, regression, model pipelines

---

### 5. **Keras**

* **Description:** High‑level neural network API (using TensorFlow backend)
* **Syntax Example:**

  ```python
  model = Sequential([
      Dense(16, activation='relu', input_dim=5),
      Dense(1, activation='sigmoid')
  ])
  ```
* **Use When:** Prototype deep-learning models quickly
* **Purpose:** Build and train neural networks simply
* **Applications:** Binary and multiclass classification tasks

---

### 6. **NumPy**

* **Description:** Core numerical library for arrays and matrices
* **Syntax Example:**

  ```python
  mean_age = np.mean(df['Age'].to_numpy())
  ```
* **Use When:** Fast numerical computation
* **Purpose:** Efficient statistics and array operations
* **Applications:** Feature engineering, performance-critical tasks

---

### 7. **SciPy**

* **Description:** Scientific computing on top of NumPy
* **Syntax Example:**

  ```python
  chi2, p, _, _ = chi2_contingency(contingency_table)
  ```
* **Use When:** Statistical hypothesis testing or signal processing
* **Purpose:** Perform scientific, statistical methods
* **Applications:** Tests, distributions, optimization, integration

---

### 8. **TensorFlow**

* **Description:** Deep learning framework designed for scale
* **Syntax Example:**

  ```python
  model = tf.keras.Sequential([...])
  model.compile(...)
  ```
* **Use When:** Deployable, complex deep neural networks at scale
* **Purpose:** Train, evaluate, and export models
* **Applications:** Deep learning pipelines, production-grade ML

---

## 🧪 Workflow Summary

1. **Dataset Preparation** (Pandas + NumPy)
2. **Exploratory Analysis** (Seaborn, Matplotlib, SciPy)
3. **Model Building**

   * Traditional ML: Scikit-learn
   * Neural Networks: Keras, TensorFlow

---

## 🚀 What Else Can We Explore?

* **Plotly / Bokeh** – Interactive visualizations
* **Statsmodels** – Statistical modeling (regression analysis)
* **XGBoost / LightGBM** – Gradient Boosted Trees
* **NLTK / SpaCy** – Text/NLP processing (if textual data available)
* **CatBoost / PyMC** – Bayesian and categorical data models

---

## 📂 Folder Structure
```

CIRF_Data-Analysis-Internship/
│
├── tested.csv                  # Common dataset used across all libraries
├── Data Visualization/
│   ├── Seaborn/                # Seaborn code & README
│   └── Matplotlib/             # Matplotlib code & README
├── Pandas/                     # Pandas code & README
├── NumPy/                      # NumPy code & README
├── Scikit-learn/               # Scikit-learn code & README
├── TensorFlow/                 # TensorFlow code & README
├── Keras/                      # Keras code & README
├── SciPy/                      # SciPy code & README
└── README.md                   # This file (overall summary)
```

---

## 🔧 How to Run

1. Open any folder in **Google Colab**
2. Upload or mount `tested.csv`
3. Run the notebook
4. Check the folder-specific README for detailed instructions

---

## 🙌 Acknowledgments

Thanks to **CIRF** for the opportunity and guidance during this internship. Through this project, I learned how to leverage a versatile Python ecosystem to analyze the same dataset using different techniques—ranging from statistics to deep learning.

---

## 📬 Next Steps

* Add more advanced libraries like **XGBoost**, **Plotly**, **Statsmodels**
* Compare model performances side-by-side
* Containerize notebooks into reproducible pipelines

---

**Prepared by:** Hemalatha. A


---

