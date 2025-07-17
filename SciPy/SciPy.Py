import pandas as pd
from scipy.stats import chi2_contingency

# Load dataset
df = pd.read_csv("tested.csv")

# Create a contingency table for Survival and Sex
contingency_table = pd.crosstab(df['Sex'], df['Survived'])

# Perform Chi-Square Test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print results
print("Contingency Table:\n", contingency_table)
print("\nExpected Frequencies:\n", expected)
print("\nChi-Square Statistic:", chi2)
print("Degrees of Freedom:", dof)
print("P-value:", p)

# Interpretation
if p < 0.05:
    print("\nConclusion: Survival and Gender are dependent (statistically significant).")
else:
    print("\nConclusion: No significant association between Survival and Gender.")
