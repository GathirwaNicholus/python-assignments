#please refer to or see pmf_lab.ipynb as I have explanations (inline) alongside the notes and 
#some more visualizations, this is like a compressed summary.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Initial data
size_and_count = {17: 10, 22: 10, 27: 18, 32: 6, 37: 8, 42: 10, 47: 5, 52: 3, 57: 4}

# Step 1: Calculate total and create actual PMF
sum_class = sum(size_and_count.values())
sizes = pd.Series(list(size_and_count.keys()))
actual_pmf = pd.Series([count/sum_class for count in size_and_count.values()])

# Step 2: Create PMF functions
def p_actual(x_i):
    return size_and_count.get(x_i, 0) / sum_class

def p_perceived(x_i):
    return p_actual(x_i) * x_i / biased.sum() if x_i in size_and_count else 0

# Step 3: Calculate means
mu = sum(sizes * actual_pmf)  # ~32.47
biased = sizes.apply(p_actual) * sizes
biased_pmf = biased / biased.sum()
mu_biased = sum(sizes * biased_pmf)  # ~36.51

# Step 4: Create comparison dataframe
pmf_df = pd.concat([sizes, actual_pmf, biased_pmf], axis=1)
pmf_df.columns = ["Class Size", "Overall Probability", "Perceived Probability"]

print(f"Actual mean class size: {mu:.2f}")
print(f"Perceived mean class size: {mu_biased:.2f}")
print(f"Difference: {mu_biased - mu:.2f} students")

# Setting up shared axes
fig, ax = plt.subplots()

# Your code here
#Basically, Overlapping bars for clear comparison
pmf_df.plot.bar(x="Class Size", y="Overall Probability", 
                ax=ax, alpha=0.7, color="tab:red", label="Actual")
pmf_df.plot.bar(x="Class Size", y="Perceived Probability", 
                ax=ax, alpha=0.7, color="tab:blue", label="Perceived")
plt.title("Class Size Paradox: Overlapping Distributions")
plt.ylabel("Probability")
plt.legend()
plt.show()