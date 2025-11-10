

import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

print("=" * 80)
print("PROBLEM 1: Marketing Campaign Effectiveness")
print("=" * 80)

# Sample data
campaign_A = [12, 15, 14, 10, 13, 15, 11, 14, 13, 16]
campaign_B = [18, 17, 16, 15, 20, 19, 18, 16, 17, 19]
campaign_C = [10, 9, 11, 10, 12, 9, 11, 8, 10, 9]

# Calculate descriptive statistics first
print("\nDescriptive Statistics:")
print(f"Campaign A: Mean = {np.mean(campaign_A):.2f}, SD = {np.std(campaign_A, ddof=1):.2f}")
print(f"Campaign B: Mean = {np.mean(campaign_B):.2f}, SD = {np.std(campaign_B, ddof=1):.2f}")
print(f"Campaign C: Mean = {np.mean(campaign_C):.2f}, SD = {np.std(campaign_C, ddof=1):.2f}")

# Perform one-way ANOVA
f_stat, p_val = f_oneway(campaign_A, campaign_B, campaign_C)

print(f"\nANOVA Results:")
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_val:.5f}")

# Interpretation
print("\nInterpretation:")
alpha = 0.05
if p_val < alpha:
    print(f"✓ p-value ({p_val:.5f}) < α ({alpha})")
    print("✓ REJECT the null hypothesis")
    print("✓ There IS a statistically significant difference in conversion rates among campaigns")
    print("\nPractical insight:")
    print("  - Campaign B has highest conversion rate (~17.5%)")
    print("  - Campaign C has lowest conversion rate (~9.9%)")
    print("  - Campaign A is in the middle (~13.3%)")
    print("  → Recommendation: Focus resources on Campaign B strategy!")
else:
    print(f"✗ p-value ({p_val:.5f}) >= α ({alpha})")
    print("✗ FAIL TO REJECT the null hypothesis")
    print("✗ No significant difference detected")

print("\n" + "=" * 80)
print("PROBLEM 2: Fertilizer Type vs Crop Yield")
print("=" * 80)

fertilizer_A = [25, 27, 26, 30, 29, 28, 30, 27]
fertilizer_B = [32, 35, 34, 33, 36, 34, 35, 32]
fertilizer_C = [22, 20, 24, 23, 25, 21, 22, 23]

# Descriptive statistics
print("\nDescriptive Statistics:")
print(f"Fertilizer A: Mean = {np.mean(fertilizer_A):.2f} kg, SD = {np.std(fertilizer_A, ddof=1):.2f}")
print(f"Fertilizer B: Mean = {np.mean(fertilizer_B):.2f} kg, SD = {np.std(fertilizer_B, ddof=1):.2f}")
print(f"Fertilizer C: Mean = {np.mean(fertilizer_C):.2f} kg, SD = {np.std(fertilizer_C, ddof=1):.2f}")

# Perform ANOVA
f_stat, p_val = f_oneway(fertilizer_A, fertilizer_B, fertilizer_C)

print(f"\nANOVA Results:")
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_val:.5f}")

# Interpretation
print("\nInterpretation:")
if p_val < alpha:
    print(f"✓ p-value ({p_val:.5f}) < α ({alpha})")
    print("✓ REJECT the null hypothesis")
    print("✓ Fertilizer type DOES significantly affect crop yield")
    print("\nPractical insight:")
    print("  - Fertilizer B: highest yield (~33.9 kg) - BEST performer")
    print("  - Fertilizer A: medium yield (~27.8 kg)")
    print("  - Fertilizer C: lowest yield (~22.5 kg) - WORST performer")
    print("  - Difference between best and worst: ~11.4 kg (50% improvement!)")
    print("  → Recommendation: Use Fertilizer B for maximum yield!")
else:
    print(f"✗ p-value ({p_val:.5f}) >= α ({alpha})")
    print("✗ No significant difference detected")

print("\nNote: ANOVA tells us groups differ, but doesn't tell us WHICH pairs differ.")
print("For detailed pairwise comparisons, we'd need post-hoc tests (e.g., Tukey HSD).")

print("\n" + "=" * 80)
print("PROBLEM 3: Teaching Method vs Student Performance")
print("=" * 80)

data = {
    'score': [78, 85, 80, 90, 88, 82, 84, 86, 92],
    'method': ['Lecture', 'Lecture', 'Lecture',
               'Discussion', 'Discussion', 'Discussion',
               'Project', 'Project', 'Project']
}
df = pd.DataFrame(data)

# Descriptive statistics
print("\nDescriptive Statistics:")
for method in df['method'].unique():
    scores = df[df['method'] == method]['score']
    print(f"{method:12s}: Mean = {scores.mean():.2f}, SD = {scores.std(ddof=1):.2f}, n = {len(scores)}")

# Perform ANOVA using statsmodels
model = ols('score ~ C(method)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nANOVA Table:")
print(anova_table)

# Interpretation
p_value = anova_table['PR(>F)']['C(method)']
f_value = anova_table['F']['C(method)']

print("\nInterpretation:")
if p_value < alpha:
    print(f"✓ p-value ({p_value:.5f}) < α ({alpha})")
    print("✓ REJECT the null hypothesis")
    print("✓ Teaching method DOES have a significant effect on student performance")
    print("\nPractical insight:")
    print("  - Project-based learning: highest scores (~87.3)")
    print("  - Group Discussion: medium scores (~86.7)")
    print("  - Lecture: lowest scores (~81.0)")
    print("  - However, note the small sample size (only 3 students per method)")
    print("  → Recommendation: Consider more interactive teaching methods!")
else:
    print(f"✗ p-value ({p_value:.5f}) >= α ({alpha})")
    print("✗ FAIL TO REJECT the null hypothesis")
    print("✗ No significant effect of teaching method detected")

print("\n" + "=" * 80)
print("PROBLEM 4: Machine Type vs Production Quality")
print("=" * 80)

machine_A = [88, 85, 87, 90, 89, 88, 86, 87]
machine_B = [82, 84, 83, 81, 80, 83, 82, 84]
machine_C = [91, 90, 92, 93, 94, 92, 91, 93]

# Descriptive statistics
print("\nDescriptive Statistics:")
print(f"Machine A: Mean = {np.mean(machine_A):.2f}, SD = {np.std(machine_A, ddof=1):.2f}")
print(f"Machine B: Mean = {np.mean(machine_B):.2f}, SD = {np.std(machine_B, ddof=1):.2f}")
print(f"Machine C: Mean = {np.mean(machine_C):.2f}, SD = {np.std(machine_C, ddof=1):.2f}")

# Perform ANOVA
f_stat, p_val = f_oneway(machine_A, machine_B, machine_C)

print(f"\nANOVA Results:")
print(f"F-statistic: {f_stat:.2f}")
print(f"P-value: {p_val:.5f}")

# Interpretation
print("\nInterpretation:")
if p_val < alpha:
    print(f"✓ p-value ({p_val:.5f}) < α ({alpha})")
    print("✓ REJECT the null hypothesis")
    print("✓ Machine type DOES significantly affect quality scores")
    print("\nPractical insight:")
    print("  - Machine C: highest quality (~92.0) - BEST performance")
    print("  - Machine A: medium quality (~87.5)")
    print("  - Machine B: lowest quality (~82.4) - NEEDS ATTENTION")
    print("  - Quality gap between C and B: ~9.6 points")
    print("  → Recommendations:")
    print("    1. Use Machine C for critical/high-quality products")
    print("    2. Investigate Machine B - maintenance needed?")
    print("    3. Consider upgrading or replacing Machine B")
else:
    print(f"✗ p-value ({p_val:.5f}) >= α ({alpha})")
    print("✗ No significant difference detected")

# Visual comparison
print("\nQuality Score Ranges:")
print(f"Machine A: {min(machine_A)} - {max(machine_A)}")
print(f"Machine B: {min(machine_B)} - {max(machine_B)}")
print(f"Machine C: {min(machine_C)} - {max(machine_C)}")

print("\n" + "=" * 80)
print("PROBLEM 5: Diet Plan vs Weight Loss")
print("=" * 80)

data = {
    'diet': ['Keto']*6 + ['Paleo']*6 + ['Vegan']*6 + ['Mediterranean']*6,
    'weight_loss': [6, 5, 7, 8, 6, 7,  # Keto
                    5, 6, 4, 5, 6, 5,  # Paleo
                    3, 4, 2, 4, 3, 2,  # Vegan
                    5, 6, 7, 6, 8, 7]  # Mediterranean
}
df_diet = pd.DataFrame(data)

# Descriptive statistics
print("\nDescriptive Statistics:")
for diet in df_diet['diet'].unique():
    weight_loss = df_diet[df_diet['diet'] == diet]['weight_loss']
    print(f"{diet:15s}: Mean = {weight_loss.mean():.2f} kg, SD = {weight_loss.std(ddof=1):.2f}, n = {len(weight_loss)}")

# Perform ANOVA using statsmodels
model = ols('weight_loss ~ C(diet)', data=df_diet).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

print("\nANOVA Table:")
print(anova_table)

# Interpretation
p_value = anova_table['PR(>F)']['C(diet)']
f_value = anova_table['F']['C(diet)']

print("\nInterpretation:")
if p_value < alpha:
    print(f"✓ p-value ({p_value:.5f}) < α ({alpha})")
    print("✓ REJECT the null hypothesis")
    print("✓ Diet type DOES significantly affect weight loss")
    print("\nPractical insight:")
    # Calculate means for each diet
    keto_mean = df_diet[df_diet['diet'] == 'Keto']['weight_loss'].mean()
    paleo_mean = df_diet[df_diet['diet'] == 'Paleo']['weight_loss'].mean()
    vegan_mean = df_diet[df_diet['diet'] == 'Vegan']['weight_loss'].mean()
    med_mean = df_diet[df_diet['diet'] == 'Mediterranean']['weight_loss'].mean()
    
    print(f"  - Keto: {keto_mean:.2f} kg average loss")
    print(f"  - Mediterranean: {med_mean:.2f} kg average loss")
    print(f"  - Paleo: {paleo_mean:.2f} kg average loss")
    print(f"  - Vegan: {vegan_mean:.2f} kg average loss")
    print("\n  → Ranking: Keto ≈ Mediterranean > Paleo > Vegan")
    print("  → Recommendation: Keto or Mediterranean diets for maximum weight loss")
    print("  → Note: Vegan diet shows significantly lower weight loss (~3 kg)")
else:
    print(f"✗ p-value ({p_value:.5f}) >= α ({alpha})")
    print("✗ FAIL TO REJECT the null hypothesis")
    print("✗ No significant difference in weight loss among diets")

print("\n" + "=" * 80)
print("SUMMARY OF ALL TESTS")
print("=" * 80)

print("\nResults Overview:")
print("-" * 80)
print(f"{'Problem':<30} {'Significant?':<15} {'Decision':<20}")
print("-" * 80)
print(f"{'1. Marketing Campaigns':<30} {'YES':<15} {'Use Campaign B':<20}")
print(f"{'2. Fertilizer Types':<30} {'YES':<15} {'Use Fertilizer B':<20}")
print(f"{'3. Teaching Methods':<30} {'CHECK P-VALUE':<15} {'Results vary':<20}")
print(f"{'4. Machine Quality':<30} {'YES':<15} {'Use Machine C':<20}")
print(f"{'5. Diet Plans':<30} {'YES':<15} {'Keto/Mediterranean':<20}")
print("-" * 80)

print("\nKey Learning Points:")
print("1. ANOVA tells us IF groups differ (but not WHICH specific pairs)")
print("2. Always check both F-statistic (size of effect) and p-value (significance)")
print("3. Look at descriptive statistics to understand practical importance")
print("4. Small p-value ≠ large practical difference (could be small but consistent)")
print("5. For pairwise comparisons, need post-hoc tests (Tukey HSD, Bonferroni, etc.)")

print("\n" + "=" * 80)
print("INTERPRETATION GUIDE")
print("=" * 80)

print("""
How to Interpret ANOVA Results:

1. CHECK P-VALUE:
   - p < 0.05 → Significant difference exists
   - p ≥ 0.05 → No significant difference detected

2. CHECK F-STATISTIC:
   - Larger F = stronger evidence of difference
   - F close to 1 = groups are similar
   - F >> 1 = groups differ substantially

3. EXAMINE DESCRIPTIVE STATISTICS:
   - Compare group means
   - Check standard deviations
   - Look for practical significance (not just statistical)

4. CONSIDER CONTEXT:
   - Sample size matters
   - Small samples → less power to detect differences
   - Large samples → may detect trivial differences

5. NEXT STEPS:
   - If significant → do post-hoc tests to find which pairs differ
   - Calculate effect sizes (Cohen's d, eta-squared)
   - Visualize data with boxplots or bar charts
   - Make practical recommendations based on findings
""")