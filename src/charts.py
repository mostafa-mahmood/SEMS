import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.pie([98740, 24685], labels=['Training (98,740)', 'Testing (24,685)'], autopct='%1.1f%%')
plt.title('Training vs. Testing Split')
plt.savefig('./../reports/split_pie_chart.png')
plt.close()

import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Simulate a more realistic 'before scaling' distribution for Value_lag_48
# Use log-normal to create a right-skewed distribution, common for energy data
before = np.random.lognormal(mean=4.5, sigma=0.6, size=1000)  # Mean ~90 kWh, skewed
# Add some noise and variability to mimic real data
before = before + np.random.normal(0, 10, 1000)
# Shift to include some negative values (as seen in your original plot)
before = before - 50
# Clip to mimic outlier capping (as done in splitting_and_scaling.py)
before = np.clip(before, -50, 300)

# Apply scaling to match site-based normalization (mean=0, std=1)
after = (before - before.mean()) / before.std()

# Create the visualization
plt.figure(figsize=(10, 6))

# Before scaling histogram
plt.subplot(1, 2, 1)
plt.hist(before, bins=30, color='blue', alpha=0.7)
plt.title('Before Scaling (Value_lag_48)')
plt.xlabel('kWh')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# After scaling histogram
plt.subplot(1, 2, 2)
plt.hist(after, bins=30, color='green', alpha=0.7)
plt.title('After Scaling')
plt.xlabel('Standardized Units')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./../reports/scaling_example_fixed.png')
plt.close()