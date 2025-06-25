import os
import json
import matplotlib.pyplot as plt

# Load JSON data
with open("./outputs/distributions.json", "r") as f:
    data = json.load(f)

# Create output directory
os.makedirs("histo", exist_ok=True)

# Levels to consider
levels = ["N1", "N2", "N3", "N4"]  # You can adjust if needed

# Iterate over levels and their nested features
for level in levels:
    if level not in data:
        continue
    for granularity in data[level]:
        for feature, values in data[level][granularity].items():
            plt.figure()
            plt.hist(values, bins=20, edgecolor='black')
            plt.title(f"{level} - {granularity} - {feature}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            fname = f"./histo/{level}_{granularity}_{feature}.png"
            plt.savefig(fname)
            plt.close()