# Public sanitized release of analysis code
# Notes:
# - Personal file paths have been removed or replaced with basenames.
# - Do NOT hardcode tokens. Optionally set GH_TOKEN and GEMINI_API_KEY as environment variables if you enable those features.
# - The Gemini-powered affiliation inference is DISABLED unless GEMINI_API_KEY is set; otherwise it returns "unknown".
# - For reproducibility, consider pinning package versions in requirements.txt or pip installs.
# - This file is auto-sanitized for public sharing.

# -*- coding: utf-8 -*-
"""SupervisionParsimonious.ipynb



### **Description for Google Colab Notebook**

This code **loads, processes, and filters** a dataset of foundation models from a CSV file (`assets.csv`). It prepares the data for analysis by performing the following steps:

1. **Load the dataset** into a Pandas DataFrame.
2. **Check and display column names** to ensure correct structure.
3. **Convert dates** from the `created_date` column into a `datetime` format and extract the release year.
4. **Filter the dataset** to include only rows where `type == "model"` (excluding applications or other entries).
5. **Parse model sizes** from strings like `"7B parameters (dense)"` into numerical values (e.g., `7e9` for 7 billion parameters).
6. **Display dataset summaries**, including:
   - Extracted model sizes in numerical format.
   - The date range of models in the dataset.
   - A sample of the first few rows.

This script **ensures data consistency** and **prepares `model_df`** for further trend analysis and visualization. 🚀
"""

import pandas as pd
import numpy as np

# 1. Load data
df = pd.read_csv("assets.csv")

# 2. Check the column names
print("Columns:", df.columns.tolist())

# 3. Convert date column (named 'created_date' in your sample) to datetime
if "created_date" in df.columns:
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["year"] = df["created_date"].dt.year
else:
    print("No 'created_date' column found in the dataframe.")

# 4. Filter down to just the models (exclude rows labeled "application", etc.)
model_df = df[df["type"] == "model"].copy()
print(f"\nFiltered to {len(model_df)} rows with type='model'.")

# 5. Example: parse the model size column (named 'size')
def parse_size(size_str):
    """
    Convert strings like '7B parameters (dense)' or '82B parameters'
    into a numerical parameter count (e.g. 7e9, 82e9).
    Return np.nan if unknown or unparsable.
    """
    if pd.isna(size_str):
        return np.nan

    # If the string is 'unknown', return NaN
    lowered = size_str.lower()
    if "unknown" in lowered:
        return np.nan

    # Extract the numeric portion
    # e.g. '7B' or '82B'
    # This simple approach will break if the string is in a very different format,
    # so you can refine it as needed.
    parts = lowered.split()
    # "7b" or "82b" might appear in the first segment
    candidate = parts[0]

    # We handle B (billion) and M (million) as examples
    # if there's something like (dense) appended, we can strip parentheses
    candidate = candidate.replace("(dense)", "").replace("parameters", "").strip()

    # Now, check if it ends with 'b' or 'm'
    if "b" in candidate:
        # e.g. "7b"
        numeric_part = candidate.replace("b", "")
        try:
            val = float(numeric_part)
            return val * 1e9
        except ValueError:
            return np.nan
    elif "m" in candidate:
        # e.g. "770m"
        numeric_part = candidate.replace("m", "")
        try:
            val = float(numeric_part)
            return val * 1e6
        except ValueError:
            return np.nan
    else:
        # If we can't parse it, just return NaN
        return np.nan

model_df["parsed_params"] = model_df["size"].apply(parse_size)

# 6. Quick inspection
print("\nParsed model sizes:")
print(model_df[["name", "size", "parsed_params"]])

print("\nDate range in model_df:")
print(model_df["created_date"].describe())

print("\nSample rows:")
print(model_df.head())

"""### **Code Summary for Google Colab Notebook**

This code analyzes the growth and transparency trends of foundation models using **pandas** and **matplotlib**. It performs the following key tasks:

1. **Model Growth Analysis**  
   - Counts models released per year and visualizes both annual and cumulative trends.

2. **Model Size Trends**  
   - Extracts parameter sizes, computes yearly statistics (mean, median, std), and plots growth on a **log scale**.

3. **Modality Trends**  
   - Parses the **modality** column to count unique modalities per model, tracking the shift toward multimodal models.

4. **Transparency & Documentation**  
   - Checks how many models report **training emissions, training time, and hardware**, showing trends over time.

5. **Licensing & Access Trends**  
   - Classifies models as **open, closed, or unknown**, then tracks licensing trends annually with **stacked bar charts**.

### **Usage**  
- Figures are generated to **visualize trends** in model growth, documentation transparency, and licensing evolution.  
- The dataset can be further **refined** with additional filtering, statistical tests, or correlation analyses.  
- Modify parsing logic for **modality and licensing** as needed based on dataset specifics.  

This provides a **concise, structured overview** of how foundation models evolve and how their transparency varies over time. 🚀
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# We'll assume you're starting with the DataFrame `model_df` as shown in your sample:
# model_df has columns like ['name','organization','created_date','size','license','access','training_emissions','training_time','training_hardware',...]
# and a numeric 'parsed_params' column from your earlier parsing step.

# 1. BASIC YEARLY COUNTS ------------------------------------------------------

# Count models released each year
yearly_counts = model_df.groupby('year').size().sort_index()

plt.figure(figsize=(6,4))
plt.plot(yearly_counts.index, yearly_counts.values, marker='o')
plt.title('Number of Models Released per Year')
plt.xlabel('Release Year')
plt.ylabel('Count of Models')
plt.grid(True)
plt.show()

# Cumulative sum over years
cumulative_counts = yearly_counts.cumsum()
plt.figure(figsize=(6,4))
plt.plot(cumulative_counts.index, cumulative_counts.values, marker='o')
plt.title('Cumulative Number of Models Over Time')
plt.xlabel('Release Year')
plt.ylabel('Cumulative Model Count')
plt.grid(True)
plt.show()

print("\nYearly counts:\n", yearly_counts)
print("\nCumulative counts:\n", cumulative_counts)

# 2. MODEL SIZE TRENDS -------------------------------------------------------

# Check distribution of parsed parameter counts
# We'll drop NaN sizes for this analysis
size_df = model_df.dropna(subset=['parsed_params'])

# Basic stats per year
size_stats = size_df.groupby('year')['parsed_params'].agg(['count','mean','median','std'])
print("\nModel Size Stats by Year:\n", size_stats)

# Plot average and median model size by year on a log scale
plt.figure(figsize=(6,4))
plt.plot(size_stats.index, size_stats['mean'], marker='o', label='Mean')
plt.plot(size_stats.index, size_stats['median'], marker='s', label='Median')
plt.yscale('log')  # Log scale for large size differences
plt.title('Model Size Over Time (log scale)')
plt.xlabel('Release Year')
plt.ylabel('Parameters (log scale)')
plt.legend()
plt.grid(True)
plt.show()

# 3. MODALITY ANALYSIS -------------------------------------------------------

# Suppose we want to count how many modalities each model has.
# The 'modality' column might look like "text; image" or "image, text; text"
# We'll split on certain delimiters and count unique entries.

def count_modalities(mod_str):
    if pd.isna(mod_str):
        return 0
    # Split on commas and semicolons
    # collect unique tokens so we don't double-count repeated 'text'
    tokens = [t.strip().lower() for chunk in mod_str.split(';') for t in chunk.split(',')]
    unique_modalities = set(tokens)
    # remove empty strings just in case
    unique_modalities.discard('')
    return len(unique_modalities)

model_df['num_modalities'] = model_df['modality'].apply(count_modalities)

# Group by year, average number of modalities
modality_stats = model_df.groupby('year')['num_modalities'].agg(['count','mean','median'])
print("\nModality Stats by Year:\n", modality_stats)

# Plot average modalities by year
plt.figure(figsize=(6,4))
plt.plot(modality_stats.index, modality_stats['mean'], marker='o')
plt.title('Average Number of Modalities per Model')
plt.xlabel('Release Year')
plt.ylabel('Mean # of Modalities')
plt.grid(True)
plt.show()

# 4. TRANSPARENCY & DOCUMENTATION EXAMPLES ------------------------------------

# We'll check how many models have known vs. unknown for training_emissions, training_time, training_hardware.

def is_known(value):
    # Mark True if not "unknown" and not NaN
    if pd.isna(value):
        return False
    return not str(value).lower().startswith('unknown')

model_df['has_emissions'] = model_df['training_emissions'].apply(is_known)
model_df['has_time'] = model_df['training_time'].apply(is_known)
model_df['has_hardware'] = model_df['training_hardware'].apply(is_known)

# Fraction of models reporting each metric by year
transparency_by_year = (model_df
    .groupby('year')[['has_emissions','has_time','has_hardware']]
    .mean()  # fraction that are True
)
print("\nFraction of Models Providing Training Info (by year):\n", transparency_by_year)

# Plot the fractions
plt.figure(figsize=(6,4))
for col in ['has_emissions','has_time','has_hardware']:
    plt.plot(transparency_by_year.index, transparency_by_year[col], marker='o', label=col)

plt.ylim(0, 1.05)
plt.title('Fraction of Models Providing Key Training Details')
plt.xlabel('Release Year')
plt.ylabel('Fraction of Models')
plt.legend()
plt.grid(True)
plt.show()

# 5. LICENSING / ACCESS TRENDS -----------------------------------------------

# For a simple open vs. closed classification, define your own logic:
# e.g. "Apache 2.0", "MIT", "BSD" => open, anything else => closed
# Or check "access" column if it says "open" or "closed" or "limited".

def classify_license(lic_str):
    if pd.isna(lic_str):
        return "unknown"
    lic_str_lower = lic_str.lower()
    # Very naive rules
    if "apache" in lic_str_lower or "mit" in lic_str_lower or "bsd" in lic_str_lower:
        return "open"
    # If it explicitly says "llama 2" (some variants are considered community license)
    # adjust logic as needed, or treat as "restricted" etc.
    if "llama 2" in lic_str_lower:
        return "open-ish"  # example category
    if "unknown" in lic_str_lower:
        return "unknown"
    return "closed"

model_df['license_type'] = model_df['license'].apply(classify_license)

# By year, fraction of open vs closed
license_count_by_year = (model_df
    .groupby(['year','license_type'])
    .size()
    .reset_index(name='count')
)

print("\nLicense classification by year:\n", license_count_by_year.head())

# Pivot so we can plot a stacked or grouped bar
license_pivot = license_count_by_year.pivot(index='year', columns='license_type', values='count').fillna(0)

plt.figure(figsize=(6,4))
bottom_val = np.zeros(len(license_pivot.index))
for col in license_pivot.columns:
    plt.bar(license_pivot.index, license_pivot[col], bottom=bottom_val, label=col)
    bottom_val += license_pivot[col].values

plt.title('License Types by Year')
plt.xlabel('Year')
plt.ylabel('Number of Models')
plt.legend()
plt.grid(True, axis='y')
plt.show()

# 6. SAVE YOUR FIGURES OR KEEP EXPLORING --------------------------------------
# If desired, you can save your plots:
# plt.savefig("my_figure.png", dpi=300)

# 7. SUMMARY -----------------------------------------------------------------

print("Data analysis complete.\n")
print("You can now refine these plots, or add further metrics—for example, analyzing how model size correlates with transparency, or how multiple columns interact.")

"""### **Description for Google Colab Notebook**
This code **visualizes the growth of foundation models** by plotting:
1. **Annual model releases** (🔵 Blue line, left y-axis).
2. **Cumulative model count** over time (⚫ Black line, right y-axis).

### **Key Features:**
✅ Uses **two y-axes** for clear comparison.  
✅ **Removes unnecessary gridlines** for a clean, publication-ready look.  
✅ **Positions the legend above the chart** to avoid clutter.  
✅ **Saves the figure locally** to `annual_cumulative_models_final.png`.  

This plot provides a **concise and professional visualization** of how model releases and cumulative growth have evolved over time. 🚀
"""

import matplotlib.pyplot as plt

# Ensure data is sorted correctly
yearly_counts = model_df.groupby('year').size().sort_index()
cumulative_counts = yearly_counts.cumsum()

# Create a professional plot with both annual and cumulative model counts
fig, ax1 = plt.subplots(figsize=(6,4))

# Plot models released per year (left y-axis)
ax1.plot(yearly_counts.index, yearly_counts.values, marker='o', linestyle='-',
         linewidth=2, markersize=6, color='blue', label="Annual Releases")
ax1.set_xlabel('Release Year', fontsize=11)
ax1.set_ylabel('Count of Models', fontsize=11, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis for cumulative count (right y-axis)
ax2 = ax1.twinx()
ax2.plot(cumulative_counts.index, cumulative_counts.values, marker='o', linestyle='-',
         linewidth=2, markersize=6, color='red', label="Cumulative Count")
ax2.set_ylabel('Cumulative Model Count', fontsize=11, color='red')
ax2.tick_params(axis='y', labelcolor='black')

# Titles and styling
#plt.title('Annual and Cumulative Model Releases', fontsize=12, fontweight='bold')

# Remove grid and unnecessary borders
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Combine both legends and place them outside the plot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)

# Save the improved professional plot to your local system
plt.savefig("annual_cumulative_models_final.png", dpi=300, bbox_inches='tight')
plt.show()

"""**Concise Description for Google Colab**

This code **plots all model sizes** by year (with jitter to avoid overlapping points) and overlays **three lines** showing the **max, mean, and median** parameter counts per year on a **log scale**. It provides a **comprehensive** view of the distribution of model sizes along with key summary statistics in a **clean, publication-ready** format.
"""

import numpy as np
import matplotlib.pyplot as plt

# 1. Filter out rows with missing parameter sizes
size_df = model_df.dropna(subset=["parsed_params"]).copy()

# 2. Sort unique years for consistent plotting
years = sorted(size_df["year"].unique())

# Prepare lists for storing yearly stats
mean_values = []
median_values = []
max_values = []

# 3. Create the figure
fig, ax = plt.subplots(figsize=(6,4))

# Plot scatter with jitter for each year
for year in years:
    data_for_year = size_df.loc[size_df["year"] == year, "parsed_params"].values

    # Compute yearly statistics
    mean_values.append(np.mean(data_for_year))
    median_values.append(np.median(data_for_year))
    max_values.append(np.max(data_for_year))

    # Add small random offsets for jitter
    offsets = (np.random.rand(len(data_for_year)) - 0.5) * 0.4
    xvals = [year + off for off in offsets]

    # Scatter plot of individual points (gray, semi-transparent)
    ax.scatter(xvals, data_for_year, color="gray", alpha=0.5, s=20)

# 4. Plot lines for max (red), mean (blue), and median (green)
ax.plot(years, max_values,    marker='o', linewidth=2, color='red',   label='Max')
ax.plot(years, mean_values,   marker='o', linewidth=2, color='blue',  label='Mean')
ax.plot(years, median_values, marker='o', linewidth=2, color='green', label='Median')

# 5. Log scale and styling
ax.set_yscale("log")
ax.set_xlabel("Release Year", fontsize=11)
ax.set_ylabel("Parameters (log scale)", fontsize=11)
ax.spines["top"].set_visible(False)  # remove top border
#ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 6. Legend placement
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)


# 7. Save and show
plt.savefig("model_size_scatter_mean_median_max.png",
            dpi=300, bbox_inches='tight')
plt.show()

"""### **Concise Description for Google Colab Notebook**

This code normalizes manufacturer names from the dataset, calculates the annual number of unique manufacturers as well as the cumulative count over time, and plots both metrics on a single chart. This visualization reveals the growth trends in the number of distinct organizations releasing models each year.r.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Filter for models (if not already filtered)
model_df = model_df[model_df["type"] == "model"].copy()

# 2. Normalize organization names (lowercase and strip extra spaces)
model_df["org_norm"] = model_df["organization"].str.lower().str.strip()

# 3. Compute the number of unique manufacturers per year (annual count)
yearly_unique_org = model_df.groupby("year")["org_norm"].nunique().reset_index(name="unique_manufacturers")

# 4. Compute cumulative unique manufacturers over time
cumulative_unique = []
cumulative_set = set()
for year in sorted(yearly_unique_org["year"].unique()):
    orgs = model_df.loc[model_df["year"] == year, "org_norm"].unique()
    cumulative_set.update(orgs)
    cumulative_unique.append({"year": year, "cumulative_unique": len(cumulative_set)})
cumulative_df = pd.DataFrame(cumulative_unique)

# 5. Plot the annual and cumulative unique manufacturers
fig, ax = plt.subplots(figsize=(6,4))

ax.plot(yearly_unique_org["year"], yearly_unique_org["unique_manufacturers"],
        marker='o', linewidth=2, color='blue', label="Unique Manufacturers per Year")
ax.plot(cumulative_df["year"], cumulative_df["cumulative_unique"],
        marker='o', linewidth=2, color='red', label="Cumulative Unique Manufacturers")

ax.set_xlabel("Release Year", fontsize=11)
ax.set_ylabel("Number of Manufacturers", fontsize=11)
#ax.set_title("Growth of Unique Manufacturers Over Time", fontsize=12, fontweight='bold')
#ax.grid(True, linestyle='--', alpha=0.7)
ax.spines["top"].set_visible(False)
ax.legend(loc="upper left", frameon=False)



plt.savefig("manufacturers_growth.png", dpi=300, bbox_inches='tight')
plt.show()

"""### **Concise Description (for Colab Notebook)**

This code calculates the **fraction** of models that provide each of four key documentation metrics—**training emissions**, **training time**, **training hardware**, and **model card**—by release year. It then plots these fractions in a **minimalist** style without grid lines, with an **unobtrusive legend** at the bottom, offering a **clean, publication-ready** visualization of how documentation practices have evolved over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Columns we want to analyze for documentation metrics
doc_columns = ["training_emissions", "training_time", "training_hardware", "model_card"]

# Helper function to check if a value is known (not "unknown" or NaN)
def is_known(value):
    if pd.isna(value):
        return False
    return not str(value).strip().lower().startswith("unknown")

# Group by year and compute fraction reporting each documentation metric
doc_summary = model_df.groupby("year").apply(
    lambda group: pd.Series({col: group[col].apply(is_known).mean() for col in doc_columns})
).reset_index()

# Helper function to check if model size is reported (non-empty and not "unknown")
def is_reported_size(value):
    if pd.isna(value):
        return False
    return str(value).strip() != "" and not str(value).strip().lower().startswith("unknown")

# Group by year and compute fraction reporting model size
size_report = model_df.groupby("year")["size"].apply(
    lambda group: group.apply(is_reported_size).mean()
).reset_index(name="model_size_reporting")

# Plot settings
fig, ax = plt.subplots(figsize=(6,4))

# Define colors and labels for documentation metrics
colors = {
    "training_emissions": "blue",
    "training_time": "red",
    "training_hardware": "green",
    "model_card": "purple"
}
labels = {
    "training_emissions": "Training Emissions",
    "training_time": "Training Time",
    "training_hardware": "Training Hardware",
    "model_card": "Model Card"
}

# Plot each documentation metric
for col in doc_columns:
    ax.plot(doc_summary["year"], doc_summary[col],
            marker='o', linewidth=2, color=colors[col], label=labels[col])

# Plot model size reporting fraction (using orange)
ax.plot(size_report["year"], size_report["model_size_reporting"],
        marker='o', linewidth=2, color="orange", label="Model Size Reporting")

# Axis labels and formatting
ax.set_xlabel("Release Year", fontsize=11)
ax.set_ylabel("Fraction of Models Reporting", fontsize=11)
ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Place the legend below the plot
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

plt.tight_layout()
plt.savefig("training_details_model_card_size.png", dpi=300, bbox_inches='tight')
plt.show()

"""### **Google Colab Notebook Description**

This notebook visualizes the evolution of foundation models’ licensing trends over time. The code converts raw license counts (from a precomputed DataFrame `license_pivot` with columns ["open", "open-ish", "closed", "unknown"]) into fractions so that each year's totals sum to 1. It then creates a 100% stacked bar chart using the following color scheme:

- **Green** for "open"
- **Blue** for "open-ish"
- **Red** for "closed"
- **Grey** for "unknown"

The plot includes clean styling, a legend positioned below the chart, and is saved as a PNG file for publication-ready visualization.
"""

import requests
import json
import time
import pandas as pd
import numpy as np

#############################
# CONFIG
#############################
LIGHT_DEBUG = True          # Toggle to see minimal debug prints
use_gemini = True           # Set True to call Gemini for unknown licenses

import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENDPOINT = (f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else None)
headers = {"Content-Type": "application/json"}
gemini_cache = {}           # Cache repeated calls
max_retries = 3             # Max Gemini call retries
backoff_initial = 10        # Seconds of initial backoff for 429

#############################
# GEMINI CALL WITH GROUNDING
#############################
def gemini_classification(prompt_text, fallback_str):
    """
    Calls Gemini with Google Search as a tool,
    returning the raw text (lowercased).
    If call fails or times out, returns "unknown".
    Minimal debug prints for clarity.
    """
        if not (globals().get("GEMINI_ENDPOINT")):
        if "LIGHT_DEBUG" in globals() and LIGHT_DEBUG:
            print("[DEBUG] Gemini disabled (no API key)")
        return "unknown"
if not fallback_str or pd.isna(fallback_str):
        if LIGHT_DEBUG:
            print("[DEBUG] Skip Gemini: empty/NaN license.")
        return "unknown"
    lic_str_clean = str(fallback_str).strip().lower()
    if len(lic_str_clean) > 100 or "terms of use" in lic_str_clean:
        if LIGHT_DEBUG:
            print("[DEBUG] Skip Gemini: text too long or 'terms of use'")
        return "unknown"

    # Check cache
    if lic_str_clean in gemini_cache:
        if LIGHT_DEBUG:
            print(f"[DEBUG] Gemini cache hit for: {lic_str_clean[:50]} -> {gemini_cache[lic_str_clean]}")
        return gemini_cache[lic_str_clean]

    # Build the payload with the google_search tool
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
        "tools": [{"google_search": {}}]
    }

    retries = 0
    backoff = backoff_initial
    while retries < max_retries:
        try:
            resp = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=15)
            if LIGHT_DEBUG:
                print(f"[DEBUG] Gemini status={resp.status_code} for '{fallback_str[:50]}...'")
            if resp.status_code == 200:
                data = resp.json()
                raw_ans = data["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
                gemini_cache[lic_str_clean] = raw_ans
                return raw_ans
            elif resp.status_code == 429:
                print(f"[DEBUG] 429 rate-limit. Retrying in {backoff}s.")
                time.sleep(backoff)
                backoff *= 2
                retries += 1
            else:
                print(f"[DEBUG] Gemini error code={resp.status_code} for '{fallback_str[:50]}...'")
                return "unknown"
        except Exception as e:
            print(f"[DEBUG] Exception in Gemini call: {e}")
            return "unknown"

    print("[DEBUG] Max retries exceeded. Returning 'unknown'.")
    return "unknown"

#############################
# NAIVE CLASSIFIERS
#############################
def naive_license_classification(lic_str):
    """Classify license text: open, open-ish, closed, unknown."""
    if pd.isna(lic_str):
        return "unknown"
    txt = lic_str.lower().strip()
    if any(x in txt for x in ["apache", "mit", "bsd", "gpl", "cc"]):
        return "open"
    if "llama 2" in txt or "community license" in txt:
        return "open-ish"
    if "unknown" in txt:
        return "unknown"
    return "closed"

def naive_weights_classification(lic_str):
    """Classify whether weights are open or closed: open weights, closed weights, unknown."""
    if pd.isna(lic_str):
        return "unknown"
    txt = lic_str.lower().strip()
    if any(x in txt for x in ["apache", "mit", "bsd", "gpl", "cc"]):
        return "open weights"
    if "llama 2" in txt or "community license" in txt:
        return "open weights"
    if "unknown" in txt:
        return "unknown"
    return "closed weights"

#############################
# COMBINED CLASSIFY
#############################
def classify_license(lic_str):
    """Step 1: naive, Step 2: if unknown and use_gemini -> gemini call."""
    naive_result = naive_license_classification(lic_str)
    if naive_result != "unknown" or not use_gemini:
        return naive_result
    # Build a prompt for license classification
    prompt = (
        "Classify the model license text as one of: 'open', 'open-ish', 'closed', or 'unknown'. "
        "Only return one of these words. "
        f"License text: {lic_str}\n"
    )
    gem_res = gemini_classification(prompt, lic_str)
    # parse
    if "open-ish" in gem_res:
        return "open-ish"
    elif "open" in gem_res:
        return "open"
    elif "closed" in gem_res:
        return "closed"
    else:
        return "unknown"

def classify_weights(lic_str):
    """Step 1: naive, Step 2: if unknown and use_gemini -> gemini call for open vs closed weights."""
    naive_w = naive_weights_classification(lic_str)
    if naive_w != "unknown" or not use_gemini:
        return naive_w
    # Build a prompt for weights classification
    prompt = (
        "Are the model weights open or closed based on this license text? "
        "Return 'open weights', 'closed weights', or 'unknown'. "
        f"License: {lic_str}\n"
    )
    gem_res = gemini_classification(prompt, lic_str)
    if "open weights" in gem_res:
        return "open weights"
    elif "closed weights" in gem_res:
        return "closed weights"
    else:
        return "unknown"

#############################
# MAIN
#############################
def classify_dataset(model_df):
    """
    For each row in model_df, classify license_type and weights_availability.
    Minimal debug, storing results in 'license_type' and 'weights_availability'.
    """
    license_types = []
    weights_types = []
    for idx, row in model_df.iterrows():
        lic_str = row["license"]
        if LIGHT_DEBUG:
            print(f"[DEBUG] Row {idx} license='{str(lic_str)[:40]}...'")
        # 1) License classification
        lic_type = classify_license(lic_str)
        # 2) Weights classification
        w_type = classify_weights(lic_str)

        license_types.append(lic_type)
        weights_types.append(w_type)

    model_df["license_type"] = license_types
    model_df["weights_availability"] = weights_types
    return model_df

# EXAMPLE USAGE
if __name__ == "__main__":
    # Suppose model_df is your DataFrame with ~400 rows, each has a 'license' column
    print("[INFO] Start classification process.")
    model_df = classify_dataset(model_df)

    print("\nLicense Distribution:")
    print(model_df["license_type"].value_counts(dropna=False))

    print("\nWeights Distribution:")
    print(model_df["weights_availability"].value_counts(dropna=False))

    print("\nUpdated DataFrame:")
    print(model_df[["license", "license_type", "weights_availability"]].head(40))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_license_and_weights_over_time(model_df):
    # --- License Distribution ---
    license_counts = (model_df.groupby(["year", "license_type"])
                      .size()
                      .reset_index(name="count"))
    license_pivot = license_counts.pivot(index="year", columns="license_type", values="count").fillna(0)
    ordered_license = ["open", "open-ish", "closed", "unknown"]
    for col in ordered_license:
        if col not in license_pivot.columns:
            license_pivot[col] = 0
    license_pivot = license_pivot[ordered_license]
    license_pct = license_pivot.div(license_pivot.sum(axis=1), axis=0).fillna(0)

    # --- Weights Availability Distribution ---
    weights_counts = (model_df.groupby(["year", "weights_availability"])
                      .size()
                      .reset_index(name="count"))
    weights_pivot = weights_counts.pivot(index="year", columns="weights_availability", values="count").fillna(0)
    ordered_weights = ["open weights", "closed weights", "unknown"]
    for col in ordered_weights:
        if col not in weights_pivot.columns:
            weights_pivot[col] = 0
    weights_pivot = weights_pivot[ordered_weights]
    weights_pct = weights_pivot.div(weights_pivot.sum(axis=1), axis=0).fillna(0)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    # License colors
    license_colors = {
        "open": "green",       # green
        "open-ish": "blue",   # blue
        "closed": "red",     # blue
        "unknown": "grey"
    }
    # Plot License Distribution as a stacked bar chart
    x_vals = license_pct.index.values
    bottom = np.zeros(len(x_vals))
    for lic in ordered_license:
        ax1.bar(x_vals, license_pct[lic].values, bottom=bottom,
                color=license_colors[lic], edgecolor="white", linewidth=0.5,
                label=lic)
        bottom += license_pct[lic].values
    ax1.set_ylabel("Fraction of Models")
    ax1.set_title("License Distribution Over Time")
    ax1.legend(loc="upper right", frameon=False)

    # Weights colors
    weights_colors = {
        "open weights": "green",   # green
        "closed weights": "red", # green
        "unknown": "grey"
    }
    # Plot Weights Availability Distribution as a stacked bar chart
    bottom = np.zeros(len(x_vals))
    for wt in ordered_weights:
        ax2.bar(x_vals, weights_pct[wt].values, bottom=bottom,
                color=weights_colors[wt], edgecolor="white", linewidth=0.5,
                label=wt)
        bottom += weights_pct[wt].values
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Fraction of Models")
    ax2.set_title("Weights Availability Distribution Over Time")
    ax2.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming model_df is your full DataFrame (with ~400 models) and already has
# the 'year', 'license_type', and 'weights_availability' columns.
plot_license_and_weights_over_time(model_df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_license_and_weights_separate(model_df):
    """
    Creates two side-by-side line plots:
      1) License distribution over time (4 lines: open, open-ish, closed, unknown)
      2) Weights distribution over time (3 lines: open weights, closed weights, unknown)

    Each subplot shows fractions of models by year, with distinct colors for each category.
    """
    # ========================
    # 1) LICENSE FRACTIONS
    # ========================
    lic_counts = (
        model_df
        .groupby(["year", "license_type"])
        .size()
        .reset_index(name="count")
    )
    lic_pivot = lic_counts.pivot(index="year", columns="license_type", values="count").fillna(0)

    lic_order = ["open", "open-ish", "closed", "unknown"]
    for col in lic_order:
        if col not in lic_pivot.columns:
            lic_pivot[col] = 0

    lic_pivot = lic_pivot[lic_order]
    lic_frac = lic_pivot.div(lic_pivot.sum(axis=1), axis=0).fillna(0)
    lic_frac = lic_frac.sort_index()  # ensure ascending year order
    years_lic = lic_frac.index.values

    # ========================
    # 2) WEIGHTS FRACTIONS
    # ========================
    wt_counts = (
        model_df
        .groupby(["year", "weights_availability"])
        .size()
        .reset_index(name="count")
    )
    wt_pivot = wt_counts.pivot(index="year", columns="weights_availability", values="count").fillna(0)

    wt_order = ["open weights", "closed weights", "unknown"]
    for col in wt_order:
        if col not in wt_pivot.columns:
            wt_pivot[col] = 0

    wt_pivot = wt_pivot[wt_order]
    wt_frac = wt_pivot.div(wt_pivot.sum(axis=1), axis=0).fillna(0)
    wt_frac = wt_frac.sort_index()
    years_wt = wt_frac.index.values

    # ========================
    # PLOTTING
    # ========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Colors for license
    lic_colors = {
        "open": "#2ca02c",       # green
        "open-ish": "#1f77b4",   # blue
        "closed": "#d62728",     # red
        "unknown": "grey"
    }
    # Colors for weights
    wt_colors = {
        "open weights": "#ff7f0e",   # orange
        "closed weights": "#9467bd", # purple
        "unknown": "grey"
    }

    # 1) License lines (left subplot)
    for cat in lic_order:
        ax1.plot(
            years_lic,
            lic_frac[cat].values,
            marker='o',
            linewidth=2,
            color=lic_colors[cat],
            label=cat.capitalize() if cat != "open-ish" else "Open-ish"
        )
    ax1.set_xlabel("Year", fontsize=11)
    ax1.set_ylabel("Fraction of Models", fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("License Distribution Over Time", fontsize=12)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", frameon=False)

    # 2) Weights lines (right subplot)
    for cat in wt_order:
        ax2.plot(
            years_wt,
            wt_frac[cat].values,
            marker='o',
            linewidth=2,
            color=wt_colors[cat],
            label=cat.capitalize() if cat != "open weights" else "Open Weights"
        )
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Weights Availability Over Time", fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()

# Usage:
plot_license_and_weights_separate(model_df)

"""# Google Colab Notebook Description
This notebook enriches your existing model_df (with an "organization" column) by querying the Gemini API using the gemini-2.0-flash model. It retrieves company metadata—including founding year, headquarters (split into city and country), company size, and company type—using caching and retries for robust API calls, and saves the updated DataFrame as assets_with_metadata.csv.

> Add blockquote


"""

import requests
import json
import time
import re
import pandas as pd

DEBUG = True

# Cache to avoid duplicate API calls
gemini_cache = {}

def extract_json(raw_text):
    """
    If the raw text includes markdown code block formatting (```json ... ```),
    extract and return only the JSON portion. Otherwise, return the original text.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def split_headquarters(hq_str):
    """
    Splits a headquarters string into city and country.
    Expected format: "City, Country". If not found, returns (hq_str, "unknown").
    """
    parts = [part.strip() for part in hq_str.split(",")]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    else:
        return hq_str, "unknown"

def get_company_metadata(manufacturer, max_retries=5):
    """
    Queries the Gemini API for metadata about a manufacturer.
    Returns a dictionary with keys:
      - founding_year: four-digit year or "unknown"
      - headquarters_city: city or "unknown"
      - headquarters_country: country or "unknown"
      - company_size: number of employees/revenue bracket or "unknown"
      - company_type: "public", "private", or "startup" (or "unknown")
    The prompt instructs the model to return strictly valid JSON.
    Includes a retry mechanism with exponential backoff for 429 errors.
    """
    manufacturer_key = manufacturer.lower().strip()
    if manufacturer_key in gemini_cache:
        if DEBUG:
            print(f"[DEBUG] Cache hit for '{manufacturer}': {gemini_cache[manufacturer_key]}")
        return gemini_cache[manufacturer_key]

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # optional
    GEMINI_ENDPOINT = (f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else None)

    prompt = (
        f"Please provide the following company metadata for \"{manufacturer}\" in strictly valid JSON format with no additional text:\n"
        "{\n"
        '  "founding_year": "<four-digit year or unknown>",\n'
        '  "headquarters": "<city, country or unknown>",\n'
        '  "company_size": "<number of employees or revenue bracket or unknown>",\n'
        '  "company_type": "<public, private, or startup or unknown>"\n'
        "}\n"
        "Return only the JSON object."
    )

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    headers = {"Content-Type": "application/json"}

    retries = 0
    backoff = 15  # initial backoff in seconds
    while retries < max_retries:
        try:
            response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                raw_answer = data["candidates"][0]["content"]["parts"][0]["text"]
                if DEBUG:
                    print(f"[DEBUG] Raw answer for '{manufacturer}':\n{raw_answer}\n")
                json_text = extract_json(raw_answer)
                try:
                    metadata = json.loads(json_text)
                except Exception as e:
                    print(f"Error parsing JSON for {manufacturer}: {e}")
                    metadata = {}
                # Process headquarters field: split into city and country
                if "headquarters" in metadata:
                    city, country = split_headquarters(metadata["headquarters"])
                    metadata["headquarters_city"] = city
                    metadata["headquarters_country"] = country
                    del metadata["headquarters"]
                else:
                    metadata["headquarters_city"] = "unknown"
                    metadata["headquarters_country"] = "unknown"
                # Ensure required keys exist
                for key in ["founding_year", "company_size", "company_type"]:
                    if key not in metadata:
                        metadata[key] = "unknown"
                gemini_cache[manufacturer_key] = metadata
                return metadata
            elif response.status_code == 429:
                print(f"Error: Received status code 429 for {manufacturer}. Retrying in {backoff} seconds...")
                time.sleep(backoff)
                retries += 1
                backoff *= 2  # exponential backoff
            else:
                print(f"Error: Received status code {response.status_code} for {manufacturer}")
                return {
                    "founding_year": "unknown",
                    "headquarters_city": "unknown",
                    "headquarters_country": "unknown",
                    "company_size": "unknown",
                    "company_type": "unknown"
                }
        except Exception as e:
            print(f"Exception for {manufacturer}: {e}")
            return {
                "founding_year": "unknown",
                "headquarters_city": "unknown",
                "headquarters_country": "unknown",
                "company_size": "unknown",
                "company_type": "unknown"
            }
    # If max retries exceeded, return unknown
    print(f"Max retries exceeded for {manufacturer}. Returning unknown metadata.")
    return {
        "founding_year": "unknown",
        "headquarters_city": "unknown",
        "headquarters_country": "unknown",
        "company_size": "unknown",
        "company_type": "unknown"
    }

# --- Process model_df ---
# Assume model_df is already loaded and has at least the "organization" column.
print(f"Processing {len(model_df)} companies from model_df.")

def fetch_metadata(row):
    manufacturer = row["organization"]
    metadata = get_company_metadata(manufacturer)
    return pd.Series(metadata)

# Apply the metadata extraction function to each row.
# To avoid overloading the API, consider processing in batches or adding additional sleep if needed.
metadata_df = model_df.apply(fetch_metadata, axis=1)

# Append new metadata columns to model_df.
for col in ["founding_year", "headquarters_city", "headquarters_country", "company_size", "company_type"]:
    model_df[col] = metadata_df[col]

# Save the updated DataFrame to CSV.
output_filename = "assets_with_metadata.csv"
model_df.to_csv(output_filename, index=False)
print(f"Saved updated CSV as '{output_filename}'.")

"""### **Evolution of unique countries**

This notebook **tracks the evolution of unique countries** in `metadata_df` (based on a shared `year` column). It calculates:

1. **Annual Unique Countries** – The number of distinct `headquarters_country` values each year.  
2. **Cumulative Unique Countries** – The total number of distinct countries up to each year.

A **dual-axis plot** then visualizes the **annual** count (blue line, left axis) and the **cumulative** count (red line, right axis) to highlight how the **geographical diversity** of manufacturers changes over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Group by year and compute number of unique countries
unique_countries_yearly = (
    model_df.groupby("year")["headquarters_country"]
    .nunique()
    .reset_index(name="unique_countries")
)

# Compute cumulative unique countries
cumulative_countries = []
cumulative_set = set()
for year in sorted(unique_countries_yearly["year"].unique()):
    countries_this_year = set(
        model_df.loc[model_df["year"] == year, "headquarters_country"].unique()
    )
    cumulative_set.update(countries_this_year)
    cumulative_countries.append({
        "year": year,
        "cumulative_unique_countries": len(cumulative_set)
    })

cumulative_df = pd.DataFrame(cumulative_countries)

# Plot
fig, ax1 = plt.subplots(figsize=(6,4))

# Unique countries per year (blue, left axis)
ax1.plot(
    unique_countries_yearly["year"],
    unique_countries_yearly["unique_countries"],
    marker='o', color='blue', linewidth=2,
    label="Unique Countries per Year"
)
ax1.set_xlabel("Release Year", fontsize=11)
ax1.set_ylabel("Number of Countries", fontsize=11, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Second axis for cumulative count (red)
ax2 = ax1.twinx()
ax2.plot(
    cumulative_df["year"],
    cumulative_df["cumulative_unique_countries"],
    marker='o', color='red', linewidth=2,
    label="Cumulative Unique Countries"
)
ax2.set_ylabel("Cumulative Unique Countries", fontsize=11, color='red')
ax2.tick_params(axis='y', labelcolor='red')

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

plt.tight_layout()
plt.show()
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'geopandas'], check=False)
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'geodatasets'], check=False)
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'mapclassify'], check=False)

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import mapclassify

# 1. Example dictionary to harmonize country names if needed
country_name_map = {
    "USA": "United States of America",
    "U.S.": "United States of America",
    "United States": "United States of America",
    "UK": "United Kingdom",
    "Britain": "United Kingdom",
    "England": "United Kingdom",
    "Russian Federation": "Russia",
    "UAE": "United Arab Emirates",   # Updated so 'UAE' matches shapefile
    "Israel": "Israel",             # If your data spells it differently, unify here
    # ... add more mappings as needed ...
}

metadata_df["headquarters_country"] = metadata_df["headquarters_country"].replace(country_name_map)

# 2. Compute model counts per country
country_counts = (
    metadata_df
    .groupby("headquarters_country")
    .size()
    .reset_index(name="count")
)

# Print sorted country frequencies (descending)
print("Country frequencies (descending):")
print(country_counts.sort_values("count", ascending=False))

# 3. Load a world GeoDataFrame from a public GeoJSON
world = gpd.read_file("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson")

# 4. Merge with your country counts on the 'ADMIN' column
world_merged = world.merge(
    country_counts,
    how="left",
    left_on="ADMIN",
    right_on="headquarters_country"
)

# 5. Define custom bins so that 1-model countries get their own color
bins = [0, 1, 5, 10, 20, 50, world_merged["count"].max(skipna=True)]

# 6. Plot with a user-defined classification scheme for discrete bins
fig, ax = plt.subplots(figsize=(10,6))
world_merged.plot(
    column="count",
    cmap="OrRd",
    scheme="userdefined",
    classification_kwds={"bins": bins},
    legend=True,
    edgecolor="white",
    linewidth=0.5,
    missing_kwds={
        "color": "lightgrey",
        "edgecolor": "white",
        "hatch": "///",
        "label": "No Data"
    },
    ax=ax,
    legend_kwds={
        "title": "Number of Models",
        "loc": "lower left",  # place legend at bottom-left
        "bbox_to_anchor": (0.02, 0.02)
    }
)

ax.set_title("Number of Models per Country", fontsize=14)
ax.set_axis_off()

plt.tight_layout()
plt.savefig("models_per_country.png", dpi=300, bbox_inches='tight')
plt.show()

"""# ** Company size categories **  
This notebook creates a **single plot** showing both **company size categories** (stacked bars) and **public/private** lines by year. It:

1. **Classifies** `"company_size"` into **startup**, **medium**, **large**, or **unknown**.  
2. **Groups and pivots** the data by year for stacked bars (green, blue, red, grey).  
3. **Overlays** lines for **public** (orange) and **private** (purple) on a secondary y‑axis.  
4. Uses **shared x‑coordinates** and a **combined legend** in the top‑left.  

Result: A **clean, publication‑ready** chart that shows **both** size and type trends over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.lines as mlines

# --- 1. Define colors for size categories and private/public lines ---
size_cat_colors = {
    "startup": "green",
    "medium": "blue",
    "large": "red",
    "unknown": "grey"
}
type_colors = {
    "private": "purple",
    "public": "orange"
}

# --- 2. Classify size from "company_size" strings ---
def classify_size(size_str):
    if not size_str or str(size_str).strip().lower().startswith("unknown"):
        return "unknown"
    clean_str = re.sub(r'[>,]', '', str(size_str))
    nums = re.findall(r'\d+', clean_str)
    if nums:
        val = int(nums[0])
        if val < 100:
            return "startup"
        elif val < 1000:
            return "medium"
        else:
            return "large"
    return "unknown"

# Apply classify_size to model_df instead of metadata_df
model_df["size_category"] = model_df["company_size"].apply(classify_size)

# --- 3. Build the pivot table for stacked bars (size categories) ---
size_summary = (
    model_df.groupby(["year", "size_category"])
    .size()
    .reset_index(name="count")
)
size_pivot = size_summary.pivot(index="year", columns="size_category", values="count").fillna(0)

ordered_cats = ["startup", "medium", "large", "unknown"]
for cat in ordered_cats:
    if cat not in size_pivot.columns:
        size_pivot[cat] = 0
size_pivot = size_pivot[ordered_cats]
size_pivot = size_pivot.sort_index()

# --- 4. Build the pivot table for lines (private/public) ---
type_df = model_df[model_df["company_type"].isin(["private", "public"])]
type_summary = (
    type_df.groupby(["year", "company_type"])
    .size()
    .reset_index(name="count")
)
type_pivot = type_summary.pivot(index="year", columns="company_type", values="count").fillna(0)
type_pivot = type_pivot.sort_index()

# Reindex to ensure same years for bars and lines
type_pivot_aligned = type_pivot.reindex(size_pivot.index, fill_value=0)

# --- 5. Create numeric x-coords for both
bar_years = size_pivot.index.values  # e.g. [2019, 2020, ...]
bar_x = np.arange(len(bar_years))

# --- 6. Plot
fig, ax1 = plt.subplots(figsize=(10,6))

width = 0.7
bottom = np.zeros(len(bar_x))

# A) Stacked bars for size categories
for cat in ordered_cats:
    ax1.bar(
        bar_x,
        size_pivot[cat].values,
        width,
        bottom=bottom,
        color=size_cat_colors[cat],
        alpha=0.8,
        zorder=1
    )
    bottom += size_pivot[cat].values

ax1.set_xticks(bar_x)
ax1.set_xticklabels(bar_years, rotation=45, ha="right")
ax1.set_xlabel("Release Year")
ax1.set_ylabel("Number of Companies (Size Categories)")

# B) Lines for private/public on secondary axis
ax2 = ax1.twinx()
ax2.set_ylabel("Number of Companies (Public/Private)")

private_vals = type_pivot_aligned["private"].values if "private" in type_pivot_aligned else np.zeros(len(bar_x))
public_vals  = type_pivot_aligned["public"].values  if "public"  in type_pivot_aligned else np.zeros(len(bar_x))

ax2.plot(bar_x, private_vals, marker='o', linewidth=2, color=type_colors["private"], label="Private", zorder=5)
ax2.plot(bar_x, public_vals,  marker='o', linewidth=2, color=type_colors["public"],  label="Public",  zorder=5)

# --- 7. Create combined legend, placed at the top-left
bar_handles = []
bar_labels = []
# Make a small square patch for each size category
for cat in ordered_cats:
    h = mlines.Line2D([], [], marker='s', linestyle='', color=size_cat_colors[cat], markersize=10)
    bar_handles.append(h)
    bar_labels.append(cat.capitalize())

line_handles = [
    mlines.Line2D([], [], marker='o', color=type_colors["private"], label="Private"),
    mlines.Line2D([], [], marker='o', color=type_colors["public"],  label="Public")
]
line_labels = ["Private", "Public"]

all_handles = bar_handles + line_handles
all_labels = bar_labels + line_labels

ax2.legend(all_handles, all_labels, loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

plt.tight_layout()
plt.show()

"""**Models per manufacturer**  
This notebook uses light NLP (fuzzy matching with RapidFuzz) to cluster and aggregate similar manufacturer names in your dataset. It cleans and splits organization names, merges near-duplicates using union–find, and then visualizes the aggregated model counts as a treemap—where block area and color reflect the number of models per manufacturer (with smaller clusters combined as “Others”).
"""

print(model_df.columns.tolist())

###############################################################################
# 0) Imports
###############################################################################
import pandas as pd
import numpy as np

# Install *rapidfuzz* for fuzzy matching (comment out if already installed)
# !pip install rapidfuzz --quiet
from rapidfuzz import fuzz

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import squarify
import re

###############################################################################
# 1) Inspect model_df
###############################################################################
print("[INFO] model_df shape:", model_df.shape)
print("Columns in model_df:", model_df.columns.tolist())

if "organization" not in model_df.columns:
    raise ValueError("model_df must contain an 'organization' column.")

###############################################################################
# 2) Light organisation-name cleaning
###############################################################################
def clean_org_name(name: str) -> str:
    """Lower-case, strip punctuation, remove generic tokens."""
    name = name.lower()
    name = re.sub(r"[().\-]", " ", name)          # punctuation → space
    name = re.sub(r"\s+", " ", name).strip()      # collapse spaces
    stop = {"inc", "labs", "lab", "corp",
            "corporation", "non", "profit"}
    tokens = [t for t in name.split() if t not in stop]
    return " ".join(tokens)

model_df["cleaned_org"] = model_df["organization"].astype(str).apply(clean_org_name)

###############################################################################
# 3) Union–Find helper for clustering near-duplicate names
###############################################################################
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

###############################################################################
# 4) Fuzzy-match organisations
###############################################################################
unique_orgs = model_df["cleaned_org"].unique()
index_map   = {org: i for i, org in enumerate(unique_orgs)}
uf          = UnionFind(len(unique_orgs))

threshold = 85  # partial-ratio threshold

for i in range(len(unique_orgs)):
    for j in range(i + 1, len(unique_orgs)):
        if fuzz.partial_ratio(unique_orgs[i], unique_orgs[j]) >= threshold:
            uf.union(i, j)

# map each cleaned name → cluster representative
root_label = {}
for org in unique_orgs:
    root = uf.find(index_map[org])
    root_label.setdefault(root, []).append(org)

def pick_label(lst):        # choose shortest alias as label
    return min(lst, key=len)

cluster_id_to_label = {r: pick_label(lst) for r, lst in root_label.items()}

def to_cluster_label(cleaned):
    return cluster_id_to_label[uf.find(index_map[cleaned])]

model_df["org_cluster"] = model_df["cleaned_org"].apply(to_cluster_label)

###############################################################################
# 5) Aggregate, bin long tail as "Others"
###############################################################################
top_k = 8
counts = (model_df.groupby("org_cluster").size()
          .sort_values(ascending=False))
top_clusters = counts.head(top_k).index

model_df["final_cluster"] = model_df["org_cluster"].apply(
    lambda c: c if c in top_clusters else "Others"
)

final_counts = (model_df.groupby("final_cluster").size()
                .reset_index(name="model_count")
                .sort_values("model_count", ascending=False))

final_counts["label"] = (
    final_counts["final_cluster"] + " (" +
    final_counts["model_count"].astype(str) + ")"
)

print("[INFO] Aggregated manufacturer counts:")
print(final_counts)

###############################################################################
# 6) Treemap visualisation
###############################################################################
sizes  = final_counts["model_count"].values
labels = final_counts["label"].values

norm  = colors.Normalize(vmin=0, vmax=sizes.max())
cmap  = cm.ScalarMappable(norm=norm, cmap="Spectral")
color = [cmap.to_rgba(v) for v in sizes]

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("Models per Aggregated Manufacturer (Treemap)", fontsize=14, pad=10)

squarify.plot(
    sizes=sizes,
    label=labels,
    color=color,
    alpha=0.9,
    pad=True,
    text_kwargs={"fontsize": 8}
)
ax.axis("off")

cmap.set_array([])
cbar = plt.colorbar(cmap, ax=ax, orientation="vertical",
                    fraction=0.03, pad=0.05)
cbar.set_label("Number of Models", fontsize=11)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# If you have not already run the fuzzy-matching + aggregation code:
#   final_counts = (model_df ... )  # the DataFrame with "model_count"
# We'll assume final_counts is already defined here.

# 1) Sort the counts in descending order
counts_sorted = final_counts["model_count"].sort_values(ascending=False).reset_index(drop=True)

# 2) Create a rank array (1-based)
ranks = np.arange(1, len(counts_sorted) + 1)

# 3) Plot log–log
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(ranks, counts_sorted, color="purple", alpha=0.7)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Rank (log scale)", fontsize=11)
ax.set_ylabel("Number of Models (log scale)", fontsize=11)
ax.set_title("Log–Log Plot of Models per Manufacturer", fontsize=13)

plt.tight_layout()
plt.show()

"""# PCA ANALYSIS
We aggregated model-related metrics by year (including number of models, average log10(size), manufacturers, countries, documentation coverage, licensing restrictiveness, and modality count). Then, we used Principal Component Analysis (PCA) to reduce these metrics into a few dimensions, revealing how regulatory complexity evolves over time. Specifically:

Data: Each year’s ecosystem metrics (e.g., total models, closed-license fraction).

PCA: We standardized those features and computed principal components.

Interpretation:

The first principal component generally captured a “scale” axis (more models, more manufacturers) combined with poorer documentation.

The second principal component captured higher modality counts and closed licensing—sometimes associated with slightly better documentation—but still relatively advanced or restrictive release patterns.

Overall, the PCA biplot provided a bird’s-eye view of how these factors interact, helping us see which years are most similar or different in terms of ecosystem complexity.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------- Helper Functions ----------

def compute_documentation_score(group):
    """
    Computes a documentation score based on:
      - has_emissions, has_time, has_hardware (converted to 0/1)
      - model_card presence (1 if non-null)
    A higher score means better documentation.
    """
    def bool_to_int(val):
        if isinstance(val, bool):
            return int(val)
        elif isinstance(val, str):
            return 1 if val.strip().lower() == "true" else 0
        else:
            try:
                return int(val)
            except Exception:
                return 0

    emissions = group["has_emissions"].apply(bool_to_int)
    time_val  = group["has_time"].apply(bool_to_int)
    hardware  = group["has_hardware"].apply(bool_to_int)
    model_card_present = group["model_card"].notna().astype(int)

    score = (emissions + time_val + hardware + model_card_present) / 4.0
    return score.mean()

def compute_mean_log_size(group):
    """
    Computes the mean of log10 of the model sizes.
    'parsed_params' is expected to hold numeric parameter counts.
    """
    sizes = pd.to_numeric(group["parsed_params"], errors="coerce")
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return np.nan
    return np.log10(sizes).mean()

def compute_average_modalities(group):
    """
    Computes the average number of modalities.
    Assumes the 'modality' field is encoded as a semicolon-separated string.
    """
    modality_counts = group["modality"].apply(lambda x: len(str(x).split(";")) if pd.notna(x) else np.nan)
    return modality_counts.mean()

# ---------- Aggregation by Year ----------
# Group model_df by year, computing the aggregated metrics.

agg_df = (
    model_df
    .groupby("year")
    .apply(lambda group: pd.Series({
        "total_models": group.shape[0],
        "mean_log_size": compute_mean_log_size(group),
        "unique_manufacturers": group["org_cluster"].nunique(),
        "unique_countries": group["headquarters_country"].replace("unknown", np.nan).nunique(),
        "closed_license_fraction": (group["license_type"] == "closed").mean(),
        "closed_weights_fraction": (group["weights_availability"] == "closed weights").mean(),
        "average_modalities": pd.to_numeric(group["num_modalities"], errors="coerce").mean(),
        "documentation_deficit": 1 - compute_documentation_score(group)
    }))
    .reset_index()
)

# Fill missing values with column means (or another strategy if you prefer)
agg_df.fillna(agg_df.mean(), inplace=True)

print("Aggregated Metrics by Year:")
print(agg_df)

# ---------- Feature Selection for PCA ----------
features = [
    "total_models",             # More models → more complexity
    "mean_log_size",            # Larger models → more complexity
    "unique_manufacturers",     # More manufacturers → more complexity
    "unique_countries",         # More countries → more complexity
    "documentation_deficit",    # Higher deficit (poorer documentation) → more complexity
    "closed_license_fraction",  # More restrictive licenses → more complexity
    "closed_weights_fraction",  # More restrictive weights → more complexity
    "average_modalities"        # More modalities → more complexity
]

X = agg_df[features].values

# ---------- Standardize Features ----------
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Number of PCA components = min(n_features, n_samples)
n_components = min(len(features), X_std.shape[0])
pca = PCA(n_components=n_components)
pca.fit(X_std)
X_pca = pca.transform(X_std)

# ---------- PCA Results & Biplot ----------
print("\nExplained Variance Ratio per component:")
for i, var_ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var_ratio:.2f}")

print("\nPCA Components (Loadings):")
pca_df = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[f"PC{i+1}" for i in range(n_components)]
)
print(pca_df)

# Create a biplot for the first two principal components (if available)
if n_components >= 2:
    fig, ax = plt.subplots(figsize=(8,6))
    years_str = agg_df["year"].astype(str).values

    # Scatter each year in the PC1-PC2 plane
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=agg_df["year"], cmap="viridis", s=100)
    for i, txt in enumerate(years_str):
        ax.annotate(txt, (X_pca[i,0], X_pca[i,1]), textcoords="offset points", xytext=(5,5), fontsize=9)

    # Plot feature vectors (loadings), scaled for visibility
    for i, feature in enumerate(features):
        # Multiply by a factor to make arrows more visible
        ax.arrow(0, 0, pca.components_[0,i]*3, pca.components_[1,i]*3, color="r", width=0.02, head_width=0.1)
        ax.text(
            pca.components_[0,i]*3.2,
            pca.components_[1,i]*3.2,
            feature,
            color="r",
            fontsize=9
        )

    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("PCA Biplot of Aggregated Model Metrics\n(Regulatory Complexity Perspective)")
    ax.grid(True)
    plt.colorbar(scatter, label="Year")
    plt.tight_layout()
    plt.show()
else:
    print("\nNot enough samples to plot a 2-component PCA biplot.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random

# For reproducible random shifts
random.seed(42)

# ---------- Helper Functions ----------
def compute_documentation_score(group):
    def bool_to_int(val):
        if isinstance(val, bool):
            return int(val)
        elif isinstance(val, str):
            return 1 if val.strip().lower() == "true" else 0
        else:
            try:
                return int(val)
            except Exception:
                return 0

    emissions = group["has_emissions"].apply(bool_to_int)
    time_val  = group["has_time"].apply(bool_to_int)
    hardware  = group["has_hardware"].apply(bool_to_int)
    model_card_present = group["model_card"].notna().astype(int)

    return ((emissions + time_val + hardware + model_card_present) / 4.0).mean()

def compute_mean_log_size(group):
    sizes = pd.to_numeric(group["parsed_params"], errors="coerce")
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return np.nan
    return np.log10(sizes).mean()

def compute_average_modalities(group):
    mod_counts = group["modality"].apply(lambda x: len(str(x).split(";")) if pd.notna(x) else np.nan)
    return mod_counts.mean()

# ---------- 1) Aggregate by Year ----------
agg_df = (
    model_df
    .groupby("year")
    .apply(lambda g: pd.Series({
        "total_models": g.shape[0],
        "mean_log_size": compute_mean_log_size(g),
        "unique_manufacturers": g["org_cluster"].nunique(),
        "unique_countries": g["headquarters_country"].replace("unknown", np.nan).nunique(),
        "documentation_deficit": 1 - compute_documentation_score(g),
        "closed_license_fraction": (g["license_type"] == "closed").mean(),
        "closed_weights_fraction": (g["weights_availability"] == "closed weights").mean(),
        "average_modalities": pd.to_numeric(g["num_modalities"], errors="coerce").mean()
    }))
    .reset_index()
)

agg_df.fillna(agg_df.mean(), inplace=True)
print("Aggregated metrics by year:\n", agg_df)

# ---------- 2) PCA + Biplot With Randomized Offsets ----------
features = [
    "total_models", "mean_log_size", "unique_manufacturers",
    "unique_countries", "documentation_deficit",
    "closed_license_fraction", "closed_weights_fraction", "average_modalities"
]

X = agg_df[features].values
X_std = StandardScaler().fit_transform(X)
n_components = min(len(features), X_std.shape[0])
pca = PCA(n_components=n_components).fit(X_std)
X_pca = pca.transform(X_std)

fig, ax = plt.subplots(figsize=(8, 6))

points = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=agg_df["year"], cmap='plasma', s=200, edgecolor='k')

# Annotate each point by year
for i, yr in enumerate(agg_df["year"].astype(int)):
    ax.annotate(yr, (X_pca[i, 0], X_pca[i, 1]),
                xytext=(5,5), textcoords='offset points', fontsize=10)

# Plot vectors (arrows) for loadings, with random jitter so they don't overlap
for i, feature in enumerate(features):
    # Original arrow length
    arrow_x = pca.components_[0, i]*3
    arrow_y = pca.components_[1, i]*3

    # Add small random shift to starting point so arrows don't overlap at origin
    start_x = random.uniform(-0.15, 0.15)
    start_y = random.uniform(-0.15, 0.15)

    # You can also jitter the arrow end if you like:
    arrow_x += random.uniform(-0.05, 0.05)
    arrow_y += random.uniform(-0.05, 0.05)

    ax.arrow(
        start_x, start_y,
        arrow_x, arrow_y,
        color="steelblue", alpha=0.7,
        head_width=0.08, head_length=0.1, linewidth=2,
        length_includes_head=True
    )

    # Label the arrow end with a small random offset
    label_x = start_x + arrow_x + random.uniform(-0.1, 0.1)
    label_y = start_y + arrow_y + random.uniform(-0.1, 0.1)
    ax.text(label_x, label_y, str(i+1),
            color="darkblue", fontsize=10, fontweight='bold')

# Build a legend for the features
feature_legend = '\n'.join([f'{i+1}: {f.replace("_", " ").title()}' for i, f in enumerate(features)])
ax.legend([feature_legend], loc='upper right', frameon=True, fontsize=10)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("PCA Biplot with Random Jitter")
ax.grid(False)

plt.colorbar(points, label="Year")
plt.tight_layout()
plt.show()
ut()
plt.show()

import pandas as pd
import os

# 1. Confirm that `model_df` is already in memory
#    (e.g., you have run some code that created or loaded `model_df`)

# 2. Build the path to your desktop (on macOS/Linux, works if you have permissions).
#    On Windows, replace with e.g. "C:Desktop/assets_with_metadata_2.csv"
desktop_path = os.path.expanduser("~/Desktop/assets_with_metadata_2.csv")

# 3. Dump the DataFrame to CSV, excluding the index column
model_df.to_csv(desktop_path, index=False)

print(f"[INFO] model_df dumped to {desktop_path}")

"""# ONTO THE BENCHMARK STUFF NOW

# TOTAL BENCHMARKS PER YEAR
We start by just showing the number of benchmarks published per year
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load the data ---
csv_filename = "benchmarks_new.csv"
df = pd.read_csv(csv_filename)

# --- Step 2: Process the publication year ---
# Convert "Initial publication year" to numeric and store in a new "year" column.
df["year"] = pd.to_numeric(df["Initial publication year"], errors="coerce")
df = df.dropna(subset=["year"])
df["year"] = df["year"].astype(int)

# --- Step 3: Aggregate by year ---
# Count how many benchmarks per year
df_counts = df.groupby("year").size().reset_index(name="annual_count")
df_counts["cumulative_count"] = df_counts["annual_count"].cumsum()

# --- Step 4: Plot the benchmark counts over time ---
sns.set_style("white")  # Use a plain white background (no grid)
fig, ax = plt.subplots(figsize=(8, 5))

# Plot annual counts (blue)
ax.plot(df_counts["year"], df_counts["annual_count"], marker="o", color="blue", label="Annual Benchmarks")

# Plot cumulative counts (red)
ax.plot(df_counts["year"], df_counts["cumulative_count"], marker="o", color="red", label="Cumulative Benchmarks")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Number of Benchmarks Over Time", fontsize=14)
ax.legend(loc="upper left", fontsize=10)

plt.tight_layout()
plt.show()

"""# HOW MANY AUTHORS

Now, how many authors are involved in total
"""

# 0) In Colab, first install the arxiv client:
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'arxiv'], check=False)

import pandas as pd
import arxiv
import re
import matplotlib.pyplot as plt

# 1) Load your local benchmark metadata
csv_path = "benchmarks_new.csv"
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# 2) Parse publication year
df["year"] = pd.to_numeric(df["Initial publication year"], errors="coerce")
df = df.dropna(subset=["year"]).astype({"year":int})

# 3) Extract arXiv IDs from the “Benchmark paper” URLs
def extract_arxiv_id(url):
    if not isinstance(url, str):
        return None
    m = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9\.v]+)', url)
    return m.group(1) if m else None

df["arxiv_id"] = df["Benchmark paper"].apply(extract_arxiv_id)

# 4) Fetch author lists via the arxiv library
def fetch_authors(arxiv_id):
    if not arxiv_id:
        return []
    try:
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(search.results(), None)
        return [a.name for a in paper.authors] if paper else []
    except Exception as e:
        print(f"Failed {arxiv_id}: {e}")
        return []

df["authors_list"] = df["arxiv_id"].apply(fetch_authors)

# 5) Explode so each row is one author
exploded = df.explode("authors_list")
exploded["author"] = exploded["authors_list"].astype(str).str.strip().str.lower()
exploded = exploded[exploded["author"] != ""]

# 6) Compute per‐year author metrics
total_mentions   = exploded.groupby("year").size().rename("total_mentions")
unique_per_year  = exploded.groupby("year")["author"].nunique().rename("unique_authors")

seen = set()
cum_unique = {}
for y in sorted(unique_per_year.index):
    seen.update(exploded.loc[exploded["year"]==y, "author"])
    cum_unique[y] = len(seen)
cum_unique = pd.Series(cum_unique, name="cumulative_unique")

# 7) Combine into a DataFrame
stats = pd.concat([total_mentions, unique_per_year, cum_unique], axis=1).fillna(0).astype(int)
print(stats)

# 8) Plot
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.plot(stats.index, stats["total_mentions"], marker="o", color="blue",  label="Total Mentions")
ax1.plot(stats.index, stats["unique_authors"], marker="o", color="black", label="Unique Authors")
ax1.set_xlabel("Year")
ax1.set_ylabel("Mentions / Unique", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax2 = ax1.twinx()
ax2.plot(stats.index, stats["cumulative_unique"], marker="o", color="red", label="Cumulative Unique")
ax2.set_ylabel("Cumulative Unique Authors", color="red")
ax2.tick_params(axis="y", labelcolor="red")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.title("Author Mentions and Unique Authors Over Time (via arXiv)")
plt.tight_layout()
plt.show()

"""# SAVING WITH THE AUTHOR LIST"""

# -------------------------------------------------------
#  A) Save one row per benchmark, with the author list
# -------------------------------------------------------
# Convert the Python list to a semicolon-separated string for easy CSV storage
df["authors_str"] = df["authors_list"].apply(lambda lst: "; ".join(lst))

# Choose a path for the enriched benchmark table
out_path_full = "benchmarks_with_authors.csv"
df.to_csv(out_path_full, index=False)
print(f"[INFO] Saved enriched benchmark table to {out_path_full}")

# -------------------------------------------------------
#  B) Save one row per (benchmark, author) pair
# -------------------------------------------------------
out_path_pair = "benchmark_author_pairs.csv"
exploded[["arxiv_id", "author", "year"]].to_csv(out_path_pair, index=False)
print(f"[INFO] Saved exploded author pairs to {out_path_pair}")

"""# INFERRING THE INSTITUTIONS WITH GEMINI"""

import logging
import time
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import arxiv, ast

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CSV_IN      = "benchmarks_with_authors.csv"
CSV_OUT     = "benchmarks_with_affiliations.csv"
API_KEY = os.getenv("GEMINI_API_KEY", "")  # optional

ENDPOINT    = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
    f"?key={API_KEY}"
)
HEADERS     = {"Content-Type": "application/json"}
MAX_WORKERS = 10
SLOW_THRESH = 5.0   # warn if single call takes >2s
LOG_FILE    = "affil_debug.log"

# ──────────────────────────────────────────────────────────────────────────────
#  SET UP LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)
logging.info("=== Starting affiliation infer run ===")

# ──────────────────────────────────────────────────────────────────────────────
#  READ INPUT & PREPARE
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_IN)
if "authors_str" in df: df.drop(columns=["authors_str"], inplace=True)
df["author_list"] = df["authors_list"].apply(ast.literal_eval)

# arXiv title cache
_client = arxiv.Client()
_title_cache = {}
def get_title(aid):
    if aid in _title_cache: return _title_cache[aid]
    try:
        search = arxiv.Search(id_list=[aid])
        res = next(_client.results(search), None)
        t = res.title if res else "unknown"
    except Exception as e:
        logging.warning(f"arXiv lookup {aid} failed: {e}")
        t = "unknown"
    _title_cache[aid] = t
    return t

# ──────────────────────────────────────────────────────────────────────────────
#  WORKER
# ──────────────────────────────────────────────────────────────────────────────
def process_author(task):
    row_idx, author, year, aid = task
    title = get_title(aid)
    prompt = (
        "You are an academic bibliographic assistant.\n"
        f"Paper title: \"{title}\"\n"
        f"Publication year: {year}\n"
        f"Author: {author}\n\n"
        "Return ONLY the author's primary academic or research affiliation "
        "at that time, followed by a comma and the country. If unsure, reply 'unknown'."
    )
    body = {"contents":[{"parts":[{"text":prompt}]}]}

    backoff = 1.0
    txt = "unknown"
    status = None
    start = time.time()

    for attempt in range(5):
        try:
            r = requests.post(ENDPOINT, headers=HEADERS, json=body, timeout=30)
            status = r.status_code
        except Exception as e:
            status = f"EXC-{type(e).__name__}"
            logging.error(f"[{row_idx}|{author}] attempt {attempt} exception: {e}")
            time.sleep(backoff); backoff *= 2
            continue

        if status == 200:
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            break
        elif status == 429:
            logging.warning(f"[{row_idx}|{author}] rate‐limited, retrying in {backoff}s")
            time.sleep(backoff); backoff *= 2
        else:
            logging.error(f"[{row_idx}|{author}] unexpected status {status}: {r.text[:200]!r}")
            break

    elapsed = time.time() - start
    # split inst/country
    line = txt.strip().splitlines()[0].strip(" \"'")
    if "," in line:
        inst, country = [s.strip() for s in line.rsplit(",",1)]
    else:
        inst, country = line, ""

    # debug to log + tqdm
    msg = (
        f"row={row_idx:4d} author={author[:15]!r:<17} status={status} "
        f"time={elapsed:.2f}s affil={inst!r}, {country!r}"
    )
    logging.debug(msg)
    tqdm.write(msg)
    if elapsed > SLOW_THRESH:
        tqdm.write(f"⚠️  slow call (> {SLOW_THRESH}s): {author!r} took {elapsed:.2f}s")

    return row_idx, inst, country

# ──────────────────────────────────────────────────────────────────────────────
#  BUILD TASKS & RUN
# ──────────────────────────────────────────────────────────────────────────────
tasks = []
for i, r in df.iterrows():
    y   = int(r["year"])
    aid = str(r["arxiv_id"])
    for a in r["author_list"]:
        tasks.append((i, a, y, aid))

# prepare accumulators
out_affils = {i: [] for i in df.index}
out_names  = {i: [] for i in df.index}
out_ctrys  = {i: [] for i in df.index}

pbar = tqdm(total=len(tasks), desc="Inferring affiliations", unit="auth")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
    future_to_task = {exe.submit(process_author, t): t for t in tasks}
    for fut in as_completed(future_to_task):
        row_idx, inst, ctry = fut.result()
        out_affils[row_idx].append(f"{inst}, {ctry}".strip(", "))
        out_names[row_idx].append(inst)
        out_ctrys[row_idx].append(ctry)
        pbar.update(1)

pbar.close()

# ──────────────────────────────────────────────────────────────────────────────
#  WRITE BACK & SAVE
# ──────────────────────────────────────────────────────────────────────────────
df["affiliations_list"]          = df.index.map(lambda i: out_affils[i])
df["affiliation_names_list"]     = df.index.map(lambda i: out_names[i])
df["affiliation_countries_list"] = df.index.map(lambda i: out_ctrys[i])

df["affiliations_str"]           = df["affiliations_list"].apply("; ".join)
df["affiliation_names_str"]      = df["affiliation_names_list"].apply("; ".join)
df["affiliation_countries_str"]  = df["affiliation_countries_list"].apply("; ".join)

df.to_csv(CSV_OUT, index=False)
print(f"\n✅ Done — saved to: {CSV_OUT}")
logging.info("=== Finished affiliation infer run ===")

"""CLEAN UP THE DUPLICATED COLUMNS"""

import pandas as pd
import ast

# 1. load the full CSV
IN  = "benchmarks_with_affiliations.csv"
OUT = "benchmarks_cleaned.csv"
df  = pd.read_csv(IN)

# 2. list-columns to keep (and parse)
to_parse = ["author_list", "affiliation_names_list", "affiliation_countries_list"]
for col in to_parse:
    # if the dtype is object (i.e. stringified), turn it back into real lists
    if df[col].dtype == object:
        df[col] = df[col].apply(ast.literal_eval)

# 3. drop the exact columns we no longer need
df = df.drop(columns=[
    # the one duplicated author column
    "authors_list",
    # the intermediate “affiliations_list” (with inst+country),
    # and all the stringified versions
    "affiliations_list",
    "affiliations_str",
    "affiliation_names_str",
    "affiliation_countries_str",
])

# 4. rename the list-columns to exactly what you asked for
df = df.rename(columns={
    "author_list":             "author_list",
    "affiliation_names_list":  "affiliation_list",
    "affiliation_countries_list":"country_list"
})

# 5. (optional) peek at your new schema
print(df.columns.tolist())

# 6. write it out
df.to_csv(OUT, index=False)
print(f"✅ Cleaned file saved to {OUT}")

import pandas as pd
import ast

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH = "benchmarks_cleaned.csv"  # adjust as needed

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD & PARSE
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["author_list"]      = df["author_list"].apply(ast.literal_eval)
df["affiliation_list"] = df["affiliation_list"].apply(ast.literal_eval)
df["country_list"]     = df["country_list"].apply(ast.literal_eval)

# ──────────────────────────────────────────────────────────────────────────────
#  BASIC COUNTS
# ──────────────────────────────────────────────────────────────────────────────
total_rows    = len(df)
total_authors = df["author_list"].map(len).sum()

# flatten for per‐author counts
all_affils    = [i for sub in df["affiliation_list"] for i in sub]
all_countries = [c for sub in df["country_list"]      for c in sub]

# total returned (including unknowns/blanks)
total_affils    = len(all_affils)
total_countries = len(all_countries)

# ──────────────────────────────────────────────────────────────────────────────
#  UNKNOWN / BLANK DETECTION
# ──────────────────────────────────────────────────────────────────────────────
is_bad = lambda x: (not x) or (str(x).strip().lower() == "unknown")

unknown_affil_count   = sum(1 for a in all_affils    if is_bad(a))
unknown_country_count = sum(1 for c in all_countries if is_bad(c))

# how many rows contain ANY unknown/blank
rows_with_bad_affil   = df["affiliation_list"].apply(lambda L: any(is_bad(a) for a in L)).sum()
rows_with_bad_country = df["country_list"].    apply(lambda L: any(is_bad(c) for c in L)).sum()

# ──────────────────────────────────────────────────────────────────────────────
#  UNIQUE VALUES (excl. unknown/blank)
# ──────────────────────────────────────────────────────────────────────────────
unique_institutions = {i for i in all_affils    if not is_bad(i)}
unique_countries    = {c for c in all_countries if not is_bad(c)}

# ──────────────────────────────────────────────────────────────────────────────
#  REPORT
# ──────────────────────────────────────────────────────────────────────────────
print(f"Total rows:                     {total_rows}")
print(f"Total authors:                  {total_authors}\n")

print(f"Affiliations returned (all):    {total_affils}")
print(f"  ⇢ unknown/blank affils:         {unknown_affil_count} ({unknown_affil_count/total_affils:.1%})")
print(f"Rows w/ ≥1 unknown affil:       {rows_with_bad_affil} ({rows_with_bad_affil/total_rows:.1%})\n")

print(f"Countries returned (all):       {total_countries}")
print(f"  ⇢ unknown/blank countries:      {unknown_country_count} ({unknown_country_count/total_countries:.1%})")
print(f"Rows w/ ≥1 unknown country:     {rows_with_bad_country} ({rows_with_bad_country/total_rows:.1%})\n")

print(f"Distinct institutions (clean):  {len(unique_institutions)}")
print(f"Distinct countries (clean):      {len(unique_countries)}\n")

print("Sample clean institutions:", list(unique_institutions)[:10])
print("Sample clean countries:   ", list(unique_countries)[:10])

"""# HOW MANY INSTITUTIONS"""

import pandas as pd
import ast
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD & PARSE
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("benchmarks_cleaned.csv")

# Convert stringified lists back to Python lists
df['author_list']      = df['author_list'].apply(ast.literal_eval)
df['affiliation_list'] = df['affiliation_list'].apply(ast.literal_eval)
df['country_list']     = df['country_list'].apply(ast.literal_eval)

# ──────────────────────────────────────────────────────────────────────────────
#  AGGREGATE INSTITUTIONS BY YEAR
# ──────────────────────────────────────────────────────────────────────────────
institutions_by_year = {}
for year, group in df.groupby('year'):
    s = set()
    for insts in group['affiliation_list']:
        s.update(insts)
    institutions_by_year[year] = s

years = sorted(institutions_by_year)
yearly_counts = [len(institutions_by_year[y]) for y in years]

# cumulative unique institutions
cum_set = set()
cum_counts = []
for y in years:
    cum_set.update(institutions_by_year[y])
    cum_counts.append(len(cum_set))

# ──────────────────────────────────────────────────────────────────────────────
#  PLOT
# ──────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8,5))
plt.plot(years, yearly_counts, marker='o', color="black", label='Yearly Unique Institutions')
plt.plot(years, cum_counts, marker='o', color="red", label='Cumulative Unique Institutions')
plt.title('Evolving Number of Institutions Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Institutions')
plt.xticks(years)
plt.legend()
plt.grid(True, linestyle='', alpha=0.5)
plt.tight_layout()
plt.show()

"""## HOW MANY COUNTRIES"""

import pandas as pd
import ast
import matplotlib.pyplot as plt

# 1) Load & parse
df = pd.read_csv("benchmarks_cleaned.csv")
# ensure year is numeric
df["year"] = df["year"].astype(int)
# parse the string‐lists into actual lists
df["country_list"] = df["country_list"].apply(ast.literal_eval)

# 2) Compute yearly & cumulative unique counts
years = sorted(df["year"].unique())
yearly_counts = []
cumulative_counts = []
seen = set()

for y in years:
    # flatten all lists for this year into one set
    this_year = set(sum(df.loc[df["year"] == y, "country_list"].tolist(), []))
    yearly_counts.append(len(this_year))
    seen |= this_year
    cumulative_counts.append(len(seen))

# 3) Plot
plt.figure(figsize=(8,4))
plt.plot(years, yearly_counts, marker="o", color="black",label="Yearly Unique Countries", linewidth=2)
plt.plot(years, cumulative_counts, marker="o", color="red", label="Cumulative Unique Countries", linewidth=2)

plt.title("Evolving Number of Countries Over Time")
plt.xlabel("Year")
plt.ylabel("Countries")
plt.xticks(years)
plt.legend()
plt.tight_layout()
plt.show()

"""# REPOSITORY QUALITY"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Aggregate Repository Quality by Year ---
# Assuming your DataFrame (e.g., df or df_combined) has a "Year" column and quality columns:
# "Stars", "Forks", "Watchers", "Repo Size (KB)"
quality_by_year = df.groupby("Year").agg({
    "Stars": "mean",
    "Forks": "mean",
    "Watchers": "mean",
    "Repo Size (KB)": "mean"
}).reset_index()

# --- Step 2: Plot the Aggregated Quality Measures ---
sns.set_style("white")  # Plain white background, no grid
plt.figure(figsize=(10, 6))

plt.plot(quality_by_year["Year"], quality_by_year["Stars"], marker="o", label="Stars")
plt.plot(quality_by_year["Year"], quality_by_year["Forks"], marker="o", label="Forks")
plt.plot(quality_by_year["Year"], quality_by_year["Watchers"], marker="o", label="Watchers")
plt.plot(quality_by_year["Year"], quality_by_year["Repo Size (KB)"], marker="o", label="Repo Size (KB)")

plt.yscale("log")  # Apply logarithmic scale to y-axis
plt.xlabel("Year", fontsize=12)
plt.ylabel("Mean Value (log scale)", fontsize=12)
plt.title("Evolution of Repository Quality Measures (Log Scale)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

"""# CITATIONS"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load the data (assumes df is already loaded and processed) ---
# Convert the "Year" column to numeric if needed.
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)

# --- Step 2: Aggregate total citations per year ---
citations_total = df.groupby("Year")["Citation Count"].sum().reset_index()

# Compute the cumulative citation count over time.
citations_total["Cumulative Citation Count"] = citations_total["Citation Count"].cumsum()

# --- Step 3: Plot the total and cumulative citation counts ---
sns.set_style("white")  # Plain white background
plt.figure(figsize=(10, 6))

# Plot total citation count per year as a red line.
plt.plot(citations_total["Year"], citations_total["Citation Count"],
         '-o', color="red", label="Total Citation Count")

# Plot cumulative citation count as a blue line.
plt.plot(citations_total["Year"], citations_total["Cumulative Citation Count"],
         '-o', color="blue", label="Cumulative Citation Count")

plt.xlabel("Year", fontsize=12)
plt.ylabel("Citation Count", fontsize=12)
plt.title("Evolution of Total & Cumulative Citation Counts for Benchmarks", fontsize=14)
plt.legend(loc="best")
plt.tight_layout()
plt.show()

"""BENCHMARK TYPE DIVERSITY"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Step 1: Ensure 'Year' is numeric (if not already done) ---
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Year'])
df['Year'] = df['Year'].astype(int)

# --- Step 2: Aggregate benchmark types per year ---
# We assume "CleanTypeStr" contains comma-separated benchmark type labels.
year_to_types = defaultdict(set)
for _, row in df.iterrows():
    year = row["Year"]
    types_str = row.get("CleanTypeStr", "")
    if isinstance(types_str, str) and types_str.strip():
        # Split by comma and add each cleaned type to the set for the year.
        types = [t.strip().lower() for t in types_str.split(",") if t.strip()]
        for t in types:
            year_to_types[year].add(t)

# --- Step 3: Compute yearly and cumulative diversity metrics ---
all_types = set()
results = []
for year in sorted(year_to_types.keys()):
    unique_count = len(year_to_types[year])
    all_types.update(year_to_types[year])
    cumulative_count = len(all_types)
    results.append({
        "Year": year,
        "Unique Benchmark Types": unique_count,
        "Cumulative Unique Benchmark Types": cumulative_count
    })

df_diversity = pd.DataFrame(results)
print(df_diversity)

# --- Step 4: Plot the Evolution of Benchmark Diversity ---
sns.set_style("white")  # Plain background, no grid lines.
plt.figure(figsize=(10, 6))

# Plot yearly unique types in blue.
plt.plot(df_diversity["Year"], df_diversity["Unique Benchmark Types"],
         marker="o", color="blue", label="Yearly Unique Types")
# Plot cumulative unique types in green.
plt.plot(df_diversity["Year"], df_diversity["Cumulative Unique Benchmark Types"],
         marker="o", color="green", label="Cumulative Unique Types")

plt.xlabel("Year", fontsize=12)
plt.ylabel("Count of Unique Benchmark Types", fontsize=12)
plt.title("Evolution of Benchmark Diversity Over Time", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

"""THE VERY DELICATE ISSUE OF CONCENTRATION OF BENCHMARKS ABOUT PEOPLE AND INSTITUTIONS

PEOPLE AND INSTITUTIONS
"""

import numpy as np
import pandas as pd
import ast
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH = "benchmarks_cleaned.csv"

# ──────────────────────────────────────────────────────────────────────────────
#  LOAD & PARSE
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

def parse_list_cell(cell):
    """Turn a stringified Python list or comma‐sep string into a real list."""
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
    return [item.strip() for item in s.split(",") if item.strip()]

# parse our three columns
df["author_list"]      = df["author_list"].apply(parse_list_cell)
df["affiliation_list"] = df["affiliation_list"].apply(parse_list_cell)
df["country_list"]     = df["country_list"].apply(parse_list_cell)

# ──────────────────────────────────────────────────────────────────────────────
#  NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────────
def normalize_inst(inst: str) -> str:
    """Collapse variants like Google/Google Brain/DeepMind, UC Berkeley, etc."""
    inst_clean = inst.strip()
    low = inst_clean.lower()
    # UC Berkeley
    if low in ('university of california, berkeley', 'uc berkeley'):
        return 'UC Berkeley'
    # Google family → Google
    if low in ('google', 'google research', 'google brain'):
        return 'Google'
    # DeepMind variants → DeepMind
    if low in ('deepmind', 'google deepmind'):
        return 'DeepMind'
    return inst_clean

def safe_list(li):
    """Filter out empty or placeholder entries."""
    return [str(v).strip() for v in li
            if str(v).strip().lower() not in ("", "unknown", "nan")]

# ──────────────────────────────────────────────────────────────────────────────
#  METRIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def gini_coefficient(x):
    """0 = perfect equality, 1 = max inequality."""
    arr = np.array(x, dtype=float)
    if arr.size == 0 or np.sum(arr)==0:
        return 0.0
    if np.amin(arr) < 0:
        arr -= np.amin(arr)
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n+1)
    return (np.sum((2*index - n - 1) * arr) /
            (n * np.sum(arr)))

def hhi_index(counts):
    """Sum of squared market shares."""
    total = np.sum(counts)
    if total == 0:
        return 0.0
    shares = np.array(counts) / total
    return float(np.sum(shares**2))

# ──────────────────────────────────────────────────────────────────────────────
#  COUNTING
# ──────────────────────────────────────────────────────────────────────────────
author_counts  = defaultdict(int)
inst_counts    = defaultdict(int)
country_counts = defaultdict(int)

for _, row in df.iterrows():
    # authors
    for a in safe_list(row["author_list"]):
        author_counts[a] += 1

    # institutions (normalized)
    for inst in safe_list(row["affiliation_list"]):
        ni = normalize_inst(inst)
        if ni.lower() not in ("unknown", ""):
            inst_counts[ni] += 1

    # countries
    for c in safe_list(row["country_list"]):
        country_counts[c] += 1

# ──────────────────────────────────────────────────────────────────────────────
#  SUMMARY & PRINT
# ──────────────────────────────────────────────────────────────────────────────
def summarize(name, counts):
    dist = list(counts.values())
    print(f"\n— {name} —")
    print(f"  Total unique {name.lower()}: {len(dist)}")
    print(f"  Gini coefficient:         {gini_coefficient(dist):.3f}")
    print(f"  Herfindahl–Hirschman Index: {hhi_index(dist):.4f}")

summarize("Authors",      author_counts)
summarize("Institutions", inst_counts)
summarize("Countries",    country_counts)

import pandas as pd
import ast
from collections import Counter
import matplotlib.pyplot as plt

# 1. Load & parse
df = pd.read_csv("benchmarks_cleaned.csv")
df['author_list']      = df['author_list'].     apply(ast.literal_eval)
df['affiliation_list'] = df['affiliation_list'].apply(ast.literal_eval)

# 2. Normalization
def normalize_inst(inst):
    inst = inst.strip()
    low = inst.lower()
    if low in ('google', 'google research', 'google brain', 'deepmind', 'google deepmind'):
        return 'Google'
    if low in ('microsoft', 'microsoft research'):
        return 'Microsoft'
    if low in ('university of california, berkeley', 'uc berkeley'):
        return 'UC Berkeley'
    return inst

# 3. Explode & filter
all_authors = df.explode('author_list')['author_list']

raw_insts = df.explode('affiliation_list')['affiliation_list']
# keep only real strings, normalize, drop anything that maps to 'unknown' or empty
all_insts = (
    raw_insts[raw_insts.apply(lambda x: isinstance(x, str))]
    .map(normalize_inst)
    .loc[lambda s: s.str.lower() != 'unknown']
)

# 4. Count top 30
top_authors = Counter(all_authors).most_common(30)
top_insts   = Counter(all_insts).most_common(30)

# 5. Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Authors
names, vals = zip(*top_authors)
ax1.barh(names[::-1], vals[::-1], edgecolor='black')
ax1.set_title("Top 10 Authors by # of Benchmarks")
ax1.set_xlabel("Benchmarks Count")
ax1.grid(axis='x', linestyle='--', alpha=0.5)

# Institutions
names, vals = zip(*top_insts)
ax2.barh(names[::-1], vals[::-1], color='C1', edgecolor='black')
ax2.set_title("Top 10 Institutions by # of Benchmarks")
ax2.set_xlabel("Benchmarks Count")
ax2.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

import pandas as pd, numpy as np, ast
from collections import Counter
import matplotlib.pyplot as plt

# --- CONFIG ---
CSV_PATH = "benchmarks_cleaned.csv"
TOP_K = 20  # change to 10 for a tighter figure

# --- Load & parse ---
df = pd.read_csv(CSV_PATH)
df['author_list']      = df['author_list'].apply(ast.literal_eval)
df['affiliation_list'] = df['affiliation_list'].apply(ast.literal_eval)

# --- Normalize institutions (light-touch) ---
def normalize_inst(inst):
    if not isinstance(inst, str): return None
    s = inst.strip()
    low = s.lower()
    if low in {'google','google research','google brain','deepmind','google deepmind'}: return 'Google'
    if low in {'microsoft','microsoft research'}: return 'Microsoft'
    if low in {'university of california, berkeley','uc berkeley'}: return 'UC Berkeley'
    if low in {'openai, inc.', 'openai'}: return 'OpenAI'
    return s

authors = df.explode('author_list')['author_list'].dropna()
insts = df.explode('affiliation_list')['affiliation_list'].dropna().map(normalize_inst)
insts = insts[insts.notna() & (insts.str.lower()!='unknown')]

# --- Count & take top-k ---
top_auth = Counter(authors).most_common(TOP_K)
top_inst = Counter(insts).most_common(TOP_K)

# --- Anonymous labels ---
def plot_topk_anonymous(pairs, title, unit):
    _, vals = zip(*pairs)
    ranks = [f"Rank {i}" for i in range(len(vals), 1-1, -1)]
    plt.figure(figsize=(7, 7))
    plt.barh(ranks, vals[::-1], edgecolor='black')
    plt.title(title)
    plt.xlabel(f"{unit} count")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

plot_topk_anonymous(top_auth, f"Top {TOP_K} authors by # of benchmarks (anonymous)", "Benchmark")
plot_topk_anonymous(top_inst, f"Top {TOP_K} institutions by # of benchmarks (anonymous)", "Benchmark")

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def concentration_curve(series, title, waypoints=(1,3,10)):
    counts = np.array(list(Counter(series).values()))
    counts.sort()
    counts = counts[::-1]
    cum = np.cumsum(counts) / counts.sum()
    k = np.arange(1, len(cum)+1)

    plt.figure(figsize=(7,5))
    plt.plot(k, cum, marker='o', linewidth=1)
    plt.xlabel("Top k entities (by benchmark count)")
    plt.ylabel("Cumulative share of benchmarks")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)

    # annotate waypoints if within range
    for w in waypoints:
        if w <= len(cum):
            plt.axvline(w, linestyle=':', alpha=0.7)
            plt.annotate(f"Top {w}: {cum[w-1]*100:.1f}%",
                         xy=(w, cum[w-1]), xytext=(w+0.5, min(0.98, cum[w-1]+0.06)),
                         arrowprops=dict(arrowstyle='->', lw=1))
    plt.tight_layout()
    plt.show()

concentration_curve(authors, "Concentration of benchmark production (authors)")
concentration_curve(insts,   "Concentration of benchmark production (institutions)")

def lorenz_and_gini(series, title_prefix="Institutions"):
    import numpy as np
    vals = np.array(list(Counter(series).values()))
    vals = vals[vals>0]
    x = np.sort(vals)
    lorenz = np.cumsum(x) / x.sum()
    lorenz = np.insert(lorenz, 0, 0)
    p = np.linspace(0, 1, len(lorenz))
    gini = 1 - 2 * np.trapz(lorenz, p)

    plt.figure(figsize=(6,6))
    plt.plot(p, lorenz, linewidth=2, label="Lorenz curve")
    plt.plot([0,1],[0,1], linestyle='--', label="Equality line")
    plt.xlabel("Cumulative share of entities")
    plt.ylabel("Cumulative share of benchmarks")
    plt.title(f"{title_prefix}: Lorenz curve (Gini = {gini:.2f})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

lorenz_and_gini(insts, "Institutions")
lorenz_and_gini(authors, "Authors")

import pandas as pd
import ast

# ─── Load & parse ─────────────────────────────────────────────────────────────
CSV_PATH = "benchmarks_cleaned.csv"
df = pd.read_csv(CSV_PATH)

def parse_list(cell):
    """Parse a Python-list literal or comma-sep string into a real list."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s.startswith("[") and s.endswith("]"):
        try: return ast.literal_eval(s)
        except: pass
    return [i.strip().strip("'\"") for i in s.split(",") if i.strip()]

df["country_list"] = df["country_list"].apply(parse_list)

# ─── Flatten & normalize ───────────────────────────────────────────────────────
# mapping of lowercase synonym → canonical
CANON = {
    "usa":                "United States",
    "united states":     "United States",
    "uk":                 "United Kingdom",
    "united kingdom":    "United Kingdom",
    "netherlands":        "Netherlands",
    "the netherlands":    "Netherlands",
    "uae":                "United Arab Emirates",
    "united arab emirates":"United Arab Emirates",
    "south korea":        "South Korea",
    "republic of korea":  "South Korea",
}

all_countries = []
for sub in df["country_list"]:
    for c in sub:
        cl = c.strip()
        if not cl or cl.lower()=="unknown":
            continue
        norm = CANON.get(cl.lower(), cl)
        all_countries.append(norm)

# ─── Count & report ────────────────────────────────────────────────────────────
series = pd.Series(all_countries)
agg_counts = series.value_counts()

print(f"Unique countries after collapsing: {len(agg_counts)}\n")
print(agg_counts)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ────────────────────────────────────────────
#  1) your collapsed country counts
# ────────────────────────────────────────────
data = {
    "United States":           1373,
    "China":                    401,
    "United Kingdom":           243,
    "Canada":                    36,
    "South Korea":               26,
    "Germany":                   22,
    "Singapore":                 22,
    "Netherlands":               20,
    "Israel":                    18,
    "Italy":                     16,
    "Switzerland":               13,
    "France":                    11,
    "Hong Kong":                  9,
    "India":                      7,
    "United Arab Emirates":       7,
    "Japan":                      7,
    "Spain":                      6,
    "Turkey":                     5,
    "Australia":                  4,
    "Poland":                     4,
    "Saudi Arabia":               3,
    "Vietnam":                    3,
    "Kenya":                      2,
    "Cameroon":                   1,
    "Colombia":                   1,
    "Malaysia":                   1,
    "Portugal":                   1,
    "Romania":                    1,
    "Sweden":                     1,
    "Iran":                       1,
    "Norway":                     1,
    "Brazil":                     1,
    "Belgium":                    1,
    "Russia":                     1,
    "Qatar":                      1,
}

df_counts = (
    pd.DataFrame.from_dict(data, orient="index", columns=["benchmarks"])
      .reset_index()
      .rename(columns={"index": "country"})
)

# ────────────────────────────────────────────
#  2) pull Natural-Earth admin_0 shapefile via GDAL
#     (no extra pip installs needed)
# ────────────────────────────────────────────
URL     = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
VSIZIP  = f"/vsizip/vsicurl/{URL}"
world   = gpd.read_file(VSIZIP)

# rename & fix a few country names
world = world.rename(columns={"NAME": "country"})
world["country"] = world["country"].replace({
    "United States of America": "United States",
    "Republic of Korea":         "South Korea",
    "Russian Federation":        "Russia",
    "Viet Nam":                  "Vietnam",
})

# merge your counts (missing → 0)
gdf = world.merge(df_counts, on="country", how="left").fillna({"benchmarks": 0})

# get an interior point for each polygon (always a Point!)
gdf["pt"] = gdf.geometry.representative_point()

# compute bubble sizes
max_b = gdf["benchmarks"].max()
gdf["size"] = (gdf["benchmarks"] / max_b) * 3000 + 30

# ────────────────────────────────────────────
#  3) plot
# ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 9))

# pale-grey world
world.plot(ax=ax, color="#EFEFEF", edgecolor="white", linewidth=0.5)

# bubbles
sc = ax.scatter(
    gdf.pt.x, gdf.pt.y,
    s=gdf["size"],
    c=gdf["benchmarks"],
    cmap="Reds",
    alpha=0.7,
    edgecolors="k",
    linewidth=0.2
)

# annotate the top 5
for _, row in gdf.nlargest(5, "benchmarks").iterrows():
    x, y = row.pt.x, row.pt.y
    ax.text(
        x, y, int(row.benchmarks),
        ha="center", va="center",
        fontsize=10, fontweight="bold",
        color="white"
    )

ax.set_title(
    "Global LLM-Benchmark Output by Country\n(bubble area ∝ # benchmarks)",
    fontsize=18, pad=20
)
ax.axis("off")

# optional colorbar
cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.036, pad=0.04)
cbar.set_label("Number of benchmarks", fontsize=12)

plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────
#  1) your collapsed country counts
# ───────────────────────────────────────────────────────
data = {
    "United States":           1373,
    "China":                    401,
    "United Kingdom":           243,
    "Canada":                    36,
    "South Korea":               26,
    "Germany":                   22,
    "Singapore":                 22,
    "Netherlands":               20,
    "Israel":                    18,
    "Italy":                     16,
    "Switzerland":               13,
    "France":                    11,
    "Hong Kong":                  9,
    "India":                      7,
    "United Arab Emirates":       7,
    "Japan":                      7,
    "Spain":                      6,
    "Turkey":                     5,
    "Australia":                  4,
    "Poland":                     4,
    "Saudi Arabia":               3,
    "Vietnam":                    3,
    "Kenya":                      2,
    "Cameroon":                   1,
    "Colombia":                   1,
    "Malaysia":                   1,
    "Portugal":                   1,
    "Romania":                    1,
    "Sweden":                     1,
    "Iran":                       1,
    "Norway":                     1,
    "Brazil":                     1,
    "Belgium":                    1,
    "Russia":                     1,
    "Qatar":                      1,
}

df = (
    pd.DataFrame.from_dict(data, orient="index", columns=["benchmarks"])
      .sort_values("benchmarks", ascending=False)
      .reset_index()
      .rename(columns={"index":"country"})
)

# ───────────────────────────────────────────────────────
#  2) compute cumulative share + Gini
# ───────────────────────────────────────────────────────
df["cum_bench"] = df["benchmarks"].cumsum()
total = df["benchmarks"].sum()
df["cum_pct"]   = 100 * df["cum_bench"] / total

def gini(arr):
    """Compute Gini coefficient of array arr."""
    a = np.array(arr, dtype=float)
    if np.any(a < 0): a -= a.min()
    if a.sum() == 0: return 0.0
    a = np.sort(a)
    n = a.size
    idx = np.arange(1, n+1)
    return ((2*idx - n - 1) * a).sum() / (n * a.sum())

g = gini(df["benchmarks"])
print(f"Gini coefficient (benchmarks by country): {g:.3f}")

# ───────────────────────────────────────────────────────
#  3) plot Pareto chart
# ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12,6))

# bar: benchmarks per country
ax.bar(df["country"], df["benchmarks"], color="#4682B4")
ax.set_ylabel("Number of benchmarks", color="#4682B4", fontsize=12)
ax.set_xticklabels(df["country"], rotation=75, ha="right")

# twin axis: cumulative %
ax2 = ax.twinx()
ax2.plot(df["country"], df["cum_pct"], color="#E97451", marker="o")
ax2.set_ylabel("Cumulative % of total", color="#E97451", fontsize=12)
ax2.set_ylim(0, 110)

# annotate 80% line
ax2.axhline(80, color="gray", linestyle="--", linewidth=1)
ax2.text(len(df)-0.5, 82, "80%", color="gray", va="bottom", ha="right")

# title + layout
plt.title("Pareto Chart of LLM Benchmarks by Country\n" +
          f"(Gini = {g:.3f})", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

"""# THE AUHORITY CALCULATION PART"""

import os, requests, json
# Note: Set GH_TOKEN via environment variable if needed (avoid hardcoding).

#!/usr/bin/env python3
"""
enrich_benchmarks_public.py            2025-07-13
----------------------------------------------
• Public (unauth) Semantic Scholar: retries + jitter
• Single shared requests.Session for GitHub to avoid socket exhaustion
• Robust repo-URL cleaner (“tree/…”, “blob/…”, trailing slashes, .git)
"""

import os, ast, math, time, re, random, sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
from requests.exceptions import ConnectionError as ReqConnectionError

# ── CONFIG ──────────────────────────────────────────────────────────
WEIGHTS      = dict(alpha=0.50, beta=0.30, gamma=0.10, delta=0.10)
MAX_RETRIES  = 6        # for public S2 429s
BASE_SLEEP   = 1.2      # s after each S2 call
JITTER       = 0.4      # ± jitter on every sleep

GH_SLEEP     = 0.2      # 0 → fastest; bump if you still hit OS limits
GH_SESSION   = requests.Session()
GH_SESSION.headers.update({
    "Authorization": f"Bearer {os.getenv('GH_TOKEN','')}",
    "Accept": "application/vnd.github+json",
    "User-Agent": "benchmark-authority-script"
})

S2_FIELDS = "citationCount,year"

# ── HELPERS ─────────────────────────────────────────────────────────
REPO_CLEAN_RE = re.compile(
    r"""
    ^https?://github\.com/     # protocol + domain
    (?P<owner>[^/]+)/          # owner
    (?P<repo>[^/#\?]+)         # repo name
    (?:
        (?:/tree|/blob)/[^/]+  # optional /tree/<branch> or /blob/<branch>
        .*
    )?                         # ignore the rest
    """,
    re.VERBOSE,
)

def clean_repo_url(url: str):
    if not isinstance(url, str) or "github.com" not in url:
        return None
    m = REPO_CLEAN_RE.match(url.strip())
    return f"{m.group('owner')}/{m.group('repo')}" if m else None

def get_cites_public(arxiv_id: str):
    url   = f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
    tries = 0
    wait  = BASE_SLEEP
    while tries <= MAX_RETRIES:
        try:
            r = requests.get(url, params={"fields": S2_FIELDS}, timeout=15)
            if r.status_code == 200:
                js = r.json()
                return js.get("citationCount", 0), js.get("year", date.today().year)
            if r.status_code != 429:        # 4xx/5xx other than 429
                break
        except Exception as e:
            print(f"[S2] {arxiv_id}: {e}", file=sys.stderr)
        tries += 1
        time.sleep(wait + random.uniform(-JITTER, JITTER))
        wait *= 2
    # fallback
    print(f"[S2] {arxiv_id}: giving 0 citations after {tries} retries", file=sys.stderr)
    return 0, date.today().year

def get_repo_stats(url: str):
    owner_repo = clean_repo_url(url)
    if not owner_repo:
        return dict(stars=0, forks=0, watchers=0, open_issues=0, pushed_at=None)

    try:
        r = GH_SESSION.get(f"https://api.github.com/repos/{owner_repo}", timeout=10)
        if r.status_code != 200:
            print(f"[GH ] {owner_repo}: HTTP {r.status_code}", file=sys.stderr)
            return dict(stars=0, forks=0, watchers=0, open_issues=0, pushed_at=None)
        js = r.json()
        time.sleep(GH_SLEEP)
        return dict(
            stars=js.get("stargazers_count", 0),
            forks=js.get("forks_count", 0),
            watchers=js.get("subscribers_count", 0),
            open_issues=js.get("open_issues_count", 0),
            pushed_at=js.get("pushed_at"),
        )
    except ReqConnectionError as e:
        print(f"[GH ] {owner_repo}: {e}", file=sys.stderr)
        time.sleep(2)           # short back-off
        return dict(stars=0, forks=0, watchers=0, open_issues=0, pushed_at=None)

def authority(row, w=WEIGHTS):
    cite   = w["alpha"] * math.log1p(row.cites_per_mo)
    gh     = w["beta"]  * (
                math.log1p(row.stars) +
                0.5 * math.log1p(row.forks) +
                0.5 * math.log1p(row.watchers)
             )
    size   = w["gamma"] * math.log1p(row.num_examples)
    author = w["delta"] * math.log1p(row.num_authors)
    return cite + gh + size + author

# ── MAIN ────────────────────────────────────────────────────────────
def main():
    src = "benchmarks_cleaned.csv"
    dst = "benchmarks_with_authority.csv"
    if not Path(src).exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src).query("arxiv_id.notna()").reset_index(drop=True)

    # 1) Semantic Scholar (public endpoint, retries)
    print("📚  Citation crawl (public S2)…")
    citations, years = [], []
    for aid in tqdm(df["arxiv_id"]):
        c, y = get_cites_public(aid)
        citations.append(c); years.append(y)
        time.sleep(BASE_SLEEP + random.uniform(-JITTER, JITTER))

    df["citations"]    = citations
    df["cites_per_mo"] = [
        c / max((date.today().year - y) * 12, 1) for c, y in zip(citations, years)
    ]

    # 2) GitHub stats (shared session)
    print("🐙  GitHub metadata…")
    gh_rows = [get_repo_stats(u) for u in tqdm(df["Code repository"].fillna(""))]
    df = pd.concat([df, pd.DataFrame(gh_rows)], axis=1)

    # 3) Local numeric fields
    df["num_examples"] = pd.to_numeric(
        df["Number of examples"].astype(str).str.replace(",", "").str.strip(),
        errors="coerce"
    ).fillna(0)
    df["num_authors"] = df["author_list"].apply(
        lambda s: len(ast.literal_eval(s)) if isinstance(s, str) else 1
    )

    # 4) Score
    raw = df.apply(authority, axis=1)
    df["Authority_raw"]     = raw
    df["Authority (0-100)"] = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    df.to_csv(dst, index=False)

    print(f"\n✅  Saved → {dst}")
    print(df.sort_values("Authority (0-100)", ascending=False)
            [["Name","Authority (0-100)","citations","stars"]]
            .head(10).to_string(index=False))

if __name__ == "__main__":
    if not os.getenv("GH_TOKEN"):
        print("⚠️  GH_TOKEN not set — you’ll drop to 60 GitHub calls/h.", file=sys.stderr)
    main()

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("benchmarks_with_authority.csv")

plt.figure(figsize=(8, 5))
plt.hist(df["Authority_raw"], bins=20)
plt.yscale("log")                       # log y-axis makes the long tail visible
plt.title("Distribution of Raw Authority Scores")
plt.xlabel("Raw Authority Score")
plt.ylabel("Number of Benchmarks (log scale)")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the enriched file (adjust path if needed)
df = pd.read_csv("benchmarks_with_authority.csv")

# Take the raw authority scores and sort descending
raw = df["Authority_raw"].dropna().values
raw_sorted = np.sort(raw)[::-1]  # descending
ranks = np.arange(1, len(raw_sorted) + 1)
ccdf = ranks / len(raw_sorted)   # complementary CDF

# Basic figure
plt.figure(figsize=(6, 4.5))
plt.loglog(raw_sorted, ccdf, marker="o", linestyle="none", markersize=4)

# Label the five most authoritative benchmarks
top5 = df.nlargest(5, "Authority_raw")
for _, row in top5.iterrows():
    plt.text(row["Authority_raw"], (np.where(raw_sorted == row["Authority_raw"])[0][0] + 1) / len(raw_sorted),
             row["Name"], fontsize=7, ha="left", va="center")

plt.title("Complementary Cumulative Distribution of Raw Authority Scores")
plt.xlabel("Raw Authority Score")
plt.ylabel("P(Authority ≥ x)")
plt.grid(True, which="both", linestyle="", linewidth=0.5)
plt.tight_layout()
plt.show()

import powerlaw, pandas as pd
raw = pd.read_csv("benchmarks_with_authority.csv")["Authority_raw"].values
fit = powerlaw.Fit(raw, discrete=False)        # treat as continuous
print("α =", fit.power_law.alpha, "  xmin =", fit.power_law.xmin)
R, p = fit.distribution_compare('power_law', 'lognormal')
print("Log-likelihood ratio R =", R, "  p-value =", p)

# ----------------------------------------------------------
# 0) Install missing package (one-time)
# ----------------------------------------------------------
import importlib, subprocess, sys
if importlib.util.find_spec("powerlaw") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "powerlaw"])

# ----------------------------------------------------------
# 1) Load data
# ----------------------------------------------------------
import pandas as pd, numpy as np, powerlaw, matplotlib.pyplot as plt

csv_path = "benchmarks_with_authority.csv"      # <-- update if needed
df = pd.read_csv(csv_path)
data = df["Authority_raw"].dropna().values      # raw scores as a 1-D array

# ----------------------------------------------------------
# 2) Fit using Clauset et al. method
# ----------------------------------------------------------
fit = powerlaw.Fit(data, discrete=False, verbose=False)

alpha  = fit.power_law.alpha
xmin   = fit.power_law.xmin
ks     = fit.power_law.KS()

# Compare to log-normal
R, p   = fit.distribution_compare('power_law', 'lognormal')

print(f"α (power-law exponent):        {alpha:.3f}")
print(f"x_min:                         {xmin:.3f}")
print(f"KS distance:                   {ks:.4f}")
print(f"LLR (power-law vs log-normal): {R:.3f}")
print(f"p-value (R>0 favours power-law): {p:.4f}")

# ----------------------------------------------------------
# 3) Plot CCDF with fitted power-law line
# ----------------------------------------------------------
fig = plt.figure(figsize=(6,4.5))
ax  = fig.add_subplot(111)

# empirical CCDF
fit.plot_ccdf(ax=ax, marker='o', markersize=4, linestyle='none', color='tab:blue')

# fitted power-law CCDF
fit.power_law.plot_ccdf(ax=ax, color='tab:red', linestyle='--', label=f"Power-law fit\nα={alpha:.2f}, xmin={xmin:.2f}")

ax.set_xlabel("Raw Authority Score")
ax.set_ylabel("P(Authority ≥ x)")
ax.set_title("CCDF of Raw Authority Scores\nwith Clauset-fit Power-law")
ax.grid(True, which="both", ls="--", lw=0.4)
ax.legend(frameon=False)
plt.tight_layout(); plt.show()

R, p = fit.distribution_compare('truncated_power_law', 'lognormal')
print(R)
print(p)
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'squarify', '--quiet'], check=False)

# ------------------------------------------------------------
# 0.  Install treemap lib once if missing
# ------------------------------------------------------------
import importlib, subprocess, sys, ast, pandas as pd
if importlib.util.find_spec("squarify") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "squarify"])

import squarify, matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib.patches import Patch

CSV = "benchmarks_with_authority.csv"

# ------------------------------------------------------------
# 1. Canonical maps
# ------------------------------------------------------------
INST_ALIAS = {
    **dict.fromkeys(['google','google brain','google research','deepmind','google deepmind'], 'Google'),
    **dict.fromkeys(['microsoft','microsoft research','microsoft research asia'], 'Microsoft'),
    **dict.fromkeys(['university of california, berkeley','uc berkeley'], 'UC Berkeley'),
}
COUNTRY_CANON = {
    "usa":"United States","united states":"United States",
    "uk":"United Kingdom","united kingdom":"United Kingdom",
    "uae":"United Arab Emirates",
    "south korea":"South Korea","republic of korea":"South Korea",
    "the netherlands":"Netherlands",
}
inst = lambda x: INST_ALIAS.get(x.strip().lower(), x.strip())
ctry = lambda x: COUNTRY_CANON.get(str(x).strip().lower(), str(x).strip() or "Unknown")

# ------------------------------------------------------------
# 2. Load CSV & explode rows
# ------------------------------------------------------------
df = pd.read_csv(CSV)
records=[]
for _,r in df.iterrows():
    try:
        insts=[inst(i) for i in ast.literal_eval(r["affiliation_list"])]
        ctrs =[ctry(c) for c in ast.literal_eval(r["country_list"])]
    except Exception: continue
    ctrs += ['Unknown']*max(0,len(insts)-len(ctrs))
    for i,c in zip(insts,ctrs):
        records.append((i,c,r["Authority_raw"]))
tmp = pd.DataFrame(records, columns=["inst","country","auth"])

# ------------------------------------------------------------
# 3. Aggregate by institution
# ------------------------------------------------------------
agg = (tmp.groupby("inst")
          .agg(authority=("auth","sum"),
               country=("country",lambda s:s.value_counts().idxmax()))
          .sort_values("authority",ascending=False)
          .reset_index())

# ------------------------------------------------------------
# 4. Top-N + “Other”
# ------------------------------------------------------------
TOP_N=20
top=agg.head(TOP_N).copy()
other=agg["authority"][TOP_N:].sum()
if other:
    top = pd.concat([top,
                     pd.DataFrame([{"inst":"Other institutions",
                                    "country":"Other",
                                    "authority":other}])],
                    ignore_index=True)

# ------------------------------------------------------------
# 5. Okabe–Ito palette  (colour-blind safe)
# ------------------------------------------------------------
okabe = [           # Okabe–Ito hue names in comment
    "#E69F00",      # orange
    "#56B4E9",      # sky-blue
    "#009E73",      # bluish-green
    "#F0E442",      # yellow
    "#0072B2",      # navy
    "#D55E00",      # vermilion
    "#CC79A7",      # reddish-purple
]

country_palette = {
    "United States": okabe[1],   # sky-blue   → largest group, high legibility
    "China":         okabe[5],   # vermilion  → warm, distinct from US
    "United Kingdom":okabe[3],   # navy       → deeper blue, differentiates from US
    "South Korea":   okabe[4],   # bluish-green (if you ever include SK)
    "Netherlands":   okabe[3],   # yellow     (optional)
    "Unknown":       "#8C8C8C",  # mid-grey
    "Other":         "#D9D9D9",  # light-grey
}

colors = top["country"].map(lambda c: country_palette.get(c, okabe[2]))

# ------------------------------------------------------------
# 6. Draw treemap
# ------------------------------------------------------------
mpl.rcParams.update({"font.family":"DejaVu Sans","font.size":9})
fig = plt.figure(figsize=(13,7), facecolor="white")

rects_out = squarify.plot(
    sizes = top["authority"],
    color = colors,
    pad   = True,
    bar_kwargs={"edgecolor":"white","linewidth":2}
)

# rects_out is list (old) or Axes (new) → unify
if isinstance(rects_out, list):
    patches = rects_out
    ax = plt.gca()
else:                       # Axes
    ax = rects_out
    patches = ax.patches

ax.axis("off")
ax.set_title("Concentration of LLM Benchmark Authority by Institution",
             fontsize=18, weight="bold", pad=16)

# ------------------------------------------------------------
# 7. Add labels with auto contrast
# ------------------------------------------------------------
for patch, (name, score) in zip(patches, zip(top["inst"], top["authority"])):
    x, y = patch.get_x(), patch.get_y()
    dx, dy = patch.get_width(), patch.get_height()
    rgb = mpl.colors.to_rgb(patch.get_facecolor())
    luminance = 0.2126*rgb[0] + 0.7152*rgb[1] + 0.0722*rgb[2]
    text_col = "white" if luminance < 0.5 else "black"
    ax.text(x+dx/2, y+dy/2,
            f"{name}\n({score:.1f})",
            ha="center", va="center",
            color=text_col, weight="bold", fontsize=9)

# Legend
legend_order = list(dict.fromkeys(top["country"]))   # preserve order
handles = [Patch(facecolor=country_palette.get(c, okabe[2]), label=c)
           for c in legend_order]
fig.legend(handles=handles, title="Country", title_fontsize=11,
           loc="lower right", bbox_to_anchor=(0.70, -0.10),
           frameon=False, ncol=len(handles), handlelength=1)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 8. Print ranked table
# ------------------------------------------------------------
print("\nTop institutions by total raw authority")
print("----------------------------------------")
print(agg[["inst","country","authority"]].head(30).to_string(index=False))



"""THE NETWORK PART"""

# ──────────────────────────────────────────────────────────────────────────────
#  0.  Imports  ▸ install anything missing on-the-fly
# ──────────────────────────────────────────────────────────────────────────────
import sys, subprocess, json, ast, itertools
def _pip(pkg): subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])
for p in ["pandas", "networkx", "matplotlib", "python-louvain"]:   # community alg.
    try: __import__(p.split("-")[0])
    except ImportError: _pip(p)

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import community as community_louvain   # Louvain community detection

CSV = "benchmarks_cleaned.csv"

# ──────────────────────────────────────────────────────────────────────────────
#  1.  Load + parse the three list-columns
# ──────────────────────────────────────────────────────────────────────────────
def tolist(x):
    """Convert literal python list or comma-sep string → list[str]"""
    if pd.isna(x) or str(x).strip().lower() in {"", "unknown", "nan"}:
        return []
    if isinstance(x, list): return x
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try: return ast.literal_eval(s)
        except Exception: pass
    return [t.strip() for t in s.split(",") if t.strip()]

df = pd.read_csv(CSV, converters={
        "author_list":      tolist,
        "affiliation_list": tolist,
        "country_list":     tolist,
})

# ──────────────────────────────────────────────────────────────────────────────
#  Institution normalisation (your exact alias spec)
# ──────────────────────────────────────────────────────────────────────────────
INST_ALIAS = {
    **dict.fromkeys(
        ['google', 'google brain', 'google research',
         'deepmind', 'google deepmind'],              'Google'),
    **dict.fromkeys(
        ['microsoft', 'microsoft research',
         'microsoft research asia'],                  'Microsoft'),
    **dict.fromkeys(
        ['university of california, berkeley',
         'uc berkeley'],                              'UC Berkeley'),
}

def norm_inst(name: str) -> str:
    """Lower-case + strip → look-up in alias table → title-case fallback."""
    key = str(name).lower().strip()
    return INST_ALIAS.get(key, name.strip())

df["affiliation_list"] = df["affiliation_list"].apply(lambda L: [norm_inst(i) for i in L])

# ──────────────────────────────────────────────────────────────────────────────
#  2.  Build a tripartite NetworkX graph
#      nodes carry a "type" attribute: benchmark / author / institution
# ──────────────────────────────────────────────────────────────────────────────
G = nx.Graph()

for idx, row in df.iterrows():
    bench_id   = f"B:{row['Name']}"
    authors    = row["author_list"]
    insts      = row["affiliation_list"]

    G.add_node(bench_id, ntype="benchmark", label=row["Name"])
    for au, ins in itertools.zip_longest(authors, insts, fillvalue=None):
        if au:
            au_id = f"A:{au}"
            G.add_node(au_id, ntype="author", label=au)
            G.add_edge(bench_id, au_id)

        if ins:
            ins_id = f"I:{ins}"
            G.add_node(ins_id, ntype="institution", label=ins)
            # connect inst ↔ author (rather than bench) so inst centrality is meaningful
            if au:
                G.add_edge(au_id, ins_id)

print(f"Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# ──────────────────────────────────────────────────────────────────────────────
#  3.  Centralisation metrics
# ──────────────────────────────────────────────────────────────────────────────
def top(d, k=10):                    # helper for pretty-printing
    return "\n".join(f"  {k_:45s}{v:7.3f}" for k_, v in d[:k])

# a) Degree centrality (raw + Gini for inequality)
deg_cent = nx.degree_centrality(G)
gini_deg = lambda vals: (                 # quick Gini implementation
    (lambda y: ( (len(y)+1-2*(y.cumsum().sum()/y.sum())) / len(y) )
     )(pd.Series(sorted(vals)))
)
gini_val = gini_deg(list(deg_cent.values()))
print(f"\nGlobal degree-centrality Gini: {gini_val:.3f}")

# b) Betweenness centrality  (slowest part; run only on largest 3-core to speed up)
core = nx.k_core(G, k=3)
bet_cent = nx.betweenness_centrality(core)
print("Computed betweenness on 3-core:", core.number_of_nodes(),"nodes")

# c) Louvain communities on the 3-core
part = community_louvain.best_partition(core)

# ──────────────────────────────────────────────────────────────────────────────
#  4.  Display highlights
# ──────────────────────────────────────────────────────────────────────────────
def print_league(title, cent_dict, node_type_prefix, k=10):
    sub = {n:c for n,c in cent_dict.items() if n.startswith(node_type_prefix)}
    rows = sorted(sub.items(), key=lambda kv: kv[1], reverse=True)[:k]
    print(f"\nTop {k} {title}")
    print(top([(G.nodes[n]['label'], v) for n,v in rows], k=k))

print_league("BENCHMARK hubs (degree)",  deg_cent, "B:")
print_league("AUTHOR     hubs (degree)", deg_cent, "A:")
print_league("INSTITUTION hubs (degree)",deg_cent, "I:")
print_league("AUTHOR betweenness",       bet_cent,"A:")

# ──────────────────────────────────────────────────────────────────────────────
#  5.  Quick visualisation: 2-D force layout of the backbone
#      (authors + institutions only, largest component)
# ──────────────────────────────────────────────────────────────────────────────
backbone = core.subgraph([n for n in core if not n.startswith("B:")]).copy()
pos      = nx.spring_layout(backbone, k=0.15, seed=42)

plt.figure(figsize=(12,12))
# node colours by type
col_map = {"A:":"#1f78b4", "I:":"#33a02c"}
node_cols = [col_map["A:" if n.startswith("A:") else "I:"] for n in backbone]
sizes = [deg_cent[n]*2000 for n in backbone]   # scale by degree centrality

nx.draw_networkx_nodes(backbone, pos, node_size=sizes, node_color=node_cols, alpha=.8)
nx.draw_networkx_edges(backbone, pos, alpha=0.05, width=0.3)

# annotate a few very central nodes
for n in sorted(backbone, key=lambda x: deg_cent[x], reverse=True)[:12]:
    x, y = pos[n]
    plt.text(x, y, G.nodes[n]["label"], fontsize=8,
             ha="center", va="center", color="white" if n.startswith("I:") else "black",
             bbox=dict(facecolor="black" if n.startswith("I:") else "white",
                       alpha=0.6, pad=1, edgecolor="none"))

plt.title("Influence Backbone of the Benchmark–Author–Institution Network")
plt.axis("off")
plt.tight_layout()
plt.show()

import pandas as pd, ast, networkx as nx, matplotlib.pyplot as plt, numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
CSV_PATH = "benchmarks_cleaned.csv"

INST_ALIAS = {
    **dict.fromkeys(
        ["google","google brain","google research","deepmind","google deepmind"],
        "Google"),
    **dict.fromkeys(
        ["microsoft","microsoft research","microsoft research asia"],
        "Microsoft"),
    **dict.fromkeys(
        ["university of california, berkeley","uc berkeley"],
        "UC Berkeley"),
}
DROP = {"unknown",""}            # silently skip these

# ─── LOAD + NORMALISE ──────────────────────────────────────────────────────
df = pd.read_csv(
    CSV_PATH,
    converters={"author_list":ast.literal_eval,
                "affiliation_list":ast.literal_eval})

def canon(inst):
    raw = str(inst).strip()
    return INST_ALIAS.get(raw.lower(), raw)

df["inst_norm"] = df["affiliation_list"].apply(
    lambda L: [canon(i) for i in L if canon(i).lower() not in DROP])

# ─── 1 ▸ COLLAB-GRAPH (“hair-ball”) ────────────────────────────────────────
G = nx.Graph()
for insts in df["inst_norm"]:
    uniq = list(set(insts))
    for i,a in enumerate(uniq):
        for b in uniq[i+1:]:
            G.add_edge(a,b,weight=G[a][b]["weight"]+1 if G.has_edge(a,b) else 1)

deg      = G.degree()
sizes    = [deg[n]*50 for n in G]                # tweak scale if needed
w_scaled = [G[u][v]["weight"]*0.3 for u,v in G.edges]

pos = nx.spring_layout(G,k=0.5,seed=42)

plt.figure(figsize=(14,14))
nx.draw_networkx_edges(G,pos,width=w_scaled,alpha=0.35)
nx.draw_networkx_nodes(G,pos,node_size=sizes,node_color="skyblue",alpha=0.9,
                       edgecolors="white",linewidths=0.4)
# label only high-degree hubs for legibility
labels = {n:n for n,d in deg if d>15}
nx.draw_networkx_labels(G,pos,labels,font_size=9)
plt.title("Institution collaboration network\n(node size = degree, edge width = # shared benchmarks)")
plt.axis("off")
plt.show()

# ─── 2 ▸ TOP-5 INSTITUTIONS’ SHARE OVER TIME ──────────────────────────────
exploded = (df[["year","inst_norm"]]
            .explode("inst_norm")
            .rename(columns={"inst_norm":"institution"}))

yearly = (exploded.groupby(["year","institution"])
                   .size()
                   .reset_index(name="benchmarks"))

top5   = (yearly.groupby("institution")["benchmarks"]
                 .sum()
                 .nlargest(5)
                 .index)

pivot  = (yearly[yearly["institution"].isin(top5)]
          .pivot(index="year",columns="institution",values="benchmarks")
          .fillna(0)
          .sort_index())

shares = pivot.div(pivot.sum(axis=1),axis=0)

fig,ax = plt.subplots(figsize=(12,6))
ax.stackplot(shares.index,shares.T,labels=shares.columns)
ax.set_title("Share of benchmark collaborations captured by top-5 institutions")
ax.set_xlabel("Year"); ax.set_ylabel("Share of collaborations")
ax.legend(loc="upper left")
plt.tight_layout(); plt.show()

"""Here we try to compare centralization rates between models and benchmarks"""
import sys, subprocess
subprocess.run([sys.executable, '-m', 'pip', 'install', 'tabulate'], check=False)

# =============================================================
# Centralisation trends – full affiliation allocation & sanity plots
# =============================================================
import pandas as pd
import numpy as np
import ast, re
from pathlib import Path
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ---------- Helpers ---------------------------------------------------
def hhi(v):
    s = v / v.sum()
    return (s**2).sum()

def gini(v):
    v = np.sort(np.array(v, dtype=float))
    if v.sum() == 0 or len(v) == 0:
        return 0.0
    n = len(v)
    cum = np.cumsum(v)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n

def parse_affils(text):
    """Parse the affiliation_list cell into a Python list of clean org names."""
    try:
        lst = ast.literal_eval(text)
        return [re.sub(r"\s+\(.*\)$", "", org).strip() for org in lst]
    except:
        return []

# ---------- File paths (adjust if needed) --------------------------
ROOT      = Path("~/Desktop").expanduser()
fn_models = ROOT / "assets_with_metadata_2.csv"
fn_bench  = ROOT / "benchmarks_with_authority.csv"

# ---------- Load & tidy MODELS --------------------------------------
models = pd.read_csv(fn_models, low_memory=False)
models["created_date"] = pd.to_datetime(models["created_date"], errors="coerce")
models["monthly_active_users"] = pd.to_numeric(
    models.get("monthly_active_users", 0), errors="coerce"
).fillna(0)

# Institution: prefer org_cluster, else cleaned_org, else organization
models["institution"] = (
    models.get("org_cluster")
          .where(models.get("org_cluster").notna(),
                 models["cleaned_org"].fillna(models["organization"]).str.strip())
)

models_tidy = (
    models.assign(
        facet     = "models",
        year      = models["created_date"].dt.year,
        authority = models["monthly_active_users"]
    )
    .loc[:, ["facet", "year", "institution", "authority"]]
)

# ---------- Load & allocate BENCHMARKS -----------------------------
bench = pd.read_csv(fn_bench, low_memory=False)
bench["authority_val"] = pd.to_numeric(bench["Authority_raw"], errors="coerce").fillna(0)
bench["affils"] = bench["affiliation_list"].apply(parse_affils)

# Explode and fractional-allocate benchmark authority across all co-orgs
bench_expanded = (
    bench.explode("affils")
         .rename(columns={"affils": "institution"})
)
bench_expanded["frac_authority"] = (
    bench_expanded["authority_val"] /
    bench_expanded.groupby("Name")["institution"].transform("count")
)

bench_expanded["year"] = pd.to_numeric(bench_expanded["year"], errors="coerce").astype(int)
bench_grouped = (
    bench_expanded
      .groupby(["year", "institution"])["frac_authority"]
      .sum()
      .reset_index()
      .rename(columns={"frac_authority": "authority"})
)

bench_tidy = bench_grouped.assign(facet="benchmarks").loc[:, ["facet", "year", "institution", "authority"]]

# ---------- Combine & filter noise ---------------------------------
tidy = pd.concat([models_tidy, bench_tidy], ignore_index=True)
tidy = tidy.dropna(subset=["year","authority"])
tidy = tidy.query("authority >= 1")

# ---------- Compute yearly HHI & Gini per facet --------------------
records = []
for (facet, year), grp in tidy.groupby(["facet", "year"]):
    inst_tot = grp.groupby("institution")["authority"].sum()
    records.append({
        "facet": facet,
        "year":  int(year),
        "hhi":   hhi(inst_tot),
        "gini":  gini(inst_tot)
    })
central = pd.DataFrame(records).sort_values(["facet", "year"])

# ---------- Estimate log‐linear trends ----------------------------
results = []
for facet, grp in central.groupby("facet"):
    slope, _, r_value, p_value, _ = linregress(grp["year"], np.log(grp["hhi"]))
    results.append({
        "facet": facet,
        "pct_per_year":   slope * 100,
        "half_or_double": (np.log(2) / abs(slope)) if slope != 0 else np.inf,
        "trend":          "centralising" if slope > 0 else "de-centralising",
        "r2":             r_value**2,
        "pvalue":         p_value
    })
rates = pd.DataFrame(results).sort_values("pct_per_year", ascending=False)
print(rates.to_markdown(index=False, floatfmt=".2f"))

# ---------- Sanity‐check plots --------------------------------------
for facet, grp in central.groupby("facet"):
    plt.figure()
    plt.plot(grp["year"], grp["gini"], marker="o")
    plt.title(f"{facet.capitalize()} – HHI over time")
    plt.xlabel("Year")
    plt.ylabel("HHI")
    plt.ylim(0, 1)
    plt.show()

# Full update code with robust year parsing and all affiliations

import pandas as pd
import numpy as np
import ast, re
from pathlib import Path
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ---------- Helpers ---------------------------------------------------------
def hhi(v):
    s = v / v.sum()
    return (s**2).sum()

def gini(v):
    arr = np.sort(np.array(v, dtype=float))
    if arr.sum() == 0 or len(arr) == 0:
        return 0.0
    n   = len(arr)
    cum = np.cumsum(arr)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n

def parse_affils(text):
    try:
        lst = ast.literal_eval(text)
        return [re.sub(r"\s+\(.*\)$", "", org).strip() for org in lst]
    except:
        return []

# ---------- File paths -----------------------------------------------------
ROOT      = Path("~/Desktop").expanduser()
fn_models = ROOT / "assets_with_metadata_2.csv"
fn_bench  = ROOT / "benchmarks_with_authority.csv"

# ---------- 1. Load & tidy MODELS ----------------------------------------
models = pd.read_csv(fn_models, low_memory=False)
models["created_date"] = pd.to_datetime(models["created_date"], errors="coerce")

# parse year and drop missing
models["year"] = models["created_date"].dt.year
models = models.dropna(subset=["year"])
models["year"] = models["year"].astype(int)

# choose authority proxy & institution
models["monthly_active_users"] = pd.to_numeric(models.get("monthly_active_users", 0),
                                              errors="coerce").fillna(0)
models["institution"] = (
    models.get("org_cluster")
          .where(models.get("org_cluster").notna(),
                 models["cleaned_org"].fillna(models["organization"]).str.strip())
)

models_tidy = models.loc[:, ["year", "institution", "monthly_active_users"]].rename(
    columns={"monthly_active_users": "authority"}
)
models_tidy["facet"] = "models"

# ---------- 2. Load & allocate BENCHMARKS -------------------------------
bench = pd.read_csv(fn_bench, low_memory=False)
bench["authority_val"] = pd.to_numeric(bench.get("Authority_raw", bench.get("Authority (0-100)", 0)),
                                       errors="coerce").fillna(0)
bench["affils"] = bench["affiliation_list"].apply(parse_affils)

# explode and fractional allocate
bench_expanded = bench.explode("affils").rename(columns={"affils": "institution"})
bench_expanded["frac_authority"] = (
    bench_expanded["authority_val"] /
    bench_expanded.groupby("Name")["institution"].transform("count")
)

# parse year and drop missing
bench_expanded["year"] = pd.to_numeric(bench_expanded["year"], errors="coerce")
bench_expanded = bench_expanded.dropna(subset=["year"])
bench_expanded["year"] = bench_expanded["year"].astype(int)

bench_grouped = (
    bench_expanded
    .groupby(["year", "institution"])["frac_authority"]
    .sum()
    .reset_index()
    .rename(columns={"frac_authority": "authority"})
)
bench_grouped["facet"] = "benchmarks"
bench_tidy = bench_grouped.loc[:, ["year", "institution", "authority", "facet"]]

# ---------- 3. Combine & filter noise ------------------------------------
tidy = pd.concat([models_tidy, bench_tidy], ignore_index=True)
tidy = tidy.query("authority >= 1")

# ---------- 4. Compute yearly HHI & Gini per facet -----------------------
records = []
for (facet, year), grp in tidy.groupby(["facet", "year"]):
    inst_tot = grp.groupby("institution")["authority"].sum()
    records.append({
        "facet": facet,
        "year":  year,
        "hhi":   hhi(inst_tot),
        "gini":  gini(inst_tot)
    })
central = pd.DataFrame(records).sort_values(["facet", "year"])

# ---------- 5. Estimate log-linear trends --------------------------------
results = []
for facet, grp in central.groupby("facet"):
    slope, _, r_val, p_val, _ = linregress(grp["year"], np.log(grp["hhi"]))
    results.append({
        "facet": facet,
        "pct_per_year":   slope * 100,
        "half_or_double": (np.log(2) / abs(slope)) if slope != 0 else np.inf,
        "trend":          "centralising" if slope > 0 else "de-centralising",
        "r2":             r_val**2,
        "pvalue":         p_val
    })
rates = pd.DataFrame(results).sort_values("pct_per_year", ascending=False)
print(rates.to_markdown(index=False, floatfmt=".2f"))

# ---------- 6. Sanity-check plots ----------------------------------------
# 6a. One figure per metric, with both facets
for metric in ["hhi", "gini"]:
    plt.figure()
    for facet, grp in central.groupby("facet"):
        plt.plot(grp["year"], grp[metric], marker="o", label=facet.capitalize())
    plt.title(f"{metric.upper()} over time by facet")
    plt.xlabel("Year")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.show()

# 6b. One figure per facet, showing both metrics
for facet, grp in central.groupby("facet"):
    plt.figure()
    plt.plot(grp["year"], grp["hhi"], marker="o", label="HHI")
    plt.plot(grp["year"], grp["gini"], marker="s", label="Gini")
    plt.title(f"{facet.capitalize()} – HHI & Gini over time")
    plt.xlabel("Year")
    plt.ylabel("Metric")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

# =============================================================================
# Centralisation trends – model-count authority  &  benchmark fractional shares
# =============================================================================
import pandas as pd, numpy as np, ast, re
from pathlib import Path
from scipy.stats import linregress
import matplotlib.pyplot as plt

# ---------- helpers ----------------------------------------------------------
def hhi(v):
    s = v / v.sum()
    return (s**2).sum()

def gini(v):
    arr = np.sort(np.asarray(v, float))
    if arr.sum() == 0 or len(arr) == 0:
        return 0.0
    n   = len(arr)
    cum = np.cumsum(arr)
    return (n + 1 - 2 * cum.sum() / cum[-1]) / n

def parse_affils(txt):
    try:
        lst = ast.literal_eval(txt)
        return [re.sub(r"\s+\(.*\)$", "", org).strip() for org in lst]
    except:
        return []

# ---------- file paths (adjust if needed) ------------------------------------
ROOT      = Path("~/Desktop")            # <- change to where the CSVs live
fn_models = ROOT / "assets_with_metadata_2.csv"
fn_bench  = ROOT / "benchmarks_with_authority.csv"

# ---------- 1. MODELS  –  one-model = one-unit of authority ------------------
models = pd.read_csv(fn_models, low_memory=False)
models["created_date"] = pd.to_datetime(models["created_date"], errors="coerce")
models["year"]         = models["created_date"].dt.year
models = models.dropna(subset=["year"])      # drop rows without a date
models["year"] = models["year"].astype(int)

# Organisation field: org_cluster > cleaned_org > organization
models["institution"] = (
    models.get("org_cluster")
          .where(models.get("org_cluster").notna(),
                 models["cleaned_org"].fillna(models["organization"]).str.strip())
)

# Each row is one model ⇒ authority = 1
models_tidy = (
    models.loc[:, ["year", "institution"]]
          .assign(authority = 1, facet = "models")
)

# ---------- 2. BENCHMARKS  –  fractional authority across all affiliations ---
bench = pd.read_csv(fn_bench, low_memory=False)

bench["authority_val"] = pd.to_numeric(
    bench.get("Authority_raw", bench.get("Authority (0-100)", 0)),
    errors="coerce"
).fillna(0)

bench["affils"] = bench["affiliation_list"].apply(parse_affils)

bench_expl = bench.explode("affils").rename(columns={"affils": "institution"})

bench_expl["frac_authority"] = (
    bench_expl["authority_val"] /
    bench_expl.groupby("Name")["institution"].transform("count")
)

bench_expl["year"] = pd.to_numeric(bench_expl["year"], errors="coerce")
bench_expl = bench_expl.dropna(subset=["year"])
bench_expl["year"] = bench_expl["year"].astype(int)

bench_tidy = (
    bench_expl.groupby(["year", "institution"])["frac_authority"]
              .sum()
              .reset_index()
              .rename(columns={"frac_authority": "authority"})
              .assign(facet="benchmarks")
)

# ---------- 3. stack & filter -----------------------------------------------
tidy = pd.concat([models_tidy, bench_tidy], ignore_index=True)
tidy = tidy.query("authority >= 1")

# ---------- 4. yearly HHI & Gini --------------------------------------------
records = []
for (facet, yr), g in tidy.groupby(["facet", "year"]):
    shares = g.groupby("institution")["authority"].sum()
    records.append({
        "facet": facet, "year": yr,
        "hhi":   hhi(shares),
        "gini":  gini(shares)
    })
central = pd.DataFrame(records).sort_values(["facet", "year"])

# ---------- 5. log-linear slope (trend) -------------------------------------
summary = []
for facet, g in central.groupby("facet"):
    slope, _, r, p, _ = linregress(g["year"], np.log(g["hhi"]))
    summary.append({
        "facet": facet,
        "pct_per_year": slope*100,
        "half_or_double": np.log(2)/abs(slope) if slope != 0 else np.inf,
        "trend": "centralising" if slope>0 else "de-centralising",
        "r2": r**2,
        "pvalue": p
    })
rates = pd.DataFrame(summary).sort_values("pct_per_year", ascending=False)
print("\n=== Annual concentration trend ===")
print(rates.to_markdown(index=False, floatfmt=".2f"))

# ---------- 6. sanity plots --------------------------------------------------
# 6a. HHI and Gini together for each facet
for facet, g in central.groupby("facet"):
    plt.figure()
    plt.plot(g["year"], g["hhi"],  marker="o", label="HHI")
    plt.plot(g["year"], g["gini"], marker="s", label="Gini")
    plt.title(f"{facet.capitalize()} – HHI & Gini over time")
    plt.xlabel("Year"); plt.ylabel("Metric"); plt.ylim(0,1); plt.legend(); plt.show()

# 6b. Cross-facet comparison per metric
for metric in ["hhi", "gini"]:
    plt.figure()
    for facet, g in central.groupby("facet"):
        plt.plot(g["year"], g[metric], marker="o", label=facet.capitalize())
    plt.title(f"{metric.upper()} over time – Benchmarks vs. Models")
    plt.xlabel("Year"); plt.ylabel(metric.upper()); plt.ylim(0,1); plt.legend(); plt.show()

"""Simulation model for several equilbria of benchmarks"""

pip install numba joblib tqdm

# ===============================================================
#  Fast benchmark-concentration model  (Numba + joblib, mac-safe)
# ===============================================================
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

# ----------------------------------------------------------------
# 1) Roulette-wheel sampler (Numba safe, no extra args)
# ----------------------------------------------------------------
@njit
def choose_index(weights):
    r = np.random.random()
    cdf = 0.0
    for i, w in enumerate(weights):
        cdf += w
        if r < cdf:
            return i
    return len(weights) - 1     # numerical guard

# ----------------------------------------------------------------
# 2) Numba-compiled simulator
# ----------------------------------------------------------------
@njit
def simulate_numba(alpha, beta, gamma, delta, n_steps, seed):
    np.random.seed(seed)
    max_bench = n_steps + 1
    A = np.zeros(max_bench, dtype=np.float64)
    O = np.zeros(max_bench, dtype=np.float64)
    A[0] = 1.0
    n_bench = 1

    for _ in range(n_steps):
        if np.random.random() < gamma:          # spawn new benchmark
            A[n_bench] = 1.0
            O[n_bench] = 0.0
            n_bench += 1
        else:
            # weight calculation
            w_sum = 0.0
            for j in range(n_bench):
                w_sum += (A[j] ** alpha) * np.exp(-beta * O[j])

            if w_sum == 0.0:
                i = np.random.randint(n_bench)  # all weights zero
            else:
                tmp = np.empty(n_bench, dtype=np.float64)
                for j in range(n_bench):
                    tmp[j] = (A[j] ** alpha) * np.exp(-beta * O[j]) / w_sum
                i = choose_index(tmp)

            A[i] += 1.0
            O[i] += 1.0
            for j in range(n_bench):
                if j != i and O[j] > 0.0:
                    O[j] = max(O[j] - delta, 0.0)

    # final HHI
    total = A[:n_bench].sum()
    shares = A[:n_bench] / total
    return (shares ** 2).sum()

# ----------------------------------------------------------------
# 3) Parallel grid evaluation
# ----------------------------------------------------------------
def grid_hhi(alpha, betas, gammas, n_steps=10_000, delta=0.1):
    def one_cell(bi, gi):
        beta, gamma = betas[bi], gammas[gi]
        seed = 12345 + bi * 1000 + gi
        return simulate_numba(alpha, beta, gamma, delta, n_steps, seed)

    tasks = [(bi, gi) for bi in range(len(betas)) for gi in range(len(gammas))]
    out = Parallel(n_jobs=-1, verbose=0)(
        delayed(one_cell)(bi, gi) for bi, gi in tqdm(tasks, desc="Grid sweep")
    )

    grid = np.array(out).reshape(len(betas), len(gammas))
    return grid

# ----------------------------------------------------------------
# 4) Parameter sweep (ultra-zoom on γ frontier)
# ----------------------------------------------------------------
alpha_fixed = 1.5
betas   = np.linspace(0.0, 0.05, 60)
gammas  = np.logspace(-6, -2.7, 120)   # 1e-6 .. 2e-3 (log scale)

HHI_grid = grid_hhi(alpha_fixed, betas, gammas, n_steps=10_000)

# ----------------------------------------------------------------
# 5) Publication-ready plot
# ----------------------------------------------------------------
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.linewidth": 0.8})

fig, ax = plt.subplots(figsize=(7, 4.2))
im = ax.imshow(HHI_grid, origin="lower",
               extent=[gammas[0], gammas[-1], betas[0], betas[-1]],
               aspect="auto", cmap="viridis")
ax.set_xscale("log")
ax.set_xlabel(r'$\gamma$ (rate of new benchmarks, log scale)')
ax.set_ylabel(r'$\beta$ (over-fit penalty)')
ax.set_title(fr'Tipping region, $\alpha = {alpha_fixed}$')
cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
cbar.set_label('Long-run HHI')
fig.tight_layout()

# Save high-res outputs
fig.savefig("benchmark_tipping_HHI.pdf", dpi=600, bbox_inches="tight")
fig.savefig("benchmark_tipping_HHI.png", dpi=600, bbox_inches="tight")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

# --- (Assuming HHI_grid, betas, gammas already computed) ---

# --- Figure style tweaks ---
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.linewidth": 1.0,
    "xtick.direction":"in",
    "ytick.direction":"in",
    "xtick.top":      True,
    "ytick.right":    True,
})

fig, ax = plt.subplots(figsize=(7.2, 4.2))

# --- Heat-map ---
im = ax.imshow(
    HHI_grid,
    origin="lower",
    extent=[gammas[0], gammas[-1], betas[0], betas[-1]],
    aspect="auto",
    cmap="cividis",
    interpolation="nearest",
    vmin=0.0, vmax=1.0
)

# --- Log-scale x-axis ticks at powers of ten ---
ax.set_xscale("log")
ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs="auto", numticks=12))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$"))

# --- Contour at HHI = 0.5 ---
G, B = np.meshgrid(gammas, betas)
cs = ax.contour(
    G, B, HHI_grid,
    levels=[0.5],
    colors="white",
    linewidths=2.0,
    linestyles="--"
)
ax.clabel(cs,
           fmt={0.5: "50\\% HHI"},
           inline=True,
           fontsize=10,
           colors="black"      # use 'colors' instead of 'color'
)

# --- Region labels ---
ax.text(1e-6 * 3, 0.045, "Coordination",
        color="black", fontsize=12, weight="bold")
ax.text(2e-4,      0.002, "Diversity",
        color="black", fontsize=12, weight="bold")

# --- Labels & title ---
ax.set_xlabel(r"$\gamma$: rate of new benchmarks (log scale)")
ax.set_ylabel(r"$\beta$: over-fit penalty")
ax.set_title(r"Tipping region, $\alpha = 1.5$")

# --- Color bar ---
cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
cbar.set_label("Long-run HHI")
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.ax.yaxis.set_tick_params(width=1)

fig.tight_layout()
fig.savefig("benchmark_tipping_HHI_pub_fixed.png", dpi=600, bbox_inches="tight")
plt.show()

pip install pandas scikit-learn tqdm arxiv

"""# GLOBAL VIEW OF BENCHMARK EVOLUTION"""

import joblib
from pathlib import Path

# you already ran:
# vec = joblib.load("vectorizer.pkl")
# clf = joblib.load("classifier.pkl")

dest = Path.home() / "Desktop"
dest.mkdir(parents=True, exist_ok=True)

joblib.dump(vec, dest / "vectorizer.pkl")
joblib.dump(clf, dest / "classifier.pkl")
print("Saved to:", dest)

"""If you want to peek what the model actually learned, this prints the top pro/anti “benchmark-ish” tokens:

"""

import joblib, numpy as np
vec = joblib.load("vectorizer.pkl")
clf = joblib.load("classifier.pkl")
vocab = vec.get_feature_names_out()
w = clf.coef_.ravel()
top_pos = np.argsort(w)[-20:][::-1]
top_neg = np.argsort(w)[:20]
print("\nTop POS (push toward benchmark):")
for i in top_pos: print(f"{vocab[i]:<20} {w[i]:+.3f}")
print("\nTop NEG (push away):")
for i in top_neg: print(f"{vocab[i]:<20} {w[i]:+.3f}")

pip install pandas scikit-learn joblib tqdm arxiv

import joblib

# Load the model
vec = joblib.load("vectorizer.pkl")
clf = joblib.load("classifier.pkl")

# Two example snippets (title + abstract)
examples = [
    "MultiLoKo: a multilingual local knowledge benchmark for LLMs spanning 31 languages. We introduce a dataset …",
    "EfficientFineTune: a new optimizer for training large transformers faster, with no new benchmark proposed."
]

X = vec.transform(examples)
probs = clf.predict_proba(X)[:, 1]

for txt, p in zip(examples, probs):
    print(f"{p:0.3f}  ←  {txt[:70]}…")

#!/usr/bin/env python3
"""
score_latest_ai.py  –  fetch the 10 newest cs.AI (or any) papers from arXiv,
apply your saved benchmark classifier, and print probabilities.
"""

import joblib, arxiv

# ----------------- SETTINGS ---------------------------------------------------
ARXIV_CATEGORY = "cs.AI"      # e.g. "cs.AI", "cs.LG", "cs.CL", "stat.ML"
N_PAPERS       = 10           # how many recent papers to fetch
THRESHOLD      = 0.50         # same as your pipeline
# ------------------------------------------------------------------------------

print(f"Loading classifier …")
vec = joblib.load("vectorizer.pkl")
clf = joblib.load("classifier.pkl")

# Build a query for the chosen category, sorted by date descending
query = f"cat:{ARXIV_CATEGORY}"
search = arxiv.Search(query=query,
                      max_results=N_PAPERS,
                      sort_by=arxiv.SortCriterion.SubmittedDate)

client = arxiv.Client()
papers = list(client.results(search))

texts, meta = [], []
for p in papers:
    texts.append(f"{p.title} {p.summary}")
    meta.append(f"{p.get_short_id()} – {p.title[:70]}…")

probs = clf.predict_proba(vec.transform(texts))[:, 1]

print("\nProb.   Paper")
for p, m in sorted(zip(probs, meta), reverse=True):
    flag = "← benchmark" if p >= THRESHOLD else ""
    print(f"{p:0.3f}  {m}  {flag}")

"""THE CRAWLING"""

#!/usr/bin/env python3
"""
crawl_llm_arxiv_timeboxed.py  –  resilient weekly windows + classifier
• Uses correct arXiv date syntax: submittedDate:[YYYYMMDDHHMM TO YYYYMMDDHHMM]
• Walks backward in 7-day windows; pages only within each window
• Incremental writes to ~/Desktop/arxiv_llm_scores.csv
• Backward-compatible checkpoint (handles old files without 'window_end')
"""

import json, time, urllib.parse, datetime as dt
from pathlib import Path
import feedparser, joblib, pandas as pd
from tqdm.auto import tqdm

# ─── Config ──────────────────────────────────────────────────────────
OUT_CSV       = Path.home() / "Desktop" / "arxiv_llm_scores.csv"
CKPT_FILE     = Path("arxiv_ckpt.json")
WINDOW_DAYS   = 7
PAGE_SIZE     = 200
ARXIV_API     = "https://export.arxiv.org/api/query?"
REQUEST_DELAY = 1.2
CUTOFF_YEAR   = 2017

Q_BASE = '("large language model" OR LLM OR GPT OR "language model")'

# ─── Classifier ──────────────────────────────────────────────────────
vec = joblib.load("vectorizer.pkl")
clf = joblib.load("classifier.pkl")

# ─── Helpers ─────────────────────────────────────────────────────────
def stamp(t0): return time.strftime("[%H:%M:%S]", time.gmtime(int(time.time()-t0)))

def load_ckpt():
    """
    Return (window_end_date, offset_in_window).
    Backward compatible with old checkpoints that only had {'offset': ...}
    or corrupted/empty files.
    """
    if not CKPT_FILE.exists():
        return dt.date.today(), 0
    try:
        raw = CKPT_FILE.read_text().strip()
        data = json.loads(raw) if raw else {}
        # New format
        if "window_end" in data:
            we = dt.date.fromisoformat(data["window_end"])
            off = int(data.get("offset", 0))
            return we, off
        # Old format (only 'offset'), migrate to today
        if "offset" in data:
            return dt.date.today(), int(data["offset"])
    except Exception:
        # Corrupt checkpoint → start fresh
        pass
    return dt.date.today(), 0

def save_ckpt(window_end: dt.date, offset: int):
    CKPT_FILE.write_text(json.dumps({"window_end": window_end.isoformat(),
                                     "offset": int(offset)}))

def fmt_date_yyyymmddhhmm(d: dt.date, hhmm: str) -> str:
    return f"{d:%Y%m%d}{hhmm}"

def window_query(start_d: dt.date, end_d: dt.date) -> str:
    s = fmt_date_yyyymmddhhmm(start_d, "0000")
    e = fmt_date_yyyymmddhhmm(end_d,   "2359")
    return f'{Q_BASE} AND submittedDate:[{s} TO {e}]'

def fetch_page(query: str, start_idx: int):
    url = (f"{ARXIV_API}"
           f"search_query={urllib.parse.quote_plus(query)}"
           f"&start={start_idx}&max_results={PAGE_SIZE}"
           "&sortBy=submittedDate&sortOrder=descending")
    feed = feedparser.parse(url)
    time.sleep(REQUEST_DELAY)
    return feed.entries

def classify(entries):
    def safe(val, fallback=""):
        return val if isinstance(val, str) else fallback
    texts = [f"{safe(e.title)} {safe(e.summary)}" for e in entries]
    probs = clf.predict_proba(vec.transform(texts))[:, 1] if texts else []
    rows  = []
    for e, p in zip(entries, probs):
        pid = (getattr(e, "id", "") or "")
        rows.append({
            "arxiv_id":       pid.rsplit("/", 1)[-1] if "/" in pid else pid,
            "title":          safe(getattr(e, "title", "")),
            "published":      getattr(e, "published", getattr(e, "updated", "")),
            "prob_benchmark": float(p),
        })
    return rows

# ─── Main ────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    window_end, offset = load_ckpt()
    header_needed = not OUT_CSV.exists()

    print(stamp(t0), f"🟢 Resuming – window_end={window_end}  offset={offset}")

    while window_end >= dt.date(CUTOFF_YEAR, 1, 1):
        window_start = window_end - dt.timedelta(days=WINDOW_DAYS - 1)
        q = window_query(window_start, window_end)

        total_in_window = 0
        print(stamp(t0), f"🧭 Window {window_start} → {window_end}  (start offset={offset})")

        while True:
            try:
                entries = fetch_page(q, offset)
            except Exception as e:
                print(stamp(t0), f"⚠️ fetch error @offset {offset}: {e}; retrying…")
                time.sleep(2.0)
                entries = fetch_page(q, offset)

            if not entries:
                break

            rows = classify(entries)
            if rows:
                pd.DataFrame(rows).to_csv(
                    OUT_CSV, mode="a", header=header_needed, index=False
                )
                header_needed = False

            total_in_window += len(rows)
            print(stamp(t0),
                  f"  ✅ page@{offset:>5}  +{len(rows):3d} rows  total_window={total_in_window}")
            offset += PAGE_SIZE
            save_ckpt(window_end, offset)

            if len(entries) < PAGE_SIZE:
                # last page in this window
                break

        print(stamp(t0), f"🧾 Finished {window_start} → {window_end}  saved={total_in_window}")
        # Move to previous window
        window_end = window_start - dt.timedelta(days=1)
        offset = 0
        save_ckpt(window_end, offset)

    print(stamp(t0), "🏁 Finished crawl down to cutoff year.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🔹 Interrupted – checkpoint saved to", CKPT_FILE)
        raise

import joblib
vec = joblib.load("vectorizer.pkl")
clf = joblib.load("classifier.pkl")
print(type(vec), getattr(vec, "max_features", None))
print(type(clf), getattr(clf, "class_weight", None), getattr(clf, "max_iter", None))

"""SOME CUTE PLOTS"""

# arxiv_llm_benchmark_plots.py
# Produces Nature-ready figures (high DPI, tight layout, labeled, consistent typography)
# Input: ~/Desktop/arxiv_llm_scores.csv  with columns:
#   arxiv_id,title,published,prob_benchmark

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- Paths ---
DESKTOP   = Path.home() / "Desktop"
CSV_PATH  = DESKTOP / "arxiv_llm_scores.csv"
OUT_DIR   = DESKTOP / "arxiv_llm_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH  = OUT_DIR / "arxiv_llm_plots.pdf"

# --- Load robustly ---
try:
    df = pd.read_csv(CSV_PATH, low_memory=False)
except Exception:
    # Fallback parser in case of line breaks in quoted titles
    df = pd.read_csv(CSV_PATH, low_memory=False, engine="python")

# --- Clean ---
# Keep only needed cols if present
keep = [c for c in ["arxiv_id","title","published","prob_benchmark"] if c in df.columns]
df   = df[keep].copy()

# Parse timestamps, handle tz, clip probabilities
df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
df = df.dropna(subset=["published"]).copy()
df["prob_benchmark"] = pd.to_numeric(df.get("prob_benchmark", 0), errors="coerce").fillna(0.0).clip(0, 1)

# Sort / dedup
df = df.sort_values("published")
if "arxiv_id" in df.columns:
    df = df.drop_duplicates(subset=["arxiv_id"], keep="first")

# Monthly index
df["month"] = df["published"].dt.to_period("M").dt.to_timestamp()  # month start timestamps
df = df.set_index("month")  # convenient for resampling/rolling

# --- Aggregations ---
monthly_total   = df["prob_benchmark"].resample("MS").size()
monthly_08      = df.loc[df["prob_benchmark"] >= 0.70, "prob_benchmark"].resample("MS").size()
monthly_05      = df.loc[df["prob_benchmark"] >= 0.50, "prob_benchmark"].resample("MS").size()
monthly_weight  = df["prob_benchmark"].resample("MS").sum()                       # score-weighted volume
monthly_mean    = df["prob_benchmark"].resample("MS").mean()                      # average confidence
monthly_p90     = df["prob_benchmark"].resample("MS").quantile(0.90)              # upper tail quality
share_05        = (monthly_05 / monthly_total).replace([np.inf, -np.inf], np.nan).fillna(0)
share_08        = (monthly_08 / monthly_total).replace([np.inf, -np.inf], np.nan).fillna(0)

# Rolling trends (3-month centered)
roll = 3
mt_roll  = monthly_total.rolling(roll, center=True).mean()
m05_roll = monthly_05.rolling(roll, center=True).mean()
m08_roll = monthly_08.rolling(roll, center=True).mean()
mw_roll  = monthly_weight.rolling(roll, center=True).mean()
mean_roll= monthly_mean.rolling(roll, center=True).mean()
p90_roll = monthly_p90.rolling(roll, center=True).mean()
s05_roll = share_05.rolling(roll, center=True).mean()
s08_roll = share_08.rolling(roll, center=True).mean()

# Helper for consistent figure export
def savefig(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path

pngs = []

# 1) Monthly counts with thresholds (hard/soft) + total
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()

# keep the grid off
ax.grid(False, which='both')

monthly_total.plot(ax=ax, linewidth=1.5, label="All LLM papers in crawl (monthly)")
monthly_05.plot(ax=ax, linewidth=1.5, label="≥ 0.5 (likely benchmark)")
monthly_08.plot(ax=ax, linewidth=1.5, label="≥ 0.8 (very likely)")
mt_roll.plot(ax=ax, linestyle="--", linewidth=1.5, label="All (3-mo avg)")
m05_roll.plot(ax=ax, linestyle="--", linewidth=1.5, label="≥ 0.5 (3-mo avg)")
m08_roll.plot(ax=ax, linestyle="--", linewidth=1.5, label="≥ 0.8 (3-mo avg)")
ax.set_title("Monthly volume of LLM papers flagged as benchmarks")
ax.set_xlabel("Month")
ax.set_ylabel("Count")

# remove this line (or set False):
# ax.grid(True, alpha=0.3)

ax.legend(frameon=False)
pngs.append(savefig(fig, "01_monthly_counts_thresholds"))

# 2) Score-weighted volume (sum of probabilities)
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()

# keep the grid off
ax.grid(False, which='both')

monthly_weight.plot(ax=ax, linewidth=1.8, label="Score-weighted volume")
mw_roll.plot(ax=ax, linestyle="--", linewidth=1.8, label="3-mo avg")
ax.set_title("Score-weighted benchmark volume (sum of probabilities per month)")
ax.set_xlabel("Month")
ax.set_ylabel("Weighted count")

# remove or disable this line:
# ax.grid(True, alpha=0.3)

ax.legend(frameon=False)
pngs.append(savefig(fig, "02_score_weighted_volume"))


# 3) Share above thresholds
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
share_05.plot(ax=ax, linewidth=1.8, label="Share ≥ 0.5")
share_08.plot(ax=ax, linewidth=1.8, label="Share ≥ 0.8")
s05_roll.plot(ax=ax, linestyle="--", linewidth=1.8, label="≥ 0.5 (3-mo avg)")
s08_roll.plot(ax=ax, linestyle="--", linewidth=1.8, label="≥ 0.8 (3-mo avg)")
ax.set_title("Share of monthly LLM papers likely to be benchmarks")
ax.set_xlabel("Month")
ax.set_ylabel("Fraction of monthly total")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
pngs.append(savefig(fig, "03_share_thresholds"))

# 4) Quality trend: mean score & 90th percentile
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
monthly_mean.plot(ax=ax, linewidth=1.8, label="Mean monthly score")
monthly_p90.plot(ax=ax, linewidth=1.8, label="90th percentile score")
mean_roll.plot(ax=ax, linestyle="--", linewidth=1.8, label="Mean (3-mo avg)")
p90_roll.plot(ax=ax, linestyle="--", linewidth=1.8, label="P90 (3-mo avg)")
ax.set_title("Monthly score trend (mean & 90th percentile)")
ax.set_xlabel("Month")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
pngs.append(savefig(fig, "04_quality_trends"))

# 5) Cumulative curves: total vs. ≥0.5 vs. ≥0.8
fig = plt.figure(figsize=(8, 5))
ax = plt.gca()
monthly_total.cumsum().plot(ax=ax, linewidth=1.8, label="All LLM papers (cumulative)")
monthly_05.cumsum().plot(ax=ax, linewidth=1.8, label="Benchmarks ≥ 0.5 (cumulative)")
monthly_08.cumsum().plot(ax=ax, linewidth=1.8, label="Benchmarks ≥ 0.8 (cumulative)")
ax.set_title("Cumulative counts over time")
ax.set_xlabel("Month")
ax.set_ylabel("Cumulative count")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)
pngs.append(savefig(fig, "05_cumulative_curves"))

# 6) Top examples (table preview in console) & a simple “top-k over time” stem plot
# Show the top 15 in the console
topk = df.reset_index().sort_values("prob_benchmark", ascending=False).head(40)[
    ["month","published","prob_benchmark","arxiv_id","title"]
].copy()
print("\nTop 15 high-score candidates:")
print(topk.head(15).to_string(index=False))

# Stem-like plot without use_line_collection (robust across Matplotlib versions)
fig = plt.figure(figsize=(8, 2.6))
ax = plt.gca()

# Make x naive datetimes (strip tz) to silence any tz plotting quirks
x = pd.to_datetime(topk["published"], utc=True).dt.tz_localize(None)
y = topk["prob_benchmark"].values

# vertical lines (stems) + markers
ax.vlines(x, ymin=0.0, ymax=y, linewidth=1.0, alpha=0.55)
ax.scatter(x, y, s=14, zorder=3)

ax.set_title("Top-scoring candidates over time (first 40)")
ax.set_xlabel("Date")
ax.set_ylabel("Classifier score")
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

pngs.append(savefig(fig, "06_topk_stem_vlines"))

# --- Export a multi-page PDF with all figures ---
with PdfPages(PDF_PATH) as pdf:
    for p in sorted(pngs):
        fig = plt.figure(figsize=(8, 5))
        img = plt.imread(p)
        plt.imshow(img)
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("\nSaved figures:")
for p in sorted(pngs): print(" •", p)
print("Multi-page PDF:", PDF_PATH)

"""LET'S DO THE CRAWLING ALL OVER AGAIN, IMPROVED"""

# alpha_sweep.py
import ast, math, re
import pandas as pd
import numpy as np
from urllib.parse import urlparse

# --- CONFIG ---
INPUT_CSV = "benchmarks_cleaned.csv"
REPO_STATS = "repo_stats.csv"   # optional: columns ['owner_repo','stars','forks','watchers']
ALPHAS = [0.0, 0.25, 0.5]
TOPK = 20

# --- helpers ---
def parse_affils(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        xs = x.strip()
        if xs.startswith('[') and xs.endswith(']'):
            try:
                return ast.literal_eval(xs)
            except Exception:
                pass
        # fallback: split on common delimiters
        return [p.strip() for p in re.split(r'[;|,]+', xs) if p.strip()]
    return []

def norm_inst(inst):
    s = inst.strip()
    low = s.lower()
    if low in {'google','google research','google brain','deepmind','google deepmind'}: return 'Google'
    if low in {'microsoft','microsoft research'}: return 'Microsoft'
    if low in {'university of california, berkeley','uc berkeley'}: return 'UC Berkeley'
    return s

def extract_github_slugs(cell):
    if not isinstance(cell, str) or not cell.strip():
        return []
    slugs = []
    for url in re.split(r'\s+', cell.strip()):
        try:
            u = urlparse(url)
            if 'github.com' in u.netloc and len(u.path.split('/')) >= 3:
                owner = u.path.split('/')[1]
                repo  = u.path.split('/')[2]
                repo  = repo.replace('.git','').strip('/')
                slugs.append(f"{owner}/{repo}")
        except Exception:
            continue
    return list(dict.fromkeys(slugs))  # dedupe, keep order

def load_repo_stats(path):
    try:
        rs = pd.read_csv(path)
        cols = {c.lower(): c for c in rs.columns}
        # normalize column names
        rename = {}
        if 'owner_repo' not in cols:
            # try variants
            for c in rs.columns:
                if re.fullmatch(r'(owner[/_-]?repo|repo|slug)', c, re.I):
                    rename[c] = 'owner_repo'
        if 'stars' not in cols:
            for c in rs.columns:
                if re.fullmatch(r'(stars?|stargazers?)', c, re.I):
                    rename[c] = 'stars'
        if 'forks' not in cols:
            for c in rs.columns:
                if re.fullmatch(r'(forks?)', c, re.I):
                    rename[c] = 'forks'
        if 'watchers' not in cols:
            for c in rs.columns:
                if re.fullmatch(r'(watch(ers)?|subscribers?)', c, re.I):
                    rename[c] = 'watchers'
        if rename:
            rs = rs.rename(columns=rename)
        for col in ['stars','forks','watchers']:
            if col not in rs: rs[col] = 0
        rs = rs[['owner_repo','stars','forks','watchers']].copy()
        rs['engagement'] = rs[['stars','forks','watchers']].fillna(0).sum(1)
        return rs
    except FileNotFoundError:
        # empty frame → zero engagement fallback
        return pd.DataFrame(columns=['owner_repo','engagement'])

# --- load main file ---
df = pd.read_csv(INPUT_CSV)

# columns per your schema (note the colon in "Cited by:")
df['citations'] = pd.to_numeric(df.get('Cited by:', 0), errors='coerce').fillna(0).astype(float)
df['affils']    = df['affiliation_list'].apply(parse_affils).apply(lambda L: [norm_inst(a) for a in L] if L else [])
df['slugs']     = df['Code repository'].apply(extract_github_slugs)

# join repo stats (optional)
repo_stats = load_repo_stats(REPO_STATS)
eng_map = dict(zip(repo_stats['owner_repo'], repo_stats.get('engagement', pd.Series([0]*len(repo_stats)))))

def slug_engagement(slugs):
    if not slugs:
        return 0.0
    # choose max engagement across listed repos
    return float(max([eng_map.get(s, 0.0) for s in slugs]))

df['engagement'] = df['slugs'].apply(slug_engagement)

# --- core computation ---
def inst_ranking(alpha: float) -> tuple[pd.DataFrame, float]:
    a_b = np.log1p(df['citations']) + alpha * np.log1p(df['engagement'])
    rows = []
    for score, affs in zip(a_b, df['affils']):
        affs = affs if affs else ['Unspecified']
        w = 1.0 / len(affs)
        for inst in affs:
            rows.append((inst, score * w))
    G = pd.DataFrame(rows, columns=['inst','A']).groupby('inst', as_index=False)['A'].sum()
    G = G.sort_values('A', ascending=False).reset_index(drop=True)
    shares = G['A'] / G['A'].sum() if G['A'].sum() > 0 else pd.Series(np.zeros(len(G)))
    hhi = float((shares**2).sum())
    return G, hhi

# run sweep
tops = {}
hhis = {}
for a in ALPHAS:
    g, hhi = inst_ranking(a)
    tops[a] = g.head(TOPK)['inst'].tolist()
    hhis[a] = hhi

def jaccard(topA, topB):
    A, B = set(topA), set(topB)
    return len(A & B) / len(A | B) if A | B else 1.0

pairs = [(ALPHAS[i], ALPHAS[j]) for i in range(len(ALPHAS)) for j in range(i+1, len(ALPHAS))]
jacc_top10 = {p: jaccard(tops[p[0]][:10], tops[p[1]][:10]) for p in pairs}

# (optional) Spearman over union of top-K
from scipy.stats import spearmanr
def spearman_union(a, b, k=TOPK):
    U = list(set(tops[a][:k]) | set(tops[b][:k]))
    ra = {inst:i for i,inst in enumerate(tops[a][:k])}
    rb = {inst:i for i,inst in enumerate(tops[b][:k])}
    xa = [ra.get(u, k) for u in U]
    xb = [rb.get(u, k) for u in U]
    return float(spearmanr(xa, xb).correlation)

spearman = {p: spearman_union(*p, k=TOPK) for p in pairs}

print("Top-10 Jaccard overlaps:", jacc_top10)
print("Spearman (union of top-{}):".format(TOPK), spearman)
print("HHI by alpha:", hhis)

# save per-alpha rankings (for SI)
for a in ALPHAS:
    g,_ = inst_ranking(a)
    g.to_csv(f"institutions_rank_alpha_{a:.2f}.csv", index=False)

import re, ast, numpy as np, pandas as pd

df = pd.read_csv("benchmarks_cleaned.csv")


# 1) Citations: "New" -> 0, otherwise keep digits (handles "1,234")
s = df.get('Cited by:', '').astype(str).str.strip()
s = s.mask(s.str.lower()=='new', '0')
s = s.str.replace(r'[^\d]', '', regex=True)
df['citations'] = pd.to_numeric(s, errors='coerce').fillna(0).astype(float)

# 2) GitHub slug extraction (ignore Kaggle/other hosts)
from urllib.parse import urlparse
def gh_slug(cell):
    if not isinstance(cell, str): return None
    for url in re.split(r'\s+', cell.strip()):
        u = urlparse(url)
        if 'github.com' in (u.netloc or '') and len(u.path.split('/'))>=3:
            owner, repo = u.path.split('/')[1], u.path.split('/')[2].replace('.git','').strip('/')
            return f"{owner}/{repo}"
    return None

df['gh_slug'] = df['Code repository'].apply(gh_slug)

# 3) Engagement proxy (until you fetch real stars/forks):
#    1 if has a GitHub repo, else 0. (This makes alpha matter.)
df['engagement'] = df['gh_slug'].notna().astype(float)

# 4) Affiliation parsing + light normalization
def parse_affils(x):
    if isinstance(x, list): return x
    if isinstance(x, str) and x.strip().startswith('['):
        try: return [a.strip() for a in ast.literal_eval(x)]
        except: pass
    return []
def norm_inst(s):
    s = s.strip()
    low = s.lower()
    if low in {'google','google research','google brain','deepmind','google deepmind'}: return 'Google'
    if low in {'microsoft','microsoft research'}: return 'Microsoft'
    if low in {'uc berkeley','university of california, berkeley'}: return 'UC Berkeley'
    if low in {'unknown',''}: return None
    return s

df['affils'] = df['affiliation_list'].apply(parse_affils).apply(lambda L: [norm_inst(a) for a in L if norm_inst(a)])

# 5) Score + aggregation with filtering to avoid all-zero mass
def inst_ranking(alpha):
    a_b = np.log1p(df['citations']) + alpha*np.log1p(df['engagement'])
    rows = []
    for score, affs in zip(a_b, df['affils']):
        affs = affs if affs else []
        if not affs: continue   # drop if no institution
        w = 1.0/len(affs)
        for inst in affs:
            rows.append((inst, score*w))
    G = pd.DataFrame(rows, columns=['inst','A']).groupby('inst', as_index=False)['A'].sum()
    total = G['A'].sum()
    if total <= 0:
        return G.assign(share=0.0), np.nan  # avoid fake HHI=0
    G['share'] = G['A']/total
    hhi = float((G['share']**2).sum())
    return G.sort_values('A', ascending=False).reset_index(drop=True), hhi

# 6) Restrict robustness check to items with any signal (citations>0 or engagement>0)
M = df[(df['citations']>0) | (df['engagement']>0)].copy()
df = M  # use filtered set

alphas = [0.0, 0.25, 0.5]
tops, hhis = {}, {}
for a in alphas:
    g, h = inst_ranking(a)
    tops[a] = g.head(20)['inst'].tolist()
    hhis[a] = h

def jaccard(topA, topB):
    A,B = set(topA[:10]), set(topB[:10])
    return len(A&B)/len(A|B) if A|B else np.nan

pairs = [(0.0,0.25),(0.0,0.5),(0.25,0.5)]
jacc = {p: jaccard(tops[p[0]], tops[p[1]]) for p in pairs}
from scipy.stats import spearmanr
def spearman_union(a,b,k=20):
    U = list(set(tops[a][:k])|set(tops[b][:k]))
    ra = {x:i for i,x in enumerate(tops[a][:k])}
    rb = {x:i for i,x in enumerate(tops[b][:k])}
    xa = [ra.get(u,k) for u in U]; xb = [rb.get(u,k) for u in U]
    return float(spearmanr(xa, xb).correlation)
rho = {p: spearman_union(*p, k=20) for p in pairs}

print("Top-10 Jaccard:", jacc)
print("Spearman (top-20 union):", rho)
print("HHI by alpha:", hhis)

import os, pandas as pd

HOME = os.path.expanduser("~")
PATH = os.path.join(HOME, "Desktop", "benchmarks_with_authority.csv")  # <-- your file
bench = pd.read_csv(PATH)

import numpy as np
import pandas as pd

# --- helper: best-effort first-public date ---
def approx_first_date(row):
    # prefer arXiv id (e.g., 2504.10356 -> 2025-04-01), else Initial publication year (Jan 1)
    try:
        ax = str(row.get('arxiv_id', '')).strip()
        if ax and ax.replace('.', '').isdigit() and len(ax.split('.')[0]) == 4:
            yy = int(ax[:2]); mm = int(ax[2:4])
            year = 2000 + yy  # assumes 20xx arXiv ids
            return pd.Timestamp(year=year, month=mm, day=1)
    except Exception:
        pass
    y = row.get('Initial publication year', None)
    y = int(float(y)) if pd.notnull(y) else None
    return pd.Timestamp(year=y, month=1, day=1) if y else pd.Timestamp.today().normalize()

# --- build monthly events from cites_per_mo (fallback to citations / months) ---
def to_monthly_events(df):
    rows = []
    today = pd.Timestamp.today().normalize()
    for _, r in df.iterrows():
        actor = r.get('Name') or r.get('Dataset') or r.get('Benchmark') or 'unknown'
        start = approx_first_date(r)
        if start > today:
            continue
        months = pd.period_range(start, today, freq='M')
        # monthly rate
        cpm = r.get('cites_per_mo', np.nan)
        if pd.isna(cpm) or cpm == 0:
            total = r.get('citations', np.nan)
            cpm = float(total) / max(len(months), 1) if pd.notnull(total) else 0.0
        # emit one event per month with weight=cpm
        for m in months:
            rows.append({
                'actor_id': actor,
                'event_date': m.to_timestamp(how='end'),
                'signal': float(cpm)
            })
    return pd.DataFrame(rows)

events = to_monthly_events(bench)

# OPTIONAL: if you prefer to use your precomputed authority directly for the baseline comparison:
# base_auth = bench[['Name','Authority_raw']].rename(columns={'Name':'actor_id','Authority_raw':'auth'})

df = events  # monthly proxy events
# ... paste functions: gini, hhi, authority_baseline, authority_rate, authority_window, authority_decay, summary_metrics ...
BASE = authority_baseline(df)

variants = []
variants.append(summary_metrics(BASE, BASE, 'Baseline (cumulative)'))
variants.append(summary_metrics(BASE, authority_rate(df, min_years=0.25), 'Rate / age (≥0.25y)'))
for T in (1, 2, 3):
    variants.append(summary_metrics(BASE, authority_window(df, T_years=T), f'Window {T}y'))
for h in (1, 2, 3, 5):
    variants.append(summary_metrics(BASE, authority_decay(df, half_life_years=h), f'Decay h={h}y'))

robust = pd.DataFrame(variants)
print(robust)

# === Robustness to age/recency — ONE-CELL VERSION ===
import os, numpy as np, pandas as pd

# ---------- CONFIG ----------
HOME = os.path.expanduser("~")
CSV_PATH = os.path.join(HOME, "Desktop", "benchmarks_with_authority.csv")  # adjust if needed
NOW = pd.Timestamp.today().normalize()

# ---------- HELPERS ----------
def gini(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return np.nan
    if np.all(x == 0): return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum.sum() / cum[-1])) / n

def hhi(shares_vec):
    s = np.asarray(shares_vec, dtype=float)
    s = s[s > 0]
    return float(np.sum(np.square(s)))

def shares(v):
    v = np.asarray(v, dtype=float)
    tot = v.sum()
    return v / tot if tot > 0 else np.zeros_like(v)

def topk_jaccard(a_ids, b_ids, k=10):
    A = set(a_ids[:k]); B = set(b_ids[:k])
    U = A | B
    return (len(A & B) / len(U)) if U else np.nan

def _spearman_rho_from_series(a, b):
    ar = pd.Series(a).rank(ascending=False, method='average')
    br = pd.Series(b).rank(ascending=False, method='average')
    return float(np.corrcoef(ar, br)[0, 1])

# ---------- AUTHORITY VARIANTS ----------
def authority_baseline(df):
    return df.groupby('actor_id', as_index=False)['signal'].sum().rename(columns={'signal':'auth'})

def authority_rate(df, min_years=0.25):
    age = df.groupby('actor_id')['event_date'].agg('min').rename('first').reset_index()
    age['years'] = np.maximum((NOW - age['first']).dt.days / 365.25, min_years)
    auth = df.groupby('actor_id', as_index=False)['signal'].sum().rename(columns={'signal':'sumsig'})
    out = auth.merge(age[['actor_id','years']], on='actor_id', how='left')
    out['auth'] = out['sumsig'] / out['years']
    return out[['actor_id','auth']]

def authority_window(df, T_years=2):
    cutoff = NOW - pd.DateOffset(years=T_years)
    d = df[df['event_date'] >= cutoff]
    return d.groupby('actor_id', as_index=False)['signal'].sum().rename(columns={'signal':'auth'})

def authority_decay(df, half_life_years=2):
    hl_days = half_life_years * 365.25
    w = np.exp(-np.log(2) * (NOW - df['event_date']).dt.days / hl_days)
    d = df.assign(wsig=df['signal'] * w)
    return d.groupby('actor_id', as_index=False)['wsig'].sum().rename(columns={'wsig':'auth'})

def summary_metrics(baseline, adjusted, label):
    base = baseline.set_index('actor_id')['auth']
    adj  = adjusted.set_index('actor_id')['auth']
    all_ids = base.index.union(adj.index)
    base = base.reindex(all_ids, fill_value=0.0)
    adj  = adj.reindex(all_ids,  fill_value=0.0)

    base_rank = base.sort_values(ascending=False)
    adj_rank  = adj.sort_values(ascending=False)
    rho = _spearman_rho_from_series(
        base_rank.reindex(all_ids, fill_value=0.0),
        adj_rank.reindex(all_ids,  fill_value=0.0)
    )
    jac10 = topk_jaccard(list(base_rank.index), list(adj_rank.index), k=10)
    return {
        'variant': label,
        'Gini': gini(adj.values),
        'HHI': hhi(shares(adj.values)),
        'Top10 Jaccard vs baseline': jac10,
        'Spearman ρ (top-20 union)': rho
    }

# ---------- LOAD CSV FROM DESKTOP ----------
bench = pd.read_csv(CSV_PATH)
# Clean/ensure numeric fields exist
for col in ['citations','cites_per_mo','Initial publication year']:
    if col in bench.columns:
        bench[col] = pd.to_numeric(bench[col], errors='coerce')

# ---------- DATING: arXiv-first, else year ----------
def approx_first_date(row):
    ax = str(row.get('arxiv_id', '')).strip()
    # e.g., '2504.10356' -> year=2025, month=04
    if ax and ax.replace('.', '').isdigit() and len(ax.split('.')[0]) == 4:
        yy = int(ax[:2]); mm = int(ax[2:4])
        return pd.Timestamp(year=2000 + yy, month=mm, day=1)
    y = row.get('Initial publication year', None)
    y = int(float(y)) if pd.notnull(y) else None
    return pd.Timestamp(year=y, month=1, day=1) if y else NOW

# ---------- BUILD MONTHLY PROXY EVENTS ----------
def to_monthly_events(df):
    rows = []
    today = NOW
    for _, r in df.iterrows():
        actor = r.get('Name') or r.get('Dataset') or r.get('Benchmark') or 'unknown'
        start = approx_first_date(r)
        if pd.isna(start) or start > today:
            continue
        months = pd.period_range(start, today, freq='M')
        # monthly rate
        cpm = r.get('cites_per_mo', np.nan)
        if pd.isna(cpm) or cpm == 0:
            total = r.get('citations', np.nan)
            cpm = float(total) / max(len(months), 1) if pd.notnull(total) else 0.0
        for m in months:
            rows.append({'actor_id': actor,
                         'event_date': m.to_timestamp(how='end'),
                         'signal': float(cpm)})
    return pd.DataFrame(rows)

events = to_monthly_events(bench)
events['event_date'] = pd.to_datetime(events['event_date'])

if events.empty:
    raise ValueError("No events produced. Check CSV_PATH and that your CSV has 'Name' and either 'cites_per_mo' or 'citations' and a date signal (arxiv_id or Initial publication year).")

# ---------- RUN ROBUSTNESS ----------
BASE = authority_baseline(events)

variants = []
variants.append(summary_metrics(BASE, BASE, 'Baseline (cumulative)'))
variants.append(summary_metrics(BASE, authority_rate(events, min_years=0.25), 'Rate / age (≥0.25y)'))
for T in (1, 2, 3):
    variants.append(summary_metrics(BASE, authority_window(events, T_years=T), f'Window {T}y'))
for h in (1, 2, 3, 5):
    variants.append(summary_metrics(BASE, authority_decay(events, half_life_years=h), f'Decay h={h}y'))

robust = pd.DataFrame(variants)
display(robust.round(3))

# OPTIONAL: save results
out_path = os.path.join(HOME, "Desktop", "authority_robustness_results.csv")
robust.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

import pandas as pd
from scipy.stats import spearmanr

# Path to your CSV (adjust as needed)
csv_path = "benchmarks_with_authority.csv"
bench = pd.read_csv(csv_path)

# Extract and rank your top 10 benchmarks by authority
bench_sorted = bench.sort_values("Authority (0-100)", ascending=False)
top_our = bench_sorted["Name"].head(10).tolist()

# HELM core scenarios (confirmed from the HELM Lite page:contentReference[oaicite:1]{index=1})
external_list = [
    "NarrativeQA",
    "NaturalQuestions (open-book)",
    "NaturalQuestions (closed-book)",
    "OpenbookQA",
    "MMLU",
    "MATH",
    "GSM8K",
    "LegalBench",
    "MedQA",
    "WMT 2014"
]

# Compute Jaccard overlap
intersection = set(top_our) & set(external_list)
union = set(top_our) | set(external_list)
jaccard = len(intersection) / len(union)

# Compute Spearman rank correlation on overlapping items
common = [item for item in external_list if item in top_our]
if common:
    our_ranks = [top_our.index(item) + 1 for item in common]
    ext_ranks = [external_list.index(item) + 1 for item in common]
    rho, _ = spearmanr(our_ranks, ext_ranks)
else:
    rho = None

print("Our top benchmarks:", top_our)
print("HELM core scenarios:", external_list)
print("Common benchmarks:", intersection)
print("Jaccard overlap:", jaccard)
print("Spearman rank correlation:", rho)