"""makes a 2x2 data quality figure from the raw and cleaned csvs."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw = pd.read_csv(os.path.join(BASE, "data", "raw_healthcare_data.csv"))
cleaned = pd.read_csv(os.path.join(BASE, "outputs", "cleaned_healthcare_data.csv"))

# normalise raw column names the same way the cleaning script does so the
# before/after comparison lines up
raw.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in raw.columns]

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("data quality - before vs after cleaning", fontsize=14)

# missing values per column, raw vs cleaned. weight and bp change names after
# cleaning so map them across
pairs = [("dob", "dob"), ("gender", "gender"), ("age", "age"),
         ("weight", "weight_kg"), ("blood_pressure", "systolic_bp"),
         ("diagnosis", "diagnosis"), ("last_visit", "last_visit"),
         ("phone", "phone"), ("email", "email")]
labels = [p[0] for p in pairs]
before = [int(raw[p[0]].isna().sum()) for p in pairs]
after = [int(cleaned[p[1]].isna().sum()) for p in pairs]
x = range(len(labels))
ax[0, 0].bar([i - 0.2 for i in x], before, width=0.4, label="raw", color="#c0504d")
ax[0, 0].bar([i + 0.2 for i in x], after, width=0.4, label="cleaned", color="#4f81bd")
ax[0, 0].set_xticks(list(x))
ax[0, 0].set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
ax[0, 0].set_title("missing values by column")
ax[0, 0].legend()

# diagnosis distribution after standardization. raw had 50+ spellings, now
# its just the canonical vocabulary
counts = cleaned["diagnosis"].value_counts()
ax[0, 1].barh(counts.index[::-1], counts.values[::-1], color="#4f81bd")
ax[0, 1].set_title("diagnosis distribution (after standardization)")
ax[0, 1].tick_params(axis="y", labelsize=8)

ax[1, 0].hist(cleaned["age"].dropna(), bins=30, color="#9bbb59", edgecolor="white")
ax[1, 0].set_title("age distribution (cleaned)")
ax[1, 0].set_xlabel("age")

visits = pd.to_datetime(cleaned["last_visit"], errors="coerce").dropna()
per_month = visits.dt.to_period("M").value_counts().sort_index()
ax[1, 1].plot(per_month.index.astype(str), per_month.values, marker="o",
              color="#8064a2", linewidth=1.5)
ax[1, 1].set_title("records per month")
ax[1, 1].tick_params(axis="x", rotation=60, labelsize=7)

plt.tight_layout()
out = os.path.join(BASE, "outputs", "data_quality.png")
plt.savefig(out, dpi=120)
print("saved", out)
