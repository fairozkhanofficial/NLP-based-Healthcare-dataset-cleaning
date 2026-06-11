"""cleans data/raw_healthcare_data.csv into one tidy csv plus a text report."""
import os
import re
from difflib import SequenceMatcher, get_close_matches

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, "data", "raw_healthcare_data.csv")
OUT_DIR = os.path.join(BASE, "outputs")
OUT_CSV = os.path.join(OUT_DIR, "cleaned_healthcare_data.csv")
OUT_REPORT = os.path.join(OUT_DIR, "cleaning_report.txt")

# the 10 conditions i actually want in the final dataset
CANONICAL = ["hypertension", "type 2 diabetes", "asthma", "migraine", "arthritis",
             "anemia", "coronary artery disease", "hypothyroidism", "copd",
             "urinary tract infection"]

# built this dict by eyeballing value_counts() on the raw diagnosis column.
# abbreviations and synonyms that regex alone cant guess
SYNONYMS = {
    "htn": "hypertension",
    "high bp": "hypertension",
    "high blood pressure": "hypertension",
    "t2dm": "type 2 diabetes",
    "diabetes type 2": "type 2 diabetes",
    "type ii diabetes": "type 2 diabetes",
    "diabetes type ii": "type 2 diabetes",
    "diabetes mellitus type 2": "type 2 diabetes",
    "bronchial asthma": "asthma",
    "migraine headache": "migraine",
    "chronic migraine": "migraine",
    "osteoarthritis": "arthritis",
    "anaemia": "anemia",
    "iron deficiency anemia": "anemia",
    "low iron": "anemia",
    "cad": "coronary artery disease",
    "heart disease": "coronary artery disease",
    "hypothyroid": "hypothyroidism",
    "low thyroid": "hypothyroidism",
    "underactive thyroid": "hypothyroidism",
    "chronic obstructive pulmonary disease": "copd",
    "uti": "urinary tract infection",
    "urine infection": "urinary tract infection",
}

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
          "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

stats = {}  # everything that ends up in the report


def parse_date(val):
    # three formats show up in this export: 12/03/2024 (day first),
    # 2024-03-12, and 12-Mar-2024. anything else becomes NaT
    if not isinstance(val, str):
        return pd.NaT
    val = val.strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", val)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})$", val)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            m = re.match(r"^(\d{1,2})-([A-Za-z]{3})-(\d{4})$", val)
            if m and m.group(2).lower() in MONTHS:
                d, mo, y = int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3))
            else:
                return pd.NaT
    try:
        return pd.Timestamp(year=y, month=mo, day=d)
    except ValueError:
        return pd.NaT


def fix_gender(val):
    if not isinstance(val, str):
        return "Unknown"
    v = val.strip().lower()
    if v in ("m", "male"):
        return "M"
    if v in ("f", "female"):
        return "F"
    return "Unknown"


def to_kg(val):
    # weights came in as "72 kg", "72kg", "159 lbs" or just "72". bare numbers
    # are assumed kg because thats what the clinic uses
    if not isinstance(val, str) or not val.strip():
        return np.nan
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(kg|kgs|lb|lbs|pounds)?$", val.strip().lower())
    if not m:
        return np.nan
    num = float(m.group(1))
    unit = m.group(2) or "kg"
    if unit.startswith(("lb", "pound")):
        num = num * 0.453592
        stats["weights_converted_from_lbs"] = stats.get("weights_converted_from_lbs", 0) + 1
    return round(num, 1)


def split_bp(val):
    # "120/80", "120 / 80", "120/80 mmHg" all hide the same two numbers
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", str(val))
    if m:
        return int(m.group(1)), int(m.group(2))
    return np.nan, np.nan


def clean_phone(val):
    # keep only digits, drop +91 / leading zero, then it should be a 10 digit
    # indian mobile. anything else gets nulled instead of kept wrong
    if not isinstance(val, str):
        return np.nan
    digits = re.sub(r"\D", "", val)
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):
        digits = digits[1:]
    if len(digits) == 10 and digits[0] in "6789":
        return digits
    return np.nan


def clean_email(val):
    if not isinstance(val, str):
        return np.nan
    v = val.strip().lower()
    # good enough check, not trying to implement the whole rfc here
    if re.match(r"^[\w.+-]+@[\w-]+\.[\w.]+$", v):
        return v
    return np.nan


def map_diagnosis(text):
    # the nlp-ish part of this project. maps free text diagnoses onto the
    # small canonical vocabulary in three layers, cheapest first:
    #   1. regex cleanup - lowercase, strip punctuation, squeeze whitespace.
    #      this alone fixes "Hypertension ", "C.O.P.D", "UTI" etc
    #   2. dictionary lookup - the SYNONYMS dict catches abbreviations like
    #      "t2dm" or "htn" that no string similarity would ever find
    #   3. difflib.get_close_matches - stdlib fuzzy matching for typos like
    #      "migrane" or "diabetess type ii". it compares character overlap,
    #      no ml involved, which is plenty for a vocabulary this small
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    t = re.sub(r"[^\w\s]", "", text.lower())
    t = re.sub(r"\s+", " ", t).strip()
    if t in CANONICAL:
        return t
    if t in SYNONYMS:
        return SYNONYMS[t]
    # fuzzy fallback, checked against canonical names and synonym keys so a
    # typo'd abbreviation still lands in the right bucket
    match = get_close_matches(t, CANONICAL + list(SYNONYMS), n=1, cutoff=0.75)
    if match:
        return SYNONYMS.get(match[0], match[0])
    return "other"


def drop_near_dupes(df):
    # same date of birth + nearly identical name = same person who got
    # registered twice. fuzzy compare with SequenceMatcher, keep the first row
    drop = set()
    for _, grp in df.groupby("dob"):
        if len(grp) < 2:
            continue
        idx = list(grp.index)
        names = grp["full_name"].str.lower().tolist()
        for i in range(len(idx)):
            if idx[i] in drop:
                continue
            for j in range(i + 1, len(idx)):
                if idx[j] in drop:
                    continue
                if SequenceMatcher(None, names[i], names[j]).ratio() >= 0.85:
                    drop.add(idx[j])
    return df.drop(index=list(drop)), len(drop)


def main():
    raw = pd.read_csv(RAW)
    stats["rows_in"] = len(raw)
    stats["missing_before"] = int(raw.isna().sum().sum())

    print("raw shape:", raw.shape)
    print("\ndtypes:")
    print(raw.dtypes.to_string())
    print("\nmissing values per column:")
    print(raw.isna().sum().to_string())

    df = raw.copy()

    # column names first: strip, lowercase, underscores. "Full Name " with the
    # trailing space was a fun one to discover
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]

    # strip stray whitespace inside every text cell
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df["full_name"] = df["full_name"].str.title()

    df["dob"] = df["dob"].apply(parse_date)
    df["last_visit"] = df["last_visit"].apply(parse_date)
    stats["unparseable_dobs"] = int(df["dob"].isna().sum())

    df["gender"] = df["gender"].apply(fix_gender)
    stats["gender_unknown"] = int((df["gender"] == "Unknown").sum())

    bp = df["blood_pressure"].apply(split_bp)
    df["systolic_bp"] = pd.to_numeric([x[0] for x in bp])
    df["diastolic_bp"] = pd.to_numeric([x[1] for x in bp])
    df = df.drop(columns=["blood_pressure"])
    stats["bp_parsed"] = int(df["systolic_bp"].notna().sum())

    df["weight_kg"] = df["weight"].apply(to_kg)
    df = df.drop(columns=["weight"])

    # ages like 999 or -1 are obviously data entry junk. null them, then
    # recompute from dob where we can
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    bad_age = (df["age"] < 0) | (df["age"] > 110)
    stats["age_outliers_nulled"] = int(bad_age.sum())
    df.loc[bad_age, "age"] = np.nan
    fillable = df["age"].isna() & df["dob"].notna() & df["last_visit"].notna()
    df.loc[fillable, "age"] = ((df.loc[fillable, "last_visit"]
                                - df.loc[fillable, "dob"]).dt.days // 365)
    stats["ages_filled_from_dob"] = int(fillable.sum())

    stats["phones_invalid"] = int(df["phone"].notna().sum()
                                  - df["phone"].apply(clean_phone).notna().sum())
    df["phone"] = df["phone"].apply(clean_phone)

    stats["emails_invalid"] = int(df["email"].notna().sum()
                                  - df["email"].apply(clean_email).notna().sum())
    df["email"] = df["email"].apply(clean_email)

    # keep the raw text around so anyone can audit what got mapped to what
    df["diagnosis_raw"] = df["diagnosis"]
    df["diagnosis"] = df["diagnosis"].apply(map_diagnosis)
    stats["diagnosis_unmatched"] = int((df["diagnosis"] == "other").sum())
    print("\ndiagnosis mapping result:")
    print(df["diagnosis"].value_counts().to_string())

    before = len(df)
    df = df.drop_duplicates()
    stats["exact_dupes_removed"] = before - len(df)

    df, near = drop_near_dupes(df)
    stats["near_dupes_removed"] = near

    # tidy column order and iso dates for the output
    df["dob"] = df["dob"].dt.strftime("%Y-%m-%d")
    df["last_visit"] = df["last_visit"].dt.strftime("%Y-%m-%d")
    df = df[["patient_id", "full_name", "gender", "dob", "age", "weight_kg",
             "systolic_bp", "diastolic_bp", "diagnosis", "diagnosis_raw",
             "last_visit", "phone", "email"]]

    stats["rows_out"] = len(df)
    stats["missing_after"] = int(df.isna().sum().sum())

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    lines = ["cleaning report", "-" * 40]
    for k, v in stats.items():
        lines.append("%s: %s" % (k, v))
    lines.append("-" * 40)
    lines.append("note: missing_after counts nulls in the cleaned file. some of")
    lines.append("those are new on purpose - invalid phones, emails and junk ages")
    lines.append("get nulled rather than kept wrong.")
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print("\nsaved", OUT_CSV)


if __name__ == "__main__":
    main()
