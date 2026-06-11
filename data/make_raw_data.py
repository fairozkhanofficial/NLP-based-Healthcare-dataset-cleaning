"""generates the raw messy ehr csv that the cleaning pipeline fixes. run this first."""
import csv
import os
import random

random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "raw_healthcare_data.csv")

first_names = ["Aarav", "Vihaan", "Aditya", "Arjun", "Rohan", "Karan", "Imran", "Faisal",
               "Rahul", "Amit", "Sanjay", "Vikram", "Nikhil", "Suresh", "Manoj", "Deepak",
               "Priya", "Ananya", "Sneha", "Pooja", "Kavya", "Divya", "Neha", "Riya",
               "Aisha", "Fatima", "Zara", "Meera", "Lakshmi", "Anita", "Sunita", "Rekha",
               "John", "David", "Maria", "Sarah", "James", "Linda", "Peter", "Anna"]

last_names = ["Sharma", "Verma", "Gupta", "Khan", "Patel", "Reddy", "Nair", "Iyer",
              "Singh", "Kumar", "Das", "Bose", "Chopra", "Mehta", "Joshi", "Kulkarni",
              "Rao", "Naidu", "Pillai", "Menon", "Shaikh", "Syed", "Ali", "Ahmed",
              "Fernandes", "Dsouza", "Thomas", "George", "Mathew", "Varghese"]

# all the messy ways each condition shows up in the export. the cleaning
# script has to map every one of these back to a single canonical name
diagnosis_pool = {
    "hypertension": ["hypertension", "Hypertension", "HTN", "htn", "high bp",
                     "High Blood Pressure", "hypertention"],
    "type 2 diabetes": ["type 2 diabetes", "T2DM", "diabetes type 2",
                        "diabetess type II", "Type II Diabetes",
                        "diabetes mellitus type 2"],
    "asthma": ["asthma", "Asthma", "asthmaa", "bronchial asthma"],
    "migraine": ["migraine", "migrane", "chronic migraine", "Migraine headache"],
    "arthritis": ["arthritis", "arthritus", "osteoarthritis", "Arthritis"],
    "anemia": ["anemia", "anaemia", "iron deficiency anemia", "low iron"],
    "coronary artery disease": ["coronary artery disease", "CAD", "heart disease",
                                "coronory artery disease"],
    "hypothyroidism": ["hypothyroidism", "hypothyroid", "low thyroid",
                       "Underactive Thyroid"],
    "copd": ["copd", "COPD", "C.O.P.D", "chronic obstructive pulmonary disease"],
    "urinary tract infection": ["urinary tract infection", "UTI", "uti",
                                "urine infection"],
}

# a few rows have notes that dont map to any real condition, the pipeline
# should shove these into an "other" bucket
junk_diagnoses = ["general checkup", "follow up visit", "fever", "viral fever",
                  "routine visit"]

months_abbr = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def messy_date(y, m, d):
    # the clinic software apparently exported dates in whatever format it felt
    # like that day. slash format is day first
    r = random.random()
    if r < 0.40:
        return "%02d/%02d/%d" % (d, m, y)
    elif r < 0.75:
        return "%d-%02d-%02d" % (y, m, d)
    else:
        return "%02d-%s-%d" % (d, months_abbr[m - 1], y)


def messy_gender():
    return random.choices(
        ["M", "Male", "male", "m", "F", "Female", "female", "f", "unknown", "U", ""],
        weights=[14, 12, 8, 5, 14, 12, 8, 5, 2, 1, 3])[0]


def messy_weight():
    if random.random() < 0.06:
        return ""
    kg = round(random.uniform(42, 110), 1)
    r = random.random()
    if r < 0.30:
        lbs = round(kg * 2.20462)
        return random.choice(["%d lbs" % lbs, "%dlbs" % lbs])
    elif r < 0.55:
        return "%s kg" % kg
    elif r < 0.70:
        return "%skg" % kg
    else:
        return str(kg)  # bare number, assume kg


def messy_bp():
    r = random.random()
    if r < 0.05:
        return ""
    if r < 0.07:
        return "n/a"
    sys = random.randint(95, 180)
    dia = random.randint(55, 110)
    if r < 0.60:
        return "%d/%d" % (sys, dia)
    elif r < 0.85:
        return "%d / %d" % (sys, dia)
    else:
        return "%d/%d mmHg" % (sys, dia)


def messy_phone():
    if random.random() < 0.05:
        return ""
    num = random.choice("6789") + "".join(random.choice("0123456789") for _ in range(9))
    r = random.random()
    if r < 0.04:
        return num[:8]  # truncated, garbage
    if r < 0.35:
        return num
    elif r < 0.55:
        return "+91 " + num
    elif r < 0.70:
        return "+91-%s-%s" % (num[:5], num[5:])
    elif r < 0.85:
        return "0" + num
    else:
        return "%s %s" % (num[:5], num[5:])


def messy_email(first, last):
    r = random.random()
    if r < 0.08:
        return ""
    if r < 0.11:
        return "not provided"
    base = "%s.%s%d" % (first.lower(), last.lower(), random.randint(1, 99))
    domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "rediffmail.com"])
    email = base + "@" + domain
    if r < 0.14:
        return base + "@" + domain.split(".")[0]  # missing the .com, invalid
    if r < 0.25:
        return email.upper()
    if r < 0.32:
        return "  " + email + " "
    return email


def maybe_pad(s):
    # random whitespace mess
    if random.random() < 0.07:
        return " " + s
    if random.random() < 0.07:
        return s + "  "
    return s


def typo(name):
    # small typo for near duplicate rows, like someone retyped the name at the desk
    chars = list(name)
    r = random.random()
    if r < 0.3:
        i = random.randint(1, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    elif r < 0.6:
        i = random.randint(1, len(chars) - 2)
        del chars[i]
    elif r < 0.8:
        i = random.randint(1, len(chars) - 2)
        chars.insert(i, chars[i])
    else:
        return name.lower()
    return "".join(chars)


# flatten diagnosis variants into one weighted pool
all_diag = []
for canon, variants in diagnosis_pool.items():
    all_diag.extend(variants)

rows = []
for i in range(8600):
    first = random.choice(first_names)
    last = random.choice(last_names)
    name = first + " " + last
    # some names got mangled by whatever system this came from
    r = random.random()
    if r < 0.05:
        name = name.upper()
    elif r < 0.10:
        name = name.lower()

    by = random.randint(1940, 2008)
    bm = random.randint(1, 12)
    bd = random.randint(1, 28)

    vy = random.randint(2024, 2025)
    vm = random.randint(1, 12)
    vd = random.randint(1, 28)

    age = vy - by + random.choice([0, 0, 0, -1])
    if random.random() < 0.005:
        age = ""
    # data entry disasters
    elif random.random() < 0.004:
        age = 999
    elif random.random() < 0.0025:
        age = -1
    elif random.random() < 0.0015:
        age = random.randint(150, 300)

    if random.random() < 0.02:
        diag = random.choice(junk_diagnoses)
    elif random.random() < 0.01:
        diag = ""
    else:
        diag = random.choice(all_diag)

    rows.append([
        "P%05d" % (i + 1),
        maybe_pad(name),
        messy_date(by, bm, bd) if random.random() > 0.015 else "",
        messy_gender(),
        age,
        messy_weight(),
        messy_bp(),
        maybe_pad(diag),
        messy_date(vy, vm, vd),
        messy_phone(),
        messy_email(first, last),
    ])

# exact duplicate rows, the same record uploaded twice
for row in random.sample(rows, 250):
    rows.append(list(row))

# near duplicates: same person registered again with a new patient id and a
# typo in the name. same date of birth though, thats the giveaway
near = random.sample(rows[:8600], 180)
next_id = 8601
for row in near:
    copy = list(row)
    copy[0] = "P%05d" % next_id
    next_id += 1
    copy[1] = typo(copy[1].strip())
    copy[9] = messy_phone()  # phone re-entered differently
    rows.append(copy)

random.shuffle(rows)

header = ["Patient ID", "Full Name ", "DOB", "Gender", "Age", "Weight",
          "Blood Pressure", "Diagnosis", "Last Visit", "Phone", "Email "]

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(rows)

print("wrote %d rows to %s" % (len(rows), OUT))
