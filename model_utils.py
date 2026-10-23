"""Shared feature definitions and data loading for the Heart Disease Prediction System.

DATASET NOTE
------------
The bundled file (data/heart_raw.csv) is a 303-row copy of the Cleveland subset of the
UCI Heart Disease data. It differs from the original UCI file in two ways, both handled here:

1. Its ``target`` column is the opposite of what the name suggests: ``target == 0`` marks
   patients WITH recorded disease (verified from the data: that group has higher ST
   depression, more exercise angina, more flagged vessels and lower max heart rate).
   We convert it to ``has_disease = 1 - target`` so 1 always means "disease recorded".
2. Some categorical codes are renumbered relative to the original UCI numbering.
   The labels in FEATURES below describe the codes as they appear in THIS file.

If you place the original ``processed.cleveland.data`` (from archive.ics.uci.edu) in the
data/ folder, load_dataset() uses it instead, with '?' treated as missing and
num > 0 treated as disease, exactly as described in the project abstract. NOTE: the
original uses different category codes, so the web form labels would need adjusting.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "model" / "heart_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERIC + CATEGORICAL

# Form definition: drives both validation and the web form.
FEATURES = {
    "age": {"label": "Age (years)", "type": "number", "min": 18, "max": 100, "step": 1},
    "sex": {"label": "Sex", "type": "choice",
            "choices": {0: "Female", 1: "Male"}},
    "cp": {"label": "Chest pain type", "type": "choice",
           "choices": {0: "Asymptomatic", 1: "Atypical angina",
                       2: "Non-anginal pain", 3: "Typical angina"}},
    "trestbps": {"label": "Resting blood pressure (mm Hg)", "type": "number",
                 "min": 80, "max": 220, "step": 1},
    "chol": {"label": "Serum cholesterol (mg/dl)", "type": "number",
             "min": 100, "max": 600, "step": 1},
    "fbs": {"label": "Fasting blood sugar > 120 mg/dl", "type": "choice",
            "choices": {0: "No", 1: "Yes"}},
    "restecg": {"label": "Resting ECG result", "type": "choice",
                "choices": {0: "Left ventricular hypertrophy", 1: "Normal",
                            2: "ST-T wave abnormality"}},
    "thalach": {"label": "Maximum heart rate achieved (bpm)", "type": "number",
                "min": 60, "max": 220, "step": 1},
    "exang": {"label": "Exercise-induced angina", "type": "choice",
              "choices": {0: "No", 1: "Yes"}},
    "oldpeak": {"label": "ST depression induced by exercise", "type": "number",
                "min": 0, "max": 7, "step": 0.1},
    "slope": {"label": "Slope of peak exercise ST segment", "type": "choice",
              "choices": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}},
    "ca": {"label": "Major vessels coloured by fluoroscopy", "type": "choice",
           "choices": {0: "0", 1: "1", 2: "2", 3: "3"}},
    "thal": {"label": "Thalassemia test result", "type": "choice",
             "choices": {1: "Fixed defect", 2: "Normal", 3: "Reversible defect"}},
}


def load_dataset():
    """Return (X, y) with y = 1 meaning heart disease recorded."""
    original = DATA_DIR / "processed.cleveland.data"
    if original.exists():
        cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
        df = pd.read_csv(original, header=None, names=cols, na_values="?")
        y = (df["num"] > 0).astype(int)  # 0 = no disease, >0 = disease
        X = df.drop(columns="num")[ALL_FEATURES]
        return X, y

    df = pd.read_csv(DATA_DIR / "heart_raw.csv")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    y = (1 - df["target"]).astype(int)  # see DATASET NOTE: target==0 is disease
    X = df[ALL_FEATURES].copy()
    # Placeholder codes for missing values in this copy of the data.
    X.loc[X["ca"] == 4, "ca"] = np.nan
    X.loc[X["thal"] == 0, "thal"] = np.nan
    return X, y


def parse_form(form):
    """Validate a submitted form. Returns (values_dict, errors_dict)."""
    values, errors = {}, {}
    for name, spec in FEATURES.items():
        raw = (form.get(name) or "").strip()
        if raw == "":
            errors[name] = "This field is required."
            continue
        try:
            num = float(raw)
        except ValueError:
            errors[name] = "Enter a valid number."
            continue
        if spec["type"] == "choice":
            if int(num) != num or int(num) not in spec["choices"]:
                errors[name] = "Choose one of the listed options."
                continue
            values[name] = int(num)
        else:
            if not (spec["min"] <= num <= spec["max"]):
                errors[name] = f"Enter a value between {spec['min']} and {spec['max']}."
                continue
            values[name] = num
    return values, errors
