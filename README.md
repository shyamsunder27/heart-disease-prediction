# Heart Disease Prediction System (Logistic Regression)

A Flask web application that estimates the likelihood of heart disease from clinical
parameters using a scikit-learn Logistic Regression model, with Patient and Admin modules.

> **Academic use only.** Predictions are not a medical diagnosis and must not be used for
> clinical decisions without independent validation and clinical assessment.

## Quick start

```bash
pip install -r requirements.txt
python train_model.py        # trains the model, writes model/heart_model.joblib + metrics.json
python create_admin.py       # creates an administrator account (prompts for a password)
python app.py                # http://127.0.0.1:5000
```

Patients register themselves at `/register`. Administrators are created only with
`create_admin.py`; there are no default credentials.

## Features

**Patient module:** register / log in, personal dashboard, submit the 13 clinical parameters
(validated server-side), see the estimated probability and predicted class, assessment
history, printable report (use the browser's *Print > Save as PDF*).

**Admin module:** list of registered patients, each patient's stored assessments and reports,
and a model-performance page (accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
5-fold cross-validation).

**Security:** passwords hashed (Werkzeug scrypt), signed server-side sessions, CSRF tokens on
every POST, role checks on every route, patients can only open their own records, login lockout
after 5 failed attempts (10 min), security headers, input range validation.

## Project layout

```
app.py              Flask routes, auth, admin
model_utils.py      feature definitions, dataset loading, form validation
train_model.py      preprocessing, training, evaluation
storage.py          thread-safe JSON storage (atomic writes)
create_admin.py     admin account creation
templates/, static/ web UI
data/heart_raw.csv  training data
tests/test_app.py   end-to-end tests  (python -m unittest discover -s tests -v)
```

## Model

- Numeric features (age, resting BP, cholesterol, max heart rate, ST depression): median
  imputation + standardisation.
- Categorical features (sex, chest pain, fasting sugar, resting ECG, exercise angina, slope,
  vessels, thalassemia): most-frequent imputation + one-hot encoding.
- Logistic Regression, stratified 80/20 train/test split (random_state 42), 5-fold CV on the
  training portion.

Current results on the 61-record test set: accuracy 0.836, precision 0.846, recall 0.786,
F1 0.815, ROC-AUC 0.939. With a test set this small these figures are noisy; 5-fold CV gives
accuracy 0.835 +/- 0.030 and ROC-AUC 0.908 +/- 0.030.

## About the dataset (please read)

The sandbox this was built in could not reach archive.ics.uci.edu, so `data/heart_raw.csv` is a
GitHub copy of the **Cleveland** subset (303 records) in the widely circulated recoded format.
That copy differs from the original UCI file:

1. **Its `target` is inverted** relative to its name: `target == 0` is the disease group (checked
   against the data: that group has higher ST depression, more exercise angina, more flagged
   vessels, lower max heart rate). The code converts it so 1 always means disease.
2. **Some categorical codes are renumbered** versus the original UCI numbering (chest pain,
   resting ECG, slope, thalassemia). The form labels in `model_utils.py` follow this file's coding,
   inferred by matching category counts to the original distribution.
3. `ca = 4` and `thal = 0` look like placeholders for missing values and are treated as missing.
4. Only a binary label is available, so the original 0-4 severity scale is lost.

**To use the original UCI file instead** (recommended for your report): download
`processed.cleveland.data` from the UCI Heart Disease page into `data/`, then re-run
`python train_model.py`. The loader will use it automatically (`?` as missing, `num > 0` as
disease, exactly as in the abstract). The original uses different category codes, so update the
`choices` in `FEATURES` (`model_utils.py`) to the original numbering (cp 1-4, restecg 0-2,
slope 1-3, ca 0-3, thal 3/6/7) and adjust the test examples.

## Deployment notes

- Set `SECRET_KEY` (env var) and `HTTPS=1` when serving over HTTPS; run with a production server
  (e.g. `gunicorn app:app`), not the Flask dev server.
- JSON files (`data/users.json`, `data/assessments.json`) need persistent disk. Many free hosting
  platforms wipe local files on restart; for real use, move to a database.
- Set `HEART_STORE_DIR` to keep user data outside the project folder.
- The lockout counter is in memory (per process); use a shared store if you run several workers.

## Possible extensions

Larger/multiple datasets with external validation, probability calibration, SQL database,
server-side PDF generation, charts in reports, email verification and password reset.
