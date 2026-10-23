"""Train and evaluate the Logistic Regression model.

Usage:  python train_model.py
Writes: model/heart_model.joblib and model/metrics.json
"""
import json

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import CATEGORICAL, METRICS_PATH, MODEL_PATH, NUMERIC, load_dataset

RANDOM_STATE = 42


def build_pipeline():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])


def main():
    X, y = load_dataset()
    print(f"Samples: {len(X)}  disease: {int(y.sum())}  no disease: {int((1 - y).sum())}")
    print(f"Missing values before imputation: {int(X.isna().sum().sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

    metrics = {
        "dataset": {"samples": int(len(X)), "disease": int(y.sum()),
                    "no_disease": int((1 - y).sum()),
                    "train_size": int(len(X_train)), "test_size": int(len(X_test))},
        "test": {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall": round(recall_score(y_test, pred), 4),
            "f1": round(f1_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "cross_validation_5fold_on_train": {
            "accuracy_mean": round(float(np.mean(cv_acc)), 4),
            "accuracy_std": round(float(np.std(cv_acc)), 4),
            "roc_auc_mean": round(float(np.mean(cv_auc)), 4),
            "roc_auc_std": round(float(np.std(cv_auc)), 4),
        },
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
