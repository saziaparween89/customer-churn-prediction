"""
Score one customer: churn probability, risk band, top reasons (SHAP) and the
recommended retention action.

Python:
    from predict import score_customer
    score_customer({"tenure": 3, "Contract": "Month-to-month", "InternetService": "Fiber optic", ...})

Command line (demo customer):
    python src/predict.py
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from features import FEATURES, pretty
from retention import recommend_action

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 80.0,
    "TotalCharges": None,
}
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


@lru_cache(maxsize=1)
def load():
    bundle = joblib.load(ROOT / "models" / "churn_model.joblib")
    clf = bundle["model"].named_steps["clf"]
    explainer = shap.TreeExplainer(clf) if hasattr(clf, "estimators_") or "XGB" in type(clf).__name__ else None
    return bundle, explainer


def build_features(customer: dict) -> pd.DataFrame:
    c = {**DEFAULTS, **customer}
    if c["InternetService"] == "No":
        for a in ADDONS:
            c[a] = "No"
    if c["TotalCharges"] is None:
        c["TotalCharges"] = c["MonthlyCharges"] * c["tenure"]
    c["num_addon_services"] = sum(c[a] == "Yes" for a in ADDONS)
    c["auto_pay"] = "Yes" if "automatic" in c["PaymentMethod"] else "No"
    avg = c["TotalCharges"] / c["tenure"] if c["tenure"] > 0 else c["MonthlyCharges"]
    c["charge_increase"] = round(c["MonthlyCharges"] - avg, 2)
    return pd.DataFrame([c])


def score_customer(customer: dict, top_n: int = 5) -> dict:
    bundle, explainer = load()
    model, threshold = bundle["model"], bundle["threshold"]
    X = build_features(customer)
    risk = float(model.predict_proba(X[FEATURES])[0, 1])

    reasons = []
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt = pd.DataFrame(prep.transform(X[FEATURES]), columns=[pretty(n) for n in prep.get_feature_names_out()])
    if explainer is not None:
        sv = explainer(Xt).values
        sv = sv[0, :, 1] if sv.ndim == 3 else sv[0]
    else:  # linear model: contribution = coefficient x standardised value
        sv = clf.coef_[0] * Xt.values[0]
    order = np.argsort(-np.abs(sv))[:top_n]
    for i in order:
        name = Xt.columns[i]
        if " = " in name:   # one-hot column: say whether the customer has it or not
            label = name if X.iloc[0][name.split(" = ")[0]] == name.split(" = ")[1] else name.replace(" = ", " is not ")
        else:
            label = f"{name} = {X.iloc[0][name]:g}"
        reasons.append({"feature": label, "impact": round(float(sv[i]), 3),
                        "direction": "raises risk" if sv[i] > 0 else "lowers risk"})

    row = X.iloc[0].copy()
    row["churn_risk"] = risk
    band = "High" if risk >= max(0.5, threshold + 0.05) else "Medium" if risk >= threshold else "Low"
    return {"churn_probability": round(risk, 3), "risk_band": band,
            "annual_revenue_at_risk": round(risk * row["MonthlyCharges"] * 12, 2),
            "recommended_action": recommend_action(row, threshold=threshold),
            "top_reasons": reasons}


if __name__ == "__main__":
    demo = {"tenure": 2, "Contract": "Month-to-month", "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check", "MonthlyCharges": 89.5, "TechSupport": "No"}
    print(json.dumps(score_customer(demo), indent=2))
    loyal = {"tenure": 60, "Contract": "Two year", "InternetService": "DSL", "Partner": "Yes",
             "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0, "TechSupport": "Yes",
             "OnlineSecurity": "Yes"}
    print(json.dumps(score_customer(loyal), indent=2))
