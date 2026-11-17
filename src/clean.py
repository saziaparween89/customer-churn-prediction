"""
Step 1 - Data cleaning, validation & feature engineering.

Input : data/raw/telco_customer_churn.csv   (IBM Telco Customer Churn, 7,043 customers)
Output: data/processed/churn_clean.csv
        reports/data_quality_report.md

Run:  python src/clean.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "telco_customer_churn.csv"
OUT = ROOT / "data" / "processed" / "churn_clean.csv"
REPORT = ROOT / "reports" / "data_quality_report.md"

ADDON_SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"]


def main() -> pd.DataFrame:
    df = pd.read_csv(RAW)
    log: list[tuple[str, str]] = [("Raw rows x columns", f"{df.shape[0]:,} x {df.shape[1]}")]

    # 1. Duplicates
    log.append(("Duplicate customerID", str(df["customerID"].duplicated().sum())))
    df = df.drop_duplicates("customerID")

    # 2. TotalCharges is stored as TEXT because of blank strings.
    blanks = df["TotalCharges"].astype(str).str.strip().eq("")
    log.append(("Blank TotalCharges (stored as text)", str(int(blanks.sum()))))
    # Every blank row has tenure = 0 -> brand-new customers who haven't been billed yet.
    assert (df.loc[blanks, "tenure"] == 0).all(), "Unexpected: blank TotalCharges for an existing customer"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True))
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    log.append(("-> filled with 0 (tenure = 0, not yet billed)", str(int(blanks.sum()))))

    # 3. Consistent encodings
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    # "No internet service" / "No phone service" carry the same info as InternetService / PhoneService
    n_collapsed = int(df[ADDON_SERVICES + ["MultipleLines"]].isin(["No internet service", "No phone service"]).sum().sum())
    df[ADDON_SERVICES] = df[ADDON_SERVICES].replace("No internet service", "No")
    df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
    log.append(("'No internet/phone service' values collapsed to 'No'", f"{n_collapsed:,}"))

    # 4. Sanity checks
    assert df["MonthlyCharges"].between(0, 500).all()
    assert df["tenure"].between(0, 100).all()
    assert df["Churn"].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0, "Missing values remain"
    # Billing consistency: TotalCharges should be roughly tenure x MonthlyCharges
    ratio = df.loc[df.tenure > 0, "TotalCharges"] / (df.loc[df.tenure > 0, "tenure"] * df.loc[df.tenure > 0, "MonthlyCharges"])
    log.append(("Customers where TotalCharges deviates >25% from tenure x monthly", str(int((~ratio.between(0.75, 1.25)).sum()))))

    # 5. Feature engineering (business-meaningful features)
    df["tenure_group"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72],
                                labels=["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49-72 m"])
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)
    df["has_family"] = np.where((df["Partner"] == "Yes") | (df["Dependents"] == "Yes"), "Yes", "No")
    df["auto_pay"] = np.where(df["PaymentMethod"].str.contains("automatic"), "Yes", "No")
    df["avg_monthly_spend"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]).round(2)
    # Price change signal: paying more now than the historical average
    df["charge_increase"] = (df["MonthlyCharges"] - df["avg_monthly_spend"]).round(2)
    df["annual_revenue"] = (df["MonthlyCharges"] * 12).round(2)
    log.append(("Engineered features added", "7"))
    log.append(("Clean rows x columns", f"{df.shape[0]:,} x {df.shape[1]}"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("# Data Quality Report\n\n| Check / step | Result |\n|---|---:|\n" +
                      "\n".join(f"| {k} | {v} |" for k, v in log) + "\n")
    for k, v in log:
        print(f"{k:<68} {v:>12}")
    return df


if __name__ == "__main__":
    main()
