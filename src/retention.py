"""Turn a churn score into a concrete next step for the retention team.

Rules are based on the drivers found in the SQL analysis and SHAP, ordered by
impact, so every high-risk customer gets the single most relevant offer."""
import pandas as pd


def recommend_action(row: pd.Series, threshold: float = 0.30) -> str:
    if row["churn_risk"] < threshold:
        return "No action - keep in regular communication"
    if row["tenure"] <= 6:
        return "Onboarding call + first-3-months welcome discount"
    if row["Contract"] == "Month-to-month" and row["churn_risk"] >= 0.5:
        return "Offer 1-year contract with 15% discount"
    if row["InternetService"] == "Fiber optic" and row["TechSupport"] == "No":
        return "Free Tech Support add-on for 3 months"
    if row["PaymentMethod"] == "Electronic check":
        return "Incentive to switch to auto-pay (bank/card)"
    if row.get("charge_increase", 0) > 5:
        return "Loyalty discount to offset recent price increase"
    if row["Contract"] == "Month-to-month":
        return "Offer 1-year contract with 10% discount"
    return "Account manager check-in call"
