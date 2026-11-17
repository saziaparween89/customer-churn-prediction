"""Feature lists and the preprocessing pipeline shared by training, scoring and the app."""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"

CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "auto_pay",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "num_addon_services", "charge_increase"]
FEATURES = CATEGORICAL + NUMERIC


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", StandardScaler() if scale else "passthrough", NUMERIC),
    ], verbose_feature_names_out=False)


def pretty(name: str) -> str:
    """Turn 'Contract_Month-to-month' into 'Contract = Month-to-month' for charts."""
    for col in CATEGORICAL:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name
