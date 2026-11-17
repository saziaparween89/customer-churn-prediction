"""
Customer Churn Prediction - interactive Streamlit app.

    streamlit run app.py

Tabs:
  1. Predict a customer - what-if simulator with SHAP reasons and next best action
  2. Churn dashboard    - KPIs and churn drivers with filters
  3. Retention list     - every active customer ranked by risk, downloadable
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import score_customer  # noqa: E402

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")


@st.cache_data
def load():
    df = pd.read_csv(ROOT / "data" / "processed" / "churn_clean.csv")
    risk = pd.read_csv(ROOT / "reports" / "at_risk_customers.csv")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    return df, risk, metrics


df, risk, metrics = load()
st.title("📉 Customer Churn Prediction & Retention")
st.caption(f"{metrics['best_model']} · test ROC-AUC {metrics['test']['roc_auc']:.2f} · "
           f"catches {metrics['test']['recall']:.0%} of churners at the profit-optimal threshold")

tab1, tab2, tab3 = st.tabs(["🔮 Predict a customer", "📊 Churn dashboard", "🎯 Retention list"])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.write("Change the customer's details and watch the churn risk update.")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Tenure (months)", 0, 72, 4)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    with c2:
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 85.0, 0.5)
        addons = st.multiselect("Add-on services", ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                                                    "TechSupport", "StreamingTV", "StreamingMovies"])
    with c3:
        senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)
        multiple = st.radio("Multiple phone lines", ["No", "Yes"], horizontal=True)

    customer = {"tenure": tenure, "Contract": contract, "PaymentMethod": payment, "PaperlessBilling": paperless,
                "InternetService": internet, "MonthlyCharges": monthly, "SeniorCitizen": senior,
                "Partner": partner, "Dependents": dependents, "MultipleLines": multiple,
                **{a: ("Yes" if a in addons else "No") for a in ["OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]}}
    r = score_customer(customer, top_n=7)

    k1, k2, k3 = st.columns(3)
    k1.metric("Churn probability", f"{r['churn_probability']:.0%}")
    k2.metric("Risk band", r["risk_band"])
    k3.metric("Annual revenue at risk", f"${r['annual_revenue_at_risk']:,.0f}")
    st.info(f"**Next best action:** {r['recommended_action']}")
    reasons = pd.DataFrame(r["top_reasons"]).iloc[::-1]
    fig = px.bar(reasons, x="impact", y="feature", orientation="h", color="direction",
                 color_discrete_map={"raises risk": "#dc2626", "lowers risk": "#2563eb"},
                 title="Why? Top factors behind this prediction (SHAP)", height=360)
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- Tab 2
with tab2:
    f1, f2 = st.columns(2)
    contracts = f1.multiselect("Contract", df["Contract"].unique().tolist(), default=df["Contract"].unique().tolist())
    nets = f2.multiselect("Internet", df["InternetService"].unique().tolist(),
                          default=df["InternetService"].unique().tolist())
    d = df[df["Contract"].isin(contracts) & df["InternetService"].isin(nets)]
    k = st.columns(4)
    k[0].metric("Customers", f"{len(d):,}")
    k[1].metric("Churn rate", f"{d['Churn'].mean():.1%}")
    k[2].metric("Monthly revenue", f"${d['MonthlyCharges'].sum():,.0f}")
    k[3].metric("Annual revenue lost", f"${d.loc[d.Churn == 1, 'MonthlyCharges'].sum() * 12:,.0f}")

    def rate(col):
        g = d.groupby(col)["Churn"].mean().mul(100).round(1).reset_index()
        return px.bar(g, x=col, y="Churn", title=f"Churn % by {col}", labels={"Churn": "Churn %"})

    a, b = st.columns(2)
    order = ["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49-72 m"]
    g = d.groupby("tenure_group")["Churn"].mean().mul(100).reindex(order).round(1).reset_index()
    a.plotly_chart(px.bar(g, x="tenure_group", y="Churn", title="Churn % by tenure",
                          labels={"Churn": "Churn %"}), width="stretch")
    b.plotly_chart(rate("PaymentMethod"), width="stretch")
    a2, b2 = st.columns(2)
    a2.plotly_chart(rate("num_addon_services"), width="stretch")
    b2.plotly_chart(px.scatter(d.sample(min(len(d), 2500), random_state=1), x="tenure", y="MonthlyCharges",
                               color=d.sample(min(len(d), 2500), random_state=1)["Churn"].map({0: "Stayed", 1: "Churned"}),
                               opacity=0.5, title="Tenure vs monthly charges"), width="stretch")

# ---------------------------------------------------------------- Tab 3
with tab3:
    s = risk.groupby("risk_band").agg(customers=("customerID", "count"),
                                      revenue_at_risk=("annual_revenue_at_risk", "sum"))
    k = st.columns(3)
    for col, band in zip(k, ["High", "Medium", "Low"]):
        if band in s.index:
            col.metric(f"{band} risk", f"{int(s.loc[band, 'customers']):,} customers",
                       f"${s.loc[band, 'revenue_at_risk']:,.0f} revenue at risk", delta_color="off")
    bands = st.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    view = risk[risk["risk_band"].isin(bands)]
    st.plotly_chart(px.histogram(view, x="recommended_action", title="Recommended actions",
                                 labels={"recommended_action": ""}).update_xaxes(categoryorder="total descending"),
                    width="stretch")
    st.dataframe(view, width="stretch", height=420)
    st.download_button("Download retention list (CSV)", view.to_csv(index=False), "retention_list.csv")
