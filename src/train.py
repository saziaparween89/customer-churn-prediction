"""
Step 4 - Train, compare and explain churn models, then turn predictions into money.

  * 3 models compared with 5-fold cross-validation: Logistic Regression,
    Random Forest, XGBoost
  * Final check on a 20% hold-out test set the models never saw
  * Decision threshold chosen by PROFIT, not by accuracy (see BUSINESS below)
  * SHAP explains which factors drive churn overall and for each customer
  * Every current customer gets a churn-risk score, a risk band and a
    recommended retention action -> reports/at_risk_customers.csv

Run:  python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from features import FEATURES, TARGET, make_preprocessor, pretty  # noqa: E402
from retention import recommend_action  # noqa: E402
from style import PALETTE, apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "churn_clean.csv"
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
SEED = 42

# Business assumptions for the retention campaign (change these to your company's numbers)
BUSINESS = {
    "offer_cost": 60.0,        # $ cost of a retention offer (discount / free add-on) per customer contacted
    "offer_success_rate": 0.30,  # share of would-be churners the offer actually saves
    "months_saved": 12,          # revenue kept for 12 months when a churner is saved
}


def candidate_models() -> dict[str, Pipeline]:
    # No class re-weighting on purpose: it would inflate the probabilities.
    # We keep probabilities honest (so "revenue at risk" is realistic) and handle
    # the 26% / 74% imbalance by choosing the decision threshold instead.
    return {
        "Logistic Regression": Pipeline([
            ("prep", make_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, C=0.5)),
        ]),
        "Random Forest": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10,
                                           random_state=SEED, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("prep", make_preprocessor(scale=False)),
            ("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                                  colsample_bytree=0.8, eval_metric="logloss",
                                  random_state=SEED, n_jobs=4)),
        ]),
    }


def campaign_value(y, proba, monthly, threshold) -> dict:
    """Net $ value of contacting every customer whose risk >= threshold."""
    target = proba >= threshold
    saved_revenue = (y[target] * monthly[target] * BUSINESS["months_saved"]
                     * BUSINESS["offer_success_rate"]).sum()
    cost = target.sum() * BUSINESS["offer_cost"]
    return {"threshold": round(float(threshold), 2), "contacted": int(target.sum()),
            "churners_reached": int(y[target].sum()), "saved_revenue": round(float(saved_revenue)),
            "campaign_cost": round(float(cost)), "net_value": round(float(saved_revenue - cost))}


def main() -> dict:
    apply_style()
    for p in (MODELS, FIGS):
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te, m_tr, m_te = train_test_split(
        X, y, df["MonthlyCharges"].to_numpy(), test_size=0.2, stratify=y, random_state=SEED)
    print(f"Train {len(X_tr):,} | Test {len(X_te):,} | churn rate {y.mean():.1%}")

    # ---- 1. Compare models with 5-fold CV (train set only) -----------------
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    comparison, oof = {}, {}
    for name, model in candidate_models().items():
        p = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        comparison[name] = {"cv_roc_auc": round(roc_auc_score(y_tr, p), 4),
                            "cv_pr_auc": round(average_precision_score(y_tr, p), 4)}
        print(f"  {name:<20} CV ROC-AUC={comparison[name]['cv_roc_auc']:.3f}  PR-AUC={comparison[name]['cv_pr_auc']:.3f}")
    best_name = max(comparison, key=lambda k: comparison[k]["cv_roc_auc"])
    print(f"Best model: {best_name}")

    # ---- 2. Pick the decision threshold that maximises campaign profit -----
    #         (on out-of-fold predictions, so the test set stays untouched)
    grid = np.round(np.arange(0.10, 0.91, 0.01), 2)
    curve = pd.DataFrame([campaign_value(y_tr, oof[best_name], m_tr, t) for t in grid])
    best_t = float(curve.loc[curve["net_value"].idxmax(), "threshold"])
    print(f"Profit-optimal threshold: {best_t:.2f}")

    # ---- 3. Fit on full train set, evaluate once on the hold-out test set --
    final = candidate_models()[best_name].fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]
    pred = (p_te >= best_t).astype(int)
    test_metrics = {
        "roc_auc": round(roc_auc_score(y_te, p_te), 4),
        "pr_auc": round(average_precision_score(y_te, p_te), 4),
        "threshold": best_t,
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
    }
    test_value = campaign_value(y_te, p_te, m_te, best_t)
    contact_all = campaign_value(y_te, np.ones_like(p_te), m_te, 0.5)
    print(f"  TEST ROC-AUC={test_metrics['roc_auc']:.3f} recall={test_metrics['recall']:.2f} "
          f"precision={test_metrics['precision']:.2f}  net value=${test_value['net_value']:,}")
    for name in comparison:  # test AUC of every model, for the README table
        m = candidate_models()[name].fit(X_tr, y_tr)
        comparison[name]["test_roc_auc"] = round(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]), 4)

    # ---- Charts: ROC comparison --------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for (name, p), color in zip(oof.items(), [PALETTE[0], PALETTE[1], PALETTE[2]]):
        fpr, tpr, _ = roc_curve(y_tr, p)
        ax.plot(fpr, tpr, lw=2.2, color=color, label=f"{name} (AUC {comparison[name]['cv_roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (churners caught)")
    ax.set_title("Model comparison - 5-fold cross-validated ROC")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "model_roc_comparison.png", dpi=150)
    plt.close(fig)

    # ---- Charts: profit vs threshold ---------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(curve["threshold"], curve["net_value"] / 1000, color=PALETTE[1], lw=2.4)
    ax.axvline(best_t, ls="--", color=PALETTE[3], lw=1.2)
    peak = curve["net_value"].max() / 1000
    ax.annotate(f"Best threshold {best_t:.2f}\nnet value ${peak:,.0f}k", xy=(best_t, peak),
                xytext=(best_t + 0.12, peak * 0.8), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#4b5563"))
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_xlabel("Churn-risk threshold for sending a retention offer")
    ax.set_ylabel("Net campaign value ($ thousands)")
    ax.set_title("Choosing who to contact: threshold that maximises profit")
    fig.tight_layout()
    fig.savefig(FIGS / "profit_vs_threshold.png", dpi=150)
    plt.close(fig)

    # ---- 4. Explainability with SHAP ---------------------------------------
    prep, clf = final.named_steps["prep"], final.named_steps["clf"]
    Xt_te = pd.DataFrame(prep.transform(X_te), columns=[pretty(c) for c in prep.get_feature_names_out()])
    if best_name == "Logistic Regression":
        explainer = shap.LinearExplainer(clf, pd.DataFrame(prep.transform(X_tr), columns=Xt_te.columns))
    else:
        explainer = shap.TreeExplainer(clf)
    sv = explainer(Xt_te)
    if sv.values.ndim == 3:          # tree classifiers return one set per class -> keep "churn"
        sv = sv[:, :, 1]

    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False, plot_size=(9, 6.5))
    plt.title("What drives churn? (SHAP values, test set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    importance = pd.Series(np.abs(sv.values).mean(0), index=Xt_te.columns).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    top = importance.head(12)[::-1]
    ax.barh(top.index, top.values, color=PALETTE[0])
    ax.set_xlabel("Mean |SHAP value| (impact on churn prediction)")
    ax.set_title("Top 12 churn drivers")
    fig.tight_layout()
    fig.savefig(FIGS / "shap_importance.png", dpi=150)
    plt.close(fig)

    # One customer explained - the kind of chart a retention agent would see
    i = int(np.argmax(p_te * (y_te == 1)))
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.title(f"Why is this customer high-risk? (predicted {p_te[i]:.0%})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGS / "shap_single_customer.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    # ---- 5. Score every customer (out-of-fold = honest scores) --------------
    all_scores = cross_val_predict(candidate_models()[best_name], X, y, cv=cv, method="predict_proba")[:, 1]
    scored = df[["customerID", "tenure", "Contract", "InternetService", "PaymentMethod",
                 "TechSupport", "MonthlyCharges", "charge_increase", "Churn"]].copy()
    scored["churn_risk"] = all_scores.round(3)
    # Low  = below the profit threshold (an offer would cost more than it saves)
    # Medium = worth an offer;  High = more likely to leave than to stay
    high_cut = max(0.5, best_t + 0.05)
    scored["risk_band"] = pd.cut(scored["churn_risk"], [-0.01, best_t, high_cut, 1.0], right=False,
                                 labels=["Low", "Medium", "High"]).astype(str)
    scored["recommended_action"] = scored.apply(recommend_action, axis=1, threshold=best_t)
    scored["annual_revenue_at_risk"] = (scored["MonthlyCharges"] * 12 * scored["churn_risk"]).round(2)
    active = scored[scored["Churn"] == 0].drop(columns="Churn").sort_values("churn_risk", ascending=False)
    active.to_csv(REPORTS / "at_risk_customers.csv", index=False)
    bands = active.groupby("risk_band").agg(customers=("customerID", "count"),
                                            revenue_at_risk=("annual_revenue_at_risk", "sum")).round(0)
    print(bands)

    # ---- Save ---------------------------------------------------------------
    final_all = candidate_models()[best_name].fit(X, y)   # production model uses all data
    joblib.dump({"model": final_all, "threshold": best_t, "name": best_name}, MODELS / "churn_model.joblib")

    metrics = {
        "best_model": best_name, "model_comparison": comparison, "test": test_metrics,
        "business_assumptions": BUSINESS, "campaign_on_test_set": test_value,
        "contact_everyone_on_test_set": contact_all,
        "top_drivers": importance.head(10).round(4).to_dict(),
        "active_customers_by_risk_band": bands.reset_index().to_dict(orient="records"),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model and metrics. Test net value ${test_value['net_value']:,} "
          f"vs contact-everyone ${contact_all['net_value']:,}")
    return metrics


if __name__ == "__main__":
    main()
