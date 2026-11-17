"""
Step 3 - Exploratory data analysis charts (saved to reports/figures).

Run:  python src/eda.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import PALETTE, apply_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "churn_clean.csv"
FIGS = ROOT / "reports" / "figures"


def rate_bar(ax, s: pd.Series, title: str, overall: float, rotate: int = 0):
    colors = [PALETTE[3] if v > overall else PALETTE[5] for v in s.values]
    bars = ax.bar(s.index.astype(str), s.values, color=colors)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in s.values], padding=2, fontsize=9)
    ax.axhline(overall, ls="--", color="#374151", lw=1)
    ax.set_title(title)
    ax.set_ylabel("Churn rate %")
    ax.set_ylim(0, max(s.values) * 1.18)
    if rotate:
        ax.tick_params(axis="x", rotation=rotate)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=150)
    plt.close(fig)


def main() -> None:
    apply_style()
    FIGS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN)
    overall = df["Churn"].mean() * 100

    # 1. The four strongest single drivers on one figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    rate_bar(axes[0, 0], df.groupby("Contract")["Churn"].mean() * 100, "Contract type", overall)
    order = ["0-6 m", "7-12 m", "13-24 m", "25-48 m", "49-72 m"]
    rate_bar(axes[0, 1], (df.groupby("tenure_group")["Churn"].mean() * 100).reindex(order),
             "Tenure (months with us)", overall)
    pm = (df.groupby("PaymentMethod")["Churn"].mean() * 100).sort_values(ascending=False)
    pm.index = pm.index.str.replace(" (automatic)", "\n(auto)")
    rate_bar(axes[1, 0], pm, "Payment method", overall)
    rate_bar(axes[1, 1], df.groupby("InternetService")["Churn"].mean() * 100, "Internet service", overall)
    fig.suptitle(f"Churn rate by key customer attributes  (dashed line = overall {overall:.1f}%, red = above)",
                 fontweight="bold", fontsize=13)
    save(fig, "eda_churn_drivers.png")

    # 2. Tenure distribution, churned vs stayed
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(df.loc[df.Churn == 0, "tenure"], bins=36, alpha=0.75, color=PALETTE[0], label="Stayed")
    ax.hist(df.loc[df.Churn == 1, "tenure"], bins=36, alpha=0.8, color=PALETTE[3], label="Churned")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Customers")
    ax.set_title("Most churn happens in the first few months")
    ax.legend()
    save(fig, "eda_tenure_distribution.png")

    # 3. Monthly charges vs tenure scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, val in [("Stayed", PALETTE[0], 0), ("Churned", PALETTE[3], 1)]:
        d = df[df.Churn == val]
        ax.scatter(d["tenure"], d["MonthlyCharges"], s=7, alpha=0.35, color=color, label=label)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly charges ($)")
    ax.set_title("New customers on expensive plans churn the most (top-left corner)")
    ax.legend(markerscale=3)
    save(fig, "eda_charges_vs_tenure.png")

    # 4. Add-on services (internet customers only)
    inet = df[df["InternetService"] != "No"]
    s = inet.groupby("num_addon_services")["Churn"].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bars = ax.bar(s.index.astype(str), s.values, color=PALETTE[1])
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in s.values], padding=2, fontsize=9)
    ax.set_xlabel("Number of add-on services (security, backup, support, streaming...)")
    ax.set_ylabel("Churn rate %")
    ax.set_title("Every extra add-on service makes customers stickier")
    save(fig, "eda_addons.png")

    # 5. Revenue lost by contract type
    lost = df[df.Churn == 1].groupby("Contract")["MonthlyCharges"].sum() * 12 / 1000
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.barh(lost.index, lost.values, color=[PALETTE[3], PALETTE[5], PALETTE[5]])
    ax.bar_label(bars, labels=[f"${v:,.0f}k" for v in lost.values], padding=3)
    ax.set_xlabel("Annual revenue lost to churn ($ thousands)")
    ax.set_title("Month-to-month contracts drive 87% of lost revenue")
    ax.set_xlim(0, lost.max() * 1.2)
    save(fig, "eda_revenue_lost.png")

    print(f"Saved EDA charts to {FIGS}")


if __name__ == "__main__":
    main()
