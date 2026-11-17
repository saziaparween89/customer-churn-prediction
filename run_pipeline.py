"""
Run the whole project end to end:

    python run_pipeline.py

1. clean & validate   ->  data/processed/churn_clean.csv, reports/data_quality_report.md
2. SQL analysis       ->  reports/sql_results.md
3. EDA charts         ->  reports/figures/eda_*.png
4. train & explain    ->  models/churn_model.joblib, reports/metrics.json,
                          reports/at_risk_customers.csv, reports/figures/*
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

STEPS = [
    ("Clean & validate", ROOT / "src" / "clean.py"),
    ("SQL churn analysis", ROOT / "src" / "sql_analysis.py"),
    ("EDA charts", ROOT / "src" / "eda.py"),
    ("Train, compare & explain models", ROOT / "src" / "train.py"),
]

for i, (name, path) in enumerate(STEPS, 1):
    print(f"\n{'=' * 70}\nStep {i}/{len(STEPS)}: {name}\n{'=' * 70}")
    runpy.run_path(str(path), run_name="__main__")

print("\nDone. Launch the app with:  streamlit run app.py")
