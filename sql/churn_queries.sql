-- ============================================================
-- Customer churn analysis (SQLite dialect)
-- Table: customers  (loaded from data/processed/churn_clean.csv)
-- ============================================================

-- name: 1. Churn KPI snapshot
SELECT
    COUNT(*)                                              AS customers,
    SUM(Churn)                                            AS churned,
    ROUND(AVG(Churn) * 100, 1)                            AS churn_rate_pct,
    ROUND(SUM(MonthlyCharges), 0)                         AS total_mrr,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges END), 0)   AS lost_mrr,
    ROUND(100.0 * SUM(CASE WHEN Churn = 1 THEN MonthlyCharges END) / SUM(MonthlyCharges), 1)
                                                          AS lost_mrr_pct
FROM customers;

-- name: 2. Churn by contract type
SELECT
    Contract,
    COUNT(*)                                   AS customers,
    ROUND(AVG(Churn) * 100, 1)                 AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)              AS avg_monthly_charge,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges * 12 END), 0) AS annual_revenue_lost
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- name: 3. Tenure cohorts - when do customers leave?
SELECT
    tenure_group,
    COUNT(*)                                   AS customers,
    ROUND(AVG(Churn) * 100, 1)                 AS churn_rate_pct,
    ROUND(100.0 * SUM(Churn) / (SELECT SUM(Churn) FROM customers), 1) AS share_of_all_churn_pct
FROM customers
GROUP BY tenure_group
ORDER BY MIN(tenure);

-- name: 4. Payment method & paperless billing
SELECT
    PaymentMethod,
    PaperlessBilling,
    COUNT(*)                                   AS customers,
    ROUND(AVG(Churn) * 100, 1)                 AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod, PaperlessBilling
HAVING COUNT(*) > 200
ORDER BY churn_rate_pct DESC;

-- name: 5. Internet service x tech support (do support add-ons keep customers?)
SELECT
    InternetService,
    TechSupport,
    COUNT(*)                                   AS customers,
    ROUND(AVG(Churn) * 100, 1)                 AS churn_rate_pct
FROM customers
WHERE InternetService <> 'No'
GROUP BY InternetService, TechSupport
ORDER BY InternetService, TechSupport;

-- name: 6. More add-on services = stickier customers (internet customers only)
-- (customers without internet can't buy add-ons, so they'd distort the 0 bucket)
SELECT
    num_addon_services,
    COUNT(*)                                   AS customers,
    ROUND(AVG(Churn) * 100, 1)                 AS churn_rate_pct,
    ROUND(AVG(MonthlyCharges), 2)              AS avg_monthly_charge
FROM customers
WHERE InternetService <> 'No'
GROUP BY num_addon_services
ORDER BY num_addon_services;

-- name: 7. Highest-risk segment: combine the 3 strongest drivers (CTE)
WITH segments AS (
    SELECT
        CASE WHEN Contract = 'Month-to-month' THEN 'M2M' ELSE 'Term' END   AS contract_type,
        CASE WHEN tenure <= 12 THEN 'New (<=12m)' ELSE 'Established' END   AS tenure_type,
        CASE WHEN InternetService = 'Fiber optic' THEN 'Fiber' ELSE 'Other' END AS internet,
        Churn, MonthlyCharges
    FROM customers
)
SELECT
    contract_type, tenure_type, internet,
    COUNT(*)                                   AS customers,
    ROUND(AVG(Churn) * 100, 1)                 AS churn_rate_pct,
    ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges * 12 END), 0) AS annual_revenue_lost,
    RANK() OVER (ORDER BY AVG(Churn) DESC)     AS risk_rank
FROM segments
GROUP BY contract_type, tenure_type, internet
HAVING COUNT(*) >= 30
ORDER BY risk_rank
LIMIT 8;

-- name: 8. Demographics: seniors and customers without family
SELECT
    SeniorCitizen,
    has_family,
    COUNT(*)                                   AS customers,
    ROUND(AVG(Churn) * 100, 1)                 AS churn_rate_pct
FROM customers
GROUP BY SeniorCitizen, has_family
ORDER BY churn_rate_pct DESC;
