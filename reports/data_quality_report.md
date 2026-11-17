# Data Quality Report

| Check / step | Result |
|---|---:|
| Raw rows x columns | 7,043 x 21 |
| Duplicate customerID | 0 |
| Blank TotalCharges (stored as text) | 11 |
| -> filled with 0 (tenure = 0, not yet billed) | 11 |
| 'No internet/phone service' values collapsed to 'No' | 9,838 |
| Customers where TotalCharges deviates >25% from tenure x monthly | 27 |
| Engineered features added | 7 |
| Clean rows x columns | 7,043 x 28 |
