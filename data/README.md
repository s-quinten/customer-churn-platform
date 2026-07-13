# Data

This project uses two datasets, kept separate because they serve different layers of the pipeline. Neither is committed to git. Download them locally into `data/raw/` following the instructions below.

## 1. TheLook eCommerce (structured / churn backbone)

Synthetic e-commerce dataset (users, orders, order items, products, distribution centers, web events) covering customers across all US states. There is no pre-made churn label: churn is derived from order recency during the ETL step, which is a more defensible modeling choice than consuming a pre-labeled column.

**Source:** [Kaggle mirror of the BigQuery `thelook_ecommerce` public dataset](https://www.kaggle.com/datasets/mustafakeser4/looker-ecommerce-bigquery-dataset)

**Download:**
1. Download the zip from the Kaggle link above (requires a free Kaggle account).
2. Extract into `data/raw/thelook_ecommerce/`.
3. Expected files: `users.csv`, `orders.csv`, `order_items.csv`, `products.csv`, `distribution_centers.csv`, `events.csv`, `inventory_items.csv`.

## 2. CFPB Consumer Complaint Database (NLP layer)

Real US consumer complaints against banks, credit card companies, and other financial services, including free-text complaint narratives. Used to build the NLP layer (text classification / sentiment on complaint text) and, potentially, as a complementary churn-risk signal (e.g. rising complaint frequency per customer segment).

**Source:** [consumerfinance.gov/data-research/consumer-complaints](https://www.consumerfinance.gov/data-research/consumer-complaints/) (no account required).

**Download:**
1. On the "Download complaint data" page, filter by date range (recommend last 2-3 years to keep volume manageable for local development) and leave State unfiltered.
2. Export as CSV.
3. Save into `data/raw/cfpb_complaints/complaints.csv`.

## Notes

- `data/raw/` and `data/processed/` are gitignored. Everyone working on this repo downloads the raw data themselves rather than committing it.
- Both datasets are US-based with a `state` field, which is what enables the state-level comparison angle of this project.
