# Customer Churn Platform

An end-to-end customer churn prediction platform: a PySpark data pipeline feeding a machine learning classifier and an NLP layer for unstructured customer feedback, surfaced through an interactive dashboard. Containerized with Docker Compose and deployable both locally and on Google Cloud Platform.

**Status:** actively in development (started July 2026). This README tracks real progress, see the roadmap below for what's done vs. planned.

## Why this project

Churn prediction sits at the intersection of the things I want to work with professionally: large-scale data processing, applied ML, and infrastructure that runs the same way locally as it does in the cloud. This project is built to demonstrate that end-to-end, not just the modeling part.

## Architecture

```
                      ┌─────────────────┐
  Raw data (CSV) ───▶ │  PySpark ETL     │ ───▶ Feature store (Postgres)
  structured +        │  ingestion +     │
  free-text sources   │  transformation  │
                      └────────┬─────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                            ▼
        ┌─────────────────┐         ┌──────────────────┐
        │  ML classifier   │         │   NLP pipeline    │
        │  (churn risk)    │         │  (text signals)   │
        └────────┬─────────┘         └─────────┬─────────┘
                  └─────────────┬───────────────┘
                                ▼
                      ┌──────────────────┐
                      │    Dashboard      │
                      │ (churn risk view) │
                      └──────────────────┘
```

A full architecture diagram (with the Docker Compose / Traefik / GCP deployment topology) lives in [`docs/architecture.md`](docs/architecture.md) and is being filled in as the infrastructure lands.

## Tech stack

| Layer | Tools |
|---|---|
| Data processing | PySpark |
| Storage | PostgreSQL |
| ML | scikit-learn (baseline), evaluated with proper train/validation/test splits |
| NLP | spaCy / scikit-learn text pipelines (TBD as this layer is built) |
| Dashboard | Streamlit |
| Infrastructure | Docker Compose, Traefik (reverse proxy), deployed on GCP |
| Language | Python 3.13 |

## Data sources

- **[TheLook eCommerce](https://www.kaggle.com/datasets/mustafakeser4/looker-ecommerce-bigquery-dataset)**: synthetic but realistic US e-commerce data (users, orders, order items, products, events) spanning all US states, used as the structured backbone for churn labeling and feature engineering.
- **[CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)**: real US consumer complaints against financial services companies, including free-text complaint narratives, used as the source for the NLP layer.

Full dataset documentation, schema notes, and download instructions are in [`data/README.md`](data/README.md).

## Roadmap

- [x] Repository structure
- [ ] Dataset ingestion and exploratory analysis
- [ ] PySpark ETL pipeline (cleaning, joins, feature engineering)
- [ ] Churn label definition
- [ ] Baseline ML classifier + evaluation
- [ ] NLP pipeline on complaint narratives
- [ ] Streamlit dashboard
- [ ] Docker Compose (multi-container, Traefik reverse proxy, Postgres)
- [ ] GCP deployment

## Getting started

Setup instructions will be added once the pipeline has a runnable first version. In the meantime, see [`data/README.md`](data/README.md) for how to obtain the raw datasets.

## License

MIT, see [LICENSE](LICENSE).
