# Customer Churn Platform

End-to-end churn prediction on e-commerce data: PySpark pipeline for the data processing, an ML classifier to predict churn risk, an NLP layer on top of real consumer complaint texts, and a dashboard to show the results. Everything runs in Docker, locally and on Google Cloud.

Started July 2026, actively being built. The roadmap below shows where the project stands.

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

The Docker and deployment topology (Traefik, Postgres, GCP) will be documented in `docs/architecture.md` once that part is built.

## Stack

| Layer | Tools |
|---|---|
| Data processing | PySpark 4 (in Docker, jupyter/pyspark-notebook image) |
| Storage | PostgreSQL, parquet for the intermediate datasets |
| ML | scikit-learn |
| NLP | spaCy / scikit-learn text pipelines |
| Dashboard | Streamlit |
| Infra | Docker Compose, Traefik, GCP |
| Language | Python 3.13 |

## Data

Two datasets, both US based so state level comparison is possible:

- [TheLook eCommerce](https://www.kaggle.com/datasets/mustafakeser4/looker-ecommerce-bigquery-dataset): synthetic e-commerce data (100k users, 2.4M web events, orders from 2019 to 2024). Used for the churn label and the customer features. The project only uses the 22.5k US customers.
- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/): 2.36M real consumer complaints against financial companies, all with free text narratives. This is the input for the NLP part.

Download instructions and schema notes: [`data/README.md`](data/README.md).

## How the churn label works

TheLook has no churn column, so the label is derived from order behavior. A cutoff date is placed 180 days before the last order in the dataset. Customers with at least one order before the cutoff are in scope, and whoever doesn't order again after the cutoff counts as churned. All features are computed from data before the cutoff only, so the model can't peek at the answer (no label leakage).

The 180 day window wasn't a random pick: the median time between two orders from the same customer is 155 days, so someone who stays quiet for 6 months is past their normal rhythm. The full analysis is in [`notebooks/02_churn_window.ipynb`](notebooks/02_churn_window.ipynb). About 85% of customers churn under this definition, which sounds high but is normal for e-commerce, most shoppers buy once and never come back. The class imbalance gets handled in the ML step.

## Roadmap

- [x] Repo structure
- [x] Ingestion with explicit Spark schemas (TheLook + CFPB)
- [x] EDA notebooks
- [x] Churn label with temporal cutoff
- [x] Feature engineering (recency, frequency, spend, browsing behavior) into one training table
- [ ] ML classifier + evaluation
- [ ] NLP pipeline on the complaint narratives
- [ ] Streamlit dashboard
- [ ] Docker Compose stack with Traefik and Postgres
- [ ] GCP deployment

## Running it

1. Download the raw data into `data/raw/`, see [`data/README.md`](data/README.md).
2. Start the Spark container (this is also the Jupyter server for the notebooks):

```powershell
docker network create qs-data
docker run -d --name qs-spark --network qs-data -p 8888:8888 `
  -v ${PWD}:/home/jovyan/work quay.io/jupyter/pyspark-notebook:latest
```

3. Build the training dataset:

```powershell
docker exec -e PYTHONPATH=/usr/local/spark/python:/usr/local/spark/python/lib/py4j-0.10.9.9-src.zip:/home/jovyan/work `
  qs-spark bash -c "cd /home/jovyan/work && python -m src.processing.build_dataset"
```

Output lands in `data/processed/churn_dataset` as parquet. For the notebooks: `docker logs qs-spark` gives the Jupyter URL with the token, or connect PyCharm to it as an external Jupyter server.

## License

MIT, see [LICENSE](LICENSE).
