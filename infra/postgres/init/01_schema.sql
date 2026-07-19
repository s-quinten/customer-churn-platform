-- Serving schema for the churn platform.
-- Two star schemas sharing dim_state and dim_date (fact constellation):
-- churn scoring per customer per scoring date, and complaint counts per
-- state, product and month.
-- Runs automatically on first container startup via docker-entrypoint-initdb.d,
-- dimensions before facts because the foreign keys need the targets to exist.

CREATE TABLE dim_state (
    state_sk    SERIAL PRIMARY KEY,
    state_code  CHAR(2) NOT NULL UNIQUE,
    state_name  TEXT NOT NULL
);

CREATE TABLE dim_date (
    date_sk     INT PRIMARY KEY,
    full_date   DATE NOT NULL UNIQUE,
    year        INT NOT NULL,
    month       INT NOT NULL,
    month_name  TEXT NOT NULL
);

CREATE TABLE dim_customer (
    user_sk         SERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL UNIQUE,
    age             INT,
    age_group       TEXT,
    gender          TEXT,
    traffic_source  TEXT
);

CREATE TABLE dim_product (
    product_sk    SERIAL PRIMARY KEY,
    product_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE fact_churn_prediction (
    user_sk            INT NOT NULL REFERENCES dim_customer (user_sk),
    state_sk           INT NOT NULL REFERENCES dim_state (state_sk),
    date_sk            INT NOT NULL REFERENCES dim_date (date_sk),
    churn_probability  DOUBLE PRECISION NOT NULL,
    churned            INT NOT NULL,
    PRIMARY KEY (user_sk, date_sk)
);

CREATE TABLE fact_complaints (
    state_sk         INT NOT NULL REFERENCES dim_state (state_sk),
    product_sk       INT NOT NULL REFERENCES dim_product (product_sk),
    date_sk          INT NOT NULL REFERENCES dim_date (date_sk),
    complaint_count  INT NOT NULL,
    PRIMARY KEY (state_sk, product_sk, date_sk)
);
