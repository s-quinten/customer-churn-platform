"""Generate infra/postgres/init/02_seed.sql from the parquet gold outputs.

This is a build-time tool. It reads the finished data (churn scores,
customer features, complaint stats) and writes a plain SQL file of
INSERTs that lives next to the schema. On any machine, the postgres
container runs 01_schema.sql then 02_seed.sql on first startup, so the
database builds and fills itself with no manual step.

The interesting part is dim_state. TheLook stores states as full names
("California"), CFPB stores two-letter codes ("CA"). dim_state holds
both, so the churn fact joins on the name and the complaint fact joins
on the code, and both land on the same state_sk. That reconciliation is
what makes dim_state a conformed dimension.

Run on the host (reads parquet, writes text, no Spark needed):
    python -m src.serving.build_seed
"""
import calendar

import pandas as pd

OUT_PATH = "infra/postgres/init/02_seed.sql"

# The scoring date is the churn label cutoff (max order date minus 180
# days), the moment "as of which" every prediction was made.
SCORING_DATE = pd.Timestamp("2023-07-21")

# Canonical US state dimension: code to name for all 50 states plus DC.
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia",
}


def age_group(age) -> str:
    if pd.isna(age):
        return "unknown"
    if age < 25:
        return "under 25"
    if age < 35:
        return "25-34"
    if age < 50:
        return "35-49"
    return "50 plus"


def sql_str(value) -> str:
    """Quote a value for SQL, doubling single quotes, NULL for missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def date_sk(ts: pd.Timestamp) -> int:
    return ts.year * 10000 + ts.month * 100 + ts.day


def insert_block(f, table, columns, rows) -> None:
    """Write batched multi-row INSERTs (1000 rows each)."""
    if not rows:
        return
    cols = ", ".join(columns)
    for start in range(0, len(rows), 1000):
        chunk = rows[start:start + 1000]
        f.write(f"INSERT INTO {table} ({cols}) VALUES\n")
        f.write(",\n".join("    (" + ", ".join(r) + ")" for r in chunk))
        f.write(";\n")
    f.write("\n")


def main() -> None:
    scores = pd.read_parquet("data/processed/churn_scores.parquet")
    features = pd.read_parquet("data/processed/churn_dataset")
    complaints = pd.read_parquet("data/processed/complaint_stats")

    # ---- dim_state: the canonical mapping, one surrogate key per state
    name_to_sk = {}
    code_to_sk = {}
    state_rows = []
    for i, (code, name) in enumerate(STATES.items(), start=1):
        name_to_sk[name] = i
        code_to_sk[code] = i
        state_rows.append([str(i), sql_str(code), sql_str(name)])

    # ---- dim_product: distinct complaint products
    products = sorted(complaints["product"].dropna().unique())
    product_to_sk = {p: i for i, p in enumerate(products, start=1)}
    product_rows = [[str(sk), sql_str(p)] for p, sk in product_to_sk.items()]

    # ---- dim_date: the scoring date plus every complaint month
    dates = {SCORING_DATE.normalize()}
    complaints = complaints.copy()
    complaints["month"] = pd.to_datetime(complaints["month"])
    dates.update(complaints["month"].dt.normalize().unique())
    date_rows = []
    for d in sorted(dates):
        d = pd.Timestamp(d)
        date_rows.append([
            str(date_sk(d)),
            sql_str(d.date().isoformat()),
            str(d.year),
            str(d.month),
            sql_str(calendar.month_name[d.month]),
        ])

    # ---- dim_customer: one row per scored customer, from the features
    feat = features[["user_id", "age", "gender", "traffic_source"]].copy()
    feat = feat.drop_duplicates("user_id").reset_index(drop=True)
    feat["user_sk"] = feat.index + 1
    user_to_sk = dict(zip(feat["user_id"], feat["user_sk"]))
    customer_rows = []
    for _, r in feat.iterrows():
        customer_rows.append([
            str(r["user_sk"]),
            str(int(r["user_id"])),
            "NULL" if pd.isna(r["age"]) else str(int(r["age"])),
            sql_str(age_group(r["age"])),
            sql_str(r["gender"]),
            sql_str(r["traffic_source"]),
        ])

    # ---- fact_churn_prediction: resolve keys, drop rows that don't map
    sc = scores.copy()
    sc["user_sk"] = sc["user_id"].map(user_to_sk)
    sc["state_sk"] = sc["state"].map(name_to_sk)
    before = len(sc)
    sc = sc.dropna(subset=["user_sk", "state_sk"])
    churn_rows = []
    for _, r in sc.iterrows():
        churn_rows.append([
            str(int(r["user_sk"])),
            str(int(r["state_sk"])),
            str(date_sk(SCORING_DATE)),
            str(float(r["churn_probability"])),
            str(int(r["churned"])),
        ])
    churn_dropped = before - len(sc)

    # ---- fact_complaints: resolve keys, drop rows that don't map
    cp = complaints.copy()
    cp["state_sk"] = cp["state"].map(code_to_sk)
    cp["product_sk"] = cp["product"].map(product_to_sk)
    cp["date_sk"] = cp["month"].map(lambda m: date_sk(pd.Timestamp(m)))
    before = len(cp)
    cp = cp.dropna(subset=["state_sk", "product_sk"])
    complaint_rows = []
    for _, r in cp.iterrows():
        complaint_rows.append([
            str(int(r["state_sk"])),
            str(int(r["product_sk"])),
            str(int(r["date_sk"])),
            str(int(r["complaint_count"])),
        ])
    complaints_dropped = before - len(cp)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("-- Generated by src/serving/build_seed.py, do not edit by hand.\n")
        f.write("-- Runs after 01_schema.sql on first container startup.\n\n")
        insert_block(f, "dim_state",
                     ["state_sk", "state_code", "state_name"], state_rows)
        insert_block(f, "dim_product",
                     ["product_sk", "product_name"], product_rows)
        insert_block(f, "dim_date",
                     ["date_sk", "full_date", "year", "month", "month_name"],
                     date_rows)
        insert_block(f, "dim_customer",
                     ["user_sk", "user_id", "age", "age_group", "gender",
                      "traffic_source"], customer_rows)
        insert_block(f, "fact_churn_prediction",
                     ["user_sk", "state_sk", "date_sk", "churn_probability",
                      "churned"], churn_rows)
        insert_block(f, "fact_complaints",
                     ["state_sk", "product_sk", "date_sk", "complaint_count"],
                     complaint_rows)

    print(f"wrote {OUT_PATH}")
    print(f"  dim_state:      {len(state_rows)}")
    print(f"  dim_product:    {len(product_rows)}")
    print(f"  dim_date:       {len(date_rows)}")
    print(f"  dim_customer:   {len(customer_rows)}")
    print(f"  fact_churn:     {len(churn_rows)} (dropped {churn_dropped} unmapped)")
    print(f"  fact_complaints:{len(complaint_rows)} (dropped {complaints_dropped} unmapped)")


if __name__ == "__main__":
    main()
