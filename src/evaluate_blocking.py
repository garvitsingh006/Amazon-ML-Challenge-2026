import time
import pandas as pd
from pathlib import Path

from normalize import normalize_name, normalize_country
from blocking import add_blocking_keys


DATA = Path("data/dataset")


# =========================================================
# Load + normalize
# =========================================================

def load_source(path):

    print(f"\nLoading {path.name}...")

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ]
    )

    df["business_name_norm"] = (
        df["business_name"].map(normalize_name)
    )

    df["country_norm"] = (
        df["country"].map(normalize_country)
    )

    df = add_blocking_keys(df)

    print(f"Rows: {len(df):,}")

    return df


# =========================================================
# Load datasets
# =========================================================

start = time.time()

s1 = load_source(
    DATA / "train/train_source1.tsv"
)

s2 = load_source(
    DATA / "train/train_source2.tsv"
)

s3 = load_source(
    DATA / "train/train_source3.tsv"
)

print(
    f"\nTotal loading time: "
    f"{time.time() - start:.2f}s"
)


# =========================================================
# Ground truth
# =========================================================

ground_truth = pd.read_csv(
    DATA / "train/train_ground_truth.tsv",
    sep="\t"
)

ground_truth = ground_truth.set_index(
    "source1_entity_id"
)


# =========================================================
# Build indexes
# =========================================================

KEYS = [
    "block_prefix",
    "block_first_token",
    "block_last_token",
    "block_exact_name",
]


indexes = {}

print("\nBuilding indexes...")

for key in KEYS:

    print(f"Building: {key}")

    s2_index = (
        s2.groupby(key)["entity_id"]
        .agg(list)
        .to_dict()
    )

    s3_index = (
        s3.groupby(key)["entity_id"]
        .agg(list)
        .to_dict()
    )

    indexes[key] = (
        s2_index,
        s3_index
    )


print("Indexes ready.")


# =========================================================
# Evaluate each blocker independently
# =========================================================

print("\n")
print("=" * 70)
print("INDIVIDUAL BLOCKING RESULTS")
print("=" * 70)


for key in KEYS:

    print(f"\nTesting: {key}")

    s2_index, s3_index = indexes[key]

    total_true = 0
    found_true = 0

    total_candidates = 0
    rows_with_candidates = 0

    for s1_id, row in s1.set_index(
        "entity_id"
    ).iterrows():

        if s1_id not in ground_truth.index:
            continue

        block = row[key]

        candidates = set(
            s2_index.get(block, [])
        )

        candidates.update(
            s3_index.get(block, [])
        )

        total_candidates += len(candidates)

        if candidates:
            rows_with_candidates += 1

        matched = ground_truth.loc[
            s1_id,
            "matched_entity_ids"
        ]

        if pd.isna(matched):
            continue

        true_ids = set(
            str(matched).split(",")
        )

        total_true += len(true_ids)

        found_true += len(
            true_ids.intersection(candidates)
        )

    recall = (
        found_true / total_true
        if total_true
        else 0
    )

    avg_candidates = (
        total_candidates / len(s1)
    )

    coverage = (
        rows_with_candidates / len(s1)
    )

    print(
        f"Recall:             {recall:.4%}"
    )

    print(
        f"Avg candidates/S1:  {avg_candidates:.2f}"
    )

    print(
        f"Coverage:           {coverage:.4%}"
    )


print("\nDone.")