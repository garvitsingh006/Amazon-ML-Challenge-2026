import time
import gc
import pandas as pd
from pathlib import Path

from normalize import normalize_name, normalize_country
from blocking import add_blocking_keys, build_block_index


DATA = Path("data/dataset")


def load_source(path):
    start = time.time()

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "country"
        ]
    )

    print(f"Loaded {path.name}: {len(df):,} rows")
    print(f"Load time: {time.time() - start:.2f}s")

    start = time.time()

    df["business_name_norm"] = (
        df["business_name"].map(normalize_name)
    )

    df["country_norm"] = (
        df["country"].map(normalize_country)
    )

    print(
        f"Normalization time: "
        f"{time.time() - start:.2f}s"
    )

    return df


def memory_mb(df):
    return (
        df.memory_usage(deep=True).sum()
        / 1024**2
    )


# =========================================================
# S2
# =========================================================

print("=" * 60)
print("LOADING S2")
print("=" * 60)

start = time.time()

s2 = load_source(
    DATA / "train/train_source2.tsv"
)

print(
    f"S2 dataframe memory: "
    f"{memory_mb(s2):.2f} MB"
)

print()


# =========================================================
# S3
# =========================================================

print("=" * 60)
print("LOADING S3")
print("=" * 60)

start = time.time()

s3 = load_source(
    DATA / "train/train_source3.tsv"
)

print(
    f"S3 dataframe memory: "
    f"{memory_mb(s3):.2f} MB"
)

print()


# =========================================================
# Add blocking keys
# =========================================================

print("=" * 60)
print("ADDING BLOCKING KEYS")
print("=" * 60)

start = time.time()

s2 = add_blocking_keys(s2)
s3 = add_blocking_keys(s3)

print(
    f"Blocking-key time: "
    f"{time.time() - start:.2f}s"
)

print(
    f"S2 dataframe memory after keys: "
    f"{memory_mb(s2):.2f} MB"
)

print(
    f"S3 dataframe memory after keys: "
    f"{memory_mb(s3):.2f} MB"
)

print()


# =========================================================
# Build indexes
# =========================================================

print("=" * 60)
print("BUILDING INDEXES")
print("=" * 60)

start = time.time()

s2_index = build_block_index(s2)

print(
    f"S2 index built in "
    f"{time.time() - start:.2f}s"
)

print(
    f"S2 index memory: "
    f"{memory_mb(s2_index):.2f} MB"
)

print()

start = time.time()

s3_index = build_block_index(s3)

print(
    f"S3 index built in "
    f"{time.time() - start:.2f}s"
)

print(
    f"S3 index memory: "
    f"{memory_mb(s3_index):.2f} MB"
)

print()


# =========================================================
# Final summary
# =========================================================

print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"S2 rows: {len(s2):,}")
print(f"S3 rows: {len(s3):,}")

print(
    f"S2 index rows: {len(s2_index):,}"
)

print(
    f"S3 index rows: {len(s3_index):,}"
)

print(
    f"S2 index memory: "
    f"{memory_mb(s2_index):.2f} MB"
)

print(
    f"S3 index memory: "
    f"{memory_mb(s3_index):.2f} MB"
)

print("\nFull-dataset blocking index build complete.")