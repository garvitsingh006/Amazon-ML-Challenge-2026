import pandas as pd
from pathlib import Path

DATA = Path("data/dataset")

# -----------------------------
# Load datasets
# -----------------------------

train_s1 = pd.read_csv(
    DATA / "train/train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    DATA / "train/train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    DATA / "train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    DATA / "train/train_ground_truth.tsv",
    sep="\t"
)

test_s1 = pd.read_csv(
    DATA / "test/test_source1.tsv",
    sep="\t"
)

test_s2 = pd.read_csv(
    DATA / "test/test_source2.tsv",
    sep="\t"
)

test_s3 = pd.read_csv(
    DATA / "test/test_source3.tsv",
    sep="\t"
)


# -----------------------------
# Basic dataset information
# -----------------------------

datasets = {
    "Train S1": train_s1,
    "Train S2": train_s2,
    "Train S3": train_s3,
    "Ground Truth": ground_truth,
    "Test S1": test_s1,
    "Test S2": test_s2,
    "Test S3": test_s3,
}

for name, df in datasets.items():
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nNull values:")
    print(df.isna().sum())


# -----------------------------
# Country distribution
# -----------------------------

print("\n\nCOUNTRY DISTRIBUTION")
print("=" * 60)

print("\nTrain S1:")
print(train_s1["country"].value_counts(dropna=False))

print("\nTest S1:")
print(test_s1["country"].value_counts(dropna=False))


# -----------------------------
# Sample records
# -----------------------------

print("\n\nTRAIN S1 SAMPLE")
print("=" * 60)
print(train_s1.head(10).to_string(index=False))

print("\n\nTRAIN S2 SAMPLE")
print("=" * 60)
print(train_s2.head(10).to_string(index=False))

print("\n\nGROUND TRUTH SAMPLE")
print("=" * 60)
print(ground_truth.head(10).to_string(index=False))