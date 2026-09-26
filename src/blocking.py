import pandas as pd
import re


# =========================================================
# Basic helpers
# =========================================================

def clean_token(token):
    token = str(token).lower()
    token = re.sub(r"[^a-z0-9]", "", token)
    return token


def get_name_tokens(name):
    if pd.isna(name):
        return []

    tokens = str(name).lower().split()

    tokens = [
        clean_token(token)
        for token in tokens
    ]

    # Ignore very short tokens
    tokens = [
        token for token in tokens
        if len(token) >= 3
    ]

    return tokens


def name_prefix(name, length=6):
    if pd.isna(name):
        return ""

    name = re.sub(
        r"[^a-z0-9]",
        "",
        str(name).lower()
    )

    return name[:length]


# =========================================================
# Blocking key generation
# =========================================================

def add_blocking_keys(df):

    df = df.copy()

    # -----------------------------------------------------
    # Existing prefix key
    # -----------------------------------------------------

    df["name_prefix"] = (
        df["business_name_norm"]
        .map(name_prefix)
    )

    df["block_prefix"] = (
        df["country_norm"].astype(str)
        + "_"
        + df["name_prefix"]
    )

    # -----------------------------------------------------
    # First token
    # -----------------------------------------------------

    df["name_first_token"] = (
        df["business_name_norm"]
        .map(
            lambda x:
            get_name_tokens(x)[0]
            if get_name_tokens(x)
            else ""
        )
    )

    df["block_first_token"] = (
        df["country_norm"].astype(str)
        + "_"
        + df["name_first_token"]
    )

    # -----------------------------------------------------
    # Last token
    # -----------------------------------------------------

    df["name_last_token"] = (
        df["business_name_norm"]
        .map(
            lambda x:
            get_name_tokens(x)[-1]
            if get_name_tokens(x)
            else ""
        )
    )

    df["block_last_token"] = (
        df["country_norm"].astype(str)
        + "_"
        + df["name_last_token"]
    )

    # -----------------------------------------------------
    # Exact normalized name
    # -----------------------------------------------------

    df["block_exact_name"] = (
        df["country_norm"].astype(str)
        + "_"
        + df["business_name_norm"].fillna("")
    )

    return df


# =========================================================
# Build index
# =========================================================

def build_block_index(df, key_column):

    index = df[
        [key_column, "entity_id"]
    ].copy()

    index = index[
        index[key_column].notna()
        & (index[key_column] != "")
    ]

    index = index.sort_values(key_column)

    return index