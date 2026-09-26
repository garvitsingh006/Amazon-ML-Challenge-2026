import re
import pandas as pd
from unidecode import unidecode


# ---------------------------------------------------------
# Generic text normalization
# ---------------------------------------------------------

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Transliterate accents / scripts
    text = unidecode(text)

    # Punctuation -> spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Legal entity suffix normalization
# ---------------------------------------------------------

LEGAL_SUFFIXES = {
    r"\bcorporation\b": "corp",
    r"\bcorp\b": "corp",

    r"\bincorporated\b": "inc",
    r"\binc\b": "inc",

    r"\blimited\b": "ltd",
    r"\bltd\b": "ltd",

    r"\bprivate limited\b": "pvt ltd",
    r"\bprivate ltd\b": "pvt ltd",
    r"\bpvt limited\b": "pvt ltd",
    r"\bpvt ltd\b": "pvt ltd",

    r"\bprivate\b": "pvt",

    r"\blimited liability company\b": "llc",
    r"\bllc\b": "llc",

    # France / Europe
    r"\bsociete anonyme\b": "sa",
    r"\bsa\b": "sa",

    r"\bsociete par actions simplifiee\b": "sas",
    r"\bsas\b": "sas",

    r"\bsociete a responsabilite limitee\b": "sarl",
    r"\bsarl\b": "sarl",

    r"\bentreprise unipersonnelle a responsabilite limitee\b": "eurl",
    r"\beurl\b": "eurl",
}


def normalize_legal_suffixes(text):
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)

    return text


# ---------------------------------------------------------
# Name normalization
# ---------------------------------------------------------

def normalize_name(name):
    text = normalize_text(name)
    text = normalize_legal_suffixes(text)

    # Collapse spaces again after replacements
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

ADDRESS_ABBREVIATIONS = {
    r"\broad\b": "rd",
    r"\brd\b": "rd",

    r"\bstreet\b": "st",
    r"\bst\b": "st",

    r"\bavenue\b": "ave",
    r"\bave\b": "ave",

    r"\bboulevard\b": "blvd",
    r"\bbd\b": "blvd",

    r"\bdrive\b": "dr",
    r"\bdr\b": "dr",

    r"\blane\b": "ln",
    r"\bln\b": "ln",

    r"\bhighway\b": "hwy",
    r"\bhwy\b": "hwy",

    r"\bapartment\b": "apt",
    r"\bapt\b": "apt",

    # French
    r"\brue\b": "rue",
    r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd",
}


def normalize_address(address):
    text = normalize_text(address)

    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Country normalization
# ---------------------------------------------------------

def normalize_country(country):
    return normalize_text(country)


# ---------------------------------------------------------
# Structured tokens
# ---------------------------------------------------------

def extract_numbers(text):
    """
    Extract numeric tokens such as:
    1795
    570/13
    56001
    """

    if pd.isna(text):
        return []

    text = str(text)

    return re.findall(r"\b\d+(?:/\d+)?\b", text)


def extract_postal_codes(text):
    """
    Generic numeric postal/PIN candidates.
    We deliberately avoid country-specific rules.
    """

    if pd.isna(text):
        return []

    text = str(text)

    return re.findall(r"\b\d{4,6}\b", text)

def normalize_dataframe(df):
    df = df.copy()

    df["business_name_norm"] = df["business_name"].map(normalize_name)
    df["business_address_norm"] = df["business_address"].map(normalize_address)
    df["country_norm"] = df["country"].map(normalize_country)

    return df


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    import pandas as pd

    sample = pd.DataFrame({
        "business_name": [
            "B+ Retail Inc",
            "Consulting Nyasa Nursing Private Limited",
            "राम मार्केटिंग प्राइवेट लिमिटेड",
        ],
        "business_address": [
            "1795 Westchester Drive, High Point, NC",
            "2505 Tower 1, Oakwood, Mumbai",
            None,
        ],
        "country": [
            "US",
            "India",
            "India",
        ],
    })

    result = normalize_dataframe(sample)

    print(result.to_string(index=False))