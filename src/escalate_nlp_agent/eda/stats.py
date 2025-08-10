import pandas as pd
from collections import Counter

def doc_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame of document lengths in tokens."""
    return pd.DataFrame({"length": df["tokens"].map(len)})

def top_terms(df: pd.DataFrame, n=30) -> pd.DataFrame:
    """Get top-N frequent tokens."""
    bag = Counter([t for row in df["tokens"] for t in row])
    return pd.DataFrame(bag.most_common(n), columns=["term", "count"])
