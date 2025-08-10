import re
import pandas as pd

def run_rule_based(df, cfg):
    results = []
    patterns = cfg["patterns"]
    for _, row in df.iterrows():
        extracted = {}
        for name, pat in patterns.items():
            matches = re.findall(pat, row["text"])
            extracted[name] = list(set(matches))
        results.append({"id": row["id"], "extracted": extracted})
    return pd.DataFrame(results)
