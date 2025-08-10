import argparse, yaml
import pandas as pd
from pathlib import Path
from escalate_nlp_agent.extract.rule_based import run_rule_based
from escalate_nlp_agent.extract.spacy_ner import run_spacy_ner

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_config", required=True)
    ap.add_argument("--extract_config", required=True)
    args = ap.parse_args()

    with open(args.dataset_config) as f:
        ds_cfg = yaml.safe_load(f)
    with open(args.extract_config) as f:
        ex_cfg = yaml.safe_load(f)

    proc_dir = Path(ds_cfg["outputs"]["processed_dir"])
    df = pd.read_parquet(proc_dir / "train.parquet")

    # Determine extractor by config contents
    if "patterns" in ex_cfg:  # rule-based
        out = run_rule_based(df, ex_cfg)
    elif "model" in ex_cfg:   # spaCy NER
        out = run_spacy_ner(df, ex_cfg)
    else:
        raise ValueError("Unknown extractor type: config must have either 'patterns' or 'model'.")

    out_file = ex_cfg["outputs"]["file"].replace("{dataset}", ds_cfg["id"])
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False)
    print(f"Saved extractions to {out_file}")
