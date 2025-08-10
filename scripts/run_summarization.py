import argparse, yaml
import pandas as pd
from pathlib import Path
from escalate_nlp_agent.summarize.textrank import run_textrank
from escalate_nlp_agent.summarize.bart_pegasus_t5 import run_bart

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_config", required=True)
    ap.add_argument("--summarize_config", required=True)
    args = ap.parse_args()

    with open(args.dataset_config) as f:
        ds_cfg = yaml.safe_load(f)
    with open(args.summarize_config) as f:
        sum_cfg = yaml.safe_load(f)

    proc_dir = Path(ds_cfg["outputs"]["processed_dir"])
    df = pd.read_parquet(proc_dir / "train.parquet")

    if sum_cfg["type"] == "extractive":
        out = run_textrank(df, sum_cfg)
    else:
        out = run_bart(df, sum_cfg)

    out_file = sum_cfg["outputs"]["file"].replace("{dataset}", ds_cfg["id"])
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_file, index=False)
    print(f"Saved summaries to {out_file}")
