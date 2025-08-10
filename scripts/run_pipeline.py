import argparse, yaml
from pathlib import Path
from escalate_nlp_agent.pipeline import run_preprocess_and_eda

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ds_cfg_path = Path(cfg["dataset_config"])
    with open(ds_cfg_path, "r") as f:
        ds_cfg = yaml.safe_load(f)

    run_preprocess_and_eda(ds_cfg)
