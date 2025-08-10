import argparse, yaml, json
from pathlib import Path
import pandas as pd

from escalate_nlp_agent.agent.planner import make_plan
from escalate_nlp_agent.agent.retriever import TfidfRetriever
from escalate_nlp_agent.agent.toolchain import run_extractors, run_summarizer, synthesize_answer
from escalate_nlp_agent.agent.memory import remember

def load_yaml(p): 
    with open(p, "r") as f: 
        return yaml.safe_load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_config", required=True)
    ap.add_argument("--agent_config", required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    ds_cfg = load_yaml(args.dataset_config)
    ag_cfg = load_yaml(args.agent_config)

    # Load processed docs
    proc_dir = Path(ds_cfg["outputs"]["processed_dir"])
    df = pd.read_parquet(proc_dir / "train.parquet")  # demo on train split

    # Plan
    plan = make_plan(args.query)

    # Build TF-IDF retriever on the fly (quick)
    ret = TfidfRetriever(
        ngram_range=tuple(ag_cfg["retriever"].get("ngram_range", [1,2])),
        max_features=ag_cfg["retriever"].get("max_features", 50000)
    ).fit(df[["id","text"]])

    # Retrieve
    top_k = int(ag_cfg["retriever"]["top_k"])
    hits = ret.search(plan.query, top_k=top_k)

    # Extract on retrieved docs only (fast)
    extracts = run_extractors(hits, ag_cfg["extractors"])

    # Summarize retrieved docs
    sum_kind = ag_cfg["summarizer"]["type"]
    sum_cfg_path = ag_cfg["summarizer"]["config"]
    summaries = run_summarizer(hits, sum_cfg_path, sum_kind)

    # Synthesize final answer
    answer = synthesize_answer(
        query=plan.query,
        docs=hits,
        summaries=summaries,
        extracts=extracts,
        n_sent=int(ag_cfg["reasoning"]["answer_sentences"])
    )

    # Persist memory (optional)
    if ag_cfg["memory"]["enabled"]:
        remember(ag_cfg["memory"]["file"], answer)

    # Save demo output
    out_json = Path(ag_cfg["outputs"]["demo_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== ANSWER ===\n{answer['answer']}\n")
    print(f"Saved demo JSON -> {out_json}")
