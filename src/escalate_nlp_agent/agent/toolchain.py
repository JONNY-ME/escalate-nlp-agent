import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from ..extract.rule_based import run_rule_based
from ..extract.spacy_ner import run_spacy_ner
from ..summarize.textrank import run_textrank
from ..summarize.bart_pegasus_t5 import run_bart

def _load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_extractors(df: pd.DataFrame, extractor_cfg_paths: list[str]) -> Dict[str, pd.DataFrame]:
    outputs = {}
    for p in extractor_cfg_paths:
        cfg = _load_cfg(p)
        name = cfg["name"].lower()
        if "rule" in name:
            outputs["rule_based"] = run_rule_based(df, cfg)
        elif "spacy" in name:
            outputs["spacy_ner"] = run_spacy_ner(df, cfg)
    return outputs

def run_summarizer(df: pd.DataFrame, sum_cfg_path: str, kind: str) -> pd.DataFrame:
    cfg = _load_cfg(sum_cfg_path)
    if kind == "extractive":
        return run_textrank(df, cfg)
    return run_bart(df, cfg)

def synthesize_answer(query: str, docs: pd.DataFrame, summaries: pd.DataFrame, extracts: Dict[str, pd.DataFrame], n_sent: int = 4) -> Dict[str, Any]:
    # Join summaries back by id
    merged = docs[["id","text","score"]].merge(summaries, on="id", how="left", suffixes=("",""))
    # Prepare a compact evidence list
    evidence = []
    for _, r in merged.iterrows():
        snippet = (r.get("summary") or r["text"])[:400]
        evidence.append({"id": r["id"], "score": float(r["score"]), "snippet": snippet})

    # Collate some extracted entities/numbers for transparency
    ents = []
    if "spacy_ner" in extracts:
        for _, row in extracts["spacy_ner"].iterrows():
            ents.extend(row["entities"][:5])  # trim

    nums = []
    if "rule_based" in extracts:
        for _, row in extracts["rule_based"].iterrows():
            numbers = row["extracted"].get("numbers", [])[:5]
            dates = row["extracted"].get("dates", [])[:3]
            if numbers or dates:
                nums.append({"id": row["id"], "numbers": numbers, "dates": dates})

    # Naive synthesis: concatenate top summaries; in a real system you might rank sentences or run a generator.
    joined = " ".join([e["snippet"] for e in evidence])[:2000]
    # Keep only ~n_sent sentences for a clean final answer
    sentences = [s.strip() for s in joined.split(". ") if s.strip()]
    answer = ". ".join(sentences[:n_sent])
    if not answer.endswith("."): answer += "."

    return {
        "query": query,
        "answer": answer,
        "support": evidence,
        "entities": ents[:20],
        "numbers_dates": nums[:10],
    }
