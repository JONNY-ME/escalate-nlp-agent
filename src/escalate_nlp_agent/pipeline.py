from pathlib import Path
import pandas as pd
import json

from .text_prep.cleaning import normalize
from .text_prep.tokenize import simple_tokenize
from .text_prep.stopwords import remove_stopwords
from .eda import stats, viz
from .eda.ner import entity_freq

def load_raw(ds_cfg):
    paths = ds_cfg["raw"]["paths"]
    if len(paths) == 1:
        return pd.read_parquet(paths[0])
    dfs = [pd.read_parquet(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)

def preprocess(df, ds_cfg):
    df = df.copy()

    # Select & rename columns to a standard view
    fields = ds_cfg["fields"]
    cols = []
    outnames = []
    for k in ("id", "split", "title", "text", "meta"):
        src = fields.get(k)
        if src:  # keep only if defined (None/null is skipped)
            cols.append(src)
            outnames.append(k)
    df = df[cols].copy()
    df.columns = outnames

    # Normalize text
    if ds_cfg["preprocess"].get("normalize_unicode", True) or ds_cfg["preprocess"].get("lowercase", True):
        df["text"] = df["text"].map(normalize)
    df = df[df["text"].str.len() > 0]

    # Tokenize + stopwords
    extras = set(ds_cfg["preprocess"]["stopwords"].get("extra", []))
    df["tokens"] = df["text"].map(simple_tokenize)
    df["tokens"] = df["tokens"].map(lambda toks: [t for t in remove_stopwords(toks) if t not in extras])

    # Min length and duplicates
    min_tokens = int(ds_cfg["preprocess"].get("min_tokens", 10))
    df = df[df["tokens"].map(len) >= min_tokens]
    if ds_cfg["preprocess"].get("drop_duplicates", True):
        df = df.drop_duplicates(subset=["text"])

    # Ensure split exists
    if "split" not in df.columns:
        df["split"] = "all"

    # Ensure optional columns exist
    for opt in ("title", "meta"):
        if opt not in df.columns:
            df[opt] = None

    return df[["id", "split", "title", "text", "tokens", "meta"]]

def run_preprocess_and_eda(ds_cfg):
    outputs = ds_cfg["outputs"]
    figures_dir = Path(outputs["figures_dir"]); figures_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(ds_cfg)
    df = preprocess(raw, ds_cfg)

    # Save interim
    interim_path = Path(outputs["interim"])
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_path, index=False)

    # Save processed by split
    proc_dir = Path(outputs["processed_dir"]); proc_dir.mkdir(parents=True, exist_ok=True)
    for sp in sorted(df["split"].unique()):
        df[df["split"] == sp][["id","split","title","text","tokens","meta"]].to_parquet(
            proc_dir / f"{sp}.parquet", index=False
        )

    # EDA plots
    lens = stats.doc_lengths(df)
    viz.plot_lengths(lens, figures_dir / f"{ds_cfg['id']}_len_hist.png")

    tops = stats.top_terms(df, n=int(ds_cfg["eda"]["top_terms_n"]))
    viz.plot_top_terms(tops, figures_dir / f"{ds_cfg['id']}_top_terms.png")

    # NER (optional)
    if ds_cfg["eda"]["ner"]["enabled"]:
        try:
            ent = entity_freq(df["text"].tolist(), limit=int(ds_cfg["eda"]["ner"]["limit"]))
            ent.head(30).to_csv(figures_dir / f"{ds_cfg['id']}_ner_top.csv", index=False)
        except Exception as e:
            print(f"NER skipped: {e}")
