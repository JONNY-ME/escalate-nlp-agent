from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    def __init__(self, ngram_range=(1,2), max_features=50000):
        self.vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self.doc_vectors = None
        self.df = None

    def fit(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.doc_vectors = self.vec.fit_transform(self.df["text"].fillna(""))
        return self

    def search(self, query: str, top_k: int = 5):
        qv = self.vec.transform([query])
        sims = cosine_similarity(qv, self.doc_vectors)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        out = self.df.iloc[top_idx].copy()
        out["score"] = sims[top_idx]
        return out

def build_retriever(ds_cfg):
    proc_dir = Path(ds_cfg["outputs"]["processed_dir"])
    # Use train split for demo; you can concat test if you want
    df = pd.read_parquet(proc_dir / "train.parquet")
    ret_cfg = ds_cfg.get("retriever_params", {})
    return TfidfRetriever(
        ngram_range=ret_cfg.get("ngram_range", (1,2)),
        max_features=ret_cfg.get("max_features", 50000),
    ).fit(df[["id","text"]])
