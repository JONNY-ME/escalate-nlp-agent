import nltk, os, pandas as pd
from pathlib import Path

RAW = Path("data/raw/reuters")
RAW.mkdir(parents=True, exist_ok=True)

nltk.download("reuters"); nltk.download("punkt")
from nltk.corpus import reuters

docs = []
for fid in reuters.fileids():
    text = reuters.raw(fid)
    split = "train" if fid.startswith("training/") else "test"
    topics = reuters.categories(fid)
    docs.append({"id": fid, "split": split, "text": text, "topics": topics})

df = pd.DataFrame(docs)
RAW.mkdir(parents=True, exist_ok=True)
df.to_parquet(RAW / "reuters_full.parquet", index=False)
print(f"Saved {len(df)} docs to data/raw/reuters/reuters_full.parquet")


from datasets import load_dataset
from pathlib import Path

RAW = Path("data/raw/amazon_polarity"); RAW.mkdir(parents=True, exist_ok=True)

ds = load_dataset("amazon_polarity")
for split in ("train", "test"):
    part = ds[split].to_pandas()[["title", "content", "label"]]
    part = part.rename(columns={"content": "text"})
    part.to_parquet(RAW / f"{split}.parquet", index=False)
    print(f"Saved {len(part)} rows -> data/raw/amazon_polarity/{split}.parquet")
