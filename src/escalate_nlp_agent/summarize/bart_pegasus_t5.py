from transformers import pipeline
import pandas as pd

def run_bart(df, cfg):
    summarizer = pipeline("summarization", model=cfg["model"])
    results = []
    for doc_id, text in zip(df["id"], df["text"]):
        # Truncate to model max length
        input_text = text[:cfg.get("max_input_tokens", 512)]
        try:
            summary = summarizer(
                input_text,
                max_length=cfg.get("max_output_tokens", 100),
                min_length=5,
                do_sample=False
            )[0]["summary_text"]
        except Exception:
            summary = text
        results.append({"id": doc_id, "summary": summary})
    return pd.DataFrame(results)
