import pandas as pd
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def run_textrank(df, cfg):
    n_sent = cfg.get("sentences", 3)
    summarizer = TextRankSummarizer()
    results = []
    for doc_id, text in zip(df["id"], df["text"]):
        try:
            parser = PlaintextParser.from_string(text or "", Tokenizer("english"))
            sentences = summarizer(parser.document, n_sent)
            summary = " ".join(str(s) for s in sentences)
            if not summary:  # very short docs fallback
                summary = (text or "")[:300]
        except Exception:
            summary = (text or "")[:300]
        results.append({"id": doc_id, "summary": summary})
    return pd.DataFrame(results)
