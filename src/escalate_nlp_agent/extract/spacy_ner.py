import spacy
import pandas as pd

def run_spacy_ner(df, cfg):
    nlp = spacy.load(cfg["model"], disable=["parser", "tagger", "lemmatizer"])
    wanted = set(cfg["labels"])
    results = []
    for doc_id, text in zip(df["id"], df["text"]):
        doc = nlp(text)
        ents = [{"label": ent.label_, "text": ent.text} for ent in doc.ents if ent.label_ in wanted]
        results.append({"id": doc_id, "entities": ents})
    return pd.DataFrame(results)
