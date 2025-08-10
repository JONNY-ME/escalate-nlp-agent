import spacy, pandas as pd
from collections import Counter

_NLP = None
def _load():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["tagger","parser","lemmatizer"])
    return _NLP

def entity_freq(texts, limit=2000):
    nlp = _load()
    kinds = {"PERSON","ORG","GPE","DATE"}
    ctr = Counter()
    for doc in nlp.pipe(texts[:limit], batch_size=64):
        for ent in doc.ents:
            if ent.label_ in kinds: ctr[(ent.label_, ent.text.lower())] += 1
    df = (pd.DataFrame([(k[0], k[1], v) for k,v in ctr.items()], columns=["label","text","count"])
            .sort_values("count", ascending=False))
    return df
