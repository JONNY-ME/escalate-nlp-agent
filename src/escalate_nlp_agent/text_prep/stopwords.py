from nltk.corpus import stopwords

EN_STOPWORDS = set(stopwords.words("english"))

EXTRA_STOPWORDS = {"reuters", "reuter", "said", "would", "also"}

def remove_stopwords(tokens):
    sw = EN_STOPWORDS | EXTRA_STOPWORDS
    return [t for t in tokens if t not in sw and len(t) > 2]
