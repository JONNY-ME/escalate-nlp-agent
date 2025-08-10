import re
import unicodedata

def normalize(text: str) -> str:
    """Lowercase, normalize unicode, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text
