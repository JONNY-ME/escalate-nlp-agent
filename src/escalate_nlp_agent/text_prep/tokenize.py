import re
from typing import List

def simple_tokenize(text: str) -> List[str]:
    """Tokenize text into alphabetic tokens."""
    return re.findall(r"[a-zA-Z]+(?:'[a-z]+)?", text)
