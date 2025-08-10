import re
from dataclasses import dataclass

@dataclass
class Plan:
    query: str
    keywords: list[str]
    need_entities: bool = True
    need_numbers: bool = True

_KEY = re.compile(r"[A-Za-z][A-Za-z\-]+")

def make_plan(query: str) -> Plan:
    # crude keywording but robust and fast
    toks = [t.lower() for t in _KEY.findall(query)]
    # keep content-like tokens (length >= 3)
    keywords = [t for t in toks if len(t) >= 3]
    return Plan(query=query.strip(), keywords=keywords)
