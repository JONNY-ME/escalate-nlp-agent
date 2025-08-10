import json
from pathlib import Path
from typing import Dict, Any

def remember(memory_path: str, item: Dict[str, Any]):
    p = Path(memory_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
