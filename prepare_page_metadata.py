"""Generate a sidecar metadata map without changing corpus text or embeddings."""
import json
from collections import OrderedDict
from pathlib import Path

from app.metadata import infer_page_metadata

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "data" / "knowledge.jsonl"
OUTPUT = ROOT / "data" / "page_metadata.json"


def main():
    pages = OrderedDict()
    with CORPUS.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("url") and row.get("text"):
                pages.setdefault(row["url"], row)
    metadata = {url: infer_page_metadata(row) for url, row in pages.items()}
    OUTPUT.write_text(
        json.dumps({"version": 1, "pages": metadata}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared metadata for {len(metadata)} pages in {OUTPUT}")


if __name__ == "__main__":
    main()
