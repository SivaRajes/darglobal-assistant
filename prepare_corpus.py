"""Clean and rechunk the existing JSONL crawl without fetching the sites again."""
import json
import shutil
from collections import OrderedDict
from pathlib import Path

from app.corpus import chunk_text, clean_text

ROOT = Path(__file__).parent
CORPUS = ROOT / "data" / "knowledge.jsonl"
BACKUP = ROOT / "data" / "knowledge.jsonl.bak"


def append_without_overlap(previous, addition):
    """Join the crawler's overlapping fixed-width fragments back into a page."""
    maximum = min(500, len(previous), len(addition))
    for size in range(maximum, 39, -1):
        if previous[-size:] == addition[:size]:
            return previous + addition[size:]
    if addition in previous:
        return previous
    return previous + " " + addition


def main():
    pages = OrderedDict()
    with CORPUS.open(encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = row.get("url", "")
            text = clean_text(row.get("text", ""))
            if not url or not text:
                continue
            if url not in pages:
                pages[url] = {**row, "text": text}
            else:
                pages[url]["text"] = append_without_overlap(pages[url]["text"], text)

    temporary = CORPUS.with_suffix(".jsonl.tmp")
    out_count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for page in pages.values():
            # Remove exact repeated adjacent sentences left by overlapping crawl chunks.
            normalized = clean_text(page["text"])
            chunks = chunk_text(normalized, max_chars=1100, overlap_sentences=1)
            for text in chunks:
                item = {key: page.get(key, "") for key in ("url", "title", "source", "scraped_at")}
                item["text"] = text
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_count += 1

    if not BACKUP.exists():
        shutil.copy2(CORPUS, BACKUP)
    temporary.replace(CORPUS)
    print(f"Rechunked {len(pages)} pages into {out_count} records.")
    print(f"Original corpus backup: {BACKUP}")


if __name__ == "__main__":
    main()
