"""Polite crawl of public DarGlobal and Wasalt pages for the HR assignment.

The crawler obeys each site's robots.txt, stays on the configured domains,
avoids account/people paths, rate limits requests, and redacts email/phone
strings before writing searchable text to data/knowledge.jsonl.
"""
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from app.corpus import chunk_text, clean_text

ROOT = Path(__file__).parent
SOURCES = ROOT / "data" / "sources.json"
OUT = ROOT / "data" / "knowledge.jsonl"
USER_AGENT = "DarGlobalWasaltAssignmentBot/1.0 (public pages; contact project maintainer)"
MAX_PAGES_PER_HOST = 1500
DELAY_SECONDS = 1.0
TIMEOUT = 25
SKIP_PATH = re.compile(r"/(login|signin|sign-in|register|signup|account|profile|agent|broker|user|users|dashboard|checkout|payment)(/|$)", re.I)
SKIP_FILE = re.compile(r"\.(pdf|jpg|jpeg|png|gif|webp|svg|zip|mp4|mp3|docx?|xlsx?|csv)(?:$|\?)", re.I)
def canonical(url):
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    # Ignore query variants to prevent crawlers entering filter/search loops.
    return parsed._replace(query="", fragment="").geturl().rstrip("/") or parsed._replace(query="", fragment="").geturl()


def robots_for(origin):
    robots_url = origin.rstrip("/") + "/robots.txt"
    try:
        response = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if response.status_code == 404:
            return RobotFileParser()  # no published robots file
        response.raise_for_status()
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        return parser
    except requests.RequestException as exc:
        print(f"Cannot verify robots.txt for {origin}; skipping host: {exc}")
        return None


def crawl_source(source, output):
    seed = canonical(source["url"])
    if not seed:
        return 0
    parsed_seed = urlparse(seed)
    host = parsed_seed.hostname.lower()
    origin = f"{parsed_seed.scheme}://{parsed_seed.netloc}"
    robots = robots_for(origin)
    if robots is None:
        return 0
    queue, seen, count = deque([seed]), set(), 0

    # Add same-host sitemap URLs advertised by robots.txt.
    try:
        rr = requests.get(origin + "/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
        if rr.ok:
            for line in rr.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sm = line.split(":", 1)[1].strip()
                    if (urlparse(sm).hostname or "").lower() == host and robots.can_fetch(USER_AGENT, sm):
                        sr = requests.get(sm, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
                        if sr.ok:
                            xml = BeautifulSoup(sr.content, "xml")
                            for loc in xml.find_all("loc"):
                                link = canonical(loc.get_text(strip=True))
                                if link and urlparse(link).hostname == host:
                                    queue.append(link)
    except requests.RequestException as exc:
        print(f"Sitemap discovery skipped for {host}: {exc}")

    while queue and len(seen) < MAX_PAGES_PER_HOST:
        url = canonical(queue.popleft())
        if not url or url in seen or urlparse(url).hostname.lower() != host:
            continue
        path = urlparse(url).path
        if SKIP_PATH.search(path) or SKIP_FILE.search(path) or not robots.can_fetch(USER_AGENT, url):
            continue
        seen.add(url)
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
            if response.status_code != 200 or "text/html" not in response.headers.get("Content-Type", "").lower():
                continue
        except requests.RequestException as exc:
            print(f"Skip {url}: {exc}")
            continue

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "svg", "button"]):
            tag.decompose()
        # Remove common repeated interface regions that pollute property/article text.
        for node in soup.select("[aria-label*=cookie i], [id*=cookie i], [class*=cookie i], [id*=breadcrumb i], [class*=breadcrumb i], [class*=pagination i]"):
            node.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = clean_text(main.get_text("\n", strip=True))
        if len(text) >= 120:
            for part in chunk_text(text):
                record = {"url": url, "title": title, "text": part,
                          "source": source["name"],
                          "scraped_at": datetime.now(timezone.utc).isoformat()}
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            output.flush()  # keep a usable JSONL corpus if the run is interrupted
        for anchor in soup.find_all("a", href=True):
            link = canonical(urljoin(url, anchor["href"]))
            if link and urlparse(link).hostname.lower() == host and link not in seen:
                queue.append(link)
        print(f"[{len(seen)}/{MAX_PAGES_PER_HOST}] {url}")
        time.sleep(DELAY_SECONDS)
    print(f"{source['name']}: visited {len(seen)} pages; retained {count} text chunks", flush=True)
    return count


def main():
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with OUT.open("w", encoding="utf-8") as output:
        for source in sources:
            print(f"Starting {source['name']} crawl", flush=True)
            counts[source["name"]] = crawl_source(source, output)
    total = sum(counts.values())
    print(f"Finished. Wrote {total} text chunks to {OUT}; source totals: {counts}", flush=True)


if __name__ == "__main__":
    main()
