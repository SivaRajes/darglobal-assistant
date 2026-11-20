"""Page-level metadata and lightweight intent parsing for scoped retrieval."""
import json
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT / "data" / "page_metadata.json"

CITY_COUNTRY = {
    "dubai": "United Arab Emirates", "abu dhabi": "United Arab Emirates",
    "ras al khaimah": "United Arab Emirates", "jeddah": "Saudi Arabia",
    "riyadh": "Saudi Arabia", "diriyah": "Saudi Arabia", "muscat": "Oman",
    "doha": "Qatar", "simaisma": "Qatar", "london": "United Kingdom",
    "benahavis": "Spain", "costa del sol": "Spain", "marbella": "Spain",
}
COUNTRY_ALIASES = {
    "uae": "United Arab Emirates", "united arab emirates": "United Arab Emirates",
    "saudi arabia": "Saudi Arabia", "oman": "Oman", "qatar": "Qatar",
    "spain": "Spain", "uk": "United Kingdom", "united kingdom": "United Kingdom",
}
PROJECT_WORDS = re.compile(r"\b(projects?|developments?|portfolio)\b", re.I)
LISTING_WORDS = re.compile(r"\b(for sale|property|properties|listing|listings|home|homes|villa|villas|apartment|apartments|land)\b", re.I)
NEWS_WORDS = re.compile(r"\b(partnership|partnered|announcement|announced|press release|launch|launched|news)\b", re.I)
PROJECT_LOCATION_SIGNAL = re.compile(
    r"\b(project|development|tower|residences?|villas?|hotel|resort|community|apartments?|homes?|waterfront|oceanfront)\b",
    re.I,
)


def _matches(text, phrase):
    return bool(re.search(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", text, re.I))


def infer_page_metadata(row):
    """Use page type and location context; never treat every city mention as a project location."""
    url = str(row.get("url", ""))
    title = str(row.get("title", ""))
    if title.lower().startswith(("http://", "https://")):
        title = ""
    text = str(row.get("text", ""))
    parsed = urlparse(url)
    path = unquote(parsed.path).lower()
    host = (parsed.hostname or "").lower()
    if "wasalt.sa" in host:
        if "/property/" in path:
            page_type = "property_listing"
        elif "/project/" in path:
            page_type = "project_page"
        elif "/corporate" in path:
            page_type = "corporate"
        else:
            page_type = "listing_search" if LISTING_WORDS.search(title) else "corporate"
    elif "darglobal.co.uk" in host:
        if path.startswith("/press/"):
            page_type = "news"
        elif path.startswith("/projects"):
            page_type = "portfolio"
        elif path.startswith(("/partners/", "/product-category/", "/play-project/")) or path in ("/marea",):
            page_type = "project_page"
        elif path.startswith(("/blog/", "/insights/")):
            page_type = "editorial"
        else:
            page_type = "corporate"
    else:
        page_type = "unknown"

    project_city = ""
    event_city = ""
    mentioned_city = ""
    city = ""
    country = ""
    city_phrases = sorted(CITY_COUNTRY, key=len, reverse=True)
    if page_type == "property_listing":
        # Listing URLs often lack a city; read the structured listing address before map details.
        location_text = text.split("Map view", 1)[0][:900]
        for name in city_phrases:
            if _matches(location_text, name):
                city, country = name.title(), CITY_COUNTRY[name]
                break
    elif page_type in {"project_page", "portfolio", "news", "editorial", "corporate"}:
        direct = f"{path} {title}".lower()
        city_mention = next((name for name in city_phrases if _matches(direct, name)), "")
        if "expo" in direct or "event" in direct:
            event_city = city_mention.title() if city_mention else ""
        elif page_type in {"project_page", "portfolio"}:
            project_city = city_mention.title() if city_mention else ""
            if project_city:
                country = CITY_COUNTRY[city_mention]
        elif page_type == "news" and city_mention and PROJECT_LOCATION_SIGNAL.search(direct):
            project_city = city_mention.title()
            country = CITY_COUNTRY[city_mention]
        else:
            mentioned_city = city_mention.title() if city_mention else ""

    # Country portfolio routes identify country coverage, not a specific city.
    country_route = re.search(r"properties-in-([a-z-]+)", path)
    project_country = country
    if country_route and page_type == "portfolio":
        slug = country_route.group(1).replace("-", " ")
        project_country = COUNTRY_ALIASES.get(slug, slug.title())
    elif not project_country and (project_city or (page_type in {"project_page", "news"}
                                                    and PROJECT_LOCATION_SIGNAL.search(f"{path} {title}")
                                                    and not event_city)):
        for alias, canonical in sorted(COUNTRY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
            if _matches(f"{path} {title}".lower(), alias):
                project_country = canonical
                break

    is_official = host in {"darglobal.co.uk", "www.darglobal.co.uk", "wasalt.sa", "www.wasalt.sa"}
    source_tier = {
        "project_page": 4, "property_listing": 4, "portfolio": 3,
        "corporate": 2, "news": 2, "editorial": 1, "listing_search": 1,
    }.get(page_type, 0)
    project_name = title if page_type in {"project_page", "property_listing"} else ""
    project_name = project_name.split("|", 1)[0].split("–", 1)[0].split("—", 1)[0].strip()
    project_name = re.sub(r"\s+for sale\s*$", "", project_name, flags=re.I).strip()
    if page_type == "portfolio":
        project_name = ""
    return {
        "page_type": page_type,
        "project_name": project_name,
        "city": city or project_city,
        "project_city": project_city,
        "event_city": event_city,
        "mentioned_city": mentioned_city,
        "country": country or project_country,
        "project_country": project_country,
        "location_role": "event" if event_city else ("project" if project_city else
                         ("property" if city else ("mentioned" if mentioned_city else "unspecified"))),
        "is_official": bool(is_official),
        "source_tier": source_tier,
    }


@lru_cache(maxsize=1)
def page_metadata():
    try:
        payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return payload.get("pages", {})
    except (OSError, ValueError, AttributeError):
        return {}


def metadata_for_url(url):
    metadata = page_metadata().get(url)
    return metadata if isinstance(metadata, dict) else {}


def classify_query(query):
    """Identify broad task and explicitly named city for type-aware ranking."""
    normalized = unquote(str(query or "")).lower().replace("’", "'")
    city = next((name.title() for name in sorted(CITY_COUNTRY, key=len, reverse=True) if _matches(normalized, name)), "")
    country = next((canonical for alias, canonical in sorted(COUNTRY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
                    if _matches(normalized, alias)), "")
    if (city or country) and PROJECT_WORDS.search(normalized):
        intent = "location_projects"
    elif _matched_project_name(normalized):
        intent = "named_project"
    elif LISTING_WORDS.search(normalized):
        intent = "property_listing"
    elif NEWS_WORDS.search(normalized):
        intent = "announcement"
    elif PROJECT_WORDS.search(normalized):
        intent = "projects"
    else:
        intent = "general"
    return {"intent": intent, "city": city, "country": country,
            "project_name": _matched_project_name(normalized)}


def _matched_project_name(normalized_query):
    query_tokens = set(re.findall(r"[\w]+", normalized_query))
    candidates = {
        str(item.get("project_name", "")).strip()
        for item in page_metadata().values()
        if item.get("page_type") == "project_page" and item.get("project_name")
    }
    for name in sorted(candidates, key=len, reverse=True):
        name_tokens = set(re.findall(r"[\w]+", name.lower()))
        if name_tokens and name_tokens.issubset(query_tokens):
            return name
    return ""


def rerank_with_metadata(query, docs):
    """Apply small, intent-specific source priors after semantic relevance ranking."""
    intent = classify_query(query)
    if not docs:
        return docs
    n = max(1, len(docs))
    for position, doc in enumerate(docs):
        metadata = doc.get("page_metadata") or {}
        doc["page_metadata"] = metadata
        relevance = 1.0 - position / n
        prior = 0.0
        page_type = metadata.get("page_type", "unknown")
        if intent["intent"] == "location_projects":
            target_city = intent.get("city", "")
            target_country = intent.get("country", "")
            if target_city and metadata.get("project_city", "").casefold() == target_city.casefold():
                prior += 0.55
            elif target_city and metadata.get("event_city", "").casefold() == target_city.casefold():
                prior -= 0.4
            if target_country and metadata.get("project_country", "").casefold() == target_country.casefold():
                prior += 0.35
            if page_type in {"project_page", "portfolio"}:
                prior += 0.12
            elif page_type in {"news", "editorial"}:
                prior -= 0.08
        elif intent["intent"] == "projects":
            prior += {"portfolio": 0.3, "project_page": 0.25, "news": 0.03,
                      "editorial": -0.08, "corporate": -0.05}.get(page_type, 0)
        elif intent["intent"] == "property_listing":
            prior += 0.25 if page_type == "property_listing" else (-0.2 if page_type == "editorial" else 0)
        elif intent["intent"] == "named_project":
            expected = set(re.findall(r"[\w]+", intent.get("project_name", "").lower()))
            actual = set(re.findall(r"[\w]+", str(metadata.get("project_name", "")).lower()))
            if expected and expected.issubset(actual):
                prior += 0.55
        elif intent["intent"] == "announcement":
            prior += 0.2 if page_type == "news" else 0
        doc["_metadata_rank"] = relevance + prior
    return sorted(docs, key=lambda d: d.get("_metadata_rank", 0), reverse=True)
