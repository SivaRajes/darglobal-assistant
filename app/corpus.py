"""Shared text preparation for crawled pages and question retrieval."""
import re
import unicodedata

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?966[\s().-]?|0)5(?:[\s().-]?\d){8}(?!\w)")
SPACE = re.compile(r"[\t\f\v ]+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟。])\s+|\n+")


def clean_text(text):
    """Normalize whitespace and redact direct contact details without losing content."""
    text = unicodedata.normalize("NFKC", text or "")
    text = EMAIL.sub("[email removed]", text)
    text = PHONE.sub("[phone removed]", text)
    lines = []
    previous_line = None
    for line in text.splitlines():
        line = SPACE.sub(" ", line).strip()
        if not line or line == previous_line:
            continue
        lines.append(line)
        previous_line = line
    value = "\n".join(lines).strip()
    paragraphs = []
    previous_sentence = None
    for line in value.splitlines():
        sentences = []
        for sentence in SENTENCE_BOUNDARY.split(line):
            sentence = sentence.strip()
            key = sentence.casefold()
            if sentence and key != previous_sentence:
                sentences.append(sentence)
            previous_sentence = key
        if sentences:
            paragraphs.append(" ".join(sentences))
    return "\n".join(paragraphs)


def chunk_text(text, max_chars=1100, overlap_sentences=0):
    """Split on paragraph/sentence boundaries, preserving a small semantic overlap."""
    sentences = [s.strip() for s in SENTENCE_BOUNDARY.split(clean_text(text)) if s.strip()]
    chunks, current = [], []
    current_len = 0
    for sentence in sentences:
        # Split unusually long blocks at word boundaries rather than dropping text.
        pieces = []
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
        for piece in pieces:
            extra = len(piece) + (1 if current else 0)
            if current and current_len + extra > max_chars:
                chunks.append(" ".join(current))
                keep = current[-overlap_sentences:] if overlap_sentences else []
                current = keep[:]
                current_len = sum(map(len, current)) + max(0, len(current) - 1)
                if current and current_len + extra > max_chars:
                    current = []
                    current_len = 0
            current.append(piece)
            current_len += len(piece) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(" ".join(current))
    if len(chunks) > 1 and len(chunks[-1]) < 80 and len(chunks[-2]) + len(chunks[-1]) + 1 <= max_chars * 1.25:
        tail = chunks.pop()
        chunks[-1] += " " + tail
    return [c for c in chunks if len(c) >= 40]


QUERY_INSTRUCTIONS = re.compile(
    r"\b(?:and\s+)?(?:include|including|provide|add|show)\s+(?:their\s+|the\s+)?(?:source\s+)?(?:links?|urls?|citations?)\b",
    re.I,
)
QUESTION_FRAME = re.compile(
    r"\b(?:what|where|how|which|who)\s+(?:does|do|did)\s+(?:(DarGlobal|Wasalt)(?:['’]s)?\s+)?(?:the\s+)?page\s+(?:say|says|mention|mentions|describe|describes)\s+(?:about\s+)?",
    re.I,
)


def prepare_question(question):
    """Remove answer-format requests while retaining the user's information need."""
    question = unicodedata.normalize("NFKC", question or "")
    question = QUESTION_FRAME.sub(lambda m: (m.group(1) + " ") if m.group(1) else "", question)
    question = QUERY_INSTRUCTIONS.sub(" ", question)
    question = re.sub(r"\band\s+it\b", " ", question, flags=re.I)
    question = re.sub(r"\b(?:please|sources?|links?|urls?|citations?)\b", " ", question, flags=re.I)
    question = re.sub(r"\s+", " ", question).strip(" \t\r\n.,!?;:")
    return question
