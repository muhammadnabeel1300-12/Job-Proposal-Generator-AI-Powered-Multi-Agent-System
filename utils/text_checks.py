"""
text_checks.py - Small text helpers used for SAFETY checks.

1. Evidence checking: confirm that a quote returned by Claude really appears
   in the source document (this helps catch invented information).
2. Prompt-injection protection: uploaded documents are untrusted. We clean
   them and warn when they look like they contain instructions for the AI.
"""

import re

MAX_DOCUMENT_CHARS = 40_000  # Limits cost and protects against huge inputs.

# Tags we use to wrap documents in prompts. Removed from uploaded text so a
# document cannot "close" the tag and escape into the instruction area.
_PROMPT_TAGS = re.compile(
    r"<\s*/?\s*(job_description|candidate_cv|proposal|job_analysis|"
    r"profile_match|requirements|instructions|system)\b[^>]*>",
    re.IGNORECASE,
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

INJECTION_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|rules)",
    r"disregard (all |any |the )?(previous|prior|above|earlier)",
    r"forget (all |any |the )?(previous|prior|above|earlier)",
    r"you are now\b",
    r"new instructions\s*:",
    r"(reveal|show|print) (your|the) (system )?prompt",
    r"override (the )?(system|instructions|rules)",
    r"do not follow (the )?(previous|above|system)",
    r"\bsystem prompt\b",
    r"act as (if you are )?(an? )?(different|unrestricted)",
]
_INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def normalize(text: str) -> str:
    """Lowercase and simplify text so two strings can be compared fairly."""
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = re.sub(r"[^\w\s+#.]", " ", text)  # keep + # . for c++, c#, node.js
    text = re.sub(r"\.+(?=\s|$)", "", text)   # drop sentence-ending periods
    return re.sub(r"\s+", " ", text).strip()


def keyword_in_text(keyword: str, source: str) -> bool:
    """True if the keyword appears in the source text."""
    kw = normalize(keyword)
    return bool(kw) and kw in normalize(source)


def evidence_in_text(evidence: str, source: str) -> bool:
    """
    True if a quote (evidence) is really found in the source text.

    Strategy (strict to lenient):
    1. The whole quote appears in the text.
    2. Every fragment of a quote written with '...' appears in the text.
    3. At least 85% of the words of a longer quote appear in the text
       (tolerates small OCR or punctuation differences).
    """
    ev = normalize(evidence)
    src = normalize(source)
    if not ev or not src:
        return False
    if ev in src:
        return True

    fragments = [normalize(f) for f in re.split(r"\.\.\.|\u2026|\n|;", evidence)]
    fragments = [f for f in fragments if len(f) >= 12]
    if fragments and all(f in src for f in fragments):
        return True

    tokens = ev.split()
    if len(tokens) >= 3:
        source_tokens = set(src.split())
        overlap = sum(1 for t in tokens if t in source_tokens) / len(tokens)
        return overlap >= 0.85
    return False


def detect_prompt_injection(text: str) -> list[str]:
    """Return the suspicious phrases found in a document (empty list = none)."""
    return sorted({m.group(0).strip().lower() for m in _INJECTION_REGEX.finditer(text)})


def sanitize_document_text(text: str, label: str = "Document") -> tuple[str, list[str]]:
    """
    Prepare untrusted document text for a prompt.
    Returns (clean_text, warnings).
    """
    warnings: list[str] = []
    clean = _CONTROL_CHARS.sub("", text)
    clean = _PROMPT_TAGS.sub("[removed]", clean)

    found = detect_prompt_injection(clean)
    if found:
        warnings.append(
            f"{label} contains text that looks like instructions to an AI "
            f"({', '.join(repr(f) for f in found[:3])}). It is treated as plain "
            "data and will not be followed."
        )
    if len(clean) > MAX_DOCUMENT_CHARS:
        clean = clean[:MAX_DOCUMENT_CHARS]
        warnings.append(
            f"{label} is very long. Only the first {MAX_DOCUMENT_CHARS:,} "
            "characters were analyzed."
        )
    return clean, warnings
