"""
Lightweight refinement hooks for scraped and processed text.

Edit the allowlists/denylists and functions below to control what gets into
your training dataset without touching core pipeline code.
"""
from __future__ import annotations
from typing import Optional, Dict
import re

# --- Simple knobs you can edit ---
MIN_WORDS_PER_PAGE = 40  # drop very thin pages
ALLOW_URL_KEYWORDS: list[str] = []  # e.g., ["fund", "insights"]
BLOCK_URL_KEYWORDS: list[str] = [
    "login", "signup", "account", "privacy", "terms", "cookie"
]
BLOCK_TEXT_PATTERNS: list[str] = [
    r"subscribe\s+to\s+our\s+newsletter",
    r"all\s+rights\s+reserved",
]
REDACTIONS: dict[str, str] = {
    # "ConfidentialTerm": "[REDACTED]",
}


def _word_count(text: str) -> int:
    return len((text or "").split())


def refine_text(text: str, url: str = "") -> str:
    """Clean or transform text. Return the updated text (can be unchanged)."""
    if not text:
        return text

    cleaned = text

    # Optional: remove boilerplate lines by pattern
    for pat in BLOCK_TEXT_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)

    # Optional: redact terms
    for k, v in REDACTIONS.items():
        cleaned = re.sub(re.escape(k), v, cleaned)

    # Collapse excessive blank lines
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip()


def refine_page(page: Dict) -> Optional[Dict]:
    """Decide whether to keep/modify a scraped page. Return None to drop it."""
    url = str(page.get("url", ""))
    content = page.get("content", "") or ""

    # URL-based filtering
    if any(k.lower() in url.lower() for k in BLOCK_URL_KEYWORDS):
        return None
    if ALLOW_URL_KEYWORDS:
        if not any(k.lower() in url.lower() for k in ALLOW_URL_KEYWORDS):
            return None

    # Content thinness check
    if _word_count(content) < MIN_WORDS_PER_PAGE:
        return None

    # Optionally mutate page fields
    page["content"] = refine_text(content, url=url)
    return page
