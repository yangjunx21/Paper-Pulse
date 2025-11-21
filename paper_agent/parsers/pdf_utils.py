from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import List, Sequence

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

LOGGER = logging.getLogger(__name__)

PDF_REQUEST_TIMEOUT_SECONDS = 30
PDF_MAX_BYTES = 25 * 1024 * 1024  # 25 MB safety cap
DEFAULT_MAX_PAGES = 15
DEFAULT_MAX_CHARS = 50_000
AFFILIATION_SCAN_MAX_PAGES = 2
AFFILIATION_SCAN_MAX_CHARS = 4_000
AFFILIATION_KEYWORDS = (
    "university",
    "institute",
    "laboratory",
    "lab",
    "college",
    "school",
    "centre",
    "center",
    "department",
    "dept.",
    "academy",
    "research",
    "company",
    "corporation",
    "corp",
    "inc.",
)


def download_pdf(
    url: str,
    *,
    timeout: int = PDF_REQUEST_TIMEOUT_SECONDS,
    max_bytes: int = PDF_MAX_BYTES,
    chunk_size: int = 64 * 1024,
) -> bytes | None:
    """
    Download a PDF document from the provided URL.

    The caller is responsible for respecting rate limits. Oversized responses are discarded.
    """

    if not url:
        return None
    response = None
    try:
        response = requests.get(
            url,
            stream=True,
            timeout=(timeout, timeout),
            headers={"User-Agent": "PaperPulse/1.0 (+https://github.com/)",},
        )
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and max_bytes:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size and declared_size > max_bytes:
                LOGGER.warning(
                    "Skipping PDF %s because declared size is %d bytes (limit=%d).",
                    url,
                    declared_size,
                    max_bytes,
                )
                return None
        chunks: List[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            total += len(chunk)
            if max_bytes and total > max_bytes:
                LOGGER.warning(
                    "Skipping PDF %s because payload exceeded %d bytes.",
                    url,
                    max_bytes,
                )
                return None
            chunks.append(chunk)
        return b"".join(chunks)
    except requests.Timeout:
        LOGGER.warning("Timed out while downloading PDF %s (timeout=%ss).", url, timeout)
        return None
    except requests.RequestException as exc:
        LOGGER.warning("Failed to download PDF %s: %s", url, exc)
        return None
    finally:
        if response is not None:
            response.close()


def extract_text_from_pdf_bytes(
    payload: bytes | None,
    *,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """
    Extract Unicode text from PDF bytes, truncating by pages and characters for prompt safety.
    """

    if not payload:
        return ""
    try:
        reader = PdfReader(BytesIO(payload))
    except PdfReadError as exc:
        LOGGER.warning("Unable to parse PDF bytes: %s", exc)
        return ""
    texts: List[str] = []
    page_limit = len(reader.pages)
    if max_pages is not None:
        page_limit = min(page_limit, max_pages)
    remaining_chars = max_chars if max_chars and max_chars > 0 else None
    for index in range(page_limit):
        try:
            page_text = reader.pages[index].extract_text() or ""
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.debug("Failed to extract text from page %s: %s", index, exc)
            continue
        normalized = _normalize_whitespace(page_text)
        if not normalized:
            continue
        if remaining_chars is not None:
            if len(normalized) > remaining_chars:
                normalized = normalized[:remaining_chars]
            remaining_chars -= len(normalized)
        texts.append(normalized)
        if remaining_chars is not None and remaining_chars <= 0:
            break
    return "\n\n".join(texts).strip()


def extract_affiliation_snippet_from_pdf_bytes(
    payload: bytes | None,
    *,
    max_pages: int = AFFILIATION_SCAN_MAX_PAGES,
    max_chars: int = AFFILIATION_SCAN_MAX_CHARS,
) -> str:
    """
    Extract a small slice of text suitable for affiliation heuristics.
    """

    return extract_text_from_pdf_bytes(
        payload,
        max_pages=max_pages,
        max_chars=max_chars,
    )


def infer_affiliations_from_text(
    text: str,
    existing: Sequence[str] | None = None,
    *,
    max_candidates: int = 6,
    scan_lines: int = 400,
) -> List[str]:
    """
    Extract additional affiliation strings from PDF text using lightweight heuristics.
    """

    existing_list = [entry.strip() for entry in existing or [] if entry and entry.strip()]
    seen = {_normalize_affiliation(entry) for entry in existing_list if entry}
    results: List[str] = list(existing_list)

    if not text:
        return results

    lines = text.splitlines()
    for raw_line in lines[:scan_lines]:
        candidate = _normalize_line_for_affiliation(raw_line)
        if not candidate:
            continue
        normalized = _normalize_affiliation(candidate)
        if not normalized:
            continue
        if normalized in seen:
            continue
        if not _contains_affiliation_keyword(normalized):
            continue
        seen.add(normalized)
        results.append(candidate)
        if len(results) >= max_candidates:
            break
    return results


def _normalize_whitespace(value: str | None) -> str:
    if not value:
        return ""
    collapsed = re.sub(r"\s+", " ", value)
    return collapsed.strip()


def _normalize_line_for_affiliation(value: str | None) -> str:
    text = _normalize_whitespace(value)
    if len(text) < 12 or len(text) > 180:
        return ""
    # Skip lines that look like section headings
    if text.endswith(":"):
        return ""
    # Prefer lines containing commas or conjunctions (common in affiliations)
    if "," not in text and " and " not in text.lower():
        return text if any(keyword in text.lower() for keyword in AFFILIATION_KEYWORDS) else ""
    return text


def _normalize_affiliation(value: str | None) -> str:
    text = _normalize_whitespace(value)
    return text.lower()


def _contains_affiliation_keyword(value: str) -> bool:
    lowered = value.lower()
    return any(keyword in lowered for keyword in AFFILIATION_KEYWORDS)



