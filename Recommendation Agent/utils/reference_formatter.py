from __future__ import annotations

import hashlib
from typing import Optional
from urllib.parse import urlparse

from utils.models import CitationRecord, CitationStyle, ReferenceEntry


def normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower().replace("www.", "")
    return f"{scheme}://{netloc}{parsed.path}".rstrip("/")


def build_citation_key(citation: dict) -> str:
    identifiers = citation.get("identifiers", {}) or {}
    doi = (identifiers.get("doi") or "").strip().lower()
    if doi:
        return f"doi:{doi}"

    url = normalize_url(identifiers.get("url", ""))
    if url:
        return f"url:{url}"

    authors = citation.get("authors", []) or []
    first_author = authors[0].get("family", "").strip().lower() if authors else "unknown"
    year = citation.get("year") or "n.d."
    publication = (citation.get("publication") or "").strip().lower()
    fingerprint = hashlib.sha1(f"{first_author}|{year}|{publication}".encode("utf-8")).hexdigest()
    return f"meta:{fingerprint}"


def _author_families(citation: dict) -> list[str]:
    return [
        author.get("family", "").strip()
        for author in (citation.get("authors", []) or [])
        if author.get("family", "").strip()
    ]


def _full_author_names(citation: dict) -> list[str]:
    names = []
    for author in citation.get("authors", []) or []:
        given = author.get("given", "").strip()
        family = author.get("family", "").strip()
        if given and family:
            names.append(f"{family}, {given}")
        elif family:
            names.append(family)
    return names


def _default_inline_label(citation: dict) -> str:
    author_families = _author_families(citation)
    year = citation.get("year")
    year_label = str(year) if year else "n.d."
    publication = (citation.get("publication") or "").strip()

    if author_families:
        if len(author_families) == 1:
            return f"{author_families[0]}, {year_label}"
        if len(author_families) == 2:
            return f"{author_families[0]} & {author_families[1]}, {year_label}"
        return f"{author_families[0]} et al., {year_label}"

    if publication and year:
        return f"{publication}, {year}"
    if publication:
        return publication
    return f"Unknown, {year_label}"


def _coerce_style(style: CitationStyle | str) -> CitationStyle:
    if isinstance(style, CitationStyle):
        return style
    try:
        return CitationStyle(style)
    except ValueError:
        return CitationStyle.APA


def format_inline_citation(citation: dict, style: CitationStyle | str, index: int | None = None) -> str:
    style = _coerce_style(style)
    if style == CitationStyle.IEEE:
        number = index or 1
        return f"[{number}]"

    api_in_text = (citation.get("inText") or "").strip()
    if api_in_text:
        return api_in_text

    return _default_inline_label(citation)


def _join_names(names: list[str]) -> str:
    if not names:
        return "Unknown"
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _url_or_doi(citation: dict) -> str:
    identifiers = citation.get("identifiers", {}) or {}
    doi = (identifiers.get("doi") or "").strip()
    if doi:
        return f"https://doi.org/{doi}"
    return (identifiers.get("url") or "").strip()


def format_reference_text(citation: dict, style: CitationStyle | str, index: int | None = None) -> str:
    style = _coerce_style(style)
    names = _full_author_names(citation)
    authors_label = _join_names(names)
    year = citation.get("year") or "n.d."
    title = (citation.get("title") or "").strip() or "Untitled source"
    publication = (citation.get("publication") or "").strip()
    locator = _url_or_doi(citation)

    if style == CitationStyle.APA:
        return f"{authors_label} ({year}). {title}. {publication}. {locator}".strip()
    if style == CitationStyle.MLA:
        return f'{authors_label}. "{title}." {publication}, {year}, {locator}'.strip()
    if style == CitationStyle.HARVARD:
        return f"{authors_label} ({year}) {title}. {publication}. Available at: {locator}".strip()
    if style == CitationStyle.CHICAGO:
        return f'{authors_label}. {year}. "{title}." {publication}. {locator}'.strip()
    if style == CitationStyle.IEEE:
        label = f"[{index or 1}]"
        return f'{label} {authors_label}, "{title}," {publication}, {year}. {locator}'.strip()
    return f"{authors_label} ({year}). {title}. {publication}. {locator}".strip()


def format_references(citations: list[dict], style: CitationStyle | str) -> list[ReferenceEntry]:
    style = _coerce_style(style)
    entries = []
    for idx, citation in enumerate(citations, start=1):
        key = citation.get("_citationKey") or build_citation_key(citation)
        entries.append(
            ReferenceEntry(
                citationKey=key,
                label=format_inline_citation(citation, style, idx),
                text=format_reference_text(citation, style, idx),
                style=style,
                sourceTitle=citation.get("title", ""),
            )
        )
    return entries


def coerce_citation_record(citation: dict) -> CitationRecord:
    return CitationRecord.model_validate(citation)
