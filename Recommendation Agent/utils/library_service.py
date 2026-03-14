from __future__ import annotations

import hashlib
import io
import uuid
from typing import Optional

import bibtexparser
from pypdf import PdfReader

from utils.citation_metadata import enrich_single_citation, fetch_crossref_metadata
from utils.metrics import metrics
from utils.models import (
    BibtexImportRequest,
    CitationIdentifiers,
    CitationRecord,
    DOIImportRequest,
    LibrarySearchHit,
    LibrarySearchRequest,
    LibrarySearchResponse,
    LibrarySourceResponse,
)
from utils.qdrant_ops import ensure_collection, search_qdrant, store_documents
from utils.store import store


def _collection_name(document_id: Optional[str]) -> str:
    return f"library_{document_id or 'global'}"


def _chunk_text(text: str, chunk_size: int = 1200) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = []
    buffer = []
    length = 0
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        buffer.append(paragraph)
        length += len(paragraph)
        if length >= chunk_size:
            chunks.append("\n".join(buffer))
            buffer = []
            length = 0
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks


async def _index_source(source_id: str, document_id: Optional[str], title: str, content_text: str):
    chunks = _chunk_text(content_text)
    if not chunks:
        return
    collection_name = _collection_name(document_id)
    ensure_collection(collection_name)
    await store_documents(
        collection_name=collection_name,
        texts=chunks,
        urls=[f"library://{source_id}"] * len(chunks),
        metadata=[
            {
                "source_id": source_id,
                "title": title,
                "document_id": document_id or "",
                "source": f"library://{source_id}",
            }
            for _ in chunks
        ],
    )


async def import_doi(payload: DOIImportRequest) -> LibrarySourceResponse:
    normalized = payload.doi.strip().lower()
    source_key = f"doi:{normalized}"
    existing = store.get_library_source_by_key(source_key)
    if existing:
        return LibrarySourceResponse(
            sourceId=existing["source_id"],
            documentId=existing.get("document_id"),
            sourceType=existing["source_type"],
            title=existing["title"],
            citation=CitationRecord.model_validate(existing["citation_json"]),
            duplicate=True,
        )

    citation = await enrich_single_citation(f"https://doi.org/{normalized}", "cite_1")
    source_id = f"src_{uuid.uuid4().hex}"
    source_title = citation.get("title", normalized)
    metadata = await fetch_crossref_metadata(normalized) or {}
    content_text = metadata.get("abstract", "") or source_title

    store.save_library_source(
        source_id=source_id,
        document_id=payload.documentId,
        source_type="doi",
        title=source_title,
        source_key=source_key,
        citation=citation,
        content_text=content_text,
        metadata={"doi": normalized},
    )
    await _index_source(source_id, payload.documentId, source_title, content_text)
    metrics.increment("library.imports.doi")
    return LibrarySourceResponse(
        sourceId=source_id,
        documentId=payload.documentId,
        sourceType="doi",
        title=source_title,
        citation=CitationRecord.model_validate(citation),
        duplicate=False,
    )


async def import_bibtex(payload: BibtexImportRequest) -> list[LibrarySourceResponse]:
    library = bibtexparser.parse_string(payload.bibtex)
    responses = []
    for entry in library.entries:
        title = str(entry.get("title", "Untitled source"))
        doi = str(entry.get("doi", "")).strip()
        url = str(entry.get("url", "")).strip()
        source_key = f"bibtex:{doi or url or hashlib.sha1(title.encode('utf-8')).hexdigest()}"
        existing = store.get_library_source_by_key(source_key)
        if existing:
            responses.append(
                LibrarySourceResponse(
                    sourceId=existing["source_id"],
                    documentId=existing.get("document_id"),
                    sourceType=existing["source_type"],
                    title=existing["title"],
                    citation=CitationRecord.model_validate(existing["citation_json"]),
                    duplicate=True,
                )
            )
            continue

        authors = []
        raw_author = str(entry.get("author", ""))
        for author in [part.strip() for part in raw_author.split(" and ") if part.strip()]:
            if "," in author:
                family, given = [value.strip() for value in author.split(",", 1)]
            else:
                parts = author.rsplit(" ", 1)
                given = parts[0] if len(parts) > 1 else ""
                family = parts[-1]
            authors.append({"family": family, "given": given})

        citation = CitationRecord(
            id="cite_1",
            inText="",
            type="Article",
            articleType="Journal",
            title=title,
            publication=str(entry.get("journal", "") or entry.get("booktitle", "")),
            year=int(entry["year"]) if str(entry.get("year", "")).isdigit() else None,
            authors=authors,
            identifiers=CitationIdentifiers(doi=doi, url=url),
        )
        source_id = f"src_{uuid.uuid4().hex}"
        content_text = str(entry.get("abstract", "") or title)
        store.save_library_source(
            source_id=source_id,
            document_id=payload.documentId,
            source_type="bibtex",
            title=title,
            source_key=source_key,
            citation=citation.model_dump(mode="json"),
            content_text=content_text,
            metadata={"entryType": str(entry.get("ENTRYTYPE", ""))},
        )
        await _index_source(source_id, payload.documentId, title, content_text)
        responses.append(
            LibrarySourceResponse(
                sourceId=source_id,
                documentId=payload.documentId,
                sourceType="bibtex",
                title=title,
                citation=citation,
                duplicate=False,
            )
        )
    metrics.increment("library.imports.bibtex", len(responses))
    return responses


async def import_pdf_bytes(filename: str, data: bytes, document_id: Optional[str] = None) -> LibrarySourceResponse:
    source_key = f"pdf:{hashlib.sha1(data).hexdigest()}"
    existing = store.get_library_source_by_key(source_key)
    if existing:
        return LibrarySourceResponse(
            sourceId=existing["source_id"],
            documentId=existing.get("document_id"),
            sourceType=existing["source_type"],
            title=existing["title"],
            citation=CitationRecord.model_validate(existing["citation_json"]),
            duplicate=True,
        )

    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    title = filename.rsplit(".", 1)[0]
    citation = CitationRecord(
        id="cite_1",
        inText="",
        type="Report",
        articleType="",
        title=title,
        publication="Uploaded PDF",
        identifiers=CitationIdentifiers(url=f"file://{filename}"),
    )
    source_id = f"src_{uuid.uuid4().hex}"
    store.save_library_source(
        source_id=source_id,
        document_id=document_id,
        source_type="pdf",
        title=title,
        source_key=source_key,
        citation=citation.model_dump(mode="json"),
        content_text=text,
        metadata={"filename": filename},
    )
    await _index_source(source_id, document_id, title, text)
    metrics.increment("library.imports.pdf")
    return LibrarySourceResponse(
        sourceId=source_id,
        documentId=document_id,
        sourceType="pdf",
        title=title,
        citation=citation,
        duplicate=False,
    )


async def search_library(payload: LibrarySearchRequest) -> LibrarySearchResponse:
    hits = []
    collection_names = [_collection_name(payload.documentId)]
    if payload.documentId:
        collection_names.append(_collection_name(None))

    seen_sources = set()
    for collection_name in collection_names:
        results = await search_qdrant(collection_name, payload.query, top_k=payload.topK)
        for item in results:
            source_id = item.get("source_id") or ""
            if source_id in seen_sources:
                continue
            seen_sources.add(source_id)
            source = store.get_library_source(source_id) if source_id else None
            citation = None
            if source:
                citation = CitationRecord.model_validate(source["citation_json"])
            hits.append(
                LibrarySearchHit(
                    sourceId=source_id,
                    title=(source or {}).get("title", item.get("source", "")),
                    score=item.get("score", 0.0),
                    snippet=item.get("text", "")[:280],
                    citation=citation,
                )
            )
            if len(hits) >= payload.topK:
                break
        if len(hits) >= payload.topK:
            break
    return LibrarySearchResponse(query=payload.query, hits=hits)
