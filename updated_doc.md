# AI Editor Feature Implementation Document

## Overview

This document captures the updated backend implementation for a Jenni-style AI academic writing system in Citeflow. The implementation focuses on the editor-facing backend and service contracts that a React + EditorJS frontend can integrate with.

The system now supports:

- AI writing suggestions over WebSocket
- Citation metadata persistence after suggestion acceptance
- Automatic references generation
- Citation style switching
- AI chat with separated `assistantMessage` and `documentPayload`
- Research library ingestion and search
- AI inline editing endpoints
- Durable document state and backend metrics

## Backend Scope

The implementation in this repository is backend-only. The current repo still does not contain the production React/EditorJS editor application, so the work here provides the APIs and persistence model that the frontend can consume.

Core service entry point:

- `Recommendation Agent/main.py`

Core backend modules added:

- `Recommendation Agent/utils/models.py`
- `Recommendation Agent/utils/store.py`
- `Recommendation Agent/utils/suggestion_service.py`
- `Recommendation Agent/utils/reference_formatter.py`
- `Recommendation Agent/utils/chat_service.py`
- `Recommendation Agent/utils/edit_service.py`
- `Recommendation Agent/utils/library_service.py`
- `Recommendation Agent/utils/ai_ops.py`
- `Recommendation Agent/utils/metrics.py`

Supporting updates:

- `Recommendation Agent/utils/qdrant_ops.py`
- `Recommendation Agent/requirements.txt`
- `docker-compose.yml`
- `env.template`
- `Recommendation Agent/tests/test_backend_api.py`

## Implemented Features

### 1. AI Writing Suggestions

The existing WebSocket suggestion pipeline has been retained and upgraded.

WebSocket endpoint:

- `ws://<host>:8000/suggest/{document_id}`

Supported request shape:

```json
{
  "documentId": "doc-123",
  "blockId": "paragraph-1",
  "paragraphText": "Climate change affects ecosystems worldwide.",
  "title": "Climate Change and Global Impact",
  "heading": "Introduction",
  "cursorContext": "Previous sentence context",
  "citationStyle": "APA",
  "retryOfSuggestionId": null
}
```

Legacy compatibility is preserved for:

```json
{
  "title": "Climate Change and Global Impact",
  "heading": "Introduction",
  "content": "Climate change affects ecosystems worldwide."
}
```

Updated response shape:

```json
{
  "suggestionId": "sg_xxx",
  "text": "Climate change affects ecosystems worldwide.",
  "citations": [],
  "status": "generated",
  "dedupeKey": "hash",
  "latencyMs": 1200,
  "debug": {
    "path": "research"
  }
}
```

Implemented improvements:

- Durable suggestion records
- Duplicate suppression through a backend dedupe key
- Retry support through `retryOfSuggestionId`
- Backward-compatible WebSocket payload handling
- Suggestion acceptance endpoint for persisting accepted content

Accept endpoint:

- `POST /suggestions/accept`

Example request:

```json
{
  "suggestionId": "sg_xxx",
  "documentId": "doc-123",
  "blockId": "paragraph-1",
  "acceptedText": "Climate change affects ecosystems worldwide."
}
```

### 2. Citation System

Accepted suggestions now persist citation metadata against documents and blocks.

Implemented citation behavior:

- Citation records are stored after suggestion acceptance
- Citations are deduplicated using DOI, normalized URL, or metadata fingerprint
- Block-to-citation relationships are preserved
- Citation style can be changed per document

Inline citation label rules implemented in the backend:

1. Use API `inText` if present
2. Otherwise use authors + year
3. Otherwise use publication + year
4. Otherwise use publication name
5. Never use article title as the citation label

Supported citation styles:

- APA
- MLA
- Harvard
- Chicago
- IEEE

### 3. References Section

The backend now computes a non-editable references payload for a document whenever citations exist.

References endpoint:

- `GET /documents/{document_id}/references`

Example response:

```json
{
  "documentId": "doc-123",
  "citationStyle": "APA",
  "visible": true,
  "editable": false,
  "entries": [
    {
      "citationKey": "doi:10.1000/testdoi",
      "label": "Amin et al., 2025",
      "text": "Amin, Md Ruhul (2025). Climate Change and Global Impact. Journal of Climate Research. https://doi.org/10.1000/testdoi",
      "style": "APA",
      "sourceTitle": "Climate Change and Global Impact"
    }
  ]
}
```

References behavior:

- Automatically visible when citations exist
- Automatically updated after suggestion acceptance
- Non-editable by default
- Intended to be rendered by the frontend at the end of the document

### 4. Citation Style Switching

Style updates are now handled by the backend.

Endpoint:

- `PATCH /documents/{document_id}/citation-style`

Example request:

```json
{
  "citationStyle": "IEEE"
}
```

Behavior:

- Updates stored document style
- Recomputes reference entries
- Updates inline citation labels where style requires it

### 5. AI Chat Backend

The backend now supports multi-turn document chat.

Endpoints:

- `POST /documents/{document_id}/chat/messages`
- `GET /documents/{document_id}/chat/threads/{thread_id}`

Implemented capabilities:

- Document-aware prompts
- Selected text support
- Optional web search context
- Optional library source context
- Persistent chat threads and messages

Important Add-to-Document behavior:

Responses separate explanatory UI text from generated document content:

```json
{
  "threadId": "thread_xxx",
  "assistantMessage": "I improved the paragraph and made the tone more academic.",
  "documentPayload": "Climate change affects ecosystems worldwide by altering biodiversity, migration patterns, and long-term habitat stability.",
  "citations": [],
  "contextSummary": {
    "document": true,
    "webResultCount": 0,
    "libraryResultCount": 0,
    "referenceCount": 1
  }
}
```

The frontend should insert only `documentPayload` when the user chooses Add to Document.

### 6. AI Edit Tools

Inline editing APIs are now available.

Endpoint:

- `POST /documents/{document_id}/edits`

Example request:

```json
{
  "selectionText": "This is a claim.",
  "instruction": "Convert to bullet list",
  "mode": "insert_below",
  "contextWindow": "This is a claim.",
  "targetBlockId": "paragraph-1"
}
```

Example response:

```json
{
  "explanation": "I generated an alternate version that can be inserted below the selected text.",
  "replacementText": "",
  "insertBelowText": "- This is a claim.",
  "regenerateToken": "edit_xxx",
  "citations": []
}
```

Supported response structure enables:

- Replace Selection
- Insert Below
- Try Again
- Discard

### 7. Research Library Integration

The backend now supports initial library ingestion and search.

Endpoints:

- `POST /library/import/doi`
- `POST /library/import/bibtex`
- `POST /library/import/pdf`
- `POST /library/search`

Implemented ingestion types:

- DOI
- BibTeX
- PDF upload

Current design supports future extension to:

- Zotero
- Mendeley

Implemented library behavior:

- Canonical source persistence
- Duplicate detection
- Qdrant indexing for searchable source text
- Document-scoped and global source search

## Persistence Model

The backend now uses a durable SQLite store for editor state.

Default database path:

- `CITEFLOW_DB_PATH=/data/citeflow.db`

Stored data includes:

- Documents
- Suggestions
- Accepted block content
- Document citations
- Block citation links
- Chat threads
- Chat messages
- Library sources

This replaces the old fully ephemeral session model for the new editor features.

## Metrics and Operational Updates

Metrics endpoint:

- `GET /metrics`

Currently tracked counters include activity for:

- generated suggestions
- duplicate suggestion hits
- accepted suggestions
- chat messages
- edit generations
- library imports

Infrastructure updates made:

- Persistent volume for recommendation-agent data in `docker-compose.yml`
- New env vars in `env.template`
- New Python dependencies for PDF upload, BibTeX parsing, multipart handling, and tests

## Testing

Backend API coverage has been added in:

- `Recommendation Agent/tests/test_backend_api.py`

Covered scenarios:

- Suggestion acceptance attaches citations and generates references
- Citation style switching updates reference output
- Legacy WebSocket payload compatibility
- Chat returns separate `assistantMessage` and `documentPayload`
- Edit endpoint supports insert-below behavior
- BibTeX duplicate detection
- DOI duplicate detection

Test command:

```bash
python3 -m pytest tests/test_backend_api.py
```

## Current Limitations

This implementation is intentionally backend-scoped and leaves some frontend-dependent behavior for the editor app:

- Inline citation rendering inside EditorJS is not implemented in this repo
- Automatic insertion of a non-editable References block is not implemented in this repo
- Suggestion accept UI and edit preview UI remain frontend responsibilities
- Zotero and Mendeley connectors are planned but not yet implemented
- SQLite is used for portability inside this repo, though a future Postgres migration would be preferable for production scale

## Recommended Frontend Integration

The frontend should integrate with these backend APIs in this order:

1. Use `/suggest/{document_id}` for suggestion generation
2. Call `/suggestions/accept` when the user accepts a suggestion
3. Fetch `/documents/{document_id}/manifest` to render current blocks and references state
4. Fetch `/documents/{document_id}/references` for explicit references rendering
5. Use `/documents/{document_id}/citation-style` for citation dropdown changes
6. Use `/documents/{document_id}/chat/messages` for the AI chat panel
7. Use `/documents/{document_id}/edits` for inline edit tools
8. Use `/library/import/*` and `/library/search` for research source workflows

## Summary

The backend now provides the core service layer required for a Jenni-style academic writing experience:

- suggestion generation
- citation-aware acceptance flows
- style-aware references
- context-aware chat
- inline edit generation
- research library ingestion
- durable state

This gives the editor frontend a stable API surface for implementing the full UX described in the original feature document.
