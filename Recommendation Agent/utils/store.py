from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self):
        db_path = os.getenv("CITEFLOW_DB_PATH", "./data/citeflow.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialized = False

    @contextmanager
    def connection(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def initialize(self):
        if self._initialized:
            return
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT DEFAULT '',
                    citation_style TEXT DEFAULT 'APA',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    paragraph_text TEXT DEFAULT '',
                    title TEXT DEFAULT '',
                    heading TEXT DEFAULT '',
                    cursor_context TEXT DEFAULT '',
                    citation_style TEXT DEFAULT 'APA',
                    retry_of_suggestion_id TEXT,
                    dedupe_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text TEXT DEFAULT '',
                    response_json TEXT DEFAULT '{}',
                    citations_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    accepted_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_suggestions_doc_dedupe
                    ON suggestions(document_id, dedupe_key, created_at DESC);

                CREATE TABLE IF NOT EXISTS document_blocks (
                    document_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    content TEXT DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, block_id)
                );

                CREATE TABLE IF NOT EXISTS document_citations (
                    document_id TEXT NOT NULL,
                    citation_key TEXT NOT NULL,
                    citation_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, citation_key)
                );

                CREATE TABLE IF NOT EXISTS block_citations (
                    document_id TEXT NOT NULL,
                    block_id TEXT NOT NULL,
                    citation_key TEXT NOT NULL,
                    suggestion_id TEXT NOT NULL,
                    occurrence_order INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (document_id, block_id, citation_key)
                );

                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message_text TEXT NOT NULL,
                    document_payload TEXT DEFAULT '',
                    context_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS library_sources (
                    source_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    source_type TEXT NOT NULL,
                    title TEXT DEFAULT '',
                    source_key TEXT NOT NULL UNIQUE,
                    citation_json TEXT DEFAULT '{}',
                    content_text TEXT DEFAULT '',
                    metadata_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
        self._initialized = True

    def ensure_document(self, document_id: str, title: str = "", citation_style: str = "APA"):
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, title, citation_style, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = CASE WHEN excluded.title != '' THEN excluded.title ELSE documents.title END,
                    citation_style = COALESCE(excluded.citation_style, documents.citation_style),
                    updated_at = excluded.updated_at
                """,
                (document_id, title, citation_style, now, now),
            )

    def get_document(self, document_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def update_citation_style(self, document_id: str, citation_style: str):
        self.ensure_document(document_id, citation_style=citation_style)
        with self.connection() as conn:
            conn.execute(
                "UPDATE documents SET citation_style = ?, updated_at = ? WHERE document_id = ?",
                (citation_style, utc_now(), document_id),
            )

    def save_suggestion(self, payload: dict[str, Any]):
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO suggestions (
                    suggestion_id, document_id, block_id, paragraph_text, title, heading,
                    cursor_context, citation_style, retry_of_suggestion_id, dedupe_key, status,
                    text, response_json, citations_json, created_at, accepted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["suggestion_id"],
                    payload["document_id"],
                    payload["block_id"],
                    payload.get("paragraph_text", ""),
                    payload.get("title", ""),
                    payload.get("heading", ""),
                    payload.get("cursor_context", ""),
                    payload.get("citation_style", "APA"),
                    payload.get("retry_of_suggestion_id"),
                    payload["dedupe_key"],
                    payload["status"],
                    payload.get("text", ""),
                    json.dumps(payload.get("response_json", {})),
                    json.dumps(payload.get("citations", [])),
                    payload.get("created_at", utc_now()),
                    payload.get("accepted_at"),
                ),
            )

    def get_suggestion(self, suggestion_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM suggestions WHERE suggestion_id = ?",
                (suggestion_id,),
            ).fetchone()
        return self._decode_suggestion_row(row)

    def get_latest_suggestion_by_dedupe(self, document_id: str, dedupe_key: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM suggestions
                WHERE document_id = ? AND dedupe_key = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (document_id, dedupe_key),
            ).fetchone()
        return self._decode_suggestion_row(row)

    def mark_suggestion_accepted(self, suggestion_id: str):
        with self.connection() as conn:
            conn.execute(
                "UPDATE suggestions SET status = 'accepted', accepted_at = ? WHERE suggestion_id = ?",
                (utc_now(), suggestion_id),
            )

    def upsert_block(self, document_id: str, block_id: str, content: str):
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO document_blocks (document_id, block_id, content, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id, block_id) DO UPDATE SET
                    content = excluded.content,
                    updated_at = excluded.updated_at
                """,
                (document_id, block_id, content, utc_now()),
            )

    def get_blocks(self, document_id: str) -> dict[str, str]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT block_id, content FROM document_blocks WHERE document_id = ? ORDER BY block_id",
                (document_id,),
            ).fetchall()
        return {row["block_id"]: row["content"] for row in rows}

    def upsert_document_citation(self, document_id: str, citation_key: str, citation: dict[str, Any]):
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO document_citations (document_id, citation_key, citation_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_id, citation_key) DO UPDATE SET
                    citation_json = excluded.citation_json,
                    updated_at = excluded.updated_at
                """,
                (document_id, citation_key, json.dumps(citation), now, now),
            )

    def link_block_citation(
        self,
        document_id: str,
        block_id: str,
        citation_key: str,
        suggestion_id: str,
        occurrence_order: int,
    ):
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO block_citations (document_id, block_id, citation_key, suggestion_id, occurrence_order, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id, block_id, citation_key) DO UPDATE SET
                    suggestion_id = excluded.suggestion_id,
                    occurrence_order = excluded.occurrence_order
                """,
                (document_id, block_id, citation_key, suggestion_id, occurrence_order, utc_now()),
            )

    def get_document_citations(self, document_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT dc.citation_key, dc.citation_json, COALESCE(MIN(bc.occurrence_order), 0) AS sort_order
                FROM document_citations dc
                LEFT JOIN block_citations bc
                  ON bc.document_id = dc.document_id AND bc.citation_key = dc.citation_key
                WHERE dc.document_id = ?
                GROUP BY dc.citation_key, dc.citation_json
                ORDER BY sort_order ASC, dc.created_at ASC
                """,
                (document_id,),
            ).fetchall()

        citations = []
        for row in rows:
            citation = json.loads(row["citation_json"])
            citation["_citationKey"] = row["citation_key"]
            citations.append(citation)
        return citations

    def create_chat_thread(self, thread_id: str, document_id: str, title: str = ""):
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_threads (thread_id, document_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (thread_id, document_id, title, now, now),
            )

    def touch_chat_thread(self, thread_id: str):
        with self.connection() as conn:
            conn.execute(
                "UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?",
                (utc_now(), thread_id),
            )

    def get_chat_thread(self, thread_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM chat_threads WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
        return dict(row) if row else None

    def add_chat_message(
        self,
        message_id: str,
        thread_id: str,
        role: str,
        message_text: str,
        document_payload: str = "",
        context: Optional[dict[str, Any]] = None,
    ):
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (message_id, thread_id, role, message_text, document_payload, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    thread_id,
                    role,
                    message_text,
                    document_payload,
                    json.dumps(context or {}),
                    utc_now(),
                ),
            )
        self.touch_chat_thread(thread_id)

    def get_chat_messages(self, thread_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            ).fetchall()

        messages = []
        for row in rows:
            record = dict(row)
            record["context_json"] = json.loads(record.get("context_json") or "{}")
            messages.append(record)
        return messages

    def save_library_source(
        self,
        source_id: str,
        document_id: Optional[str],
        source_type: str,
        title: str,
        source_key: str,
        citation: dict[str, Any],
        content_text: str,
        metadata: Optional[dict[str, Any]] = None,
    ):
        now = utc_now()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO library_sources (
                    source_id, document_id, source_type, title, source_key,
                    citation_json, content_text, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_id,
                    document_id,
                    source_type,
                    title,
                    source_key,
                    json.dumps(citation),
                    content_text,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )

    def get_library_source_by_key(self, source_key: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM library_sources WHERE source_key = ?",
                (source_key,),
            ).fetchone()
        return self._decode_library_row(row)

    def get_library_source(self, source_id: str) -> Optional[dict[str, Any]]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM library_sources WHERE source_id = ?",
                (source_id,),
            ).fetchone()
        return self._decode_library_row(row)

    def list_library_sources(self, document_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self.connection() as conn:
            if document_id:
                rows = conn.execute(
                    "SELECT * FROM library_sources WHERE document_id = ? ORDER BY created_at DESC",
                    (document_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM library_sources ORDER BY created_at DESC",
                ).fetchall()
        return [self._decode_library_row(row) for row in rows]

    def _decode_suggestion_row(self, row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if not row:
            return None
        record = dict(row)
        record["response_json"] = json.loads(record.get("response_json") or "{}")
        record["citations_json"] = json.loads(record.get("citations_json") or "[]")
        return record

    def _decode_library_row(self, row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if not row:
            return None
        record = dict(row)
        record["citation_json"] = json.loads(record.get("citation_json") or "{}")
        record["metadata_json"] = json.loads(record.get("metadata_json") or "{}")
        return record


store = SQLiteStore()
